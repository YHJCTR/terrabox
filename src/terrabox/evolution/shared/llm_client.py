"""Thin LLM client for evolution distillation/analysis calls.

Calls the vLLM Docker service directly via OpenAI-compatible HTTP API.
Falls back to the agent LLM factory if no Docker service is detected.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove Qwen3 chain-of-thought blocks from LLM output."""
    text = _THINK_RE.sub("", text).strip()
    text = _THINK_UNCLOSED_RE.sub("", text).strip()
    return text

logger = logging.getLogger(__name__)

# Default vLLM service URL; override with EVOLUTION_LLM_URL env var
_DEFAULT_LLM_URL = "http://localhost:9100"


def _get_no_proxy_opener():
    """Create a urllib opener that bypasses proxy for localhost."""
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _call_vllm_api(base_url: str, messages: list[dict], max_tokens: int = 512,
                    enable_thinking: bool = True) -> str:
    """Call vLLM OpenAI-compatible /v1/chat/completions endpoint directly."""
    import urllib.request
    import urllib.error

    opener = _get_no_proxy_opener()

    url = base_url.rstrip("/") + "/v1/chat/completions"
    # Auto-detect model name
    try:
        with opener.open(base_url.rstrip("/") + "/v1/models", timeout=5) as r:
            models_data = json.loads(r.read())
        model = models_data["data"][0]["id"]
    except Exception:
        model = "default"

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body).encode()

    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=600) as r:
            result = json.loads(r.read())
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"vLLM call failed: {e}") from e


class EvolutionLLMClient:
    """LLM client for evolution-specific calls.

    Prioritizes the vLLM Docker service (set via EVOLUTION_LLM_URL env var),
    falling back to the agent LLM factory if Docker is unavailable.
    All methods call the LLM synchronously and return strings.
    """

    def __init__(self, config=None, llm_url: Optional[str] = None):
        self._config = config
        self._llm = None  # lazy-initialized fallback
        self._llm_url = llm_url or os.environ.get("EVOLUTION_LLM_URL", _DEFAULT_LLM_URL)
        self._use_docker = self._detect_docker_llm()

    def _detect_docker_llm(self) -> bool:
        """Check if the Docker vLLM service is reachable."""
        try:
            opener = _get_no_proxy_opener()
            with opener.open(self._llm_url + "/v1/models", timeout=3) as r:
                r.read()
            logger.info(f"EvolutionLLMClient: using Docker vLLM at {self._llm_url}")
            return True
        except Exception:
            logger.info(f"EvolutionLLMClient: Docker vLLM not available at {self._llm_url}, "
                        f"falling back to agent LLM")
            return False

    def _get_fallback_llm(self):
        if self._llm is None:
            from ...agent.config import load_config
            from ...agent.llm import get_llm
            if self._config is None:
                self._config = load_config()
            self._llm = get_llm(self._config)
        return self._llm

    def call(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512,
             enable_thinking: bool = True) -> str:
        """Single-turn LLM call; return the text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self._use_docker:
            try:
                return _strip_think(_call_vllm_api(self._llm_url, messages, max_tokens,
                                                   enable_thinking=enable_thinking))
            except Exception as e:
                logger.warning(f"Docker vLLM call failed: {e}, trying fallback")
                self._use_docker = False

        # Fallback: use agent LLM factory (requires langchain_core)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = self._get_fallback_llm()
            lc_messages = []
            if system:
                lc_messages.append(SystemMessage(content=system))
            lc_messages.append(HumanMessage(content=prompt))
            response = llm.invoke(lc_messages)
            return _strip_think(response.content.strip())
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    def call_json(self, prompt: str, system: Optional[str] = None,
                   max_tokens: int = 2048):
        """Call LLM and parse response as JSON; return None on failure.

        Returns dict, list, or None depending on the LLM response.
        """
        raw = self.call(prompt, system=system, max_tokens=max_tokens)
        # <think> already stripped by call(); extract JSON from response
        # Extract JSON from markdown code blocks if present
        match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
        # Try to find a JSON array or object in the response
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"Failed to parse JSON from LLM response: {raw[:200]}")
            return None

    def extract_skills_from_success(self, question: str, task_type: str,
                                     tools_used: list[str], label: str) -> str:
        """Extract a reusable geospatial strategy from a successful trajectory."""
        system = "You are a geospatial AI expert. Extract concise, reusable strategies."
        prompt = f"""Given this successful geospatial agent trajectory, extract a concise reusable strategy (1-3 sentences).

Task type: {task_type}
Question: {question}
Tools used in order: {' → '.join(tools_used)}
Final answer quality: good

Strategy (focus on tool selection principles, not specific values):"""
        return self.call(prompt, system=system, max_tokens=800)

    def extract_lesson_from_failure(self, question: str, task_type: str,
                                     tools_used: list[str], error_msg: str) -> str:
        """Extract a mistake pattern and lesson from a failed trajectory."""
        system = "You are a geospatial AI expert. Extract concise mistake patterns and lessons."
        prompt = f"""Given this failed geospatial agent trajectory, extract a concise mistake pattern and lesson (1-3 sentences).

Task type: {task_type}
Question: {question}
Tools called: {' → '.join(tools_used) if tools_used else 'none'}
Error: {error_msg[:300]}

Lesson (what to avoid and what to try instead):"""
        return self.call(prompt, system=system, max_tokens=800)

    def analyze_failure(self, question: str, tools_called: list[str],
                        expected_tools: list[str], errors: list[str]) -> Optional[dict]:
        """EvoSkill: Analyze failure and return structured FailureAnalysis JSON."""
        system = "You are a geospatial AI expert. Analyze agent failures precisely."
        prompt = f"""Analyze this failed geospatial agent trajectory.

Question: {question}
Tools called: {tools_called}
Errors encountered: {errors[:3] if errors else ['no explicit errors']}
Expected tools: {expected_tools}

Respond in JSON:
{{
    "failure_mode": "wrong_tool | missing_capability | bad_args | hallucination",
    "failed_at": "tool_slug or step description",
    "needed_capability": "description of what skill would fix this",
    "suggested_tool_sequence": ["slug1", "slug2"]
}}"""
        return self.call_json(prompt, system=system)

    def build_skill_module(self, needed_capability: str,
                           available_tools: list[str],
                           trajectory_summary: str) -> Optional[dict]:
        """EvoSkill: Create a structured SkillModule from a failure analysis."""
        system = "You are a geospatial AI expert. Create reusable skill modules."
        prompt = f"""Create a reusable geospatial skill module for the following need.

Needed capability: {needed_capability}
Available tools (slugs): {available_tools[:20]}
Failed trajectory summary: {trajectory_summary[:500]}

Respond in JSON:
{{
    "name": "skill_identifier",
    "description": "brief description",
    "trigger_condition": "when to apply this skill",
    "tool_sequence": ["slug1", "slug2"],
    "parameter_hints": {{"slug1": "hint about key parameters"}},
    "preconditions": ["condition1"],
    "domain": "geo_perception | spatial | raster | code | multi"
}}"""
        return self.call_json(prompt, system=system)

    def synthesize_task_variants(self, template: dict, n: int = 5) -> list[str]:
        """AgentEvolver: Generate N new task questions from a template."""
        system = "You are a geospatial AI expert. Generate diverse, realistic questions."
        prompt = f"""Generate {n} diverse geospatial analysis questions based on this template.

Template: {template.get('pattern', '')}
Task type: {template.get('task_type', '')}
Example tools needed: {template.get('tool_sequence', [])}

Output one question per line, no numbering:"""
        response = self.call(prompt, system=system, max_tokens=400)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[:n]

    def extract_key_insight(self, question: str, tool_sequence: list[str],
                            final_answer: str) -> str:
        """MemRL: Extract a key insight from a trajectory for memory storage."""
        system = "You are a geospatial AI expert. Extract brief, actionable insights."
        prompt = f"""Extract one key insight from this geospatial agent trajectory.

Question: {question[:300]}
Tools used: {' → '.join(tool_sequence)}
Answer quality: {'good' if final_answer else 'incomplete'}

Key insight (one sentence, actionable for future similar tasks):"""
        return self.call(prompt, system=system, max_tokens=150)
