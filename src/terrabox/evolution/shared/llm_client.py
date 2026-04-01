"""Thin LLM client for evolution distillation/analysis calls.

Reuses the running vLLM instance via the existing agent.llm.get_llm() factory.
Does NOT start a new process — shares the existing inference service.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EvolutionLLMClient:
    """Wrap the existing agent LLM for evolution-specific calls.

    All methods call the LLM synchronously (not streaming) and return strings.
    The LLM is lazily initialized on first use.
    """

    def __init__(self, config=None):
        """
        Args:
            config: AgentConfig instance; if None, loaded from agent_config.yaml.
        """
        self._config = config
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from ...agent.config import load_config
            from ...agent.llm import get_llm
            if self._config is None:
                self._config = load_config()
            self._llm = get_llm(self._config)
        return self._llm

    def call(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        """Single-turn LLM call; return the text response."""
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = self._get_llm()
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    def call_json(self, prompt: str, system: Optional[str] = None) -> Optional[dict]:
        """Call LLM and parse response as JSON; return None on failure."""
        raw = self.call(prompt, system=system)
        # Extract JSON from markdown code blocks if present
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
        # Try to find a JSON object in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
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
        return self.call(prompt, system=system, max_tokens=200)

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
        return self.call(prompt, system=system, max_tokens=200)

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
