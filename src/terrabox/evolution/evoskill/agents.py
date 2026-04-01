"""EvoSkill three-agent architecture: BaseAgent, ProposerAgent, SkillBuilderAgent.

From: EvoSkill — Automated Skill Discovery for Multi-Agent Systems (arXiv 2603.02766)

Three-agent loop:
  1. BaseAgent executes task with current skill set → Trajectory
  2. ProposerAgent analyzes failures → FailureAnalysis JSON
  3. SkillBuilderAgent generates SkillModule from failure analysis
"""
from __future__ import annotations

import logging
from typing import Optional

from ..shared.llm_client import EvolutionLLMClient
from ..shared.trajectory import Trajectory, Turn
from .skill_module import SkillModule

logger = logging.getLogger(__name__)


class BaseAgent:
    """Executor: run a geospatial task using existing agent infrastructure.

    Wraps build_langchain_tools + create_react_agent directly.
    Does NOT touch existing graph.py.
    """

    def __init__(self, config=None):
        self._config = config
        self._llm = None
        self._tools = None

    def execute(
        self,
        question: str,
        images: list[str],
        system_prompt: str,
        expected_tools: Optional[list[str]] = None,
    ) -> Trajectory:
        """Run agent and return Trajectory."""
        from langchain_core.messages import HumanMessage
        from langgraph.prebuilt import create_react_agent
        from ..shared.mock_user import MockUser
        from ...agent.config import load_config
        from ...agent.llm import get_llm
        from ...agent.tools import build_langchain_tools

        if self._config is None:
            self._config = load_config()
        if self._llm is None:
            self._llm = get_llm(self._config)

        from ...extensions import load_builtin_toolkits
        load_builtin_toolkits()

        user = MockUser()
        tools = build_langchain_tools(user)

        content = question
        if images:
            content += f"\n\n[Images: {', '.join(images)}]"

        agent = create_react_agent(self._llm, tools, prompt=system_prompt)
        result = agent.invoke({"messages": [HumanMessage(content=content)]})

        tools_called = []
        turns: list[Turn] = [Turn(role="human", content=content)]
        for msg in result.get("messages", []):
            content_text = getattr(msg, "content", "")
            role = type(msg).__name__.lower().replace("message", "")

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    slug = tc.get("name", "").replace("__", ".")
                    if slug:
                        tools_called.append(slug)
                        turns.append(Turn(
                            role="tool",
                            content="",
                            tool_name=slug,
                            tool_args=tc.get("args", {}),
                        ))
            elif content_text:
                turns.append(Turn(role="assistant", content=content_text))

        final_answer = ""
        msgs = result.get("messages", [])
        if msgs and hasattr(msgs[-1], "content"):
            final_answer = msgs[-1].content

        return Trajectory(
            task_id="",
            question=question,
            images=images,
            turns=turns,
            tools_called=tools_called,
            expected_tools=expected_tools or [],
            final_answer=final_answer,
            success=bool(tools_called),
            source="evoskill_eval",
        )


class ProposerAgent:
    """Analyzer: identify failure modes in a failed trajectory.

    Returns structured FailureAnalysis dict:
    {
        failure_mode: "wrong_tool" | "missing_capability" | "bad_args" | "hallucination",
        failed_at: "tool_slug or step description",
        needed_capability: "description",
        suggested_tool_sequence: ["slug1", "slug2"],
    }
    """

    def __init__(self, llm_client: EvolutionLLMClient):
        self._llm = llm_client

    def analyze(self, trajectory: Trajectory) -> Optional[dict]:
        """Return structured failure analysis, or None if LLM call fails."""
        error_tools = [t for t in trajectory.turns if t.is_error and t.tool_name]
        errors = [t.tool_result or "" for t in error_tools[:3]]

        analysis = self._llm.analyze_failure(
            question=trajectory.question,
            tools_called=trajectory.tools_called,
            expected_tools=trajectory.expected_tools,
            errors=errors,
        )

        if analysis is None:
            # Fallback: simple heuristic analysis
            missing = set(trajectory.expected_tools) - set(trajectory.tools_called)
            extra = set(trajectory.tools_called) - set(trajectory.expected_tools)
            return {
                "failure_mode": "wrong_tool" if extra else "missing_capability",
                "failed_at": list(extra)[0] if extra else "unknown",
                "needed_capability": f"Use {list(missing)} instead" if missing else "Better tool selection",
                "suggested_tool_sequence": trajectory.expected_tools,
            }
        return analysis


class SkillBuilderAgent:
    """Generator: create a SkillModule from a failure analysis.

    Uses the LLM to synthesize a structured skill definition.
    """

    def __init__(self, llm_client: EvolutionLLMClient):
        self._llm = llm_client

    def build(
        self,
        failure_analysis: dict,
        trajectory: Trajectory,
        available_tools: Optional[list[str]] = None,
    ) -> Optional[SkillModule]:
        """Return a new SkillModule, or None if generation fails."""
        import uuid

        if available_tools is None:
            available_tools = self._get_available_tools()

        trajectory_summary = (
            f"Question: {trajectory.question[:200]}\n"
            f"Tools called: {trajectory.tools_called}\n"
            f"Expected: {trajectory.expected_tools}"
        )

        raw = self._llm.build_skill_module(
            needed_capability=failure_analysis.get("needed_capability", ""),
            available_tools=available_tools,
            trajectory_summary=trajectory_summary,
        )

        if raw is None:
            # Fallback: build from failure analysis directly
            raw = {
                "name": f"skill_{trajectory.task_type}",
                "description": failure_analysis.get("needed_capability", ""),
                "trigger_condition": f"When {failure_analysis.get('failure_mode', 'unknown')} occurs",
                "tool_sequence": failure_analysis.get("suggested_tool_sequence", []),
                "parameter_hints": {},
                "preconditions": [],
                "domain": self._infer_domain(failure_analysis.get("suggested_tool_sequence", [])),
            }

        skill = SkillModule(
            id=str(uuid.uuid4()),
            name=raw.get("name", "unnamed_skill"),
            description=raw.get("description", ""),
            trigger_condition=raw.get("trigger_condition", ""),
            tool_sequence=raw.get("tool_sequence", []),
            parameter_hints=raw.get("parameter_hints", {}),
            preconditions=raw.get("preconditions", []),
            domain=raw.get("domain", "multi"),
        )
        return skill

    def _get_available_tools(self) -> list[str]:
        try:
            from ...core.registry import registry
            return [spec.slug for spec in registry.list_tools()]
        except Exception:
            return []

    def _infer_domain(self, tool_sequence: list[str]) -> str:
        seq = " ".join(tool_sequence)
        if "geo_perception" in seq:
            return "geo_perception"
        if "osm_gis" in seq:
            return "spatial"
        if "georaster" in seq or "raster" in seq:
            return "raster"
        if "ipython_code" in seq:
            return "code"
        return "multi"
