"""PromptAugmenter ABC and formatting utilities shared across all methods."""
from __future__ import annotations

from abc import ABC, abstractmethod

# Base system prompt imported from existing agent session module
_REACT_SYSTEM_PROMPT = """You are a geospatial analysis assistant with access to the tools provided in the current run.

## Objective
Answer the user's request using the user's inputs, the current conversation, and concrete tool observations.

## Tool-use principles
- Choose tool calls based only on the user's request, the conversation state, and the tool schemas available to you.
- Do not assume a tool exists unless it appears in the available tool list.
- Do not invent tool outputs, file paths, measurements, counts, distances, areas, coordinates, or geospatial facts.
- **If a tool can compute, measure, detect, or look up something, you MUST call that tool to obtain the result — do NOT make it up or estimate it from your own reasoning.** Reasoning may plan the steps, but every reported value must come from an actual tool observation, not from your own calculation or guesswork.
- Use additional tool calls when the task still requires missing evidence, transformation, computation, or generated artifacts.
- If you decide that tool evidence is needed, make an actual tool call instead of only describing a hypothetical plan.
- If a tool returns an error, treat the error message as evidence. Retry only when changing the inputs or approach is justified by the conversation and tool schema.
- Avoid repeating the same tool call with the same arguments after it has failed.
- Only give a final answer after the required tool observations exist; do not short-circuit to an answer that a tool should have produced.
- When enough evidence is available, provide a clear final answer grounded in that evidence."""


def _try_import_base_prompt() -> str:
    """Try to import _REACT_SYSTEM_PROMPT from agent.session; fall back to local copy."""
    try:
        from ...agent.session import _REACT_SYSTEM_PROMPT as imported
        return imported
    except ImportError:
        return _REACT_SYSTEM_PROMPT


class PromptAugmenter(ABC):
    """Base class for all evolution prompt injectors.

    Each subclass implements augment() to return an augmented system prompt
    that can be passed as state_modifier to create_react_agent().
    """

    BASE_SYSTEM: str = _try_import_base_prompt()

    @abstractmethod
    def augment(self, user_query: str, **kwargs) -> str:
        """Return augmented system prompt for this query."""

    def record_outcome(self, reward: float = 0.5, **kwargs) -> None:
        """Update knowledge store after an episode (online learning).

        Default no-op. Subclasses that support online updates override this.
        Args:
            reward: Episode reward in [0, 1].
            **kwargs: May include query, tools_called, task_type.
        """

    def _format_skill_block(self, items: list[str], header: str) -> str:
        """Format a list of skills/insights into a labeled section."""
        if not items:
            return ""
        lines = [f"\n\n## {header}"]
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)

    def _format_memory_block(self, memories: list[dict]) -> str:
        """Format retrieved memory entries for injection."""
        if not memories:
            return ""
        lines = ["\n\n## Relevant Past Experiences (Memory)"]
        for i, mem in enumerate(memories, 1):
            intent = mem.get("intent", {})
            exp = mem.get("experience", {})
            utility = mem.get("utility", 0.0)
            task_type = intent.get("task_type", "unknown") if isinstance(intent, dict) else "unknown"
            tool_seq = exp.get("tool_sequence", []) if isinstance(exp, dict) else []
            insight = exp.get("key_insights", "") if isinstance(exp, dict) else ""

            lines.append(f"\nExperience {i} (utility={utility:.2f}, task: {task_type}):")
            if tool_seq:
                lines.append(f"  - Used tools: {' → '.join(tool_seq)}")
            if insight:
                lines.append(f"  - Key insight: {insight}")
        return "\n".join(lines)
