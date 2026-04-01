"""PromptAugmenter ABC and formatting utilities shared across all methods."""
from __future__ import annotations

from abc import ABC, abstractmethod

# Base system prompt imported from existing agent session module
_REACT_SYSTEM_PROMPT = """You are a geospatial analysis assistant with access to various tools for Earth observation data.

## When to STOP calling tools:
1. You have successfully answered the user's question
2. You have provided a complete analysis
3. A tool error indicates the task cannot be completed with available tools
4. The user's request is simple and does not require multiple tools

## Important rules:
- For image analysis tasks, ONE successful vlm_analyze call is usually sufficient
- Do NOT chain multiple perception tools unless explicitly asked
- If a tool fails, explain why and provide the best answer you can with available information
- Always provide a clear, final answer to the user's question
- Do NOT keep calling tools hoping for different results

## Tool selection guide:
- vlm_analyze: General image description and analysis (START HERE for image tasks)
- sam2_segment: Instance segmentation (only when you need object masks)
- remoteclip_analysis: Zero-shot classification (only when you need class labels)
- strip_rcnn_detect: Rotated object detection (only for specific object detection)
- remotesam_segment: Text-prompted segmentation (only when you have specific text prompts)

Remember: Quality over quantity. A single well-chosen tool is better than many unnecessary calls."""


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
