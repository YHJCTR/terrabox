"""Prompt injection for the reflection baseline."""
from __future__ import annotations

from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter
from .memory import ReflectionMemoryBank


class ReflectionPromptInjector(PromptAugmenter):
    """Inject top-k retrieved reflections into the standard agent prompt."""

    def __init__(self, store_dir: str | Path, *, top_k: int = 5):
        self.store_dir = Path(store_dir)
        self.top_k = top_k
        self.memory = ReflectionMemoryBank(self.store_dir / "memory" / "reflection_memory.jsonl")

    def augment(self, user_query: str, **kwargs) -> str:
        retrieved = self.memory.retrieve(user_query, top_k=self.top_k)
        if not retrieved:
            return ""
        lines = [
            "## Retrieved Reflection Memories",
            "Use only the relevant lessons below. Ignore reflections that do not match the current task.",
        ]
        for index, entry in enumerate(retrieved, 1):
            tools = entry.tools_called or entry.expected_tools
            lines.append(f"{index}. [{entry.kind}; F1={entry.f1:.2f}; task={entry.task_type}] {entry.reflection}")
            if tools:
                lines.append(f"   Related tools: {' -> '.join(tools[:8])}")
        return "\n".join(lines)
