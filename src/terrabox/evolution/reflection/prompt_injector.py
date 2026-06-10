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
            "## Lessons from your past similar attempts",
            "These are your own self-reflections from earlier tasks. Use only the relevant ones.",
        ]
        for index, entry in enumerate(retrieved, 1):
            # Only inject the self-written lesson (+ what you did last time).
            # No gold expected_tools / F1 are shown — those are never available.
            lines.append(f"{index}. {entry.reflection}")
            if entry.tools_called:
                lines.append(f"   (Last time you used: {' -> '.join(entry.tools_called[:8])})")
        return "\n".join(lines)
