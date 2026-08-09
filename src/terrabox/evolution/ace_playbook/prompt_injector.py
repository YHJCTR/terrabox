"""ACE-style playbook prompt injector."""
from __future__ import annotations

from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter
from .playbook import ACEPlaybook


class ACEPlaybookPromptInjector(PromptAugmenter):
    """Inject retrieved ACE-style playbook bullets into the system prompt."""

    def __init__(self, store_dir: str | Path, top_k: int = 5):
        self.playbook = ACEPlaybook(store_dir)
        self.top_k = top_k

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        matches = self.playbook.retrieve(
            user_query,
            task_type=task_type,
            available_tools=kwargs.get("available_tools"),
            top_k=self.top_k,
        )
        if not matches:
            return self.BASE_SYSTEM

        lines = [
            "\n\n## ACE-Style Evolved Playbook",
            "These bullets are distilled from prior train rollouts. Use them as transferable strategy guidance only; do not copy historical task ids, exact artifact paths, or place-specific facts.",
        ]
        for score, bullet in matches:
            tools = " -> ".join(bullet.tools[:8])
            suffix = f" | tools: {tools}" if tools else ""
            lines.append(
                f"- {bullet.as_prompt_line()} (retrieval_score={score:.2f}, support={bullet.support}{suffix})"
            )

        return self.BASE_SYSTEM + "\n".join(lines)
