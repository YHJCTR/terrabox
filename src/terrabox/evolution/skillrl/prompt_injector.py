"""SkillRLPromptInjector: prepend retrieved hierarchical skills to system prompt."""
from __future__ import annotations

from ..shared.prompt_builder import PromptAugmenter
from .retriever import SkillRetriever
from .skill_bank import HierarchicalSkillBank


class SkillRLPromptInjector(PromptAugmenter):
    """Inject retrieved skills from HierarchicalSkillBank into system prompt.

    Appends three sections after the base ReAct system prompt:
      ## Retrieved Geospatial Skills     (general tier)
      ## Task-Specific Strategies        (task-specific tier)
      ## Common Mistakes to Avoid        (mistakes tier)

    Only non-empty tiers are included.
    """

    def __init__(
        self,
        bank: HierarchicalSkillBank,
        retriever: SkillRetriever,
        top_k: int = 3,
    ):
        self._bank = bank
        self._retriever = retriever
        self.top_k = top_k

    def augment(
        self,
        user_query: str,
        task_type: str = "unknown",
        **kwargs,
    ) -> str:
        skills = self._retriever.retrieve(user_query, task_type, self.top_k)
        augmentation = ""

        general = skills.get("general", [])
        specific = skills.get("specific", [])
        mistakes = skills.get("mistakes", [])

        augmentation += self._format_skill_block(general, "Retrieved Geospatial Skills")
        augmentation += self._format_skill_block(specific, "Task-Specific Strategies")
        augmentation += self._format_skill_block(mistakes, "Common Mistakes to Avoid")

        if not augmentation.strip():
            return self.BASE_SYSTEM

        return self.BASE_SYSTEM + augmentation
