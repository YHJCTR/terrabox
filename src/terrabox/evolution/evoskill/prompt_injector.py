"""EvoSkillPromptInjector: inject Pareto-optimal skills into system prompt."""
from __future__ import annotations

from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .pareto_manager import ParetoManager
from .transfer_engine import CrossDomainTransfer


class EvoSkillPromptInjector(PromptAugmenter):
    """Inject active skills from the Pareto frontier into system prompt.

    Injected section format:
    ## Discovered Geospatial Skills
    **skill_name**: description
      When to use: trigger condition
      Tool sequence: slug1 → slug2 → slug3
      Parameter hints: slug1: hint; slug2: hint
      Preconditions: condition1

    Also includes transferred skills from related domains if relevant.
    """

    def __init__(
        self,
        pareto_manager: ParetoManager,
        transfer_engine: Optional[CrossDomainTransfer] = None,
        top_n: int = 5,
    ):
        self._pareto = pareto_manager
        self._transfer = transfer_engine or CrossDomainTransfer()
        self.top_n = top_n

    def augment(
        self,
        user_query: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        active_skills = self._pareto.get_active_skills(top_n=self.top_n)

        # Domain-aware transfer: add relevant transferred skills
        target_domain = CrossDomainTransfer.classify_domain(user_query, images or [])
        transferred = self._transfer.find_transferable_skills(active_skills, target_domain)

        # Combine: filter active for relevance + add transferred
        q_lower = user_query.lower()
        relevant_active = [
            s for s in active_skills
            if any(word in q_lower for word in s.trigger_condition.lower().split() if len(word) > 3)
            or s.domain in (target_domain, "multi")
        ]

        all_skills = relevant_active[:self.top_n] + transferred[:2]

        if not all_skills:
            return self.BASE_SYSTEM

        lines = ["\n\n## Discovered Geospatial Skills"]
        for skill in all_skills:
            lines.append(skill.to_prompt_text())

        return self.BASE_SYSTEM + "\n".join(lines)
