"""AgentEvolverPromptInjector: combine navigation guidance + credit hints."""
from __future__ import annotations

from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .self_attributing import ADCAGRPOAttributor
from .self_navigating import ExperiencePool, HybridPolicy


class AgentEvolverPromptInjector(PromptAugmenter):
    """Inject navigation guidance and credit-aware hints into system prompt.

    Injected sections:
    1. Navigation Guidance (from HybridPolicy + ExperiencePool)
    2. Credit-Aware Tool Guidance (from ADCA-GRPO attribution data)
    """

    def __init__(
        self,
        pool: ExperiencePool,
        policy: HybridPolicy,
        attributor: ADCAGRPOAttributor,
        credit_experiences: Optional[list[dict]] = None,
    ):
        self._pool = pool
        self._policy = policy
        self._attributor = attributor
        self._credit_experiences: list[dict] = credit_experiences or []

    def augment(
        self,
        user_query: str,
        task_type: str = "unknown",
        **kwargs,
    ) -> str:
        nav_hint = self._policy.get_navigation_hint(user_query, task_type, self._pool)
        credit_hint = self._attributor.generate_credit_hint(
            [e for e in self._credit_experiences if e.get("task_type") == task_type]
        )
        augmentation = nav_hint + credit_hint
        if not augmentation.strip():
            return self.BASE_SYSTEM
        return self.BASE_SYSTEM + augmentation

    def update_credit_experiences(self, new_experiences: list[dict]) -> None:
        """Add newly attributed experiences for future injection."""
        self._credit_experiences.extend(new_experiences)
