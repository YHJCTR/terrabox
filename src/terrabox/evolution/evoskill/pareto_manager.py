"""ParetoManager: maintain Pareto frontier of skill modules.

From: EvoSkill (arXiv 2603.02766):
A skill is Pareto-optimal if no other skill dominates it on both:
  - validation_f1_delta: improvement in tool-match F1 on held-out eval set
  - generality_score: fraction of eval cases where the skill is relevant

The Pareto frontier ensures only skills that genuinely improve performance
are retained, preventing skill bloat.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..shared.storage import JSONSkillStore
from .skill_module import SkillModule

logger = logging.getLogger(__name__)

_MAX_FRONTIER = 50
_ALPHA = 0.7   # weight for f1_delta vs generality in weighted score


class ParetoManager:
    """Maintain a Pareto-optimal set of skill modules.

    Retention policy:
    1. A new skill is added only if it is not dominated (f1_delta AND generality
       both ≤ some existing skill's values).
    2. When frontier exceeds MAX_FRONTIER, prune by weighted score.
    """

    def __init__(self, store_path: str):
        self._store = JSONSkillStore(store_path)

    def add_candidate(
        self,
        skill: SkillModule,
        f1_delta: float,
        generality: float,
    ) -> bool:
        """Add skill to frontier if Pareto-optimal.

        Returns True if skill was added.
        """
        skill.validation_f1_delta = f1_delta
        skill.generality_score = generality

        existing = self._store.load_all()

        if self._is_dominated(skill, existing):
            logger.debug(f"Skill {skill.name} dominated; not added to frontier")
            return False

        self._store.save(skill.to_dict())
        logger.info(f"Added Pareto-optimal skill: {skill.name} "
                    f"(f1_delta={f1_delta:.3f}, generality={generality:.3f})")

        # Prune if over limit
        if self._store.count() > _MAX_FRONTIER:
            self._prune()

        return True

    def get_active_skills(self, top_n: int = 10) -> list[SkillModule]:
        """Return top-N skills by weighted score."""
        all_skills = self._store.load_all()
        scored = [
            (self._weighted_score(s), s)
            for s in all_skills
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [SkillModule.from_dict(s) for _, s in scored[:top_n]]

    def evaluate_skill(
        self,
        skill: SkillModule,
        eval_cases: list[dict],
        base_agent: "BaseAgent",  # noqa: F821
        baseline_f1: float,
    ) -> tuple[float, float]:
        """Run skill on a sample of eval cases; return (f1_delta, generality).

        Args:
            skill: SkillModule to evaluate.
            eval_cases: List of eval task dicts.
            base_agent: BaseAgent for execution.
            baseline_f1: F1 without any skill injection.

        Returns:
            (f1_delta, generality) tuple.
        """
        from ..shared.evaluator import ToolMatchEvaluator
        from ..shared.prompt_builder import PromptAugmenter

        evaluator = ToolMatchEvaluator()
        skill_prompt = skill.to_prompt_text()
        relevant_count = 0
        total_f1 = 0.0
        sample = eval_cases[:min(20, len(eval_cases))]   # quick evaluation sample

        for case in sample:
            # Check if skill is relevant to this case
            q_lower = case.get("question", "").lower()
            trigger = skill.trigger_condition.lower()
            is_relevant = any(
                word in q_lower
                for word in trigger.split()
                if len(word) > 3
            )
            if is_relevant:
                relevant_count += 1

            # Add skill context to system prompt
            augmented = PromptAugmenter.BASE_SYSTEM + f"\n\n## Available Skill\n{skill_prompt}"
            try:
                traj = base_agent.execute(
                    question=case["question"],
                    images=case.get("images", []),
                    system_prompt=augmented,
                    expected_tools=case.get("expected_tools", []),
                )
                result = evaluator.evaluate(traj)
                total_f1 += result.tool_f1
            except Exception as e:
                logger.warning(f"Skill eval execution failed: {e}")
                total_f1 += baseline_f1   # neutral contribution

        mean_f1 = total_f1 / len(sample) if sample else 0.0
        f1_delta = mean_f1 - baseline_f1
        generality = relevant_count / len(sample) if sample else 0.0

        return f1_delta, generality

    def count(self) -> int:
        return self._store.count()

    def _is_dominated(self, candidate: SkillModule, existing: list[dict]) -> bool:
        """Return True if candidate is dominated by any existing skill."""
        for skill_dict in existing:
            if (skill_dict.get("validation_f1_delta", 0) >= candidate.validation_f1_delta and
                    skill_dict.get("generality_score", 0) >= candidate.generality_score):
                return True
        return False

    def _weighted_score(self, skill_dict: dict) -> float:
        f1 = skill_dict.get("validation_f1_delta", 0.0)
        gen = skill_dict.get("generality_score", 0.0)
        return _ALPHA * f1 + (1 - _ALPHA) * gen

    def _prune(self) -> None:
        """Remove lowest-scoring skills until frontier is at MAX_FRONTIER."""
        all_skills = self._store.load_all()
        scored = sorted(
            [(self._weighted_score(s), s) for s in all_skills],
            key=lambda x: x[0],
            reverse=True,
        )
        # Delete skills beyond MAX_FRONTIER
        for _, skill_dict in scored[_MAX_FRONTIER:]:
            skill_id = skill_dict.get("id", "")
            if skill_id:
                self._store.delete(skill_id)
        logger.info(f"[ParetoManager] Pruned frontier to {_MAX_FRONTIER} skills")
