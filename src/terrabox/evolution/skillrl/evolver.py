"""SkillEvolver: monitor performance and trigger recursive skill evolution.

From SkillRL (arXiv 2602.08234):
Recursive evolution: when the agent's recent F1 drops below a threshold
relative to baseline, collect failing trajectories and synthesize new skills.
This closes the feedback loop: Experience → Distillation → Evolution → Policy.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from ..shared.evaluator import ToolMatchEvaluator
from ..shared.trajectory import EpisodeResult, Trajectory
from .distiller import ExperienceDistiller
from .skill_bank import HierarchicalSkillBank

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 20
_DROP_THRESHOLD = 0.10   # 10% relative drop triggers re-evolution
_MAX_NEW_SKILLS_PER_CYCLE = 3


class SkillEvolver:
    """Monitor rolling F1 and trigger recursive skill distillation on drift.

    Maintains a sliding window of recent episode F1 scores.
    When the window mean drops more than DROP_THRESHOLD below the baseline
    (computed from all skills' performance_delta), it:
      1. Collects recent failing episodes
      2. Calls distiller.distill_batch() on them
      3. Resets the drift counter
    """

    def __init__(
        self,
        bank: HierarchicalSkillBank,
        distiller: ExperienceDistiller,
        window_size: int = _WINDOW_SIZE,
        drop_threshold: float = _DROP_THRESHOLD,
    ):
        self._bank = bank
        self._distiller = distiller
        self._window_size = window_size
        self._drop_threshold = drop_threshold
        self._recent_window: deque[float] = deque(maxlen=window_size)
        self._recent_episodes: deque[EpisodeResult] = deque(maxlen=window_size * 2)
        self._evolution_count = 0

    def record_episode(self, result: EpisodeResult) -> bool:
        """Record an episode result; trigger evolution if performance drops.

        Returns:
            True if an evolution cycle was triggered, False otherwise.
        """
        self._recent_window.append(result.tool_f1)
        self._recent_episodes.append(result)

        if len(self._recent_window) < self._window_size:
            return False   # not enough data yet

        if self._detect_drift():
            self._trigger_evolution()
            return True
        return False

    def _detect_drift(self) -> bool:
        """Compare recent window mean F1 to skill bank baseline."""
        baseline = self._bank.get_performance_baseline()
        if baseline <= 0:
            return False   # no baseline yet
        recent_mean = sum(self._recent_window) / len(self._recent_window)
        relative_drop = (baseline - recent_mean) / max(baseline, 0.01)
        if relative_drop > self._drop_threshold:
            logger.info(
                f"[SkillEvolver] Performance drift detected: "
                f"baseline={baseline:.3f}, recent={recent_mean:.3f}, "
                f"drop={relative_drop:.1%}"
            )
            return True
        return False

    def _trigger_evolution(self) -> None:
        """Run distiller on recent failures to generate new skills."""
        failing = [e for e in self._recent_episodes if e.tool_f1 <= 0.3]
        if not failing:
            return

        # Limit to most recent failures
        failing = list(failing)[-_MAX_NEW_SKILLS_PER_CYCLE * 3:]
        logger.info(f"[SkillEvolver] Triggering evolution on {len(failing)} failing episodes")

        created = self._distiller.distill_batch(failing)
        self._evolution_count += 1
        logger.info(
            f"[SkillEvolver] Evolution cycle {self._evolution_count}: "
            f"created {created} new skills"
        )

        # Clear recent window to avoid immediate re-triggering
        self._recent_window.clear()

    @property
    def evolution_count(self) -> int:
        return self._evolution_count

    @property
    def recent_f1_mean(self) -> Optional[float]:
        if not self._recent_window:
            return None
        return sum(self._recent_window) / len(self._recent_window)
