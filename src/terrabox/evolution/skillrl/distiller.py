"""ExperienceDistiller: convert trajectories into skills via LLM calls.

From SkillRL (arXiv 2602.08234):
- Successful trajectories → strategic behavioral patterns (general/task skills)
- Failed trajectories → concise lessons from failure (mistakes tier)
"""
from __future__ import annotations

import logging

from ..shared.evaluator import ToolMatchEvaluator
from ..shared.llm_client import EvolutionLLMClient
from ..shared.trajectory import EpisodeResult, Trajectory
from .skill_bank import HierarchicalSkillBank

logger = logging.getLogger(__name__)

# Only distill high-signal episodes
_SUCCESS_F1_THRESHOLD = 0.8
_FAILURE_F1_THRESHOLD = 0.2


class ExperienceDistiller:
    """Distill agent trajectories into hierarchical skills.

    High-quality successes (F1 ≥ 0.8) → general + task-specific strategies.
    Clear failures (F1 ≤ 0.2) → mistake patterns + lessons.
    Middle-ground episodes are skipped (low signal-to-noise).
    """

    def __init__(self, bank: HierarchicalSkillBank, llm_client: EvolutionLLMClient):
        self._bank = bank
        self._llm = llm_client

    def distill_batch(self, episodes: list[EpisodeResult]) -> int:
        """Distill all high-signal episodes; return count of skills created."""
        import os
        try:
            import tqdm as _tqdm_mod
            tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
            _iter = _tqdm_mod.tqdm(
                episodes,
                desc="[skillrl distill]", unit="ep", ncols=90,
                position=tqdm_pos, leave=True,
            )
        except ImportError:
            _iter = iter(episodes)
        created = 0
        for episode in _iter:
            if not self._should_distill(episode):
                continue
            try:
                n = self._distill_one(episode)
                created += n
            except Exception as e:
                logger.warning(f"Distillation failed for {episode.trajectory.task_id}: {e}")
        return created

    def _should_distill(self, episode: EpisodeResult) -> bool:
        traj = episode.trajectory
        # Need at least one tool call to be interesting
        if not traj.tools_called:
            return False
        # Only high-quality successes or clear failures
        if traj.expected_tools:
            return episode.tool_f1 >= _SUCCESS_F1_THRESHOLD or episode.tool_f1 <= _FAILURE_F1_THRESHOLD
        # Train split: distill all (no F1 signal, use success flag)
        return True

    def _distill_one(self, episode: EpisodeResult) -> int:
        traj = episode.trajectory
        is_success = episode.tool_f1 >= _SUCCESS_F1_THRESHOLD or (
            not traj.expected_tools and traj.success
        )
        created = 0

        if is_success:
            # Tier 1: general skill
            strategy = self._llm.extract_skills_from_success(
                traj.question,
                traj.task_type,
                traj.tools_called,
                traj.final_answer,
            )
            if strategy and len(strategy) > 10:
                tags = traj.tools_called[:3] + [traj.task_type]
                self._bank.add_general_skill(
                    content=strategy,
                    tags=tags,
                    performance_delta=episode.tool_f1,
                    source_tasks=[traj.task_id],
                )
                created += 1

            # Tier 2: task-specific skill (only if task_type is known)
            if traj.task_type and traj.task_type != "unknown":
                self._bank.add_task_skill(
                    task_type=traj.task_type,
                    content=strategy or f"For {traj.task_type}: use {' → '.join(traj.tools_called)}",
                    performance_delta=episode.tool_f1,
                    source_tasks=[traj.task_id],
                )
                created += 1

        else:
            # Tier 3: mistake pattern
            error_turns = [t for t in traj.turns if t.is_error and t.tool_name]
            failed_tool = error_turns[0].tool_name if error_turns else (
                traj.tools_called[-1] if traj.tools_called else "unknown"
            )
            error_msg = error_turns[0].tool_result or "low tool-match F1" if error_turns else "low tool-match F1"

            lesson = self._llm.extract_lesson_from_failure(
                traj.question,
                traj.task_type,
                traj.tools_called,
                error_msg,
            )
            if lesson and len(lesson) > 10:
                self._bank.add_mistake(
                    pattern=f"Task type '{traj.task_type}': tools {traj.tools_called} failed",
                    failed_tool=failed_tool,
                    lesson=lesson,
                    source_tasks=[traj.task_id],
                )
                created += 1

        return created
