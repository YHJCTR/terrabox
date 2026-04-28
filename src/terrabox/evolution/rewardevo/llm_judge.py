"""LLM-as-Judge for RewardEvo: Compare and rank trajectories without ground-truth labels.

Core idea: Instead of training a reward model, use the agent LLM to compare trajectory pairs
and provide quality scores. This enables annotation-free self-evolution.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..shared.llm_client import EvolutionLLMClient
from ..shared.trajectory import Trajectory

logger = logging.getLogger(__name__)


class LLMJudge:
    """Use LLM to compare and rank trajectories.

    Core scoring: For geospatial tasks, we compare trajectories on:
    1. Tool coverage: did it call the necessary tools?
    2. Tool order: is the sequence logically sound?
    3. No redundancy: unnecessary duplicate calls?

    Uses call() with small max_tokens (matching SkillRL pattern) instead of
    call_json() with large max_tokens, to avoid vLLM timeout issues.
    """

    def __init__(self, llm_client: Optional[EvolutionLLMClient] = None):
        self._llm = llm_client or EvolutionLLMClient()

    def judge_trajectory_pair(
        self,
        trajectory_a: Trajectory,
        trajectory_b: Trajectory,
        question: str,
    ) -> dict:
        """Compare two trajectories for the same question; return preference + score.

        Uses call() + text parsing (like SkillRL's extract_skills_from_success)
        instead of call_json() to keep max_tokens small and avoid timeout.

        Returns:
            {
                "a_score": 0.0–1.0,
                "b_score": 0.0–1.0,
            }
        """
        # Short-circuit: identical tool sequences → equal scores (avoids LLM confusion)
        if trajectory_a.tools_called == trajectory_b.tools_called:
            return {"a_score": 0.5, "b_score": 0.5}

        system = "You are an expert geospatial AI evaluator."

        prompt = f"""Compare two tool-calling trajectories for: "{question[:200]}"

A tools: {' → '.join(trajectory_a.tools_called)}
B tools: {' → '.join(trajectory_b.tools_called)}

Score each 0.0-1.0 on correctness, order, completeness, efficiency.
Reply ONLY: A=<score> B=<score>"""

        raw = self._llm.call(prompt, system=system, max_tokens=1000,
                             enable_thinking=False)

        # Parse "A=0.8 B=0.4" or "A: 0.8, B: 0.4" etc.
        a_match = re.search(r"A\s*[=:]\s*([\d.]+)", raw)
        b_match = re.search(r"B\s*[=:]\s*([\d.]+)", raw)

        if a_match and b_match:
            try:
                a_score = max(0.0, min(1.0, float(a_match.group(1))))
                b_score = max(0.0, min(1.0, float(b_match.group(1))))
                return {"a_score": a_score, "b_score": b_score}
            except ValueError:
                pass

        logger.warning("LLM judge parse failed, using heuristic fallback")
        return self._heuristic_compare(trajectory_a, trajectory_b)

    def rank_trajectories(
        self,
        question: str,
        trajectories: list[Trajectory],
        max_comparisons_per_traj: int = 3,
    ) -> list[float]:
        """Rank trajectories for the same question; return scores [0,1] per trajectory.

        Uses Swiss-system tournament: each trajectory competes in limited rounds
        against adjacent opponents, reducing O(n²) to O(n·rounds).
        """
        if len(trajectories) <= 1:
            return [1.0] * len(trajectories)

        n = len(trajectories)
        scores = [0.0] * n

        # Swiss-system: compare each trajectory against a limited set of opponents
        import random
        rounds = min(max_comparisons_per_traj, n - 1)

        for round_idx in range(rounds):
            # Pair up trajectories (shifted by round_idx+1 for diverse matchups)
            for i in range(n):
                j = (i + round_idx + 1) % n
                if i == j:
                    continue
                # Only compute each pair once per round
                if i > j:
                    continue

                comparison = self.judge_trajectory_pair(
                    trajectories[i],
                    trajectories[j],
                    question,
                )
                a_score = comparison.get("a_score", 0.5)
                b_score = comparison.get("b_score", 0.5)

                scores[i] += a_score
                scores[j] += b_score

        # Normalize to [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized = [s / max_score for s in scores]

        logger.info(
            f"Ranked {len(trajectories)} trajectories ({rounds} rounds): "
            f"{', '.join(f'{score:.2f}' for score in normalized)}"
        )

        return normalized

    def _heuristic_compare(self, traj_a: Trajectory, traj_b: Trajectory) -> dict:
        """Simple heuristic when LLM judge fails."""
        # Prefer shorter, more direct sequences
        len_a = len(traj_a.tools_called)
        len_b = len(traj_b.tools_called)

        # Penalize tool duplication
        unique_a = len(set(traj_a.tools_called))
        unique_b = len(set(traj_b.tools_called))

        score_a = 0.5 + (unique_a / (len_a + 1)) * 0.3 - (len_a / 10) * 0.1
        score_b = 0.5 + (unique_b / (len_b + 1)) * 0.3 - (len_b / 10) * 0.1

        # Clamp to [0, 1]
        score_a = max(0.0, min(1.0, score_a))
        score_b = max(0.0, min(1.0, score_b))

        winner = "a" if score_a > score_b else "b"

        return {
            "winner": winner,
            "a_score": score_a,
            "b_score": score_b,
            "reasoning": "(fallback heuristic: preferred unique, efficient tool sequences)",
        }
