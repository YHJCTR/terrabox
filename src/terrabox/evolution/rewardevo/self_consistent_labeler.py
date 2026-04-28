"""Self-Consistent Pseudo-Labeling for RewardEvo.

For tasks with multiple valid trajectories, use LLM-as-judge to identify high-quality ones
without ground-truth labels. Group by task_type, compare within group, label winners as positive.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from ..memrl.episodic_memory import EpisodicMemory
from ..shared.llm_client import EvolutionLLMClient
from ..shared.trajectory import Trajectory
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class SelfConsistentLabeler:
    """Pseudo-label trajectories using self-consistent voting via LLM judge."""

    def __init__(
        self,
        llm_client: Optional[EvolutionLLMClient] = None,
        high_quality_threshold: float = 0.7,
        low_quality_threshold: float = 0.3,
    ):
        """
        Args:
            high_quality_threshold: Score ≥ this → mark as positive
            low_quality_threshold: Score ≤ this → mark as negative
        """
        self._judge = LLMJudge(llm_client)
        self._high_threshold = high_quality_threshold
        self._low_threshold = low_quality_threshold

    def label_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> dict:
        """Pseudo-label trajectories by task type; return {
            "labeled_positive": [traj, ...],  # high-quality
            "labeled_negative": [traj, ...],  # low-quality
            "uncertain": [traj, ...],         # middle-ground
        }"""

        # Group by task_type
        by_type = defaultdict(list)
        for traj in trajectories:
            by_type[traj.task_type].append(traj)

        labeled_positive = []
        labeled_negative = []
        uncertain = []

        for task_type, trajs in by_type.items():
            if len(trajs) < 2:
                # Single example; treat as uncertain
                uncertain.extend(trajs)
                logger.info(f"Task type {task_type}: only 1 sample, marking as uncertain")
                continue

            logger.info(f"Labeling {len(trajs)} trajectories of type {task_type}...")

            # Rank within this task type
            scores = self._judge.rank_trajectories(
                question=trajs[0].question,  # representative question
                trajectories=trajs,
            )

            for traj, score in zip(trajs, scores):
                if score >= self._high_threshold:
                    labeled_positive.append((traj, score))
                    logger.debug(f"  ✓ {traj.task_id}: {score:.2f} (positive)")
                elif score <= self._low_threshold:
                    labeled_negative.append((traj, score))
                    logger.debug(f"  ✗ {traj.task_id}: {score:.2f} (negative)")
                else:
                    uncertain.append(traj)
                    logger.debug(f"  ? {traj.task_id}: {score:.2f} (uncertain)")

        return {
            "labeled_positive": labeled_positive,
            "labeled_negative": labeled_negative,
            "uncertain": uncertain,
        }

    def write_to_memrl(
        self,
        labeled: dict,
        memory_db: str,
        llm_client: Optional[EvolutionLLMClient] = None,
    ) -> dict:
        """Write labeled trajectories into MemRL episodic memory.

        Positive examples: initial_utility = (score + 1) / 2  (range [0.5, 1.0])
        Negative examples: initial_utility = score / 2  (range [0.0, 0.15])
        """
        memory = EpisodicMemory(memory_db, llm_client=llm_client)

        result = {
            "positive_written": 0,
            "negative_written": 0,
            "uncertain_written": 0,
            "memory_ids": [],
        }

        # Positive samples: high utility
        for traj, score in labeled.get("labeled_positive", []):
            utility = (score + 1.0) / 2.0  # map [0.7, 1.0] → [0.85, 1.0]
            mid = memory.add_memory(traj, initial_utility=utility, skip_if_exists=True)
            if mid:
                result["positive_written"] += 1
                result["memory_ids"].append(mid)
                logger.info(f"Added positive memory {mid} with utility {utility:.2f}")

        # Negative samples: low utility (but still in memory for learning)
        for traj, score in labeled.get("labeled_negative", []):
            utility = score / 2.0  # map [0.0, 0.3] → [0.0, 0.15]
            mid = memory.add_memory(traj, initial_utility=utility, skip_if_exists=True)
            if mid:
                result["negative_written"] += 1
                result["memory_ids"].append(mid)
                logger.info(f"Added negative memory {mid} with utility {utility:.2f}")

        # Uncertain samples: neutral utility
        for traj in labeled.get("uncertain", []):
            utility = 0.5
            mid = memory.add_memory(traj, initial_utility=utility, skip_if_exists=True)
            if mid:
                result["uncertain_written"] += 1
                result["memory_ids"].append(mid)

        logger.info(
            f"MemRL write summary: {result['positive_written']} positive, "
            f"{result['negative_written']} negative, {result['uncertain_written']} uncertain"
        )

        return result

    def generate_pseudo_labels_jsonl(
        self,
        labeled: dict,
    ) -> list[dict]:
        """Convert labels to JSONL-compatible format for analysis.

        Each record: {
            "task_id": str,
            "task_type": str,
            "label": "positive" | "negative" | "uncertain",
            "score": float,
            "question": str,
            "tools_called": [...],
        }
        """
        records = []

        for traj, score in labeled.get("labeled_positive", []):
            records.append({
                "task_id": traj.task_id,
                "task_type": traj.task_type,
                "label": "positive",
                "score": score,
                "question": traj.question[:300],
                "tools_called": traj.tools_called,
            })

        for traj, score in labeled.get("labeled_negative", []):
            records.append({
                "task_id": traj.task_id,
                "task_type": traj.task_type,
                "label": "negative",
                "score": score,
                "question": traj.question[:300],
                "tools_called": traj.tools_called,
            })

        for traj in labeled.get("uncertain", []):
            records.append({
                "task_id": traj.task_id,
                "task_type": traj.task_type,
                "label": "uncertain",
                "score": 0.5,
                "question": traj.question[:300],
                "tools_called": traj.tools_called,
            })

        return records
