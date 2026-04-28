"""Sequential pattern miner for tool sequences.

Mines frequent ordered subsequences (contiguous) from agent trajectories.
Separates success patterns from anti-patterns (failure co-occurrences).

No external dependencies — pure Python.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SeqPatternMiner:
    """Mine frequent sequential tool patterns and anti-patterns from trajectory data."""

    def __init__(self):
        # (tool_seq, task_type, f1) for successes and failures
        self._success_seqs: list[tuple[list[str], str, float]] = []
        self._failure_seqs: list[tuple[list[str], str, float]] = []
        self._trajectory_count = 0

    def add_trajectory(
        self,
        tool_seq: list[str],
        task_type: str = "general",
        f1: float = 1.0,
    ) -> None:
        """Record one trajectory's tool sequence."""
        if not tool_seq:
            return
        self._trajectory_count += 1
        # Deduplicate preserving order
        seen = list(dict.fromkeys(tool_seq))
        if f1 >= 0.5:
            self._success_seqs.append((seen, task_type, f1))
        else:
            self._failure_seqs.append((seen, task_type, f1))

    @classmethod
    def load_from_trajectories(
        cls,
        traj_file: str,
        min_f1: float = 0.0,
    ) -> "SeqPatternMiner":
        """Load trajectories from a JSONL file."""
        miner = cls()
        path = Path(traj_file)
        if not path.exists():
            logger.warning(f"Trajectory file not found: {traj_file}")
            return miner
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tool_seq = traj.get("tool_sequence") or traj.get("tools_called", [])
                f1 = float(traj.get("f1", traj.get("reward", 1.0)))
                task_type = traj.get("task_type", "general")
                if not tool_seq or f1 < min_f1:
                    continue
                miner.add_trajectory(tool_seq, task_type=task_type, f1=f1)

        logger.info(
            f"Loaded {miner._trajectory_count} trajectories from {traj_file}: "
            f"{len(miner._success_seqs)} success, {len(miner._failure_seqs)} failure"
        )
        return miner

    # ------------------------------------------------------------------
    # Pattern mining
    # ------------------------------------------------------------------

    @staticmethod
    def _contiguous_subseqs(seq: list[str], length: int) -> list[tuple[str, ...]]:
        """Return all contiguous subsequences of given length."""
        return [tuple(seq[i:i + length]) for i in range(len(seq) - length + 1)]

    def mine_patterns(
        self,
        min_support: int = 5,
        max_len: int = 4,
    ) -> list[dict]:
        """Mine frequent contiguous sequential patterns from successful trajectories.

        Returns a list of pattern dicts:
        {
            "pattern": ["tool_a", "tool_b", "tool_c"],
            "support": int,           # number of trajectories containing this subseq
            "task_types": {"flood_detection": 3, ...},
            "avg_f1": float,
            "length": int,
        }
        Sorted by support descending.
        """
        counter: dict[tuple[str, ...], dict] = defaultdict(
            lambda: {"count": 0, "f1_sum": 0.0, "task_types": defaultdict(int)}
        )

        for seq, task_type, f1 in self._success_seqs:
            for length in range(2, min(max_len + 1, len(seq) + 1)):
                for subseq in self._contiguous_subseqs(seq, length):
                    counter[subseq]["count"] += 1
                    counter[subseq]["f1_sum"] += f1
                    counter[subseq]["task_types"][task_type] += 1

        results = []
        for pattern, data in counter.items():
            if data["count"] < min_support:
                continue
            results.append({
                "pattern": list(pattern),
                "support": data["count"],
                "task_types": dict(data["task_types"]),
                "avg_f1": round(data["f1_sum"] / data["count"], 4),
                "length": len(pattern),
            })

        results.sort(key=lambda x: x["support"], reverse=True)
        logger.info(
            f"Mined {len(results)} sequential patterns "
            f"(min_support={min_support}, max_len={max_len})"
        )
        return results

    def mine_anti_patterns(
        self,
        max_f1: float = 0.3,
        min_support: int = 3,
        max_len: int = 3,
    ) -> list[dict]:
        """Mine tool co-occurrences associated with failure trajectories.

        Returns patterns that appear frequently in low-F1 trajectories —
        these signal "what to avoid" when similar tools are triggered.
        """
        counter: dict[tuple[str, ...], dict] = defaultdict(
            lambda: {"count": 0, "f1_sum": 0.0, "task_types": defaultdict(int)}
        )

        for seq, task_type, f1 in self._failure_seqs:
            if f1 > max_f1:
                continue
            for length in range(2, min(max_len + 1, len(seq) + 1)):
                for subseq in self._contiguous_subseqs(seq, length):
                    counter[subseq]["count"] += 1
                    counter[subseq]["f1_sum"] += f1
                    counter[subseq]["task_types"][task_type] += 1

        results = []
        for pattern, data in counter.items():
            if data["count"] < min_support:
                continue
            results.append({
                "pattern": list(pattern),
                "support": data["count"],
                "task_types": dict(data["task_types"]),
                "avg_f1": round(data["f1_sum"] / data["count"], 4),
                "length": len(pattern),
            })

        results.sort(key=lambda x: x["support"], reverse=True)
        logger.info(
            f"Mined {len(results)} anti-patterns "
            f"(max_f1={max_f1}, min_support={min_support})"
        )
        return results

    def get_directed_edges(
        self, min_count: int = 3
    ) -> list[dict]:
        """Get directed tool transition edges (A → B) from successful trajectories.

        Returns:
        [{"source": str, "target": str, "count": int, "avg_f1": float,
          "task_types": dict}, ...]
        """
        edges: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"count": 0, "f1_sum": 0.0, "task_types": defaultdict(int)}
        )
        for seq, task_type, f1 in self._success_seqs:
            for i in range(len(seq) - 1):
                key = (seq[i], seq[i + 1])
                edges[key]["count"] += 1
                edges[key]["f1_sum"] += f1
                edges[key]["task_types"][task_type] += 1

        result = []
        for (src, tgt), data in edges.items():
            if data["count"] < min_count:
                continue
            result.append({
                "source": src,
                "target": tgt,
                "count": data["count"],
                "avg_f1": round(data["f1_sum"] / data["count"], 4),
                "task_types": dict(data["task_types"]),
            })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def get_tool_stats(self) -> dict[str, dict]:
        """Return per-tool statistics from successful trajectories."""
        stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "f1_sum": 0.0, "task_types": defaultdict(int)}
        )
        for seq, task_type, f1 in self._success_seqs:
            for tool in seq:
                stats[tool]["count"] += 1
                stats[tool]["f1_sum"] += f1
                stats[tool]["task_types"][task_type] += 1
        return {
            slug: {
                "count": d["count"],
                "avg_f1": round(d["f1_sum"] / d["count"], 4) if d["count"] else 0.0,
                "task_types": dict(d["task_types"]),
            }
            for slug, d in stats.items()
        }
