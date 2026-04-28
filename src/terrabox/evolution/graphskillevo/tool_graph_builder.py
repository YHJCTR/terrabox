"""ToolGraphBuilder: Build a tool co-occurrence graph from trajectory data.

Replaces LLM-based relation extraction with statistical co-occurrence analysis
over real execution trajectories. No LLM calls required — fully offline.

Graph semantics:
  Node = tool slug (e.g. "geo_raster.calculate_index")
  Edge = these two tools frequently appear in the same trajectory
  Edge weight = co-occurrence count + 0.5 × sequential adjacency count
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ToolGraphBuilder:
    """Statistically build a tool co-occurrence graph from trajectories."""

    def __init__(self):
        # (tool_a, tool_b) sorted tuple → co-occurrence count
        self._cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        # (tool_a, tool_b) ordered pair → sequential adjacency count
        self._sequential: dict[tuple[str, str], int] = defaultdict(int)
        # tool_slug → {"count": int, "success_count": int, "task_types": set}
        self._tool_stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "success_count": 0, "task_types": set()}
        )
        self._trajectory_count = 0

    def add_trajectory(
        self,
        tool_seq: list[str],
        success: bool = True,
        task_type: str = "general",
    ) -> None:
        """Record one trajectory's tool sequence."""
        if not tool_seq:
            return
        self._trajectory_count += 1

        # Per-tool stats
        for tool in tool_seq:
            self._tool_stats[tool]["count"] += 1
            if success:
                self._tool_stats[tool]["success_count"] += 1
            self._tool_stats[tool]["task_types"].add(task_type)

        # Co-occurrence: all pairs within trajectory (order-independent)
        seen = list(dict.fromkeys(tool_seq))  # deduplicate preserving order
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                pair = tuple(sorted([seen[i], seen[j]]))
                self._cooccurrence[pair] += 1  # type: ignore[index]

        # Sequential adjacency: consecutive pairs (ordered)
        for i in range(len(tool_seq) - 1):
            self._sequential[(tool_seq[i], tool_seq[i + 1])] += 1

    @classmethod
    def load_from_trajectories(
        cls,
        traj_file: str,
        min_f1: float = 0.0,
    ) -> "ToolGraphBuilder":
        """Load trajectories from a JSONL file and build statistics.

        Supports both experience_pool.jsonl (agentevolver format) and
        generic JSONL with 'tool_sequence'/'tools_called' + 'f1'/'reward'.
        """
        builder = cls()
        path = Path(traj_file)
        if not path.exists():
            logger.warning(f"Trajectory file not found: {traj_file}")
            return builder

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

                success = f1 >= 0.5
                builder.add_trajectory(tool_seq, success=success, task_type=task_type)

        logger.info(
            f"Loaded {builder._trajectory_count} trajectories from {traj_file}, "
            f"{len(builder._tool_stats)} unique tools"
        )
        return builder

    def build_graph_data(
        self, min_cooccurrence: int = 3
    ) -> dict:
        """Return graph data dict ready to pass into ToolGraph.

        Returns:
            {
                "nodes": [{"tool_slug": ..., "f1_gain": ..., "use_count": ...,
                           "task_types": [...]}, ...],
                "edges": [{"source": ..., "target": ..., "weight": ...,
                           "cooccurrence": ..., "sequential": ...}, ...],
                "trajectory_count": int,
            }
        """
        nodes = []
        for slug, stats in self._tool_stats.items():
            n = stats["count"]
            success_rate = stats["success_count"] / n if n > 0 else 0.0
            nodes.append({
                "tool_slug": slug,
                "f1_gain": round(success_rate, 4),
                "use_count": n,
                "task_types": sorted(stats["task_types"]),
            })

        edges = []
        for (tool_a, tool_b), count in self._cooccurrence.items():
            if count < min_cooccurrence:
                continue
            # Sequential strength: how often these two appear consecutively
            seq = (
                self._sequential.get((tool_a, tool_b), 0)
                + self._sequential.get((tool_b, tool_a), 0)
            )
            weight = count + 0.5 * seq
            edges.append({
                "source": tool_a,
                "target": tool_b,
                "weight": round(weight, 2),
                "cooccurrence": count,
                "sequential": seq,
            })

        # Sort edges by weight descending for readability
        edges.sort(key=lambda e: e["weight"], reverse=True)

        logger.info(
            f"Tool graph: {len(nodes)} nodes, {len(edges)} edges "
            f"(min_cooccurrence={min_cooccurrence})"
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "trajectory_count": self._trajectory_count,
        }

    def get_top_cooccurrences(self, top_k: int = 20) -> list[tuple[str, str, int]]:
        """Return top-k tool pairs by co-occurrence count."""
        sorted_pairs = sorted(
            self._cooccurrence.items(), key=lambda x: x[1], reverse=True
        )
        return [(a, b, cnt) for (a, b), cnt in sorted_pairs[:top_k]]

    def get_task_type_tool_map(self) -> dict[str, list[str]]:
        """Return {task_type: [top tools sorted by use_count]}."""
        tt_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for slug, stats in self._tool_stats.items():
            for tt in stats["task_types"]:
                tt_map[tt][slug] += stats["count"]
        result = {}
        for tt, tool_counts in tt_map.items():
            result[tt] = sorted(tool_counts, key=tool_counts.get, reverse=True)  # type: ignore
        return result
