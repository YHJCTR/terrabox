"""SeqGraph: directed tool transition graph + sequential pattern library.

Unlike the undirected co-occurrence ToolGraph in GraphSkillEvo, SeqGraph records:
  - Directed edges: tool A → tool B (B often follows A in successful trajectories)
  - Anti-pattern pairs: tools that co-occur in failed trajectories
  - Pattern library: frequent contiguous subsequences (length 2–4)

The combination enables cross-task composition:
  keyword seed tools → find matching patterns (possibly from different task types)
  → assemble a multi-step ordered workflow
  → filter out anti-pattern combinations
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SeqGraph:
    """Directed sequential tool graph with pattern library and anti-patterns.

    Nodes  = tool slugs
    Fwd edges  = frequent A→B transitions in successful trajectories
    Anti edges = tool pairs that appear together in failed trajectories
    Patterns   = frequent contiguous tool subsequences (ordered workflows)
    """

    def __init__(self):
        self._nodes: dict[str, dict] = {}          # tool_slug → {count, avg_f1, task_types}
        self._fwd_adj: dict[str, dict[str, dict]] = {}   # src → {tgt → edge_data}
        self._anti_pairs: set[frozenset] = set()   # frozenset({tool_a, tool_b})
        self._patterns: list[dict] = []            # list of pattern dicts
        self._anti_patterns: list[dict] = []       # list of anti-pattern dicts
        self._trajectory_count: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_tool(
        self,
        tool_slug: str,
        count: int = 0,
        avg_f1: float = 0.0,
        task_types: Optional[dict] = None,
    ) -> None:
        if tool_slug not in self._nodes:
            self._nodes[tool_slug] = {
                "count": count,
                "avg_f1": avg_f1,
                "task_types": task_types or {},
            }
            self._fwd_adj[tool_slug] = {}

    def add_forward_edge(
        self,
        source: str,
        target: str,
        count: int,
        avg_f1: float,
        task_types: Optional[dict] = None,
    ) -> None:
        """Add a directed edge source → target."""
        if source not in self._nodes:
            self.add_tool(source)
        if target not in self._nodes:
            self.add_tool(target)
        self._fwd_adj[source][target] = {
            "count": count,
            "avg_f1": avg_f1,
            "task_types": task_types or {},
        }

    def add_anti_pattern(self, tool_a: str, tool_b: str) -> None:
        """Register a tool pair as an anti-pattern (appears in failures)."""
        self._anti_pairs.add(frozenset([tool_a, tool_b]))

    def set_patterns(self, patterns: list[dict]) -> None:
        self._patterns = patterns

    def set_anti_patterns(self, anti_patterns: list[dict]) -> None:
        self._anti_patterns = anti_patterns
        for p in anti_patterns:
            tools = p.get("pattern", [])
            for i in range(len(tools)):
                for j in range(i + 1, len(tools)):
                    self._anti_pairs.add(frozenset([tools[i], tools[j]]))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_next_tools(self, tool_slug: str, top_k: int = 5) -> list[str]:
        """Return top-k tools that most frequently follow this tool."""
        neighbors = [
            (tgt, data["count"])
            for tgt, data in self._fwd_adj.get(tool_slug, {}).items()
        ]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in neighbors[:top_k]]

    def is_anti_pattern(self, tool_a: str, tool_b: str) -> bool:
        return frozenset([tool_a, tool_b]) in self._anti_pairs

    def find_patterns_containing(
        self, seed_tools: list[str], top_k: int = 5
    ) -> list[dict]:
        """Find frequent patterns that contain at least one seed tool."""
        seed_set = set(seed_tools)
        scored = []
        for p in self._patterns:
            overlap = len(set(p["pattern"]) & seed_set)
            if overlap == 0:
                continue
            score = p["support"] * p["avg_f1"] * overlap
            scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    def expand_with_composition(
        self,
        seed_tools: list[str],
        task_type: str = "general",
        top_k: int = 8,
    ) -> list[str]:
        """Cross-task composition: assemble an ordered tool workflow.

        Algorithm:
        1. Find patterns that contain any seed tool (may come from different task types)
        2. Score patterns by support × avg_f1 × task_type_boost × seed_overlap
        3. Compose workflow by merging high-scoring patterns preserving order
        4. Fill gaps using directed fwd edges from seed tools
        5. Filter out tool pairs flagged as anti-patterns
        """
        seed_set = set(seed_tools)

        # Step 1+2: score patterns
        scored_patterns: list[tuple[dict, float]] = []
        for p in self._patterns:
            overlap = len(set(p["pattern"]) & seed_set)
            if overlap == 0:
                continue
            # Boost patterns relevant to the task type
            task_count = p["task_types"].get(task_type, 0)
            task_boost = 1.0 + task_count / max(p["support"], 1)
            score = p["support"] * p["avg_f1"] * task_boost + overlap * 5.0
            scored_patterns.append((p, score))

        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        # Step 3: compose workflow
        # tool → (score, position_hint)
        workflow: dict[str, float] = {}

        for p, pscore in scored_patterns[:6]:  # top-6 patterns
            for pos, tool in enumerate(p["pattern"]):
                pos_weight = len(p["pattern"]) - pos  # earlier = higher
                new_score = pscore * (1.0 + 0.1 * pos_weight)
                if tool not in workflow or workflow[tool] < new_score:
                    workflow[tool] = new_score

        # Seeds always get highest priority
        for t in seed_tools:
            workflow[t] = max(workflow.get(t, 0.0), 9999.0)

        # Step 4: for each seed, add its top-2 forward successors
        for tool in seed_tools:
            for succ in self.get_next_tools(tool, top_k=2):
                if succ not in workflow:
                    edge = self._fwd_adj.get(tool, {}).get(succ, {})
                    workflow[succ] = edge.get("count", 1) * edge.get("avg_f1", 0.5)

        # Step 5: remove anti-pattern violations (greedy)
        candidates = sorted(workflow.items(), key=lambda x: x[1], reverse=True)
        result: list[str] = []
        for tool, _ in candidates:
            # Check if adding this tool would create an anti-pattern with existing results
            has_conflict = any(self.is_anti_pattern(tool, existing) for existing in result)
            if not has_conflict:
                result.append(tool)
            if len(result) >= top_k:
                break

        return result

    def get_tools_for_task_type(self, task_type: str, top_k: int = 6) -> list[str]:
        """Return top tools for a given task type, ranked by use count."""
        candidates = []
        for slug, data in self._nodes.items():
            count = data["task_types"].get(task_type, 0)
            if count > 0:
                candidates.append((slug, count * (1.0 + data["avg_f1"])))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [slug for slug, _ in candidates[:top_k]]

    def top_forward_edges(self, top_k: int = 10) -> list[tuple[str, str, int]]:
        """Return top directed edges by count."""
        all_edges = []
        for src, neighbors in self._fwd_adj.items():
            for tgt, data in neighbors.items():
                all_edges.append((src, tgt, data["count"]))
        all_edges.sort(key=lambda x: x[2], reverse=True)
        return all_edges[:top_k]

    def top_patterns(self, top_k: int = 10) -> list[dict]:
        return self._patterns[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save graph to JSON."""
        # Serialize forward adjacency
        fwd_edges = []
        for src, neighbors in self._fwd_adj.items():
            for tgt, data in neighbors.items():
                fwd_edges.append({"source": src, "target": tgt, **data})

        data = {
            "nodes": [{"id": slug, **info} for slug, info in self._nodes.items()],
            "forward_edges": fwd_edges,
            "anti_pairs": [list(pair) for pair in self._anti_pairs],
            "patterns": self._patterns,
            "anti_patterns": self._anti_patterns,
            "trajectory_count": self._trajectory_count,
        }

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved SeqGraph to {path}: "
            f"{self.count_nodes()} nodes, {self.count_fwd_edges()} edges, "
            f"{len(self._patterns)} patterns, {len(self._anti_patterns)} anti-patterns"
        )

    @classmethod
    def load(cls, path: str) -> "SeqGraph":
        """Load SeqGraph from JSON."""
        with open(path) as f:
            data = json.load(f)

        g = cls()
        g._trajectory_count = data.get("trajectory_count", 0)

        for n in data.get("nodes", []):
            slug = n.pop("id")
            g.add_tool(slug, **n)

        for e in data.get("forward_edges", []):
            g.add_forward_edge(
                e["source"], e["target"],
                count=e.get("count", 1),
                avg_f1=e.get("avg_f1", 0.5),
                task_types=e.get("task_types"),
            )

        for pair in data.get("anti_pairs", []):
            if len(pair) == 2:
                g._anti_pairs.add(frozenset(pair))

        g._patterns = data.get("patterns", [])
        g._anti_patterns = data.get("anti_patterns", [])

        logger.info(
            f"Loaded SeqGraph from {path}: "
            f"{g.count_nodes()} nodes, {g.count_fwd_edges()} edges, "
            f"{len(g._patterns)} patterns"
        )
        return g

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count_nodes(self) -> int:
        return len(self._nodes)

    def count_fwd_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self._fwd_adj.values())

    def count_patterns(self) -> int:
        return len(self._patterns)

    def count_anti_patterns(self) -> int:
        return len(self._anti_patterns)

    def connectivity_stats(self) -> dict:
        """Return connectivity statistics."""
        nodes_with_outedge = sum(1 for nbrs in self._fwd_adj.values() if nbrs)
        nodes_with_inedge: set[str] = set()
        for nbrs in self._fwd_adj.values():
            nodes_with_inedge.update(nbrs.keys())
        connected = len(set(
            s for s in self._fwd_adj if self._fwd_adj[s]
        ) | nodes_with_inedge)
        return {
            "nodes": self.count_nodes(),
            "fwd_edges": self.count_fwd_edges(),
            "nodes_with_outedge": nodes_with_outedge,
            "connected_nodes": connected,
            "patterns": len(self._patterns),
            "anti_patterns": len(self._anti_patterns),
            "anti_pairs": len(self._anti_pairs),
        }
