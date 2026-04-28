"""Skill Relation Graph for GraphSkillEvo.

Extends SkillRL's flat 3-tier skill bank with explicit typed edges representing
relationships: precedes, enables, conflicts_with, generalizes, alternative_to.
Uses NetworkX for graph operations and BFS-based retrieval instead of pure BM25.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


class SkillNode:
    """A skill in the graph with metadata."""

    def __init__(
        self,
        skill_id: str,
        text: str,
        task_type: str,
        f1_gain: float = 0.0,
        use_count: int = 0,
        embedding: Optional[list] = None,
    ):
        """
        Args:
            skill_id: Unique identifier
            text: Skill description text
            task_type: Task type this skill applies to (e.g., "flood_detection")
            f1_gain: Estimated F1 improvement from using this skill
            use_count: How many times this skill has been used
            embedding: Optional embedding vector for similarity
        """
        self.skill_id = skill_id
        self.text = text
        self.task_type = task_type
        self.f1_gain = f1_gain
        self.use_count = use_count
        self.embedding = embedding or []

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "text": self.text,
            "task_type": self.task_type,
            "f1_gain": self.f1_gain,
            "use_count": self.use_count,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillNode:
        return cls(
            skill_id=data["skill_id"],
            text=data["text"],
            task_type=data["task_type"],
            f1_gain=data.get("f1_gain", 0.0),
            use_count=data.get("use_count", 0),
            embedding=data.get("embedding", []),
        )


class SkillGraph:
    """Graph of skills with typed relations."""

    RELATION_TYPES = {"precedes", "enables", "conflicts_with", "generalizes", "alternative_to"}

    def __init__(self):
        """Initialize empty directed graph."""
        self._graph = nx.DiGraph()
        self._nodes: dict[str, SkillNode] = {}  # skill_id -> SkillNode

    def add_skill(
        self,
        skill_text: str,
        task_type: str,
        skill_id: Optional[str] = None,
        f1_gain: float = 0.0,
    ) -> str:
        """Add a skill node; return its ID."""
        if skill_id is None:
            skill_id = str(uuid.uuid4())[:8]

        node = SkillNode(
            skill_id=skill_id,
            text=skill_text,
            task_type=task_type,
            f1_gain=f1_gain,
        )

        self._nodes[skill_id] = node
        self._graph.add_node(skill_id, **node.to_dict())
        logger.debug(f"Added skill {skill_id}: {skill_text[:50]}...")

        return skill_id

    def add_relation(
        self,
        src_id: str,
        dst_id: str,
        rel_type: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a typed edge from src to dst.

        Args:
            rel_type: One of RELATION_TYPES
            confidence: Edge weight in [0, 1]
        """
        if rel_type not in self.RELATION_TYPES:
            logger.warning(f"Unknown relation type {rel_type}; skipping")
            return

        if src_id not in self._nodes or dst_id not in self._nodes:
            logger.warning(f"One of {src_id}, {dst_id} not in graph; skipping edge")
            return

        self._graph.add_edge(
            src_id,
            dst_id,
            rel_type=rel_type,
            confidence=confidence,
        )
        logger.debug(f"Added edge {src_id} --[{rel_type}]--> {dst_id}")

    def get_related_skills(
        self,
        skill_id: str,
        rel_types: Optional[set[str]] = None,
        max_depth: int = 2,
    ) -> set[str]:
        """BFS from skill_id following specific relation types.

        Returns all reachable skills within max_depth steps.
        """
        if skill_id not in self._graph:
            return set()

        rel_types = rel_types or self.RELATION_TYPES
        visited = set()
        queue = [(skill_id, 0)]  # (node_id, depth)

        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_depth:
                continue

            visited.add(node)

            # Explore neighbors with matching relation types
            for neighbor in self._graph.successors(node):
                edge = self._graph.edges[node, neighbor]
                if edge.get("rel_type") in rel_types:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        visited.discard(skill_id)  # Don't include the seed
        return visited

    def get_conflicting_skills(self, skill_id: str) -> set[str]:
        """Return skills that conflict with the given skill."""
        return self.get_related_skills(skill_id, rel_types={"conflicts_with"}, max_depth=1)

    def get_prerequisite_skills(self, skill_id: str) -> set[str]:
        """Return skills that should precede this one."""
        # Reverse graph: find nodes that have "precedes" edge pointing to skill_id
        result = set()
        for src_id in self._graph.predecessors(skill_id):
            edge = self._graph.edges[src_id, skill_id]
            if edge.get("rel_type") == "precedes":
                result.add(src_id)
        return result

    def get_enabled_by(self, skill_id: str) -> set[str]:
        """Return skills enabled by the given skill."""
        return self.get_related_skills(skill_id, rel_types={"enables"}, max_depth=1)

    def save(self, path: str) -> None:
        """Save graph to JSON using node-link format."""
        data = {
            "nodes": [{"id": nid, **self._nodes[nid].to_dict()} for nid in self._nodes],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "rel_type": edge["rel_type"],
                    "confidence": edge.get("confidence", 1.0),
                }
                for u, v, edge in self._graph.edges(data=True)
            ],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved skill graph to {path}")

    @classmethod
    def load(cls, path: str) -> SkillGraph:
        """Load graph from JSON."""
        with open(path) as f:
            data = json.load(f)

        graph = cls()

        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            node = SkillNode.from_dict({**node_data, "skill_id": node_id})
            graph._nodes[node_id] = node
            graph._graph.add_node(node_id, **node.to_dict())

        # Add edges
        for edge_data in data.get("edges", []):
            graph.add_relation(
                src_id=edge_data["source"],
                dst_id=edge_data["target"],
                rel_type=edge_data["rel_type"],
                confidence=edge_data.get("confidence", 1.0),
            )

        logger.info(f"Loaded skill graph from {path}")
        return graph

    def subgraph_for_query(
        self,
        query_skill_ids: list[str],
        max_related: int = 10,
    ) -> list[str]:
        """Extract a relevant subgraph for a query.

        Given a list of matching skills, expand with related skills.
        Ordering: prerequisites, then seed, then enabled, then alternatives.
        """
        subgraph = set(query_skill_ids)

        for skill_id in query_skill_ids:
            # Add prerequisites (skills that must come first)
            prereqs = self.get_prerequisite_skills(skill_id)
            subgraph.update(list(prereqs)[:2])

            # Add enabled (skills that benefit from this)
            enabled = self.get_enabled_by(skill_id)
            subgraph.update(list(enabled)[:2])

        # Limit total size
        if len(subgraph) > max_related:
            # Keep seed + highest f1_gain
            seed_set = set(query_skill_ids)
            candidates = subgraph - seed_set
            sorted_candidates = sorted(
                candidates,
                key=lambda x: self._nodes[x].f1_gain,
                reverse=True,
            )
            subgraph = seed_set | set(sorted_candidates[: max_related - len(seed_set)])

        return sorted(list(subgraph))

    def count_nodes(self) -> int:
        return len(self._nodes)

    def count_edges(self) -> int:
        return self._graph.number_of_edges()


# ---------------------------------------------------------------------------
# ToolGraph: tool-level co-occurrence graph (replaces SkillGraph as primary)
# ---------------------------------------------------------------------------

class ToolNode:
    """A tool as a graph node."""

    def __init__(
        self,
        tool_slug: str,
        f1_gain: float = 0.0,
        use_count: int = 0,
        task_types: Optional[list] = None,
    ):
        self.tool_slug = tool_slug
        self.f1_gain = f1_gain
        self.use_count = use_count
        self.task_types: list[str] = task_types or []

    def to_dict(self) -> dict:
        return {
            "tool_slug": self.tool_slug,
            "f1_gain": self.f1_gain,
            "use_count": self.use_count,
            "task_types": self.task_types,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolNode":
        return cls(
            tool_slug=data["tool_slug"],
            f1_gain=data.get("f1_gain", 0.0),
            use_count=data.get("use_count", 0),
            task_types=data.get("task_types", []),
        )


class ToolGraph:
    """Undirected tool co-occurrence graph (pure Python, no networkx).

    Nodes  = tool slugs (e.g. "geo_raster.calculate_index")
    Edges  = tools that frequently co-appear in trajectories
    Weight = co-occurrence count + 0.5 × sequential adjacency count
    """

    def __init__(self):
        # adjacency: tool_slug → {neighbor_slug: {"weight": float, ...}}
        self._adj: dict[str, dict[str, dict]] = {}
        self._nodes: dict[str, ToolNode] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_tool(self, tool_slug: str, f1_gain: float = 0.0,
                 use_count: int = 0, task_types: Optional[list] = None) -> None:
        if tool_slug not in self._nodes:
            node = ToolNode(tool_slug, f1_gain, use_count, task_types or [])
            self._nodes[tool_slug] = node
            self._adj[tool_slug] = {}

    def add_cooccurrence(self, tool_a: str, tool_b: str,
                         weight: float = 1.0, cooccurrence: int = 1,
                         sequential: int = 0) -> None:
        """Add or update a co-occurrence edge (undirected)."""
        if tool_a not in self._nodes:
            self.add_tool(tool_a)
        if tool_b not in self._nodes:
            self.add_tool(tool_b)
        edge_data = {"weight": weight, "cooccurrence": cooccurrence, "sequential": sequential}
        self._adj[tool_a][tool_b] = edge_data
        self._adj[tool_b][tool_a] = edge_data

    @classmethod
    def from_graph_data(cls, data: dict) -> "ToolGraph":
        """Build from the dict returned by ToolGraphBuilder.build_graph_data()."""
        g = cls()
        for n in data.get("nodes", []):
            g.add_tool(
                n["tool_slug"],
                f1_gain=n.get("f1_gain", 0.0),
                use_count=n.get("use_count", 0),
                task_types=n.get("task_types", []),
            )
        for e in data.get("edges", []):
            g.add_cooccurrence(
                e["source"], e["target"],
                weight=e.get("weight", 1.0),
                cooccurrence=e.get("cooccurrence", 1),
                sequential=e.get("sequential", 0),
            )
        return g

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_cooccurring_tools(self, tool_slug: str, top_k: int = 5) -> list[str]:
        """Return top-k most co-occurring neighbors sorted by edge weight."""
        if tool_slug not in self._adj:
            return []
        neighbors = [
            (nbr, data.get("weight", 0.0))
            for nbr, data in self._adj[tool_slug].items()
        ]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in neighbors[:top_k]]

    def expand_toolset(self, seed_tools: list[str], top_k: int = 8) -> list[str]:
        """Expand a set of seed tools by adding their top co-occurring neighbors.

        Returns an ordered list: seeds first, then neighbors ranked by
        edge weight × neighbor f1_gain.
        """
        expanded: dict[str, float] = {t: 999.0 for t in seed_tools}  # seeds get high priority

        for tool in seed_tools:
            for nbr in self.get_cooccurring_tools(tool, top_k=3):
                if nbr not in expanded:
                    edge_data = self._adj.get(tool, {}).get(nbr, {})
                    edge_weight = edge_data.get("weight", 0.0)
                    node_f1 = self._nodes[nbr].f1_gain if nbr in self._nodes else 0.0
                    expanded[nbr] = edge_weight * (1.0 + node_f1)

        # Sort: seeds first (score=999), then by combined score
        ordered = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ordered[:top_k]]

    def get_tools_for_task_type(self, task_type: str, top_k: int = 6) -> list[str]:
        """Return top tools for a given task type, ranked by use_count × f1_gain."""
        candidates = [
            (slug, node)
            for slug, node in self._nodes.items()
            if task_type in node.task_types
        ]
        candidates.sort(
            key=lambda x: x[1].use_count * (1.0 + x[1].f1_gain),
            reverse=True,
        )
        return [slug for slug, _ in candidates[:top_k]]

    def top_cooccurrence_pairs(self, top_k: int = 10) -> list[tuple[str, str, float]]:
        """Return top-k edges by weight for prompt display."""
        seen: set[frozenset] = set()
        edges = []
        for u, neighbors in self._adj.items():
            for v, d in neighbors.items():
                key = frozenset([u, v])
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v, d.get("weight", 0.0)))
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        seen: set[frozenset] = set()
        edge_list = []
        for u, neighbors in self._adj.items():
            for v, d in neighbors.items():
                key = frozenset([u, v])
                if key not in seen:
                    seen.add(key)
                    edge_list.append({
                        "source": u,
                        "target": v,
                        "weight": d.get("weight", 1.0),
                        "cooccurrence": d.get("cooccurrence", 1),
                        "sequential": d.get("sequential", 0),
                    })

        data = {
            "nodes": [{"id": slug, **node.to_dict()} for slug, node in self._nodes.items()],
            "edges": edge_list,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            import json as _json
            _json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved ToolGraph to {path}")

    @classmethod
    def load(cls, path: str) -> "ToolGraph":
        with open(path) as f:
            import json as _json
            data = _json.load(f)
        g = cls()
        for n in data.get("nodes", []):
            node_id = n.pop("id", n.get("tool_slug", ""))
            n.setdefault("tool_slug", node_id)
            g.add_tool(
                n["tool_slug"],
                f1_gain=n.get("f1_gain", 0.0),
                use_count=n.get("use_count", 0),
                task_types=n.get("task_types", []),
            )
        for e in data.get("edges", []):
            g.add_cooccurrence(
                e["source"], e["target"],
                weight=e.get("weight", 1.0),
                cooccurrence=e.get("cooccurrence", 1),
                sequential=e.get("sequential", 0),
            )
        logger.info(f"Loaded ToolGraph from {path}: "
                    f"{g.count_nodes()} tools, {g.count_edges()} edges")
        return g

    def count_nodes(self) -> int:
        return len(self._nodes)

    def count_edges(self) -> int:
        seen: set[frozenset] = set()
        for u, neighbors in self._adj.items():
            for v in neighbors:
                seen.add(frozenset([u, v]))
        return len(seen)
