"""SeqGraphEvo PromptAugmenter: cross-task sequential pattern composition.

Compared with GraphSkillEvo (undirected co-occurrence), this method:
1. Uses DIRECTED transition edges (A→B order matters)
2. Mines frequent sequential patterns (ordered multi-step workflows)
3. Detects anti-patterns (tool combinations associated with failures)
4. Performs CROSS-TASK COMPOSITION: finds patterns from different task types
   and assembles them into a novel workflow for the current query

Key differentiator vs raw chain retrieval (MemRL-style):
  - MemRL retrieves entire episode memories similar to the query
  - SeqGraphEvo assembles workflows from FRAGMENTS of patterns across tasks
    e.g., for an unseen task combining flood + vegetation analysis:
    takes the flood-detection ordered sequence [boundary→raster→NDWI→threshold]
    and the vegetation-type sequence [calculate_index(NDVI)→statistics]
    and composes them into a combined workflow
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .seq_graph import SeqGraph
from .seq_pattern_miner import SeqPatternMiner

logger = logging.getLogger(__name__)


def _keyword_score(query: str, tool_slug: str) -> float:
    """Simple keyword overlap score between query and tool slug."""
    slug_words = set(re.split(r"[._]", tool_slug.lower()))
    query_words = set(re.split(r"\W+", query.lower()))
    if not slug_words or not query_words:
        return 0.0
    return len(slug_words & query_words) / len(slug_words)


def _rank_tools_by_query(
    query: str, tool_slugs: list[str], top_k: int = 5
) -> list[str]:
    scored = [(s, _keyword_score(query, s)) for s in tool_slugs]
    scored.sort(key=lambda x: x[1], reverse=True)
    positives = [s for s, sc in scored if sc > 0]
    return positives[:top_k] if positives else [s for s, _ in scored[:top_k]]


class SeqGraphEvoPromptInjector(PromptAugmenter):
    """Augment prompts using sequential pattern mining + cross-task composition."""

    def __init__(
        self,
        seq_graph_path: Optional[str] = None,
        traj_file: Optional[str] = None,
        top_k: int = 6,
    ):
        self._top_k = top_k
        self._graph: Optional[SeqGraph] = None

        if seq_graph_path and Path(seq_graph_path).exists():
            self._graph = SeqGraph.load(seq_graph_path)
        elif traj_file and Path(traj_file).exists():
            self._graph = self._build_from_trajectories(traj_file)
            if seq_graph_path:
                self._graph.save(seq_graph_path)
        else:
            logger.warning(
                "SeqGraphEvoPromptInjector: no seq_graph_path or traj_file; "
                "augment() will return base system prompt only."
            )

        self._all_tools: list[str] = (
            list(self._graph._nodes.keys()) if self._graph else []
        )

    def _build_from_trajectories(self, traj_file: str) -> SeqGraph:
        miner = SeqPatternMiner.load_from_trajectories(traj_file)
        return _build_seq_graph(miner)

    def augment(self, user_query: str, task_type: str = "general", **kwargs) -> str:
        if self._graph is None or not self._all_tools:
            return self.BASE_SYSTEM

        # Step 1: keyword seed tools
        seed_tools = _rank_tools_by_query(user_query, self._all_tools, top_k=4)

        # Step 2: cross-task composition
        composed = self._graph.expand_with_composition(
            seed_tools, task_type=task_type, top_k=self._top_k
        )

        # Step 3: task-type top tools
        task_tools = self._graph.get_tools_for_task_type(task_type, top_k=4)

        # Step 4: top sequential patterns for context
        matching_patterns = self._graph.find_patterns_containing(seed_tools, top_k=5)

        # Step 5: top forward edges for workflow hint
        top_edges = self._graph.top_forward_edges(top_k=6)

        return self._format_system_prompt(
            composed, task_tools, matching_patterns, top_edges, task_type
        )

    def _format_system_prompt(
        self,
        composed_tools: list[str],
        task_tools: list[str],
        patterns: list[dict],
        top_edges: list[tuple[str, str, int]],
        task_type: str,
    ) -> str:
        tool_list = "\n".join(f"  • {t}" for t in composed_tools) or "  (none identified)"

        task_section = ""
        if task_tools:
            task_section = (
                f"\nFREQUENTLY USED TOOLS for '{task_type}' tasks:\n"
                + "\n".join(f"  • {t}" for t in task_tools)
            )

        pattern_section = ""
        if patterns:
            pattern_section = "\nFREQUENT SEQUENTIAL WORKFLOWS (cross-task patterns):\n"
            for p in patterns[:4]:
                seq_str = " → ".join(p["pattern"])
                dominant_task = max(
                    p["task_types"].items(), key=lambda x: x[1], default=("", 0)
                )
                pattern_section += (
                    f"  [{p['support']} uses, F1={p['avg_f1']:.2f}] "
                    f"{seq_str}"
                )
                if dominant_task[0]:
                    pattern_section += f"  (mainly {dominant_task[0]})"
                pattern_section += "\n"

        edge_section = ""
        if top_edges:
            edge_section = "\nCOMMON TOOL TRANSITIONS (what follows what):\n"
            for src, tgt, cnt in top_edges[:5]:
                edge_section += f"  • {src}  →  {tgt}  ({cnt} times)\n"

        return (
            self.BASE_SYSTEM
            + f"""

## Sequential Pattern-Based Tool Recommendations

Tools identified via cross-task sequential pattern mining from
{getattr(self._graph, '_trajectory_count', 'historical')} geospatial analysis trajectories.

RECOMMENDED TOOLS (composed from matching sequential patterns):
{tool_list}
{task_section}
{pattern_section}
{edge_section}
GUIDANCE:
1. Follow the tool transitions shown above — order matters for sequential tasks
2. Sequential patterns show proven multi-step workflows; use them as templates
3. Tools from patterns that span multiple task types provide robust coverage
4. You may add tools not listed if the task requires them
"""
        )


def _build_seq_graph(
    miner: SeqPatternMiner,
    min_pattern_support: int = 5,
    max_pattern_len: int = 4,
    min_edge_count: int = 3,
    min_anti_support: int = 3,
) -> SeqGraph:
    """Build a SeqGraph from a SeqPatternMiner."""
    g = SeqGraph()
    g._trajectory_count = miner._trajectory_count

    # Add nodes from tool stats
    for slug, stats in miner.get_tool_stats().items():
        g.add_tool(
            slug,
            count=stats["count"],
            avg_f1=stats["avg_f1"],
            task_types=stats["task_types"],
        )

    # Add directed forward edges
    for edge in miner.get_directed_edges(min_count=min_edge_count):
        g.add_forward_edge(
            edge["source"], edge["target"],
            count=edge["count"],
            avg_f1=edge["avg_f1"],
            task_types=edge["task_types"],
        )

    # Add sequential patterns
    patterns = miner.mine_patterns(
        min_support=min_pattern_support,
        max_len=max_pattern_len,
    )
    g.set_patterns(patterns)

    # Add anti-patterns
    anti_patterns = miner.mine_anti_patterns(
        max_f1=0.3,
        min_support=min_anti_support,
    )
    g.set_anti_patterns(anti_patterns)

    return g
