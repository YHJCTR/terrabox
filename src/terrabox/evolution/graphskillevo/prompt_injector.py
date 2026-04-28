"""GraphSkillEvo PromptAugmenter: Tool co-occurrence graph retrieval.

Instead of storing LLM-distilled skill texts, this method:
1. Builds a tool co-occurrence graph from real execution trajectories
2. Identifies relevant tools from the query via keyword/BM25 matching
3. Expands the tool set by following co-occurrence edges in the graph
4. Injects structured tool recommendations into the system prompt

This provides statistically-grounded tool selection hints without LLM noise.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .skill_graph import ToolGraph
from .tool_graph_builder import ToolGraphBuilder

logger = logging.getLogger(__name__)

# Geo tool prefix patterns for filtering
_GEO_PREFIXES = (
    "geo_", "stac_", "osm_", "earth_", "disaster_",
    "bash.", "ipython_", "bing_", "github.",
)


def _is_geo_tool(slug: str) -> bool:
    return any(slug.startswith(p) for p in _GEO_PREFIXES)


def _keyword_score(query: str, tool_slug: str) -> float:
    """Simple keyword overlap score between query and tool slug."""
    # Split slug into words: "geo_raster.calculate_index" → ["geo", "raster", "calculate", "index"]
    slug_words = set(re.split(r"[._]", tool_slug.lower()))
    query_words = set(re.split(r"\W+", query.lower()))
    if not slug_words or not query_words:
        return 0.0
    return len(slug_words & query_words) / len(slug_words)


def _rank_tools_by_query(
    query: str,
    tool_slugs: list[str],
    top_k: int = 5,
) -> list[str]:
    """Rank tools by keyword relevance to the query."""
    scored = [(slug, _keyword_score(query, slug)) for slug in tool_slugs]
    scored.sort(key=lambda x: x[1], reverse=True)
    # Return all tools with score > 0, up to top_k; fallback to top tools by index
    positives = [s for s, sc in scored if sc > 0]
    if positives:
        return positives[:top_k]
    # No keyword match — return top_k by order (highest use_count assumed first)
    return [s for s, _ in scored[:top_k]]


class GraphSkillEvoPromptInjector(PromptAugmenter):
    """Augment prompts using tool co-occurrence graph retrieval.

    Compared with SkillRL (flat skill text retrieval), this method:
    1. Uses statistically-derived tool co-occurrence instead of LLM guesses
    2. Expands the relevant tool set via graph neighborhood traversal
    3. Shows frequency-based evidence ("used together N times") to the LLM
    4. Distinguishes task-type-specific vs general tool patterns
    """

    def __init__(
        self,
        tool_graph_path: Optional[str] = None,
        traj_file: Optional[str] = None,
        top_k: int = 6,
    ):
        """
        Args:
            tool_graph_path: Path to saved ToolGraph JSON. If None/missing,
                             will try to build from traj_file.
            traj_file: Trajectory JSONL for building the graph on-the-fly.
            top_k: Number of tools to recommend.
        """
        self._top_k = top_k
        self._graph: Optional[ToolGraph] = None
        self._task_type_map: dict[str, list[str]] = {}

        # Try to load pre-built graph
        if tool_graph_path and Path(tool_graph_path).exists():
            self._graph = ToolGraph.load(tool_graph_path)
            logger.info(f"Loaded ToolGraph from {tool_graph_path}")
        elif traj_file and Path(traj_file).exists():
            self._graph = self._build_from_trajectories(traj_file)
            if tool_graph_path:
                self._graph.save(tool_graph_path)
        else:
            logger.warning(
                "GraphSkillEvoPromptInjector: no tool_graph_path or traj_file provided; "
                "augment() will return base system prompt only."
            )

        if self._graph is not None:
            # Pre-build task_type → top tools index for fast lookup
            self._task_type_map = self._graph.get_tools_for_task_type.__self__.__class__  # type: ignore
            self._all_tools = list(self._graph._nodes.keys())
        else:
            self._all_tools = []

    def _build_from_trajectories(self, traj_file: str) -> ToolGraph:
        builder = ToolGraphBuilder.load_from_trajectories(traj_file)
        graph_data = builder.build_graph_data(min_cooccurrence=2)
        return ToolGraph.from_graph_data(graph_data)

    def build_tool_graph_from_trajectories(
        self, traj_file: str, save_path: Optional[str] = None,
        min_cooccurrence: int = 2,
    ) -> ToolGraph:
        """Public method: build/rebuild graph from trajectory file."""
        builder = ToolGraphBuilder.load_from_trajectories(traj_file)
        graph_data = builder.build_graph_data(min_cooccurrence=min_cooccurrence)
        self._graph = ToolGraph.from_graph_data(graph_data)
        self._all_tools = list(self._graph._nodes.keys())
        if save_path:
            self._graph.save(save_path)
        return self._graph

    def augment(self, user_query: str, task_type: str = "general", **kwargs) -> str:
        """Retrieve and inject tool co-occurrence recommendations."""
        if self._graph is None or not self._all_tools:
            logger.warning("No tool graph available; returning base system prompt")
            return self.BASE_SYSTEM

        # Step 1: keyword-based seed tool selection
        seed_tools = _rank_tools_by_query(user_query, self._all_tools, top_k=4)

        # Step 2: expand via co-occurrence graph
        expanded = self._graph.expand_toolset(seed_tools, top_k=self._top_k)

        # Step 3: also get task-type specific top tools
        task_tools = self._graph.get_tools_for_task_type(task_type, top_k=4)

        # Step 4: get top co-occurrence pairs for context
        top_pairs = self._graph.top_cooccurrence_pairs(top_k=5)

        # Step 5: format system prompt
        system_prompt = self._format_system_prompt(
            expanded, task_tools, top_pairs, task_type, user_query
        )
        logger.debug(
            f"GraphSkillEvo augmented: {len(expanded)} tools, "
            f"task_type={task_type}"
        )
        return system_prompt

    def _format_system_prompt(
        self,
        expanded_tools: list[str],
        task_tools: list[str],
        top_pairs: list[tuple[str, str, float]],
        task_type: str,
        query: str,
    ) -> str:
        tool_list = "\n".join(f"  • {t}" for t in expanded_tools) or "  (none identified)"

        task_section = ""
        if task_tools:
            task_section = (
                f"\nFREQUENTLY USED TOOLS for '{task_type}' tasks:\n"
                + "\n".join(f"  • {t}" for t in task_tools)
            )

        pairs_section = ""
        if top_pairs:
            pairs_section = "\nCOMMON TOOL COMBINATIONS (from trajectory statistics):\n"
            for a, b, w in top_pairs:
                pairs_section += f"  • {a}  +  {b}  (co-used {int(w)} times)\n"

        return (
            self.BASE_SYSTEM
            + f"""

## Graph-Based Tool Recommendations

The following tools were identified as relevant based on co-occurrence patterns
extracted from {getattr(self._graph, '_trajectory_count', 'historical')} geospatial analysis trajectories.

RECOMMENDED TOOLS for this query:
{tool_list}
{task_section}
{pairs_section}
GUIDANCE:
1. Prefer tools from the recommended list when applicable
2. Tools listed in "common combinations" work well together
3. You may use additional tools not in this list if the task requires them
4. Tool order matters: follow common sequential patterns when possible
"""
        )

    # Keep backward-compatible stub for old build_graph_from_skillrl callers
    def build_graph_from_skillrl(self, *args, **kwargs) -> None:
        logger.warning(
            "build_graph_from_skillrl() is deprecated. "
            "Use build_tool_graph_from_trajectories() instead."
        )
