"""Causal Graph Synthesis from CTFM.

Synthesizes an execution plan for a new query by:
  1. Scoring each tool by:  precondition keyword match  +  task-type affinity
                          +  CCA-weighted prior
  2. Keeping the top-k candidates above a relevance threshold
  3. Ordering them by causal dependency (Kahn topological sort on
     downstream_tools edges) with mean_position as tiebreaker
  4. Returning the ordered slug list as the synthesized plan

Key distinction from retrieval-based methods (SkillRL, MemRL):
  We do NOT look for a *similar past episode*.  Instead we compose a *new*
  plan from individual per-tool causal models, enabling compositional
  generalization to tool combinations never seen together in training.
"""
from __future__ import annotations

import logging
from typing import Optional

from .tool_function_model import ToolFunctionModel, _STOP_WORDS

logger = logging.getLogger(__name__)

_MIN_MATCH_SCORE = 0.04   # tools below this relevance threshold are excluded
_MAX_PLAN_LENGTH = 6


class CausalGraphSynthesizer:
    """Compose an execution plan from a CTFM library."""

    def __init__(
        self,
        models: dict[str, ToolFunctionModel],
        min_match_score: float = _MIN_MATCH_SCORE,
        max_plan_length: int = _MAX_PLAN_LENGTH,
    ) -> None:
        self._models = models
        self._min_match = min_match_score
        self._max_len = max_plan_length

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        query: str,
        task_type: str = "unknown",
        top_k: int = 5,
    ) -> list[str]:
        """Return an ordered list of tool slugs forming the synthesized plan."""
        if not self._models:
            return []

        candidates = self._score_candidates(query, task_type)
        if not candidates:
            return []

        top_slugs = {slug for slug, _ in candidates[:top_k]}
        ordered = self._topological_sort(top_slugs)
        return ordered[: self._max_len]

    def format_plan_hint(
        self,
        query: str,
        task_type: str = "unknown",
        top_k: int = 5,
    ) -> str:
        """Return a prompt section describing the synthesized causal plan."""
        plan = self.synthesize(query, task_type, top_k=top_k)
        if not plan:
            return ""

        lines = ["\n\n## Causal Execution Plan (CausalEvo)"]
        lines.append(
            "Based on causal analysis of past trajectories, "
            "the recommended tool order for this task is:"
        )
        for i, slug in enumerate(plan, 1):
            m = self._models.get(slug)
            line = f"  {i}. {slug}"
            if m and m.precondition_keywords:
                line += f"  (triggers: {', '.join(m.precondition_keywords[:3])})"
            lines.append(line)

        # Highlight causally critical tools
        critical = [
            slug for slug in plan
            if self._models.get(slug) and self._models[slug].avg_cca_score >= 0.70
        ]
        if critical:
            lines.append(
                f"\nCritically important (high causal necessity, do not skip): "
                f"{', '.join(critical)}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _score_candidates(
        self,
        query: str,
        task_type: str,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for slug, m in self._models.items():
            kw = m.precondition_match_score(query)           # keyword match [0,1]
            tt = m.task_type_affinity.get(task_type, 0.0)   # task-type affinity [0,1]
            ca = m.avg_cca_score * 0.3                       # causal prior (down-weighted)
            score = 0.50 * kw + 0.30 * tt + 0.20 * ca
            if score >= self._min_match:
                scored.append((slug, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _topological_sort(self, slugs: set[str]) -> list[str]:
        """Kahn's algorithm on downstream_tools edges within the candidate set."""
        in_degree = {s: 0 for s in slugs}
        adj: dict[str, list[str]] = {s: [] for s in slugs}

        for slug in slugs:
            m = self._models.get(slug)
            if m is None:
                continue
            for ds in m.downstream_tools:
                if ds in slugs and ds != slug:
                    adj[slug].append(ds)
                    in_degree[ds] += 1

        # Initial queue: zero-in-degree nodes sorted by mean_position
        def pos(s: str) -> float:
            return self._models[s].mean_position if s in self._models else 0.5

        queue = sorted([s for s in slugs if in_degree[s] == 0], key=pos)
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for nb in adj[node]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    # Insert in position order
                    p = pos(nb)
                    for i, s in enumerate(queue):
                        if p < pos(s):
                            queue.insert(i, nb)
                            break
                    else:
                        queue.append(nb)

        # Append any remaining (cycle fallback), sorted by position
        seen = set(result)
        tail = sorted([s for s in slugs if s not in seen], key=pos)
        result.extend(tail)
        return result
