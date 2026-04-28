"""CausalTextEvo PromptAugmenter: knowledge-state-driven prompt injection.

Uses the optimized KnowledgeState θ to augment agent system prompts.
Supports ablation flags for controlled experiments.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .knowledge_state import KnowledgeState

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "for",
    "and", "or", "with", "at", "on", "by", "what", "how", "where", "which",
    "this", "that", "it", "i", "you", "we", "they", "be", "do", "does",
    "please", "find", "get", "use", "can", "will", "if", "my", "your",
    "me", "us", "them", "its", "from", "about", "some", "all", "into",
})

# Ablation flags
ABLATION_NO_TEXTGRAD = "no_textgrad"
ABLATION_NO_CCA_GUIDE = "no_cca_guide"
ABLATION_NO_PATTERN = "no_pattern"
ABLATION_NO_ANTI_PATTERN = "no_anti_pattern"
ABLATION_NO_SKILL_TEXT = "no_skill_text"
ABLATION_STATIC_KEYWORDS = "static_keywords"

ALL_ABLATIONS = [
    ABLATION_NO_TEXTGRAD, ABLATION_NO_CCA_GUIDE, ABLATION_NO_PATTERN,
    ABLATION_NO_ANTI_PATTERN, ABLATION_NO_SKILL_TEXT, ABLATION_STATIC_KEYWORDS,
]


def _keyword_match_score(query: str, keywords: list[str]) -> float:
    """Word-level overlap between query and keyword list."""
    if not keywords:
        return 0.0
    q_words = {w.strip(".,!?;:\"'()[]").lower() for w in query.split()} - _STOP_WORDS
    kw_set = {kw.lower() for kw in keywords}
    if not q_words or not kw_set:
        return 0.0
    return len(q_words & kw_set) / len(kw_set)


def _slug_keyword_score(query: str, tool_slug: str) -> float:
    """Keyword overlap between query words and tool slug components."""
    slug_words = set(re.split(r"[._]", tool_slug.lower()))
    query_words = set(re.split(r"\\W+", query.lower()))
    if not slug_words or not query_words:
        return 0.0
    return len(slug_words & query_words) / len(slug_words)


class CausalTextEvoPromptInjector(PromptAugmenter):
    """Augment prompts using CausalTextEvo knowledge state."""

    def __init__(
        self,
        state: KnowledgeState,
        top_k: int = 8,
        ablation: Optional[str] = None,
    ):
        self._state = state
        self._top_k = top_k
        self._ablation = ablation
        self._last_predicted_tools: list[str] = []

    def predict_tools(
        self,
        query: str,
        task_type: str = "general",
    ) -> list[str]:
        """Predict tool sequence using knowledge state θ. Used for offline eval."""
        state = self._state

        # Step 1: Score tools by keyword match + CCA
        scored: dict[str, float] = {}
        for tool, keywords in state.tool_keywords.items():
            kw_score = _keyword_match_score(query, keywords)
            slug_score = _slug_keyword_score(query, tool)
            combined_kw = max(kw_score, slug_score)

            if self._ablation == ABLATION_NO_CCA_GUIDE:
                cca = 0.5
            else:
                cca = state.cca_scores.get(tool, 0.5)

            # Task type affinity
            tool_stat = state.tool_stats.get(tool, {})
            task_types = tool_stat.get("task_types", {})
            total_uses = sum(task_types.values()) or 1
            task_affinity = task_types.get(task_type, 0) / total_uses

            score = 0.40 * combined_kw + 0.30 * task_affinity + 0.30 * cca
            if score > 0.01:
                scored[tool] = score

        # Step 2: Pattern expansion (if not ablated)
        if self._ablation != ABLATION_NO_PATTERN and state.seq_patterns:
            seed_tools = sorted(scored.keys(), key=lambda t: scored[t], reverse=True)[:4]
            seed_set = set(seed_tools)
            for p in state.seq_patterns:
                overlap = len(set(p["pattern"]) & seed_set)
                if overlap == 0:
                    continue
                task_count = p.get("task_types", {}).get(task_type, 0)
                task_boost = 1.0 + task_count / max(p.get("support", 1), 1)
                p_score = p.get("support", 1) * p.get("avg_f1", 0.5) * task_boost
                for pos, tool in enumerate(p["pattern"]):
                    pos_weight = len(p["pattern"]) - pos
                    bonus = p_score * (1.0 + 0.1 * pos_weight) * 0.01
                    scored[tool] = max(scored.get(tool, 0.0), bonus)

            # Forward edge expansion from seeds
            for tool in seed_tools:
                for succ in state.get_downstream_tools(tool, top_k=2):
                    if succ not in scored:
                        scored[succ] = 0.05

        # Step 3: Anti-pattern filtering (if not ablated)
        candidates = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        if self._ablation == ABLATION_NO_ANTI_PATTERN:
            result = [t for t, _ in candidates[:self._top_k]]
        else:
            anti_set = {frozenset(p) for p in state.anti_pairs}
            result: list[str] = []
            for tool, _ in candidates:
                has_conflict = any(
                    frozenset([tool, existing]) in anti_set
                    for existing in result
                )
                if not has_conflict:
                    result.append(tool)
                if len(result) >= self._top_k:
                    break

        self._last_predicted_tools = result
        return result

    def augment(self, user_query: str, task_type: str = "general", **kwargs) -> str:
        """Build augmented system prompt from knowledge state θ."""
        state = self._state

        # Predict tools for prompt
        recommended = self.predict_tools(user_query, task_type)

        # Task-type tools
        task_tools = self._get_task_tools(task_type, exclude=set(recommended))

        # Matching patterns
        matching_patterns = self._get_matching_patterns(recommended, task_type)

        # Forward edges
        top_edges = self._get_top_edges(recommended)

        # Strategy text
        skill_text = ""
        if self._ablation != ABLATION_NO_SKILL_TEXT:
            skill_text = state.skill_texts.get(task_type, "")

        return self._format_prompt(
            recommended, task_tools, matching_patterns,
            top_edges, task_type, skill_text,
        )

    def _get_task_tools(self, task_type: str, exclude: set, top_k: int = 4) -> list[str]:
        candidates = []
        for tool, stats in self._state.tool_stats.items():
            if tool in exclude:
                continue
            count = stats.get("task_types", {}).get(task_type, 0)
            if count > 0:
                candidates.append((tool, count * (1.0 + stats.get("avg_f1", 0.5))))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in candidates[:top_k]]

    def _get_matching_patterns(
        self, seed_tools: list[str], task_type: str, top_k: int = 5
    ) -> list[dict]:
        if self._ablation == ABLATION_NO_PATTERN:
            return []
        seed_set = set(seed_tools)
        scored = []
        for p in self._state.seq_patterns:
            overlap = len(set(p["pattern"]) & seed_set)
            if overlap == 0:
                continue
            score = p.get("support", 1) * p.get("avg_f1", 0.5) * overlap
            scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    def _get_top_edges(self, tools: list[str], top_k: int = 6) -> list[dict]:
        tool_set = set(tools)
        relevant = [
            e for e in self._state.forward_edges
            if e.get("source") in tool_set or e.get("target") in tool_set
        ]
        relevant.sort(key=lambda e: e.get("count", 0), reverse=True)
        return relevant[:top_k]

    def _format_prompt(
        self,
        recommended: list[str],
        task_tools: list[str],
        patterns: list[dict],
        edges: list[dict],
        task_type: str,
        skill_text: str,
    ) -> str:
        tool_list = "\n".join(f"  • {t}" for t in recommended) or "  (none)"

        task_section = ""
        if task_tools:
            task_section = (
                f"\nFREQUENTLY USED TOOLS for '{task_type}' tasks:\n"
                + "\n".join(f"  • {t}" for t in task_tools)
            )

        pattern_section = ""
        if patterns:
            pattern_section = "\nPROVEN SEQUENTIAL WORKFLOWS:\n"
            for p in patterns[:4]:
                seq_str = " → ".join(p["pattern"])
                dominant = max(
                    p.get("task_types", {}).items(),
                    key=lambda x: x[1], default=("", 0)
                )
                source = " [learned]" if p.get("source") == "textgrad" else ""
                pattern_section += (
                    f"  [{p.get('support', 0)} uses, F1={p.get('avg_f1', 0):.2f}] "
                    f"{seq_str}"
                )
                if dominant[0]:
                    pattern_section += f"  (mainly {dominant[0]})"
                pattern_section += f"{source}\n"

        edge_section = ""
        if edges:
            edge_section = "\nCOMMON TOOL TRANSITIONS:\n"
            for e in edges[:5]:
                edge_section += (
                    f"  • {e['source']}  →  {e['target']}  ({e.get('count', 0)} times)\n"
                )

        skill_section = ""
        if skill_text:
            skill_section = f"\nSTRATEGY for '{task_type}' tasks:\n  {skill_text}\n"

        traj_count = self._state.trajectory_count or "historical"
        epoch_info = f" (optimized, epoch {self._state.epoch})" if self._state.epoch > 0 else ""

        return (
            self.BASE_SYSTEM
            + f"""

## CausalTextEvo Tool Recommendations{epoch_info}

Tools identified via causal attribution + textual gradient optimization from
{traj_count} geospatial analysis trajectories.

RECOMMENDED TOOLS (CCA-weighted, pattern-composed):
{tool_list}
{task_section}
{pattern_section}
{edge_section}
{skill_section}
GUIDANCE:
1. Prefer recommended tools — they are causally linked to task success
2. Follow sequential workflows when the task matches
3. Tool order matters: follow transition patterns
4. You may add tools not listed if the task requires them
"""
        )

    def record_outcome(self, reward: float = 0.5, **kwargs) -> None:
        """No-op for offline; optimizer handles updates."""
        pass
