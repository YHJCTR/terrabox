"""Prompt injector for CausalPolicyEvo."""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .policy_state import PolicyState

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "for",
    "and", "or", "with", "at", "on", "by", "what", "how", "where", "which",
    "this", "that", "it", "i", "you", "we", "they", "be", "do", "does",
    "please", "find", "get", "use", "can", "will", "if", "my", "your",
    "me", "us", "them", "its", "from", "about", "some", "all", "into",
})


def infer_task_type(question: str, task_id: str = "") -> str:
    """Infer task type from case id or question text."""
    if task_id:
        parts = task_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    q = question.lower()
    if any(kw in q for kw in ["change", "compare", "difference", "before", "after"]):
        return "change_detection"
    if any(kw in q for kw in ["flood", "inundation", "water level"]):
        return "flood_detection"
    if any(kw in q for kw in ["fire", "burn", "wildfire"]):
        return "fire_detection"
    if any(kw in q for kw in ["earthquake", "damage", "collapse", "structural"]):
        return "earthquake_assessment"
    if any(kw in q for kw in ["drought", "vegetation", "ndvi"]):
        return "drought_monitoring"
    if any(kw in q for kw in ["landslide", "slope", "debris"]):
        return "landslide_detection"
    if any(kw in q for kw in ["nearest", "route", "closest", "station", "poi"]):
        return "poi_routing"
    if any(kw in q for kw in ["detect", "segment", "count", "object"]):
        return "segmentation"
    if any(kw in q for kw in ["index", "ndvi", "ndwi", "nbr", "raster"]):
        return "index_calculation"
    return "general"


def _keyword_match_score(query: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    q_words = {w.strip(".,!?;:\"'()[]").lower() for w in query.split()} - _STOP_WORDS
    kw_set = {kw.lower() for kw in keywords}
    if not q_words or not kw_set:
        return 0.0
    return len(q_words & kw_set) / len(kw_set)


def _slug_match_score(query: str, tool_slug: str) -> float:
    slug_words = set(re.split(r"[._]", tool_slug.lower()))
    query_words = set(re.split(r"\W+", query.lower()))
    if not slug_words or not query_words:
        return 0.0
    return len(slug_words & query_words) / len(slug_words)


class CausalPolicyEvoPromptInjector(PromptAugmenter):
    """Augment prompts using an explicit external policy state."""

    def __init__(
        self,
        state: PolicyState,
        top_k: int = 8,
    ):
        self._state = state
        self._top_k = top_k
        self._last_predicted_tools: list[str] = []

    def predict_tools(self, query: str, task_type: str = "general") -> list[str]:
        state = self._state
        scored: dict[str, float] = {}

        for tool in state.get_all_tools():
            kw_score = _keyword_match_score(query, state.tool_keywords.get(tool, []))
            slug_score = _slug_match_score(query, tool)
            task_score = state.get_task_prior(tool, task_type)
            prior = float(state.tool_priors.get(tool, 0.0))
            cca = float(state.cca_scores.get(tool, 0.5))
            score = 0.35 * max(kw_score, slug_score) + 0.30 * task_score + 0.20 * prior + 0.15 * cca
            if score > 0.01:
                scored[tool] = score

        ranked = [tool for tool, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]
        anti_set = {frozenset(pair) for pair in state.anti_pairs if len(pair) == 2}

        result: list[str] = []
        for tool in ranked:
            if any(frozenset([tool, existing]) in anti_set for existing in result):
                continue
            result.append(tool)
            if len(result) >= min(3, self._top_k):
                break

        if not result and ranked:
            result = ranked[:1]

        expanded = list(result)
        cursor = 0
        while cursor < len(expanded) and len(expanded) < self._top_k:
            current = expanded[cursor]
            for edge in state.get_next_tools(current, task_type=task_type, top_k=3):
                candidate = edge.get("target", "")
                if not candidate or candidate in expanded:
                    continue
                if any(frozenset([candidate, existing]) in anti_set for existing in expanded):
                    continue
                expanded.append(candidate)
                if len(expanded) >= self._top_k:
                    break
            cursor += 1

        if len(expanded) < self._top_k:
            for tool in ranked:
                if tool in expanded:
                    continue
                if any(frozenset([tool, existing]) in anti_set for existing in expanded):
                    continue
                expanded.append(tool)
                if len(expanded) >= self._top_k:
                    break

        self._last_predicted_tools = expanded[: self._top_k]
        return self._last_predicted_tools

    def augment(
        self,
        user_query: str,
        task_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        task_type = task_type or infer_task_type(user_query)
        predicted = self.predict_tools(user_query, task_type=task_type)
        task_tools = self._state.get_top_task_tools(task_type, top_k=4)
        transitions = self._get_transition_hints(predicted, task_type)

        start_block = "\n".join(f"  • {tool}" for tool in predicted[: min(4, len(predicted))]) or "  (none)"
        transition_block = (
            "\n".join(
                f"  • {edge['source']} → {edge['target']} "
                f"(score={edge.get('rank_score', edge.get('score', 0.0)):.2f})"
                for edge in transitions
            )
            or "  (no strong transition priors)"
        )
        task_block = "\n".join(f"  • {tool}" for tool in task_tools) or "  (none)"
        stop_text = self._state.stop_rules.get(task_type) or self._state.stop_rules.get("general", "")
        recovery_text = self._state.recovery_rules.get(task_type) or self._state.recovery_rules.get("general", "")
        policy_text = self._state.policy_texts.get(task_type, "")

        stop_section = f"\nSTOP GUIDANCE:\n  {stop_text}" if stop_text else ""
        recovery_section = f"\nRECOVERY GUIDANCE:\n  {recovery_text}" if recovery_text else ""
        policy_section = f"\nTASK POLICY:\n  {policy_text}" if policy_text else ""

        return (
            self.BASE_SYSTEM
            + f"""

## CausalPolicyEvo Recommendations

This guidance is derived from a learned external policy state built from
{self._state.trajectory_count or 'historical'} trajectories.

RECOMMENDED STARTING TOOLS:
{start_block}

LIKELY NEXT-STEP TRANSITIONS:
{transition_block}

FREQUENT TOOLS FOR '{task_type}' TASKS:
{task_block}{policy_section}{stop_section}{recovery_section}

GUIDANCE:
1. Start with the recommended tools when they match the query.
2. Prefer transition-consistent next steps instead of adding unrelated tools.
3. Stop once the required evidence is gathered; avoid speculative extra calls.
4. If the current path is weak, switch to the recovery guidance instead of repeating the same pattern.
"""
        )

    def _get_transition_hints(self, predicted: list[str], task_type: str, top_k: int = 5) -> list[dict]:
        hinted: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()
        for tool in predicted[:4]:
            for edge in self._state.get_next_tools(tool, task_type=task_type, top_k=2):
                pair = (edge.get("source", ""), edge.get("target", ""))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                hinted.append(edge)
        hinted.sort(key=lambda x: x.get("rank_score", x.get("score", 0.0)), reverse=True)
        return hinted[:top_k]

    def record_outcome(self, reward: float = 0.5, **kwargs) -> None:
        """Reserved for future online policy updates."""
        return None
