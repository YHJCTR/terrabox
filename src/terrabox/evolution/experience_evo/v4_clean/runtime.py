"""v4 soft guidance with strict label-free hybrid retrieval."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from ....agent.artifacts.signatures import state_overlap, state_satisfies
from ..store import _tokens
from ..v2.models import ToolPolicy, TransitionFamily
from ..v2.store import ExperienceEvoV2Store
from ..v3.runtime import _family_tokens, _hard_mismatch, _intent_score, _preconditions_match, _query_profile
from ..v4.runtime import ExperienceEvoV4Runtime


class ExperienceEvoV4CleanRuntime(ExperienceEvoV4Runtime):
    """Strict v4 runtime: state hard filter plus lexical/structured RRF retrieval.

    ``task_type`` is accepted only for interface compatibility and is discarded.
    Runtime fallbacks from v3 are intentionally not used: this mode measures the
    rollout-distilled store rather than a hand-written transition library.
    """

    def __init__(self, store_dir: str | Path, **kwargs: Any):
        super().__init__(store_dir, **kwargs)
        self.store = ExperienceEvoV2Store(store_dir)
        self._families = self.store.load_families()

    def _retrieval_family_allowed(self, family: TransitionFamily) -> bool:
        """Hook for causal controls; the default keeps v4-clean filtering."""
        exp = family.product_experience
        return exp.q >= self.min_q and exp.risk <= self.max_risk

    def _retrieval_policies(
        self,
        family: TransitionFamily,
        *,
        available_tools: set[str] | None,
    ) -> list[dict[str, object]]:
        return self._rank_tool_policies(family, available_tools=available_tools)

    def _retrieval_quality_score(self, family: TransitionFamily) -> float:
        exp = family.product_experience
        return exp.q + min(exp.n, 12) * 0.04 + (0.2 if family.status == "positive" else 0.0)

    def _retrieval_risk_penalty(self, family: TransitionFamily) -> float:
        return family.product_experience.risk * 3.0

    def retrieve(
        self,
        user_query: str,
        *,
        task_type: str = "unknown",
        current_product_state: list[str] | None = None,
        available_tools: set[str] | None = None,
        images: object = None,
        data_files: object = None,
    ) -> list[TransitionFamily]:
        del task_type
        q_tokens = _tokens(user_query)
        current_state = current_product_state or ["task_request"]
        profile = _query_profile(user_query, q_tokens, images=images, data_files=data_files)
        has_tool_product = any(":from:" in token for token in current_state)
        has_perception_product = any("geo_perception." in token for token in current_state)

        candidates: list[tuple[TransitionFamily, float, float, set[str]]] = []
        for family in self._families:
            if not self._retrieval_family_allowed(family):
                continue
            if not _preconditions_match(current_state, family.input_product_state):
                continue
            if state_satisfies(current_state, family.target_product_state):
                continue
            policies = self._retrieval_policies(family, available_tools=available_tools)
            if not policies:
                continue
            tools = [str(item["policy"].tool) for item in policies if isinstance(item["policy"], ToolPolicy)]
            if _hard_mismatch(profile, tools, current_state=current_state):
                continue
            f_tokens = _family_tokens(family)
            overlap = len(q_tokens & f_tokens)
            intent_score = _intent_score(profile, family, tools)
            if q_tokens and overlap == 0 and intent_score <= 0.0:
                continue
            structured = intent_score + state_overlap(current_state, family.input_product_state) * 4.0
            if has_tool_product:
                structured += min(len(family.input_product_state), 3) * 0.8
                if family.input_product_state == ["task_request"]:
                    structured -= 2.5
            if has_perception_product and bool(profile["calc"]):
                structured += 3.0 if any(tool.startswith("compute.") for tool in tools) else -1.5
            candidates.append((family, structured, float(overlap), f_tokens))

        if not candidates:
            return []
        lexical = self._bm25_scores(q_tokens, [item[3] for item in candidates])
        lexical_order = sorted(range(len(candidates)), key=lambda idx: lexical[idx], reverse=True)
        structured_order = sorted(
            range(len(candidates)), key=lambda idx: candidates[idx][1], reverse=True
        )
        lexical_rank = {idx: rank for rank, idx in enumerate(lexical_order, 1)}
        structured_rank = {idx: rank for rank, idx in enumerate(structured_order, 1)}
        scored: list[tuple[float, TransitionFamily]] = []
        for idx, (family, structured, _overlap, _tokenset) in enumerate(candidates):
            rrf = 1.0 / (60 + lexical_rank[idx]) + 1.0 / (60 + structured_rank[idx])
            exp = family.product_experience
            quality = self._retrieval_quality_score(family)
            risk_penalty = self._retrieval_risk_penalty(family)
            scored.append((rrf * 100.0 + quality + structured * 0.08 - risk_penalty, family))

        scored.sort(key=lambda item: item[0], reverse=True)
        output: list[TransitionFamily] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
        for _, family in scored:
            ranked = self._rank_tool_policies(family, available_tools=available_tools)
            tools = tuple(str(item["policy"].tool) for item in ranked if isinstance(item["policy"], ToolPolicy))
            key = (tuple(family.input_product_state), tuple(family.target_product_state), tools)
            if key not in seen:
                seen.add(key)
                output.append(family)
            if len(output) >= self.top_k:
                break
        return output

    @staticmethod
    def _bm25_scores(query: set[str], documents: list[set[str]]) -> list[float]:
        if not query or not documents:
            return [0.0] * len(documents)
        doc_freq = Counter(token for doc in documents for token in doc)
        avg_len = sum(len(doc) for doc in documents) / len(documents) or 1.0
        scores: list[float] = []
        for doc in documents:
            length_norm = 1.2 * (1.0 - 0.75 + 0.75 * len(doc) / avg_len)
            score = 0.0
            for token in query & doc:
                idf = math.log(1.0 + (len(documents) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
                score += idf * 2.2 / (1.0 + length_norm)
            scores.append(score)
        return scores
