"""ExperienceEvo v5 with Qwen semantic + lexical/structured hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....agent.artifacts.signatures import state_overlap, state_satisfies
from ..store import _tokens
from ..v2.models import ToolPolicy, TransitionFamily
from ..v3.runtime import _family_tokens, _hard_mismatch, _intent_score, _preconditions_match, _query_profile
from ..v4.runtime import _env_enabled
from ..v5.runtime import ExperienceEvoV5Runtime
from .semantic_index import ExperienceEvoFamilyEmbeddingIndex


class ExperienceEvoV5HybridRuntime(ExperienceEvoV5Runtime):
    """v5 final verifier plus Qwen embedding retrieval over transition families.

    The candidate gate is still the strict v4-clean gate: current product state,
    visible tools, query profile, Q/R thresholds, and hard mismatch filters.  The
    embedding signal only reranks candidates that already satisfy those gates.
    """

    def __init__(self, store_dir: str | Path, **kwargs: Any):
        super().__init__(store_dir, **kwargs)
        self._semantic_index = ExperienceEvoFamilyEmbeddingIndex(store_dir, required=True)
        missing = self._semantic_index.missing_documents(self._families)
        if missing and not _env_enabled("TERRABOX_EXPEVO_HYBRID_ALLOW_PARTIAL_INDEX"):
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"ExperienceEvo hybrid embedding index is incomplete: missing {len(missing)} family vectors "
                f"(first: {preview}). Rebuild with build-v5-hybrid-index or set "
                "TERRABOX_EXPEVO_HYBRID_ALLOW_PARTIAL_INDEX=1 for debugging only."
            )

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        return _rename_guidance_titles(super().augment(user_query, task_type=task_type, **kwargs))

    def step_hint(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        return _rename_guidance_titles(super().step_hint(user_query, task_type=task_type, **kwargs))

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
            exp = family.product_experience
            if exp.q < self.min_q or exp.risk > self.max_risk:
                continue
            if not _preconditions_match(current_state, family.input_product_state):
                continue
            if state_satisfies(current_state, family.target_product_state):
                continue
            policies = self._rank_tool_policies(family, available_tools=available_tools)
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

        families = [item[0] for item in candidates]
        lexical = self._bm25_scores(q_tokens, [item[3] for item in candidates])
        semantic = self._semantic_index.similarity_scores(
            _semantic_query_text(user_query, current_state=current_state, profile=profile),
            families,
        )
        lexical_order = sorted(range(len(candidates)), key=lambda idx: lexical[idx], reverse=True)
        structured_order = sorted(range(len(candidates)), key=lambda idx: candidates[idx][1], reverse=True)
        semantic_order = sorted(range(len(candidates)), key=lambda idx: semantic[idx], reverse=True)
        lexical_rank = {idx: rank for rank, idx in enumerate(lexical_order, 1)}
        structured_rank = {idx: rank for rank, idx in enumerate(structured_order, 1)}
        semantic_rank = {idx: rank for rank, idx in enumerate(semantic_order, 1)}
        use_semantic = any(score > 0.0 for score in semantic) and not _env_enabled(
            "TERRABOX_EXPEVO_HYBRID_DISABLE_SEMANTIC_RANK"
        )

        scored: list[tuple[float, TransitionFamily]] = []
        for idx, (family, structured, _overlap, _tokenset) in enumerate(candidates):
            exp = family.product_experience
            rrf = 1.0 / (60 + lexical_rank[idx]) + 1.0 / (60 + structured_rank[idx])
            if use_semantic:
                rrf += 1.0 / (60 + semantic_rank[idx])
            quality = exp.q + min(exp.n, 12) * 0.04 + (0.2 if family.status == "positive" else 0.0)
            score = rrf * 100.0 + quality + structured * 0.08 + semantic[idx] * 0.5 - exp.risk * 3.0
            scored.append((score, family))

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


def _semantic_query_text(user_query: str, *, current_state: list[str], profile: dict[str, object]) -> str:
    flags = [key for key, value in sorted(profile.items()) if bool(value)]
    lines = [
        "task " + str(user_query or ""),
        "current_state " + " ".join(str(item) for item in current_state),
    ]
    if flags:
        lines.append("task_shape " + " ".join(flags))
    return "\n".join(lines)


def _rename_guidance_titles(text: str) -> str:
    return (
        text.replace("ExperienceEvo v4: Soft Product-Transition Guidance", "ExperienceEvo v5 Hybrid: Soft Product-Transition Guidance")
        .replace("ExperienceEvo v4 Step Guidance", "ExperienceEvo v5 Hybrid Step Guidance")
        .replace("ExperienceEvo v4 schema guard", "ExperienceEvo v5 Hybrid schema guard")
    )
