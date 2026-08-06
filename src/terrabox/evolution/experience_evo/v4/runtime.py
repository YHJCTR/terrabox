"""ExperienceEvo v4 runtime.

v4 keeps the v2/v3 product-transition store and Quse ranking, but removes the
hard answer-ready behavior that made v3 stop too early. Guidance is soft by
default: it can recommend next products/tools and flag missing evidence, while
the agent loop remains responsible for deciding whether to call a tool or answer.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..v2.models import ToolPolicy, TransitionFamily
from ..v3.runtime import (
    ExperienceEvoV3Runtime,
    _available_tool_set,
    _as_list,
    _calculator_expression_needs_rewrite,
    _infer_initial_state,
    _invalid_image_arg_guard,
    _policy_hints,
    _query_profile,
    _required_result_count,
    _task_shape_guidance,
)
from ..store import _tokens


@dataclass(frozen=True)
class V4Verification:
    """Lightweight verifier output used to shape v4 prompts."""

    status: str
    missing_slots: tuple[str, ...]
    risk_notes: tuple[str, ...]
    recommended_focus: str
    llm_checker_note: str = ""


class ExperienceEvoV4Runtime(ExperienceEvoV3Runtime):
    """Soft verifier-guided product-transition runtime.

    The public method name is ``experience_evo_v4``. It deliberately preserves
    v3's store compatibility while disabling three hard loop behaviors through
    attributes consumed by ``agent.eval_modes.common``:

    - no answer-ready hard guard;
    - no premature-final hard guard just because recommendations exist;
    - no synthetic final answer created by code.

    A bounded optional LLM checker can be enabled with:
    ``TERRABOX_EXPEVO_V4_LLM_CHECKER=1`` and
    ``TERRABOX_EXPEVO_V4_CHECKER_PROVIDER=longcat``. The default is disabled so
    full OEA rollouts do not double LongCat traffic.
    """

    answer_ready_guard_enabled = False
    premature_final_guard_enabled = False
    allow_synthetic_final_answer = False
    hard_guard_enabled = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._llm_checker_enabled = _env_enabled("TERRABOX_EXPEVO_V4_LLM_CHECKER")
        self._llm_checker_provider = os.environ.get("TERRABOX_EXPEVO_V4_CHECKER_PROVIDER", "longcat")
        self._llm_checker_max_calls = _env_int("TERRABOX_EXPEVO_V4_LLM_CHECKER_MAX_CALLS", 0)
        self._llm_checker_calls = 0
        self._llm_checker = None

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        current_state = kwargs.get("current_product_state")
        if not isinstance(current_state, list):
            current_state = _infer_initial_state(
                images=kwargs.get("images"),
                data_files=kwargs.get("data_files"),
            )
        else:
            current_state = [str(item) for item in current_state if str(item)]

        available_tools = _available_tool_set(kwargs.get("available_tools"))
        profile = _query_profile(
            user_query,
            _tokens(user_query),
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        families = self.retrieve(
            user_query,
            task_type=task_type,
            current_product_state=current_state,
            available_tools=available_tools,
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        if not families:
            return ""

        verification = self._verify_state(
            user_query=user_query,
            profile=profile,
            current_state=current_state,
            families=families,
            artifact_state=kwargs.get("artifact_state"),
        )
        families = _prioritize_families_for_missing_slots(families, verification, current_state)

        lines = [
            "## ExperienceEvo v4: Soft Product-Transition Guidance",
            (
                "Retrieved lessons are rollout-derived product-state transitions. "
                "They are not gold trajectories, answers, task ids, or reusable file paths."
            ),
            (
                "Use matching transitions as planning evidence, not as a forced script. "
                "Bind every downstream call to current-run artifacts and values only."
            ),
            (
                "Verifier checkpoint: treat missing-evidence notes as a checklist before "
                "answering; if evidence is already sufficient, answer normally without "
                "extra validation loops."
            ),
            f"Current inferred product state: {', '.join(current_state)}",
            _format_verification(verification),
            "",
        ]
        guidance = _task_shape_guidance(profile)
        if guidance:
            lines.append("Task-shape priors (soft):")
            lines.extend(f"- {item}" for item in guidance[:4])
            lines.append("")

        for index, family in enumerate(families, 1):
            lines.extend(self._format_family(index, family, available_tools))
        return "\n".join(lines).strip()

    def step_hint(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        current_state = kwargs.get("current_product_state")
        if not isinstance(current_state, list):
            return ""
        current_state = [str(item) for item in current_state if str(item)]
        if not current_state:
            return ""

        available_tools = _available_tool_set(kwargs.get("available_tools"))
        profile = _query_profile(
            user_query,
            _tokens(user_query),
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        families = self.retrieve(
            user_query,
            task_type=task_type,
            current_product_state=current_state,
            available_tools=available_tools,
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        verification = self._verify_state(
            user_query=user_query,
            profile=profile,
            current_state=current_state,
            families=families,
            artifact_state=kwargs.get("artifact_state"),
        )
        families = _prioritize_families_for_missing_slots(families, verification, current_state)

        lines = [
            "## ExperienceEvo v4 Step Guidance",
            (
                "Soft guidance only: use current-run product state and verifier notes; "
                "do not copy historical paths, answers, or task-specific examples."
            ),
            f"Current product state: {', '.join(current_state)}",
            _format_verification(verification),
        ]

        if verification.status == "likely_ready":
            lines.append(
                "Readiness note: current evidence appears sufficient. Prefer a concise "
                "final answer unless the user explicitly requested a missing file, plot, "
                "map, or annotation artifact."
            )
            return "\n".join(lines).strip()

        if not families:
            lines.append(
                "No matching transition family passed the current-state filter; continue "
                "with the most schema-grounded tool that fills the missing evidence slots."
            )
            return "\n".join(lines).strip()

        for index, family in enumerate(families, 1):
            lines.extend(self._format_step_family(index, family, available_tools))
        lines.append(
            "Stop only after the requested answer or requested visual/output artifact is "
            "supported by current-run evidence."
        )
        return "\n".join(lines).strip()

    def guard_tool_call(self, user_query: str, selected_tool: str, task_type: str = "unknown", **kwargs: Any) -> str:
        """Block only directly observable schema/path mistakes in v4.

        v3 blocked many task-shape choices before execution. v4 keeps only guard
        checks that are independent of the benchmark label: current-run image
        paths and calculator syntax.
        """
        del user_query, task_type
        tool = str(selected_tool or "").replace("__", ".")
        selected_args = kwargs.get("selected_args")
        if not isinstance(selected_args, dict):
            selected_args = {}
        image_guard = _invalid_image_arg_guard(
            selected_args,
            images=kwargs.get("images"),
            artifact_state=kwargs.get("artifact_state"),
        )
        if image_guard:
            return image_guard.replace("ExperienceEvo v3", "ExperienceEvo v4")

        if tool == "compute.calculator":
            expression = str(selected_args.get("expression") or "").strip()
            if not expression:
                return (
                    "ExperienceEvo v4 schema guard: `compute.calculator` requires "
                    "an `expression` argument. Provide one arithmetic expression "
                    "using current-run values."
                )
            if _calculator_expression_needs_rewrite(expression):
                return (
                    "ExperienceEvo v4 schema guard: `compute.calculator` evaluates "
                    "one safe Python math expression. Rewrite imports, assignments, "
                    "comments, or multi-line code into a single expression using "
                    "current-run values."
                )
        return ""

    def _format_family(
        self,
        index: int,
        family: TransitionFamily,
        available_tools: set[str] | None,
    ) -> list[str]:
        rows = super()._format_family(index, family, available_tools)
        extra = self._risk_and_fallback_note(family, available_tools)
        if extra:
            rows.append("   Verifier note: " + extra)
        return [row.replace("ExperienceEvo v3", "ExperienceEvo v4") for row in rows]

    def _format_step_family(
        self,
        index: int,
        family: TransitionFamily,
        available_tools: set[str] | None,
    ) -> list[str]:
        exp = family.product_experience
        ranked = self._rank_tool_policies(family, available_tools=available_tools)
        target = " + ".join(family.target_product_state)
        source = " + ".join(family.input_product_state)
        lines = [
            f"{index}. Next transition: {source} -> {target} "
            f"(Qsig={exp.q:.2f}, Nsig={exp.n}, Rsig={exp.risk:.2f})"
        ]
        if ranked:
            chunks: list[str] = []
            for item in ranked:
                policy = item.get("policy")
                if not isinstance(policy, ToolPolicy):
                    continue
                chunks.append(
                    f"{policy.tool} Quse={float(item['q_use']):.2f} "
                    f"(Qtool={policy.q:.2f}, Rtool={policy.risk:.2f})"
                )
            if chunks:
                lines.append("   Tool ranking: " + " | ".join(chunks))
        hints = _policy_hints([item["policy"] for item in ranked])
        if hints:
            lines.append("   Binding: " + "; ".join(hints[:2]))
        checks = exp.output_checks or ["observation should include the target product"]
        lines.append("   Check: " + "; ".join(checks[:1]))
        extra = self._risk_and_fallback_note(family, available_tools)
        if extra:
            lines.append("   Verifier note: " + extra)
        return lines

    def _risk_and_fallback_note(
        self,
        family: TransitionFamily,
        available_tools: set[str] | None,
    ) -> str:
        ranked = self._rank_tool_policies(family, available_tools=available_tools)
        if not ranked:
            return ""
        exp = family.product_experience
        top = ranked[0].get("policy")
        q_use = float(ranked[0].get("q_use", 0.0))
        notes: list[str] = []
        risky = exp.risk >= 0.35 or any(
            isinstance(item.get("policy"), ToolPolicy)
            and float(getattr(item["policy"], "risk", 0.0)) >= 0.35
            for item in ranked
        )
        if risky:
            notes.append(
                "high-R transition/tool; verify argument binding and output artifact before reusing it"
            )
        if isinstance(top, ToolPolicy) and top.q + 0.05 < exp.q:
            notes.append(
                "Qtool is below Qsig, so also apply the product-level checks instead of trusting the tool alone"
            )
        if q_use < 0.45:
            notes.append("low-Quse recommendation; prefer a schema-grounded alternative if available")
        return "; ".join(notes[:2])

    def _verify_state(
        self,
        *,
        user_query: str,
        profile: dict[str, object],
        current_state: list[str],
        families: list[TransitionFamily],
        artifact_state: object,
    ) -> V4Verification:
        verification = _deterministic_verification(user_query, profile, current_state)
        if not self._should_call_llm_checker(profile, verification):
            return verification

        note = self._call_llm_checker(
            user_query=user_query,
            current_state=current_state,
            families=families,
            artifact_state=artifact_state,
            verification=verification,
        )
        if not note:
            return verification
        return V4Verification(
            status=verification.status,
            missing_slots=verification.missing_slots,
            risk_notes=verification.risk_notes,
            recommended_focus=verification.recommended_focus,
            llm_checker_note=note,
        )

    def _should_call_llm_checker(self, profile: dict[str, object], verification: V4Verification) -> bool:
        if not self._llm_checker_enabled or self._llm_checker_max_calls <= 0:
            return False
        if self._llm_checker_calls >= self._llm_checker_max_calls:
            return False
        if verification.status == "not_ready" and not verification.risk_notes:
            return False
        return bool(
            verification.status == "likely_ready"
            or verification.risk_notes
            or profile.get("multi_target")
            or profile.get("localized_attribute")
            or profile.get("visual")
        )

    def _call_llm_checker(
        self,
        *,
        user_query: str,
        current_state: list[str],
        families: list[TransitionFamily],
        artifact_state: object,
        verification: V4Verification,
    ) -> str:
        try:
            if self._llm_checker is None:
                from ....agent.llm_provider import make_llm_client

                self._llm_checker = make_llm_client(self._llm_checker_provider)
            family_lines = []
            for family in families[:3]:
                top_tools = [policy.tool for policy in family.tool_policies[:3]]
                family_lines.append(
                    {
                        "source": family.input_product_state,
                        "target": family.target_product_state,
                        "q": family.product_experience.q,
                        "risk": family.product_experience.risk,
                        "tools": top_tools,
                    }
                )
            prompt = {
                "task": user_query[:800],
                "current_product_state": current_state,
                "deterministic_status": verification.status,
                "deterministic_missing_slots": list(verification.missing_slots),
                "recent_observations": _recent_observation_summaries(artifact_state),
                "candidate_transitions": family_lines,
            }
            system = (
                "You are a verifier sub-agent for a geospatial tool-use agent. "
                "Do not solve the task from scratch and do not use gold answers. "
                "Check whether the current evidence is enough, what evidence slot is missing, "
                "and whether the next transition/tool recommendation is risky. "
                "Return compact JSON with keys: allow_answer, missing_slots, risk_observed, reason."
            )
            data = self._llm_checker.call_json(
                json.dumps(prompt, ensure_ascii=False),
                system=system,
                max_tokens=350,
            )
            self._llm_checker_calls += 1
            if not isinstance(data, dict):
                return ""
            missing = data.get("missing_slots")
            if isinstance(missing, list):
                missing_text = ", ".join(str(item)[:80] for item in missing[:3])
            else:
                missing_text = str(missing or "")[:160]
            reason = str(data.get("reason") or "")[:220]
            allow = bool(data.get("allow_answer"))
            risk = data.get("risk_observed")
            return (
                f"LLM verifier: allow_answer={allow}, risk_observed={risk}, "
                f"missing={missing_text or 'none'}, reason={reason}"
            )
        except Exception as exc:
            return f"LLM verifier unavailable: {type(exc).__name__}: {str(exc)[:120]}"


def _deterministic_verification(
    user_query: str,
    profile: dict[str, object],
    current_state: list[str],
) -> V4Verification:
    state = set(current_state)
    counts = Counter(current_state)
    required_count = _required_result_count(profile)
    missing: list[str] = []
    risk_notes: list[str] = []

    has_compute_result = bool(
        counts["result:from:compute.calculator"]
        or counts["result:from:compute.solver"]
        or counts["result:from:osm_gis.compute_route_dist"]
        or counts["result:from:osm_gis.compute_index_change"]
    )
    has_arithmetic_result = bool(
        counts["result:from:compute.calculator"]
        or counts["result:from:compute.solver"]
    )
    has_visual_artifact = any(
        token.startswith("image:from:")
        or token.startswith("map:from:")
        or token.startswith("figure:from:")
        for token in state
    )
    has_localization = bool(counts["result:from:geo_perception.instructsam"])
    has_attribute = bool(counts["result:from:geo_perception.region_attribute_description"])
    has_count = bool(counts["result:from:geo_perception.count_given_object"])
    has_sam2 = bool(counts["result:from:geo_perception.sam2_segment"])

    if profile.get("visual") and not has_visual_artifact:
        missing.append("requested visual/map/plot/annotation artifact")
    if _needs_aggregate_calculation(user_query) and not has_arithmetic_result:
        missing.append("aggregate/arithmetic computation over returned tool values")
    if profile.get("pixel_threshold_measurement"):
        if not has_sam2:
            missing.append("segmentation evidence for pixel-threshold filtering")
        if counts["result:from:compute.solver"] < 2:
            missing.append("threshold/filter solver result")
        if counts["result:from:compute.calculator"] < 2:
            missing.append("area or percentage calculator result")
    elif profile.get("precise_measurement") or profile.get("calc") or profile.get("size_selection_visual"):
        if not has_compute_result:
            missing.append("numeric computation result")
    if profile.get("count_distribution") and counts["result:from:geo_perception.count_given_object"] < required_count:
        missing.append("object count result for each requested class/category")
    if profile.get("per_object_count") and not has_localization:
        missing.append("parent object/region localization before per-object counting")
    if profile.get("attribute") and not has_attribute:
        missing.append("localized/region-level attribute evidence")
    if profile.get("localized_attribute") and not has_localization:
        missing.append("localized object or region evidence")
    if profile.get("multi_target"):
        if profile.get("attribute") and counts["result:from:geo_perception.region_attribute_description"] < required_count:
            missing.append("one attribute result per target")
        if (profile.get("precise_measurement") or profile.get("calc")) and counts["result:from:compute.calculator"] < required_count:
            missing.append("one computation result per target")
    if profile.get("plot_distribution") and "image:from:compute.plot" not in state:
        missing.append("plot artifact generated from current-run counts")

    if has_localization and not has_attribute and profile.get("localized_attribute"):
        risk_notes.append("localized evidence exists but attribute evidence is still absent")
    if has_count and profile.get("plot_distribution") and "image:from:compute.plot" not in state:
        risk_notes.append("count evidence exists but requested plot is not generated")
    if profile.get("visual") and has_compute_result and not has_visual_artifact:
        risk_notes.append("analytic result exists but requested visual output is missing")

    if missing:
        focus = _focus_from_missing(missing)
        return V4Verification(
            status="not_ready",
            missing_slots=tuple(dict.fromkeys(missing[:5])),
            risk_notes=tuple(dict.fromkeys(risk_notes[:3])),
            recommended_focus=focus,
        )

    # Some simple perception/search tasks can be answerable without calculator.
    if has_compute_result:
        focus = "answer with computed current-run value"
    elif has_visual_artifact:
        focus = "answer with the generated current-run artifact path"
    elif has_attribute:
        focus = "answer with localized attribute evidence"
    elif has_count:
        focus = "answer with current-run count evidence"
    elif any(token.startswith("result:from:") for token in state):
        focus = "answer only if the latest observation directly supports the question"
    else:
        return V4Verification(
            status="not_ready",
            missing_slots=("first evidence-producing tool result",),
            risk_notes=tuple(risk_notes),
            recommended_focus="produce the first task-relevant evidence artifact",
        )

    return V4Verification(
        status="likely_ready",
        missing_slots=(),
        risk_notes=tuple(dict.fromkeys(risk_notes[:3])),
        recommended_focus=focus,
    )


def _format_verification(verification: V4Verification) -> str:
    missing = ", ".join(verification.missing_slots) if verification.missing_slots else "none"
    risk = "; ".join(verification.risk_notes) if verification.risk_notes else "none"
    parts = [
        f"Verifier checkpoint: status={verification.status}; missing_evidence={missing}; focus={verification.recommended_focus}; risk_notes={risk}."
    ]
    if verification.llm_checker_note:
        parts.append(verification.llm_checker_note)
    return " ".join(parts)


def _focus_from_missing(missing: list[str]) -> str:
    text = " | ".join(missing).lower()
    if "visual" in text or "plot" in text or "map" in text or "annotation" in text:
        return "produce the requested final visual artifact after evidence is ready"
    if "attribute" in text or "localized" in text:
        return "get localized evidence and region-level attribute descriptions"
    if "count" in text:
        return "obtain current-run count evidence before aggregating or plotting"
    if "numeric" in text or "calculator" in text or "solver" in text or "computation" in text:
        return "compute the requested numeric value from current-run evidence"
    if "segmentation" in text or "pixel" in text:
        return "obtain pixel/segmentation evidence before numeric conversion"
    return "fill the missing evidence slot before answering"


def _prioritize_families_for_missing_slots(
    families: list[TransitionFamily],
    verification: V4Verification,
    current_state: list[str],
) -> list[TransitionFamily]:
    if not families or verification.status == "likely_ready":
        return families
    missing_text = " ".join(verification.missing_slots).lower()
    has_tool_product = any(":from:" in token or token.startswith("result:from:") for token in current_state)

    def bonus(family: TransitionFamily) -> tuple[int, int]:
        tools = " ".join(policy.tool for policy in family.tool_policies).lower()
        targets = " ".join(family.target_product_state).lower()
        text = f"{tools} {targets}"
        score = 0
        if "aggregate" in missing_text or "numeric" in missing_text or "computation" in missing_text:
            if "compute." in text:
                score += 10
            if has_tool_product and family.input_product_state == ["task_request"]:
                score -= 4
        if "visual" in missing_text or "plot" in missing_text or "map" in missing_text or "annotation" in missing_text:
            if any(name in text for name in ("draw_bboxes", "add_text", "plot", "display_on", "show_index")):
                score += 10
        if "attribute" in missing_text or "localized" in missing_text:
            if "region_attribute_description" in text:
                score += 10
            if "instructsam" in text and "localized" in missing_text:
                score += 4
        if "count" in missing_text and "count_given_object" in text:
            score += 10
        if "segmentation" in missing_text or "pixel" in missing_text:
            if "sam2_segment" in text:
                score += 10
            if "instructsam" in text:
                score += 4
        return (score, int(family.product_experience.n))

    return sorted(families, key=bonus, reverse=True)


def _needs_aggregate_calculation(user_query: str) -> bool:
    text = f" {str(user_query or '').lower()} "
    phrases = (
        " average ",
        " mean ",
        " median ",
        " total ",
        " sum ",
        " combined ",
        " percentage ",
        " percent ",
        " ratio ",
        " difference ",
        " how far ",
        " how much ",
        " per ",
    )
    return any(phrase in text for phrase in phrases)


def _recent_observation_summaries(artifact_state: object) -> list[str]:
    if not isinstance(artifact_state, dict):
        return []
    summaries: list[str] = []
    raw_results = artifact_state.get("results")
    if not isinstance(raw_results, list):
        return []
    for item in raw_results[-4:]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("summary") or item.get("content") or item.get("result") or "").strip()
        if text:
            summaries.append(text[:500])
    return summaries


def _env_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
