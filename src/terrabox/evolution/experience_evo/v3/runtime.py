"""ExperienceEvo v3 runtime.

This version intentionally reuses the v2 product-transition store, but changes
runtime retrieval and formatting. The key constraint is that static rollout
injection should only include transitions whose preconditions match the current
task-start product state.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from ....agent.artifacts.signatures import state_overlap, state_satisfies
from ...shared.prompt_builder import PromptAugmenter
from ..store import _tokens
from ..v2.models import ProductExperience, ToolPolicy, TransitionFamily
from ..v2.runtime import _available_tool_set
from ..v2.store import ExperienceEvoV2Store


_RASTER_SUFFIXES = {".tif", ".tiff", ".geotiff", ".vrt"}
_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".json", ".shp", ".kml"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_OSM_TERMS = {
    "amenity",
    "area",
    "boundary",
    "buffer",
    "buffered",
    "closest",
    "distance",
    "fire",
    "hospital",
    "landmark",
    "nearest",
    "park",
    "poi",
    "pois",
    "police",
    "restaurant",
    "route",
    "school",
    "station",
    "within",
}
_INDEX_TERMS = {
    "burn",
    "burned",
    "change",
    "decrease",
    "enhanced",
    "growth",
    "index",
    "layer",
    "nbr",
    "ndbi",
    "ndvi",
    "raster",
    "regrowth",
    "severity",
    "unburned",
    "urban",
}
_PERCEPTION_TERMS = {
    "aerial",
    "airplane",
    "bbox",
    "building",
    "count",
    "detect",
    "garbage",
    "gsd",
    "helicopter",
    "image",
    "mask",
    "object",
    "pixel",
    "pixels",
    "pool",
    "region",
    "regions",
    "segment",
    "ship",
    "storage",
    "tank",
    "tennis",
    "vehicle",
}
_CALC_TERMS = {
    "area",
    "calculate",
    "combined",
    "convert",
    "distance",
    "integer",
    "measure",
    "meters",
    "percentage",
    "ratio",
    "rounded",
    "square",
}
_VIS_TERMS = {"annotate", "display", "draw", "map", "plot", "visualize"}
_SEARCH_TERMS = {"search", "web", "internet", "current", "latest"}
_VISUAL_OUTPUT_TOOLS = {
    "compute.plot",
    "geo_perception.add_text",
    "geo_perception.draw_bboxes",
    "osm_gis.display_on_geotiff",
    "osm_gis.display_on_map",
    "osm_gis.show_index_layer",
}


class ExperienceEvoV3Runtime(PromptAugmenter):
    """Retrieve compact, task-start-applicable product transitions.

    v3 is deliberately compatible with stores produced by ``build-v2``. It does
    not overwrite v1/v2 and can be selected independently via
    ``--evolution-method experience_evo_v3``.
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        top_k: int = 3,
        min_q: float = 0.0,
        max_risk: float = 0.75,
        q_use_smoothing_k: float = 5.0,
        tool_recommendations_per_family: int = 3,
    ):
        self.store = ExperienceEvoV2Store(store_dir)
        self.top_k = top_k
        self.min_q = min_q
        self.max_risk = max_risk
        self.q_use_smoothing_k = q_use_smoothing_k
        self.tool_recommendations_per_family = tool_recommendations_per_family
        self._families = self.store.load_families()

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
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

        lines = [
            "## ExperienceEvo v3: Applicable Product Transitions",
            (
                "Retrieved lessons are rollout-derived product-state transitions, "
                "plus narrow runtime fallbacks only when the current product state "
                "and tool contract make a missing downstream transition explicit. "
                "They are not gold trajectories, answers, task ids, or reusable file paths."
            ),
            (
                "Use only transitions whose Preconditions match the current state. "
                "Do not execute a later-stage transition until its required product "
                "state is actually returned by a tool."
            ),
            (
                "After a tool succeeds, bind downstream calls to the exact returned "
                "artifact/path/layer/value. If enough evidence exists for the user "
                "request, stop and answer instead of repeating validation tools."
            ),
            (
                "If a tool fails with no matching features or invalid arguments, "
                "retry only with a concrete schema-grounded fix; do not loop on the "
                "same tool and arguments."
            ),
            f"Current inferred product state: {', '.join(current_state)}",
            "",
        ]
        guidance = _task_shape_guidance(profile)
        if guidance:
            lines.extend(guidance)
            lines.append("")

        for index, family in enumerate(families, 1):
            lines.extend(self._format_family(index, family, available_tools))
        return "\n".join(lines).strip()

    def step_hint(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        current_state = kwargs.get("current_product_state")
        if not isinstance(current_state, list):
            return ""
        current_state = [str(item) for item in current_state if str(item)]
        if not current_state:
            return ""

        available_tools = _available_tool_set(kwargs.get("available_tools"))
        lines = [
            "## ExperienceEvo v3 Step Guidance",
            (
                "Use the current-run product state only; never copy historical paths or answers. "
                "Runtime fallback transitions, when shown, are tool-contract guards for store gaps."
            ),
            f"Current product state: {', '.join(current_state)}",
        ]
        profile = _query_profile(
            user_query,
            _tokens(user_query),
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        ready_reason = _answer_ready_reason(profile, current_state)
        if ready_reason:
            lines.extend(
                [
                    f"Answer-ready signal: {ready_reason}.",
                    (
                        "Prefer writing the final answer now. Call another tool only if "
                        "the user explicitly asks for a missing visualization/file or a "
                        "required value is absent from the latest observations."
                    ),
                ]
            )
            return "\n".join(lines).strip()

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

        for index, family in enumerate(families, 1):
            exp = family.product_experience
            ranked = self._rank_tool_policies(family, available_tools=available_tools)
            target = " + ".join(family.target_product_state)
            source = " + ".join(family.input_product_state)
            lines.append(
                f"{index}. Next transition: {source} -> {target} "
                f"(Qsig={exp.q:.2f}, Nsig={exp.n}, Rsig={exp.risk:.2f})"
            )
            if ranked:
                chunks = []
                for item in ranked:
                    policy = item["policy"]
                    if not isinstance(policy, ToolPolicy):
                        continue
                    chunks.append(f"{policy.tool} Quse={float(item['q_use']):.2f}")
                if chunks:
                    lines.append("   Tool ranking: " + " | ".join(chunks))
            hints = _policy_hints([item["policy"] for item in ranked])
            if hints:
                lines.append("   Binding: " + "; ".join(hints[:2]))
            checks = exp.output_checks or ["observation should include the target product"]
            lines.append("   Check: " + "; ".join(checks[:1]))
        lines.append("Stop if the requested answer is already supported by the current evidence.")
        return "\n".join(lines).strip()

    def guard_tool_call(self, user_query: str, selected_tool: str, task_type: str = "unknown", **kwargs) -> str:
        """Return a corrective message when a proposed tool violates v3 task shape.

        This is intentionally narrow: it does not replace the agent's planner,
        but prevents high-cost first-step calls that the current task shape
        makes clearly wrong.
        """
        del task_type
        current_state = kwargs.get("current_product_state")
        if not isinstance(current_state, list):
            current_state = _infer_initial_state(
                images=kwargs.get("images"),
                data_files=kwargs.get("data_files"),
            )
        current_state = [str(item) for item in current_state if str(item)]
        tool = str(selected_tool or "").replace("__", ".")
        if not tool:
            return ""

        profile = _query_profile(
            user_query,
            _tokens(user_query),
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        selected_args = kwargs.get("selected_args")
        if not isinstance(selected_args, dict):
            selected_args = {}
        image_guard = _invalid_image_arg_guard(
            selected_args,
            images=kwargs.get("images"),
            artifact_state=kwargs.get("artifact_state"),
        )
        if image_guard:
            return image_guard

        state = set(current_state)
        counts = Counter(current_state)
        required_count = _required_result_count(profile)
        instructsam_count = counts["result:from:geo_perception.instructsam"]
        calculator_count = (
            counts["result:from:compute.calculator"]
            + counts["result:from:compute.solver"]
        )
        attribute_count = counts["result:from:geo_perception.region_attribute_description"]
        has_instructsam = "result:from:geo_perception.instructsam" in state
        has_compute_result = calculator_count > 0
        has_tool_product = any(":from:" in token for token in current_state)

        if tool == "bing_search.search" and not profile["search"]:
            return (
                "ExperienceEvo v3 tool guard: this task does not ask for web/current "
                "information, so `bing_search.search` is an unrelated detour. Continue "
                "with the product-state transition recommended for the current evidence; "
                "for image measurement/classification tasks this usually means "
                "`compute.calculator` and/or `geo_perception.region_attribute_description`."
            )

        if tool in _VISUAL_OUTPUT_TOOLS and not profile["visual"]:
            return (
                "ExperienceEvo v3 tool guard: the user did not request a plot, map, "
                f"annotation, or visualization, so `{tool}` is an unnecessary output "
                "tool for this task. Continue with the next analytic transition or "
                "write the final answer if the requested evidence is already present."
            )

        if profile["index"]:
            index_layer_count = counts["raster_layer:from:osm_gis.add_index_layer"]
            if (
                tool == "osm_gis.get_area_boundary"
                and (
                    "gpkg:from:osm_gis.get_area_boundary" in state
                    or "table:from:osm_gis.get_area_boundary" in state
                )
            ):
                next_tool = (
                    "`osm_gis.compute_index_change`"
                    if index_layer_count >= 2
                    else "`osm_gis.add_index_layer` for the missing temporal/input raster"
                )
                return (
                    "ExperienceEvo v3 index guard: a current-run AOI/boundary product "
                    "already exists. Do not call `osm_gis.get_area_boundary` again. "
                    f"Continue with {next_tool}, using current-run artifacts only."
                )
            if tool == "osm_gis.add_index_layer" and index_layer_count >= 2:
                return (
                    "ExperienceEvo v3 index guard: two current-run index layers already "
                    "exist. Do not add another index layer; call "
                    "`osm_gis.compute_index_change` next using those current-run layers."
                )
            if tool == "osm_gis.add_index_layer" and index_layer_count >= 1:
                layer_name = _selected_arg_value(
                    selected_args,
                    ("layer_name", "output_layer", "target_layer", "name"),
                )
                previous_layers = _previous_arg_values(
                    kwargs.get("artifact_state"),
                    tool,
                    ("layer_name", "output_layer", "target_layer", "name"),
                )
                if layer_name and any(_same_semantic_target(layer_name, prev) for prev in previous_layers):
                    return (
                        "ExperienceEvo v3 index guard: this `osm_gis.add_index_layer` "
                        f"call appears to reuse the existing layer name/target `{layer_name}`. "
                        "Create only the missing comparison layer with a distinct current-run "
                        "layer name, then call `osm_gis.compute_index_change`. Do not retry "
                        "the same layer name after a table-exists/tool error."
                    )

        wrong_first_measurement_tools = {
            "geo_perception.vlm_analyze",
            "geo_perception.strip_rcnn_detect",
        }
        if not profile["bulk_segmentation"]:
            wrong_first_measurement_tools.add("geo_perception.sam2_segment")
        if (
            profile["precise_measurement"]
            and not has_instructsam
            and tool in wrong_first_measurement_tools
        ):
            return (
                "ExperienceEvo v3 tool guard: the proposed tool "
                f"`{tool}` is not the right first evidence tool for this GSD/pixel "
                "measurement task. Use `geo_perception.instructsam` first to get "
                "measurable region/pixel evidence, then use `compute.calculator` "
                "for the numeric conversion. Do not use VLM/descriptive analysis "
                "or generic segmentation before measurable evidence exists."
            )

        if (
            profile["localized_attribute"]
            and not has_instructsam
            and tool
            in {
                "geo_perception.vlm_analyze",
                "geo_perception.region_attribute_description",
                "geo_perception.sam2_segment",
                "geo_perception.strip_rcnn_detect",
            }
        ):
            return (
                "ExperienceEvo v3 tool guard: this image task asks for a localized "
                "attribute/comparison over specific objects or damage regions. Use "
                "`geo_perception.instructsam` first to localize the relevant region(s), "
                "then call `geo_perception.region_attribute_description` on each "
                "current-run region. Do not replace this with a global VLM description "
                "or an unlabeled generic segmentation."
            )

        if (
            profile["localized_attribute"]
            and has_instructsam
            and attribute_count < required_count
            and tool
            in {
                "geo_perception.vlm_analyze",
                "geo_perception.sam2_segment",
                "geo_perception.strip_rcnn_detect",
                "geo_perception.count_given_object",
            }
        ):
            remaining = required_count - attribute_count
            return (
                "ExperienceEvo v3 tool guard: localized evidence already exists "
                "from `geo_perception.instructsam`, so a global/broad perception "
                f"tool `{tool}` would add an ungrounded detour for this localized "
                "attribute/comparison task. Call "
                f"`geo_perception.region_attribute_description` for {remaining} "
                "remaining current-run region/object(s) instead. For two-object "
                "comparisons, bind each call to a current-run region or bbox from "
                "the localization output; do not use VLM scene descriptions as a "
                "substitute for region-level evidence."
            )

        downstream_after_localization: list[str] = []
        if profile["precise_measurement"] and calculator_count < required_count:
            remaining = required_count - calculator_count
            downstream_after_localization.append(
                f"`compute.calculator` for {remaining} remaining numeric result(s), "
                "one safe arithmetic expression per target"
            )
        if profile["attribute"] and attribute_count < required_count:
            remaining = required_count - attribute_count
            downstream_after_localization.append(
                f"`geo_perception.region_attribute_description` for {remaining} "
                "remaining localized region/object(s)"
            )
        if (
            profile["multi_target"]
            and profile["precise_measurement"]
            and instructsam_count >= required_count
            and downstream_after_localization
            and tool
            in {
                "geo_perception.instructsam",
                "geo_perception.vlm_analyze",
                "geo_perception.sam2_segment",
                "geo_perception.strip_rcnn_detect",
                "geo_perception.count_given_object",
            }
        ):
            return (
                "ExperienceEvo v3 multi-target guard: the current state already "
                f"has {instructsam_count} InstructSAM localization result(s), which "
                f"satisfies the estimated {required_count} target slot(s). Do not "
                f"call `{tool}` again or switch to another broad perception tool. "
                "Continue downstream with "
                + "; ".join(downstream_after_localization)
                + ". Bind every call to current-run artifacts and values only."
            )

        if (
            profile["precise_measurement"]
            and has_instructsam
            and not has_compute_result
            and tool
            in {
                "geo_perception.vlm_analyze",
                "geo_perception.sam2_segment",
                "geo_perception.strip_rcnn_detect",
            }
        ):
            return (
                "ExperienceEvo v3 tool guard: measurable region evidence already exists "
                "from `geo_perception.instructsam`. Continue with `compute.calculator` "
                "using the current-run pixel/area values instead of calling another "
                "perception evidence tool."
            )

        if (
            profile["localized_attribute"]
            and has_instructsam
            and tool == "geo_perception.instructsam"
        ):
            previous_targets = _previous_semantic_targets(
                kwargs.get("artifact_state"),
                tool,
            )
            previous_text = ", ".join(previous_targets) if previous_targets else "current localized region(s)"
            remaining = max(required_count - attribute_count, 1)
            return (
                "ExperienceEvo v3 tool guard: localization evidence already exists "
                f"from `geo_perception.instructsam` for {previous_text}. Do not call "
                "InstructSAM again for this localized attribute/comparison step. Call "
                "`geo_perception.region_attribute_description` on the current-run "
                f"localized region(s) now. The current state still needs {remaining} "
                "region-level attribute description call(s); for two-object comparisons, "
                "call `geo_perception.region_attribute_description` once per remaining "
                "region/object."
            )

        if (
            profile["precise_measurement"]
            and profile["attribute"]
            and not profile["multi_target"]
            and has_instructsam
            and not has_compute_result
            and tool == "geo_perception.region_attribute_description"
        ):
            return (
                "ExperienceEvo v3 tool guard: this task needs the numeric measurement "
                "before the attribute assessment. Call `compute.calculator` first with "
                "one safe expression using the current-run pixel/area values, then call "
                "`geo_perception.region_attribute_description`."
            )

        if (
            profile["index"]
            and tool == "osm_gis.compute_index_change"
            and current_state.count("raster_layer:from:osm_gis.add_index_layer") < 2
        ):
            return (
                "ExperienceEvo v3 tool guard: `osm_gis.compute_index_change` needs "
                "two current-run index layers. Call `osm_gis.add_index_layer` for "
                "the missing comparison layer first, then compute the index change. "
                "Do not reuse historical layer names."
            )

        if tool == "compute.calculator":
            expression = str(selected_args.get("expression") or "").strip()
            if not expression:
                return (
                    "ExperienceEvo v3 calculator guard: `compute.calculator` requires "
                    "an `expression` argument. Provide a single arithmetic expression."
                )
            if _calculator_expression_needs_rewrite(expression):
                return (
                    "ExperienceEvo v3 calculator guard: `compute.calculator` evaluates "
                    "one safe Python math expression via eval. Do not send import "
                    "statements, assignments, comments, or multi-line code. Rewrite the "
                    "calculation as one expression, for example "
                    "`round(2 * sqrt((30568 * 0.1402**2) / pi))`. Use returned "
                    "current-run values in that single expression."
                )

        repeated_single_evidence_tools = {
            "geo_perception.instructsam",
            "geo_perception.sam2_segment",
            "geo_perception.vlm_analyze",
            "geo_perception.strip_rcnn_detect",
        }
        if has_tool_product and tool in repeated_single_evidence_tools:
            token = f"result:from:{tool}"
            if token in set(current_state):
                selected_target = _semantic_target_from_args(selected_args)
                previous_targets = _previous_semantic_targets(
                    kwargs.get("artifact_state"),
                    tool,
                )
                if (
                    profile["multi_target"]
                    and selected_target
                    and not any(_same_semantic_target(selected_target, previous) for previous in previous_targets)
                ):
                    return ""
                previous_text = ", ".join(previous_targets) if previous_targets else "unknown"
                return (
                    "ExperienceEvo v3 tool guard: the current state already contains "
                    f"`{token}`. Do not repeat the same evidence tool for the same "
                    "semantic target. "
                    + (
                        f"Already localized target(s): {previous_text}. If this multi-object task "
                        "still needs another object/region, call the same tool with a different "
                        "`text`/target argument; otherwise continue with the next required "
                        "product transition or write the final answer."
                        if profile["multi_target"]
                        else "Continue with the next required product transition or write the "
                        "final answer if the requested evidence is complete."
                    )
                )

        return ""

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
        q_tokens = _tokens(user_query)
        current_state = current_product_state or ["task_request"]
        profile = _query_profile(
            user_query,
            q_tokens,
            images=images,
            data_files=data_files,
        )
        task_filter = task_type if task_type and task_type != "unknown" else None
        has_tool_product = any(
            ":from:" in token or token.startswith("result:from:")
            for token in current_state
        )
        has_perception_product = any("geo_perception." in token for token in current_state)
        scored: list[tuple[float, TransitionFamily]] = []

        for family in self._families:
            exp = family.product_experience
            if task_filter and family.task_type not in {task_filter, "general", "unknown"}:
                continue
            if exp.q < self.min_q or exp.risk > self.max_risk:
                continue
            if not _preconditions_match(current_state, family.input_product_state):
                continue
            if state_satisfies(current_state, family.target_product_state):
                continue

            policies = self._rank_tool_policies(family, available_tools=available_tools)
            if not policies:
                continue
            tool_names = [str(item["policy"].tool) for item in policies if isinstance(item["policy"], ToolPolicy)]
            if _hard_mismatch(profile, tool_names, current_state=current_state):
                continue

            f_tokens = _family_tokens(family)
            overlap = len(q_tokens & f_tokens)
            intent_score = _intent_score(profile, family, tool_names)
            if q_tokens and overlap == 0 and intent_score <= 0.0:
                continue

            state_score = state_overlap(current_state, family.input_product_state) * 4.0
            specific_inputs = [
                token for token in family.input_product_state if token and token != "task_request"
            ]
            specificity_bonus = min(len(specific_inputs), 3) * 1.5
            if has_tool_product and specific_inputs:
                specificity_bonus += 2.0
            generic_start_penalty = (
                2.5 if has_tool_product and family.input_product_state == ["task_request"] else 0.0
            )
            support = min(exp.n, 12) * 0.04
            status_bonus = 0.6 if family.status == "positive" else 0.0
            risk_penalty = exp.risk * 3.0
            dynamic_calc_bonus = 0.0
            if has_perception_product and profile["calc"]:
                if any(tool.startswith("compute.") for tool in tool_names):
                    dynamic_calc_bonus += 6.0
                elif any(tool.startswith("geo_perception.") for tool in tool_names):
                    dynamic_calc_bonus -= 3.0
            score = (
                overlap * 1.2
                + intent_score
                + state_score
                + specificity_bonus
                + dynamic_calc_bonus
                + exp.q
                + support
                + status_bonus
                - risk_penalty
                - generic_start_penalty
            )
            scored.append((score, family))

        fallback_families = _fallback_families(
            profile=profile,
            current_state=current_state,
            available_tools=available_tools,
        )
        if fallback_families:
            return fallback_families[: self.top_k]

        scored.sort(key=lambda item: item[0], reverse=True)
        output: list[TransitionFamily] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
        for _, family in scored:
            ranked = self._rank_tool_policies(family, available_tools=available_tools)
            tools = tuple(
                str(item["policy"].tool)
                for item in ranked
                if isinstance(item["policy"], ToolPolicy)
            )
            key = (tuple(family.input_product_state), tuple(family.target_product_state), tools)
            if key in seen:
                continue
            seen.add(key)
            output.append(family)
            if len(output) >= self.top_k:
                break
        return output

    def _rank_tool_policies(
        self,
        family: TransitionFamily,
        *,
        available_tools: set[str] | None,
    ) -> list[dict[str, object]]:
        ranked: list[dict[str, object]] = []
        sig = family.product_experience
        for policy in family.tool_policies:
            if available_tools is not None and policy.tool not in available_tools:
                continue
            lam = policy.n / (policy.n + self.q_use_smoothing_k) if policy.n >= 0 else 0.0
            q_use = lam * policy.q + (1.0 - lam) * sig.q
            ranked.append({"policy": policy, "lambda": lam, "q_use": q_use})
        ranked.sort(
            key=lambda item: (
                float(item["q_use"]),
                int(getattr(item["policy"], "n", 0)),
                -float(getattr(item["policy"], "risk", 0.0)),
            ),
            reverse=True,
        )
        return ranked[: max(1, self.tool_recommendations_per_family)]

    def _format_family(
        self,
        index: int,
        family: TransitionFamily,
        available_tools: set[str] | None,
    ) -> list[str]:
        exp = family.product_experience
        policies = self._rank_tool_policies(family, available_tools=available_tools)
        target = " + ".join(family.target_product_state)
        source = " + ".join(family.input_product_state)
        intent = _clean_intent(family.intent_signature)
        rows = [
            (
                f"{index}. Transition: {source} -> {target} "
                f"(intent={intent}; Qsig={exp.q:.2f}; Nsig={exp.n}; Rsig={exp.risk:.2f})"
            )
        ]
        preconditions = exp.preconditions or [
            "Current state must contain " + ", ".join(family.input_product_state)
        ]
        checks = exp.output_checks or ["Tool observation must include " + target]
        rows.append("   Preconditions: " + "; ".join(preconditions[:2]))
        rows.append("   Product checks: " + "; ".join(checks[:2]))

        tool_chunks: list[str] = []
        for item in policies:
            policy = item["policy"]
            if not isinstance(policy, ToolPolicy):
                continue
            q_use = float(item.get("q_use", 0.0))
            lam = float(item.get("lambda", 0.0))
            tool_chunks.append(
                f"{policy.tool} Quse={q_use:.2f} "
                f"(lambda={lam:.2f}, Qtool={policy.q:.2f}, Ntool={policy.n}, Rtool={policy.risk:.2f})"
            )
        if tool_chunks:
            rows.append("   Tool ranking: " + " | ".join(tool_chunks))

        hints = _policy_hints([item["policy"] for item in policies])
        if hints:
            rows.append("   Binding hints: " + "; ".join(hints[:3]))
        rows.append(
            "   Stop rule: once this product directly supports the requested answer, answer; "
            "otherwise continue only with transitions whose preconditions are now satisfied."
        )
        return rows


def _infer_initial_state(*, images: object, data_files: object) -> list[str]:
    state = ["task_request"]
    for item in _as_list(images):
        suffix = Path(item).suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            state.append("input:image")
    for item in _as_list(data_files):
        suffix = Path(item).suffix.lower()
        if suffix in _RASTER_SUFFIXES:
            state.append("input:raster")
        elif suffix in _VECTOR_SUFFIXES:
            if suffix == ".gpkg":
                state.append("input:gpkg")
            else:
                state.append("input:vector")
        elif suffix in _IMAGE_SUFFIXES:
            state.append("input:image")
        else:
            state.append("input:file")
    return sorted(dict.fromkeys(state))


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _preconditions_match(current: list[str], required: list[str]) -> bool:
    required_clean = [item for item in required if item]
    if not required_clean:
        return True
    if state_satisfies(current, required_clean):
        return True
    if required_clean == ["task_request"] and current:
        return True
    return False


def _fallback_families(
    *,
    profile: dict[str, object],
    current_state: list[str],
    available_tools: set[str] | None,
) -> list[TransitionFamily]:
    """Create narrow contract-backed transitions for known store coverage gaps."""

    state = set(current_state)
    counts = Counter(current_state)
    families: list[TransitionFamily] = []
    required_count = _required_result_count(profile)

    if (
        profile["multi_target"]
        and profile["has_image"]
        and profile["precise_measurement"]
        and _tool_available("geo_perception.instructsam", available_tools)
        and 0 < counts["result:from:geo_perception.instructsam"] < required_count
    ):
        families.append(
            _runtime_family(
                family_id="v3_fallback_multitarget_instructsam",
                intent="runtime:multi-target+localize",
                source=["result:from:geo_perception.instructsam"],
                target=["result:from:geo_perception.instructsam"] * required_count,
                tool="geo_perception.instructsam",
                goal="Localize another current-task object or region before computing multi-target measurements.",
                binding_rules=[
                    "Use a different concrete text target than any target already localized in this run.",
                    "For distance/comparison tasks, localize every required object or reference region before computing.",
                ],
                output_checks=[
                    "Current product state should contain separate InstructSAM results for the required task targets."
                ],
            )
        )

    if (
        profile["index"]
        and _tool_available("osm_gis.add_index_layer", available_tools)
        and counts["raster_layer:from:osm_gis.add_index_layer"] == 1
        and "raster_layer:from:osm_gis.compute_index_change" not in state
        and "result:from:osm_gis.compute_index_change" not in state
    ):
        families.append(
            _runtime_family(
                family_id="v3_fallback_second_index_layer",
                intent="runtime:index+second-layer",
                source=["raster_layer:from:osm_gis.add_index_layer"],
                target=[
                    "raster_layer:from:osm_gis.add_index_layer",
                    "raster_layer:from:osm_gis.add_index_layer",
                ],
                tool="osm_gis.add_index_layer",
                goal="Create the second current-run index layer required before index-change computation.",
                binding_rules=[
                    "Use the other temporal/input raster requested by the current task.",
                    "Do not compute index change until both current-run index layers exist.",
                ],
                output_checks=[
                    "Current product state should contain two raster_layer entries from osm_gis.add_index_layer."
                ],
            )
        )

    if (
        profile["index"]
        and _tool_available("osm_gis.compute_index_change", available_tools)
        and counts["raster_layer:from:osm_gis.add_index_layer"] >= 2
        and "raster_layer:from:osm_gis.compute_index_change" not in state
        and "result:from:osm_gis.compute_index_change" not in state
    ):
        families.append(
            _runtime_family(
                family_id="v3_fallback_index_change",
                intent="runtime:index+change",
                source=["raster_layer:from:osm_gis.add_index_layer"],
                target=["raster_layer:from:osm_gis.compute_index_change"],
                tool="osm_gis.compute_index_change",
                goal="Compute the requested index-change product from the current run's index layers.",
                binding_rules=[
                    "Use the raster/index layer artifacts returned in this run; do not use historical layer names.",
                    "Use compute_index_change when the request asks to assess increase, decrease, growth, or change.",
                ],
                output_checks=[
                    "Observation should include an index-change result or raster_layer from osm_gis.compute_index_change."
                ],
            )
        )

    perception_evidence = _first_present(
        state,
        [
            "result:from:geo_perception.instructsam",
            "result:from:geo_perception.strip_rcnn_detect",
            "result:from:geo_perception.sam2_segment",
            "result:from:geo_perception.vlm_analyze",
        ],
    )
    if (
        profile["attribute"]
        and _tool_available("geo_perception.region_attribute_description", available_tools)
        and perception_evidence
        and (
            "result:from:compute.calculator" in state
            or profile["localized_attribute"]
            or not profile["precise_measurement"]
        )
        and counts["result:from:geo_perception.region_attribute_description"] < required_count
    ):
        source = [perception_evidence]
        if "result:from:compute.calculator" in state:
            source.append("result:from:compute.calculator")
        families.append(
            _runtime_family(
                family_id="v3_fallback_region_attribute_after_measurement",
                intent="runtime:attribute+measurement",
                source=source,
                target=["result:from:geo_perception.region_attribute_description"] * required_count,
                tool="geo_perception.region_attribute_description",
                goal="Describe every requested current-task region attribute after measurable evidence exists.",
                binding_rules=[
                    "Bind the region input to the current run's perception output, not to a historical mask or bbox.",
                    "For multi-object comparison tasks, call this once per required current-run region/object.",
                ],
                output_checks=[
                    "Observation should include the requested region attribute/condition description tied to current evidence."
                ],
            )
        )

    if (
        profile["multi_target"]
        and profile["precise_measurement"]
        and _tool_available("compute.calculator", available_tools)
        and counts["result:from:geo_perception.instructsam"] >= required_count
        and counts["result:from:compute.calculator"] < required_count
    ):
        families.append(
            _runtime_family(
                family_id="v3_fallback_multitarget_calculator",
                intent="runtime:multi-target+calculate",
                source=["result:from:geo_perception.instructsam"],
                target=["result:from:compute.calculator"] * required_count,
                tool="compute.calculator",
                goal="Compute the remaining numeric result(s) for each localized current-task target.",
                binding_rules=[
                    "Use one safe arithmetic expression per missing target/result.",
                    "Use only current-run pixel, distance, area, or GSD values returned earlier in this task.",
                ],
                output_checks=[
                    "Current product state should contain separate calculator results for the required target measurements."
                ],
            )
        )

    return families


def _runtime_family(
    *,
    family_id: str,
    intent: str,
    source: list[str],
    target: list[str],
    tool: str,
    goal: str,
    binding_rules: list[str],
    output_checks: list[str],
) -> TransitionFamily:
    target_text = " + ".join(target)
    product = ProductExperience(
        goal=goal,
        preconditions=["Current product state must contain " + ", ".join(source) + "."],
        output_checks=output_checks,
        downstream_rule="Continue only with current-run artifacts returned by this transition.",
        recovery=["If the tool rejects an argument, repair the argument binding rather than changing task intent."],
        experience=(
            "Runtime fallback generated from current product state and tool contracts "
            "because the offline transition store lacks this exact downstream family."
        ),
        q=0.88,
        n=1,
        risk=0.0,
    )
    policy = ToolPolicy(
        tool=tool,
        required_input_roles=source,
        parameter_binding_rules=binding_rules,
        output_contract=target,
        post_checks=output_checks,
        downstream_rule="Use returned current-run artifacts only.",
        recovery=["Do not repeat identical failed calls."],
        experience=f"Use {tool} to produce {target_text} when the source products already exist.",
        q=0.88,
        n=1,
        risk=0.0,
        source_event_ids=[],
    )
    return TransitionFamily(
        family_id=family_id,
        schema_version=3,
        task_type="general",
        intent_signature=intent,
        input_product_state=source,
        target_product_state=target,
        product_experience=product,
        tool_policies=[policy],
        provenance_summary={"source": "experience_evo_v3_runtime_fallback"},
        status="candidate",
    )


def _tool_available(tool: str, available_tools: set[str] | None) -> bool:
    return available_tools is None or tool in available_tools


def _calculator_expression_needs_rewrite(expression: str) -> bool:
    text = str(expression or "").strip()
    if not text:
        return True
    lowered = text.lower()
    statement_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "for ",
        "while ",
        "try:",
        "with ",
        "return ",
    )
    if "\n" in text or ";" in text or "#" in text:
        return True
    if any(lowered.startswith(marker) for marker in statement_markers):
        return True
    if any(f"\n{marker}" in lowered for marker in statement_markers):
        return True
    return bool(re.search(r"(?<![<>=!])=(?!=)", text))


def _invalid_image_arg_guard(
    selected_args: dict[str, object],
    *,
    images: object,
    artifact_state: object,
) -> str:
    selected = _selected_image_arg(selected_args)
    if not selected:
        return ""
    known = _known_image_paths(images=images, artifact_state=artifact_state)
    if selected in known:
        return ""
    selected_path = Path(selected)
    if selected_path.exists() and not _looks_like_external_dataset_path(selected, known):
        return ""
    candidates = ", ".join(sorted(known)[:3]) if known else "the current task image/artifact"
    return (
        "ExperienceEvo v3 image-path guard: the proposed image path "
        f"`{selected}` is not a current-task image/artifact or does not exist. "
        f"Use the current-run image path instead: {candidates}. Do not copy or "
        "invent historical dataset paths."
    )


def _selected_image_arg(selected_args: dict[str, object]) -> str:
    for key in ("image", "image_path", "input_image", "input_path"):
        value = selected_args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key, value in selected_args.items():
        if "image" in str(key).lower() and isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _known_image_paths(*, images: object, artifact_state: object) -> set[str]:
    known = set(_as_list(images))
    if isinstance(artifact_state, dict):
        for artifact in artifact_state.get("artifacts", []):
            if not isinstance(artifact, dict):
                continue
            if str(artifact.get("kind") or "").lower() != "image":
                continue
            path = str(artifact.get("path") or "").strip()
            if path:
                known.add(path)
    return known


def _looks_like_external_dataset_path(path: str, known_paths: set[str]) -> bool:
    if not known_paths:
        return False
    lowered = path.lower()
    if "openearthagent/data/" in lowered or "/data/test/" in lowered or "/data/train/" in lowered:
        return True
    return False


def _semantic_target_from_args(args: dict[str, object]) -> str:
    for key in (
        "text",
        "text_prompt",
        "prompt",
        "label",
        "object",
        "target",
        "query",
        "attribute",
        "region",
    ):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_semantic_target(value)
    return ""


def _previous_semantic_targets(artifact_state: object, tool: str) -> list[str]:
    if not isinstance(artifact_state, dict):
        return []
    targets: list[str] = []
    for record in artifact_state.get("successful_call_records", []):
        if not isinstance(record, dict):
            continue
        if str(record.get("tool") or "").replace("__", ".") != tool:
            continue
        args = record.get("args")
        if not isinstance(args, dict):
            continue
        target = _semantic_target_from_args(args)
        if target and target not in targets:
            targets.append(target)
    return targets


def _selected_arg_value(args: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_semantic_target(value)
    return ""


def _previous_arg_values(
    artifact_state: object,
    tool: str,
    keys: tuple[str, ...],
) -> list[str]:
    if not isinstance(artifact_state, dict):
        return []
    values: list[str] = []
    for record in artifact_state.get("successful_call_records", []):
        if not isinstance(record, dict):
            continue
        if str(record.get("tool") or "").replace("__", ".") != tool:
            continue
        args = record.get("args")
        if not isinstance(args, dict):
            continue
        value = _selected_arg_value(args, keys)
        if value and value not in values:
            values.append(value)
    return values


def _normalize_semantic_target(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _same_semantic_target(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return overlap >= 0.8


def _first_present(state: set[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in state:
            return candidate
    return ""


def _family_tokens(family: TransitionFamily) -> set[str]:
    parts: list[str] = [
        _clean_intent(family.intent_signature),
        " ".join(family.input_product_state),
        " ".join(family.target_product_state),
        family.product_experience.goal,
        " ".join(family.product_experience.output_checks),
    ]
    for policy in family.tool_policies:
        parts.extend(
            [
                policy.tool,
                " ".join(policy.required_input_roles),
                " ".join(policy.parameter_binding_rules),
                " ".join(policy.output_contract),
                " ".join(policy.post_checks),
            ]
        )
    return _tokens("\n".join(part for part in parts if part))


def _clean_intent(intent_signature: str) -> str:
    intent = str(intent_signature or "general")
    if ":" in intent:
        intent = intent.split(":", 1)[1]
    return intent.replace("+", " ").replace("_", " ")


def _query_profile(
    query: str,
    q_tokens: set[str],
    *,
    images: object,
    data_files: object,
) -> dict[str, object]:
    query_lower = (query or "").lower()
    image_items = _as_list(images)
    data_items = _as_list(data_files)
    image_count = len(image_items) + sum(
        1 for item in data_items if Path(item).suffix.lower() in _IMAGE_SUFFIXES
    )
    has_image = bool(image_items) or any(
        Path(item).suffix.lower() in _IMAGE_SUFFIXES for item in _as_list(data_files)
    )
    has_data = bool(data_items)
    has_raster = any(Path(item).suffix.lower() in _RASTER_SUFFIXES for item in data_items)
    index_intent = bool(q_tokens & _INDEX_TERMS)
    osm_intent = bool(q_tokens & _OSM_TERMS) and not has_image
    perception_intent = has_image or bool(q_tokens & _PERCEPTION_TERMS)
    calc_intent = bool(q_tokens & _CALC_TERMS)
    vis_intent = bool(q_tokens & _VIS_TERMS)
    search_intent = bool(q_tokens & _SEARCH_TERMS)
    change_intent = bool({"change", "changed", "growth", "decrease", "between"} & q_tokens)
    region_mask_intent = bool({"area", "gsd", "mask", "pixel", "pixels", "region", "regions", "segment"} & q_tokens)
    precise_measurement = bool(
        has_image
        and (
            "ground sampling distance" in query_lower
            or "gsd" in q_tokens
            or "pixel distance" in query_lower
            or "square meters" in query_lower
            or bool({"pixel", "pixels", "area", "distance", "diameter"} & q_tokens)
        )
    )
    bulk_segmentation = bool(
        has_image
        and region_mask_intent
        and (
            "segment all" in query_lower
            or "sum their pixel" in query_lower
            or "sum the pixel" in query_lower
            or "total pixel" in query_lower
            or "pixel areas" in query_lower
        )
    )
    attribute_terms = {"attribute", "condition", "health", "classify", "classification"}
    attribute_intent = bool(attribute_terms & q_tokens)
    # "Assess" is common in GIS/index-change questions ("assess urban growth")
    # and should not force a geo_perception.region_attribute_description step.
    if "assess" in q_tokens and has_image and not index_intent:
        attribute_intent = True
    localized_attribute = bool(
        has_image
        and attribute_intent
        and (
            "damage pattern" in query_lower
            or "two buildings" in query_lower
            or "between the two" in query_lower
            or bool(
                {
                    "affected",
                    "both",
                    "building",
                    "buildings",
                    "damage",
                    "damaged",
                    "roof",
                    "similarly",
                    "symmetry",
                    "symmetric",
                }
                & q_tokens
            )
        )
    )
    multi_target = bool(
        has_image
        and (
            "both" in q_tokens
            or "two" in q_tokens
            or "each" in q_tokens
            or "types" in q_tokens
            or "between the two" in query_lower
            or "both garbage types" in query_lower
            or "similarly affected" in query_lower
            or bool({"compare", "comparison"} & q_tokens)
        )
    )

    # Some phrases are stronger than token overlap because the v1 tokenizer
    # expands distance toward GIS terms even for image-measurement questions.
    if "fire station" in query_lower or "police station" in query_lower:
        osm_intent = True
    if "ground sampling distance" in query_lower or "gsd" in q_tokens:
        perception_intent = True
        calc_intent = True
        if has_image:
            osm_intent = False
    if index_intent:
        osm_intent = True

    return {
        "has_image": has_image,
        "has_data": has_data,
        "has_raster": has_raster,
        "image_count": image_count,
        "osm": osm_intent,
        "index": index_intent,
        "perception": perception_intent,
        "calc": calc_intent,
        "visual": vis_intent,
        "search": search_intent,
        "change": change_intent,
        "region_mask": region_mask_intent,
        "precise_measurement": precise_measurement,
        "bulk_segmentation": bulk_segmentation,
        "attribute": attribute_intent,
        "localized_attribute": localized_attribute,
        "multi_target": multi_target,
    }


def _hard_mismatch(
    profile: dict[str, object],
    tools: list[str],
    *,
    current_state: list[str],
) -> bool:
    if not tools:
        return True
    all_osm = all(tool.startswith("osm_gis.") for tool in tools)
    all_compute = all(tool.startswith("compute.") for tool in tools)
    any_search = any(tool.startswith("bing_search.") for tool in tools)
    any_geo = any(tool.startswith("geo_perception.") for tool in tools)
    all_geo = all(tool.startswith("geo_perception.") for tool in tools)
    only_vlm = all(tool == "geo_perception.vlm_analyze" for tool in tools)
    all_visual_output = all(tool in _VISUAL_OUTPUT_TOOLS for tool in tools)
    has_tool_product = any(":from:" in token or token.startswith("result:from:") for token in current_state)
    has_raster_state = any(
        token.startswith("input:raster") or token.startswith("raster:from:")
        for token in current_state
    )
    if profile["has_image"] and all_osm and not profile["osm"] and not profile["index"]:
        return True
    if not profile["has_image"] and all_geo and not profile["perception"]:
        return True
    if profile["index"] and any_geo and not any(tool.startswith("osm_gis.") for tool in tools):
        return True
    if profile["index"] and any(
        "add_pois_layer" in tool or "compute_route_dist" in tool or "display_on_map" in tool
        for tool in tools
    ):
        return True
    if all_visual_output and not profile["visual"]:
        return True
    if any("get_bbox_from_raster" in tool for tool in tools) and not (profile["has_raster"] or has_raster_state):
        return True
    if any("change_os_detect" in tool for tool in tools) and not profile["change"]:
        return True
    if any("change_os_detect" in tool for tool in tools) and int(profile["image_count"] or 0) < 2:
        return True
    if all_compute and (profile["osm"] or profile["perception"] or profile["index"]) and not has_tool_product:
        return True
    if profile["precise_measurement"] and only_vlm:
        return True
    if any_search and not profile["search"]:
        return True
    return False


def _intent_score(
    profile: dict[str, object],
    family: TransitionFamily,
    tools: list[str],
) -> float:
    score = 0.0
    tool_text = " ".join(tools)
    target_text = " ".join(family.target_product_state)
    any_osm = any(tool.startswith("osm_gis.") for tool in tools)
    any_geo = any(tool.startswith("geo_perception.") for tool in tools)
    any_compute = any(tool.startswith("compute.") for tool in tools)
    any_search = any(tool.startswith("bing_search.") for tool in tools)

    if profile["osm"] and any_osm:
        score += 5.0
    if profile["index"] and (
        "add_index_layer" in tool_text or "compute_index_change" in tool_text or "raster" in target_text
    ):
        score += 4.0
    if profile["perception"] and any_geo:
        score += 5.0
    if profile["calc"] and any_compute:
        score += 2.0
    if profile["visual"] and ("display" in tool_text or "plot" in tool_text or "draw" in tool_text):
        score += 2.5
    if profile["search"] and any_search:
        score += 3.0

    if "geo_perception.instructsam" in tool_text and profile["region_mask"]:
        score += 3.0
    if "geo_perception.instructsam" in tool_text and profile["precise_measurement"]:
        score += 2.0 if profile["bulk_segmentation"] else 8.0
    if "geo_perception.sam2_segment" in tool_text and profile["region_mask"]:
        score += 2.0
    if "geo_perception.sam2_segment" in tool_text and profile["precise_measurement"]:
        score += 8.0 if profile["bulk_segmentation"] else 1.0
    if "geo_perception.strip_rcnn_detect" in tool_text and profile["region_mask"]:
        score -= 1.5
    if "geo_perception.strip_rcnn_detect" in tool_text and profile["precise_measurement"]:
        score -= 5.0
    if "geo_perception.vlm_analyze" in tool_text and profile["precise_measurement"]:
        score -= 7.0
    if "geo_perception.change_os_detect" in tool_text and profile["change"]:
        score += 3.0

    if profile["has_image"] and any_osm and not profile["osm"] and not profile["index"]:
        score -= 6.0
    if not profile["has_image"] and any_geo and not profile["perception"]:
        score -= 4.0
    if profile["osm"] and any_geo and not profile["perception"]:
        score -= 3.0
    return score


def _task_shape_guidance(profile: dict[str, object]) -> list[str]:
    lines: list[str] = []
    if profile["has_image"] and profile["precise_measurement"]:
        preferred_tool = (
            "geo_perception.sam2_segment for bulk all-object segmentation"
            if profile["bulk_segmentation"]
            else "geo_perception.instructsam as the first evidence-producing tool"
        )
        lines.append(
            "Task-shape rule: this image task needs measurable pixel/mask evidence "
            f"for GSD/area/distance conversion. Prefer {preferred_tool}."
        )
        lines.append(
            "Do not use geo_perception.vlm_analyze as the first numeric measurement "
            "tool; descriptive text is not sufficient for pixel area/distance math."
        )
    return lines


def _answer_ready_reason(profile: dict[str, object], current_state: list[str]) -> str:
    state = set(current_state)
    counts = Counter(current_state)
    required_count = _required_result_count(profile)
    wants_visual = bool(profile["visual"])
    if wants_visual and not any(
        token.startswith("image:from:")
        or token.startswith("map:from:")
        or token.startswith("figure:from:")
        for token in state
    ):
        return ""

    has_compute_result = any(
        token in {
            "result:from:compute.calculator",
            "result:from:compute.solver",
        }
        for token in state
    )
    has_index_change = any(
        token in {
            "result:from:osm_gis.compute_index_change",
            "raster_layer:from:osm_gis.compute_index_change",
        }
        for token in state
    )
    has_route_distance = "result:from:osm_gis.compute_route_dist" in state
    has_attribute_result = "result:from:geo_perception.region_attribute_description" in state
    needs_numeric_result = bool(profile["precise_measurement"] or profile["calc"])
    needs_attribute_result = bool(profile["attribute"])

    if needs_numeric_result and not (has_compute_result or has_route_distance or has_index_change):
        return ""
    if (
        needs_numeric_result
        and profile["has_image"]
        and profile["multi_target"]
        and counts["result:from:compute.calculator"] < required_count
        and not (has_route_distance or has_index_change)
    ):
        return ""
    if needs_attribute_result and not has_attribute_result:
        return ""
    if (
        needs_attribute_result
        and profile["multi_target"]
        and counts["result:from:geo_perception.region_attribute_description"] < required_count
    ):
        return ""

    if has_compute_result and has_attribute_result:
        return "computed scalar/result and region attribute evidence are available"
    if has_compute_result:
        return "computed scalar/result is available"
    if has_route_distance:
        return "route/distance result is available"
    if has_index_change:
        return "index-change result is available"

    answer_tools = {
        "geo_perception.count_given_object": "object count is available",
        "geo_perception.ocr_extract": "OCR text is available",
        "geo_perception.vlm_analyze": "VLM analysis is available",
        "geo_perception.region_attribute_description": "region attribute description is available",
    }
    for token in state:
        if not token.startswith("result:from:"):
            continue
        tool = token.removeprefix("result:from:")
        if tool in answer_tools:
            return answer_tools[tool]

    if wants_visual and any(
        token.startswith("image:from:")
        or token.startswith("map:from:")
        or token.startswith("figure:from:")
        for token in state
    ):
        return "requested visual artifact is available"
    return ""


def _required_result_count(profile: dict[str, object]) -> int:
    return 2 if profile.get("multi_target") else 1


def _policy_hints(policies: list[object]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for policy in policies:
        if not isinstance(policy, ToolPolicy):
            continue
        for hint in list(policy.parameter_binding_rules) + list(policy.post_checks):
            text = str(hint).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            hints.append(text)
            if len(hints) >= 4:
                return hints
    return hints
