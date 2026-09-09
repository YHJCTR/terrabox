"""Strict controls for isolating ExperienceEvo v4-clean contributions.

These runtimes deliberately contain no transition family, historical tool
policy, rollout text, or retrieval result.  They are used only for causal
ablations against the same sequential OEA execution loop.
"""

from __future__ import annotations

import os
import random
from typing import Any

from ....agent.artifacts.signatures import state_satisfies
from ...shared.prompt_builder import PromptAugmenter
from ..v3.runtime import _calculator_expression_needs_rewrite, _infer_initial_state, _invalid_image_arg_guard, _query_profile
from ..store import _tokens
from ..v2.models import ToolPolicy, TransitionFamily
from ..v4.runtime import _deterministic_verification, _format_verification
from .runtime import ExperienceEvoV4CleanRuntime


class _V4CleanAblationBase(PromptAugmenter):
    """Match v4-clean loop semantics without enabling v3 hard-stop behavior."""

    answer_ready_guard_enabled = False
    premature_final_guard_enabled = False
    allow_synthetic_final_answer = False
    hard_guard_enabled = False

    def __init__(self, *_: Any, **__: Any):
        # The factory keeps a common ``store_dir`` interface for all methods.
        # These controls intentionally ignore it and never open an experience store.
        pass


class ExperienceEvoV4NoStoreSoftRuntime(_V4CleanAblationBase):
    """Generic evidence-state prompt with no offline experience retrieval.

    This control keeps the soft, current-run verification wording but contains
    neither product transitions nor recommended tools.  It therefore measures
    whether generic evidence reminders alone explain v4-clean gains.
    """

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        del task_type
        current_state = self._current_state(kwargs)
        verification = self._verification(user_query, current_state, kwargs)
        return "\n".join(
            [
                "## Generic Evidence-State Checklist",
                "Use only current-run observations and the visible tool schemas. Do not assume historical workflows, paths, answers, or tool sequences.",
                f"Current inferred product state: {', '.join(current_state)}",
                _format_verification(verification),
                "Before answering, make sure the requested result is supported by current-run evidence. This checklist does not recommend a tool or a transition.",
            ]
        )

    def step_hint(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        del task_type
        current_state = kwargs.get("current_product_state")
        if not isinstance(current_state, list) or not current_state:
            return ""
        normalized = [str(item) for item in current_state if str(item)]
        verification = self._verification(user_query, normalized, kwargs)
        return "\n".join(
            [
                "## Generic Evidence-State Checkpoint",
                f"Current product state: {', '.join(normalized)}",
                _format_verification(verification),
                "This is a generic current-run checklist, not an experience retrieval result. Choose the next action from visible tool schemas and returned evidence.",
            ]
        )

    @staticmethod
    def _current_state(kwargs: dict[str, Any]) -> list[str]:
        current_state = kwargs.get("current_product_state")
        if isinstance(current_state, list):
            normalized = [str(item) for item in current_state if str(item)]
            if normalized:
                return normalized
        return _infer_initial_state(images=kwargs.get("images"), data_files=kwargs.get("data_files"))

    @staticmethod
    def _verification(user_query: str, current_state: list[str], kwargs: dict[str, Any]):
        profile = _query_profile(
            user_query,
            _tokens(user_query),
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        return _deterministic_verification(user_query, profile, current_state)


class ExperienceEvoV4GenericGuardRuntime(_V4CleanAblationBase):
    """Only the benchmark-independent image-path and calculator-schema guards."""

    hard_guard_enabled = True

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        del user_query, task_type, kwargs
        return ""

    def guard_tool_call(self, user_query: str, selected_tool: str, task_type: str = "unknown", **kwargs: Any) -> str:
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
            return image_guard.replace("ExperienceEvo v3", "Generic schema")

        if tool == "compute.calculator":
            expression = str(selected_args.get("expression") or "").strip()
            if not expression:
                return "Generic schema guard: `compute.calculator` requires an `expression` argument."
            if _calculator_expression_needs_rewrite(expression):
                return (
                    "Generic schema guard: `compute.calculator` evaluates one safe Python math expression. "
                    "Rewrite imports, assignments, comments, or multi-line code into one expression using current-run values."
                )
        return ""


class ExperienceEvoV4CleanNoQnrQuseRuntime(ExperienceEvoV4CleanRuntime):
    """Keep state/query retrieval and transition text, but remove Q/N/R and Quse."""

    def _retrieval_family_allowed(self, family: TransitionFamily) -> bool:
        return True

    def _retrieval_policies(
        self,
        family: TransitionFamily,
        *,
        available_tools: set[str] | None,
    ) -> list[dict[str, object]]:
        policies: list[dict[str, object]] = []
        for policy in family.tool_policies:
            if available_tools is not None and policy.tool not in available_tools:
                continue
            policies.append({"policy": policy, "lambda": 0.0, "q_use": 0.0})
        return policies[: max(1, self.tool_recommendations_per_family)]

    def _rank_tool_policies(
        self,
        family: TransitionFamily,
        *,
        available_tools: set[str] | None,
    ) -> list[dict[str, object]]:
        return self._retrieval_policies(family, available_tools=available_tools)

    def _retrieval_quality_score(self, family: TransitionFamily) -> float:
        del family
        return 0.0

    def _retrieval_risk_penalty(self, family: TransitionFamily) -> float:
        del family
        return 0.0

    def _format_family(
        self,
        index: int,
        family: TransitionFamily,
        available_tools: set[str] | None,
    ) -> list[str]:
        return _format_quality_free_family(index, family, available_tools, self)

    def _format_step_family(
        self,
        index: int,
        family: TransitionFamily,
        available_tools: set[str] | None,
    ) -> list[str]:
        return _format_quality_free_family(index, family, available_tools, self)

    def _risk_and_fallback_note(self, family: TransitionFamily, available_tools: set[str] | None) -> str:
        del family, available_tools
        return ""


class ExperienceEvoV4CleanNoStepHintRuntime(ExperienceEvoV4CleanRuntime):
    """Keep task-start experience, but disable post-tool dynamic step hints."""

    def __init__(self, store_dir: str | os.PathLike[str], **kwargs: Any):
        super().__init__(store_dir, **kwargs)
        self._disable_step_hint = True


class ExperienceEvoV4CleanNoVerifierRuntime(ExperienceEvoV4CleanRuntime):
    """Keep transition retrieval and Quse, but disable deterministic verification."""

    def __init__(self, store_dir: str | os.PathLike[str], **kwargs: Any):
        super().__init__(store_dir, **kwargs)
        self._disable_verifier = True


class ExperienceEvoV4CleanRandomRetrievalRuntime(ExperienceEvoV4CleanRuntime):
    """Inject state-compatible transitions chosen independently of the query."""

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
        del user_query, task_type, images, data_files
        current_state = current_product_state or ["task_request"]
        candidates: list[TransitionFamily] = []
        for family in sorted(self._families, key=lambda item: item.family_id):
            if not _preconditions_match_for_control(current_state, family.input_product_state):
                continue
            if state_satisfies(current_state, family.target_product_state):
                continue
            policies = self._rank_tool_policies(family, available_tools=available_tools)
            if policies:
                candidates.append(family)
        if not candidates:
            return []
        seed = _env_int("TERRABOX_EXPEVO_RANDOM_RETRIEVAL_SEED", 42)
        rng = random.Random(seed + _state_seed(current_state))
        rng.shuffle(candidates)
        return candidates[: self.top_k]


def _format_quality_free_family(
    index: int,
    family: TransitionFamily,
    available_tools: set[str] | None,
    runtime: ExperienceEvoV4CleanRuntime,
) -> list[str]:
    source = " + ".join(family.input_product_state)
    target = " + ".join(family.target_product_state)
    rows = [f"{index}. Transition: {source} -> {target}"]
    exp = family.product_experience
    rows.append("   Preconditions: " + "; ".join((exp.preconditions or ["Current state must match the input products"])[:2]))
    rows.append("   Product checks: " + "; ".join((exp.output_checks or ["Observation must include the target product"])[:2]))
    policies = runtime._candidate_tool_policies(family, available_tools=available_tools)
    if policies:
        rows.append("   Tool candidates: " + " | ".join(policy.tool for policy in policies))
        hints = _policy_hints_for_control(policies)
        if hints:
            rows.append("   Binding: " + "; ".join(hints[:2]))
    rows.append("   Use current-run values and continue only when the next transition preconditions are satisfied.")
    return rows


def _policy_hints_for_control(policies: list[ToolPolicy]) -> list[str]:
    hints: list[str] = []
    for policy in policies:
        rules = policy.parameter_binding_rules or policy.post_checks
        if rules:
            hints.append(f"{policy.tool}: {rules[0]}")
    return hints


def _preconditions_match_for_control(current_state: list[str], required_state: list[str]) -> bool:
    current = set(str(item) for item in current_state)
    required = set(str(item) for item in required_state)
    return not required or required.issubset(current) or required == {"task_request"}


def _state_seed(current_state: list[str]) -> int:
    return sum((index + 1) * sum(ord(char) for char in token) for index, token in enumerate(current_state))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
