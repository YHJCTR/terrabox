"""Strict, label-free construction of product-transition families."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Callable

from ..v2.builder import (
    _distill_family,
    _resolved_targets,
    _running_statistics,
    _template_product,
    _template_tool,
)
from ..v2.models import ToolPolicy, TransitionEvent, TransitionFamily


def _family_id(intent: str, inputs: list[str], targets: list[str]) -> str:
    """Family identity is only intent and artifact-state transition."""
    raw = "|".join([intent, "+".join(inputs), "+".join(targets)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:18]


def build_clean_transition_families(
    events: list[TransitionEvent],
    *,
    llm: Any | None = None,
    min_support: int = 1,
    max_families: int | None = None,
    max_examples: int = 6,
    allow_template_fallback: bool = False,
    alpha0: float = 1.0,
    risk_alpha0: float = 1.0,
    skip_family_ids: set[str] | None = None,
    progress: Callable[[int, int, TransitionFamily], None] | None = None,
) -> list[TransitionFamily]:
    """Build families without task labels, gold rewards, or task-id grouping."""
    targets = _resolved_targets(events)
    grouped: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[TransitionEvent]] = defaultdict(list)
    for event in events:
        target = targets.get(event.event_id)
        if target:
            grouped[(event.intent_signature, tuple(event.input_product_state), target)].append(event)

    candidates = [
        (key, family_events)
        for key, family_events in grouped.items()
        if sum(1 for event in family_events if not event.evidence.infra_error) >= min_support
    ]
    candidates.sort(
        key=lambda item: (
            sum(1 for event in item[1] if not event.evidence.infra_error),
            sum(event.evidence.reward for event in item[1]),
            -sum(event.evidence.risk_observed for event in item[1]),
        ),
        reverse=True,
    )
    if max_families is not None:
        candidates = candidates[:max_families]

    skip_family_ids = skip_family_ids or set()
    families: list[TransitionFamily] = []
    for index, (key, family_events) in enumerate(candidates, 1):
        intent, input_state, target_state = key
        family_id = _family_id(intent, list(input_state), list(target_state))
        if family_id in skip_family_ids:
            continue
        qsig, nsig, rsig = _running_statistics(
            family_events, alpha0=alpha0, risk_alpha0=risk_alpha0
        )
        by_tool: dict[str, list[TransitionEvent]] = defaultdict(list)
        for event in family_events:
            by_tool[event.tool].append(event)
        policies: list[ToolPolicy] = []
        for tool, tool_events in sorted(by_tool.items()):
            qtool, ntool, rtool = _running_statistics(
                tool_events, alpha0=alpha0, risk_alpha0=risk_alpha0
            )
            if ntool:
                policies.append(
                    _template_tool(
                        tool, tool_events, list(input_state), list(target_state), qtool, ntool, rtool
                    )
                )
        if not policies:
            continue
        policies.sort(key=lambda item: (item.q - item.risk, item.n), reverse=True)
        status = "positive" if qsig >= 0.75 and rsig <= 0.25 and nsig >= 2 else "candidate"
        if rsig >= 0.5:
            status = "negative"
        family = TransitionFamily(
            family_id=family_id,
            schema_version=4,
            task_type="general",
            intent_signature=intent,
            input_product_state=list(input_state),
            target_product_state=list(target_state),
            product_experience=_template_product(list(input_state), list(target_state), qsig, nsig, rsig),
            tool_policies=policies,
            provenance_summary={
                "events": len(family_events),
                "non_infra_events": nsig,
                "infra_events": sum(event.evidence.infra_error for event in family_events),
                "locally_attributable_failures": sum(
                    event.evidence.risk_observed > 0 for event in family_events
                ),
                "source_tasks": "withheld",
                "strict_rollout_only": True,
            },
            status=status,
        )
        if llm is not None:
            try:
                family = _distill_family(family, family_events, llm, max_examples=max_examples)
            except Exception:
                if not allow_template_fallback:
                    raise
                family.provenance_summary["template_fallback"] = True
        families.append(family)
        if progress is not None:
            progress(index, len(candidates), family)
    return families
