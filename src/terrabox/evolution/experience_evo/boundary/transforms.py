"""Deterministic counterexample templates used by the boundary v1 mode.

The first version deliberately does not call an extra judge model.  It records
auditable boundary hypotheses from the product-state schema and lets the
runtime enforce only checks that are observable in the current rollout.
"""

from __future__ import annotations

from typing import Any
from collections import Counter


def _semantic_slots(state: list[str]) -> list[str]:
    slots: list[str] = []
    for token in state:
        text = str(token)
        if any(key in text for key in ("image", "raster", "gpkg", "vector", "layer")):
            slots.append(text)
    return slots


def make_boundary_rule(family: Any) -> dict[str, Any]:
    """Create a portable rule from one TransitionFamily.

    Rules are sidecar data, so the parent v4-clean JSONL/SQLite files remain
    byte-for-byte untouched.  They are intentionally conservative: unknown
    metadata does not reject a candidate, while a missing required product does.
    """
    inputs = [str(x) for x in family.input_product_state]
    targets = [str(x) for x in family.target_product_state]
    required = list(inputs)
    semantic = _semantic_slots(inputs)
    return {
        "family_id": str(family.family_id),
        "version": 1,
        "required_product_state": required,
        "semantic_slots": semantic,
        "counterexamples": {
            "semantic_preserving": ["rename_artifact_reference", "alias_same_product"],
            "semantic_changing": ["swap_object_or_time", "change_unit_or_crs", "swap_layer"],
            "precondition_breaking": ["remove_required_product"],
        },
        "output_contract": targets,
        "policy": "reject_missing_required_state; unknown_metadata_is_not_rejected",
        "status": "active",
        "risk_delta": 0.0,
    }


def boundary_allows(rule: dict[str, Any], current_state: list[str]) -> tuple[bool, str]:
    if str(rule.get("status", "active")) in {"quarantined", "disabled"}:
        return False, "rule_disabled"
    current = Counter(str(x) for x in current_state)
    required = [str(x) for x in (rule.get("required_product_state") or [])]
    if required == ["task_request"] and current:
        return True, "task_request_available"
    missing = [token for token, count in Counter(required).items() if current[token] < count]
    if missing:
        return False, "missing_required:" + ",".join(missing[:3])
    return True, "ok"
