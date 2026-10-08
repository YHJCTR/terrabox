"""Deterministic audit/update policy for boundary v1.

This is intentionally auditable and cheap.  It does not pretend that a
synthetic rename proves semantic equivalence; those cases are recorded as
unknown hypotheses for later real-tool replay.
"""

from __future__ import annotations

from typing import Any


def audit_rule(rule: dict[str, Any]) -> dict[str, Any]:
    required = list(rule.get("required_product_state") or [])
    output = list(rule.get("output_contract") or [])
    if not required or not output:
        return {
            "family_id": rule.get("family_id"),
            "decision": "quarantine",
            "reason": "missing_input_or_output_contract",
            "semantic_preserving": "unknown",
            "semantic_changing": "unknown",
            "precondition_breaking": "reject",
        }
    return {
        "family_id": rule.get("family_id"),
        "decision": "keep",
        "reason": "schema_contract_present",
        "semantic_preserving": "defer_real_replay",
        "semantic_changing": "require_rebinding_or_rejection",
        "precondition_breaking": "reject",
    }


def apply_audit(rule: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    updated = dict(rule)
    if audit.get("decision") == "quarantine":
        updated["status"] = "quarantined"
        updated["risk_delta"] = 1.0
    updated["audit_decision"] = audit.get("decision")
    updated["audit_reason"] = audit.get("reason")
    return updated
