"""Schemas for ExperienceEvo v2 transition events and atomic families."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _strings(value: Any) -> list[str]:
    return [str(item) for item in (value or [])]


@dataclass
class LocalEvidence:
    input_binding_ok: float
    output_valid: float
    target_completed: float
    downstream_consumed: float | None
    terminal_usable: float
    reward: float
    risk_observed: float
    risk_reasons: list[str] = field(default_factory=list)
    infra_error: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalEvidence":
        downstream = data.get("downstream_consumed")
        return cls(
            input_binding_ok=float(data.get("input_binding_ok", 0.0) or 0.0),
            output_valid=float(data.get("output_valid", 0.0) or 0.0),
            target_completed=float(data.get("target_completed", 0.0) or 0.0),
            downstream_consumed=None if downstream is None else float(downstream),
            terminal_usable=float(data.get("terminal_usable", 0.0) or 0.0),
            reward=float(data.get("reward", 0.0) or 0.0),
            risk_observed=float(data.get("risk_observed", 0.0) or 0.0),
            risk_reasons=_strings(data.get("risk_reasons")),
            infra_error=bool(data.get("infra_error", False)),
        )


@dataclass
class TransitionEvent:
    event_id: str
    task_id: str
    source: str
    task_type: str
    intent_signature: str
    task_hint: str
    step_index: int
    tool: str
    input_product_state: list[str]
    target_product_state: list[str]
    output_product_state: list[str]
    args_summary: dict[str, Any]
    observation_summary: dict[str, Any]
    evidence: LocalEvidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionEvent":
        return cls(
            event_id=str(data.get("event_id", "")),
            task_id=str(data.get("task_id", "")),
            source=str(data.get("source", "rollout")),
            task_type=str(data.get("task_type", "general")),
            intent_signature=str(data.get("intent_signature", "general")),
            task_hint=str(data.get("task_hint", "general")),
            step_index=int(data.get("step_index", 0) or 0),
            tool=str(data.get("tool", "")),
            input_product_state=_strings(data.get("input_product_state")),
            target_product_state=_strings(data.get("target_product_state")),
            output_product_state=_strings(data.get("output_product_state")),
            args_summary=dict(data.get("args_summary") or {}),
            observation_summary=dict(data.get("observation_summary") or {}),
            evidence=LocalEvidence.from_dict(dict(data.get("evidence") or {})),
        )


@dataclass
class ProductExperience:
    goal: str
    preconditions: list[str]
    output_checks: list[str]
    downstream_rule: str
    recovery: list[str]
    experience: str
    q: float
    n: int
    risk: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductExperience":
        return cls(
            goal=str(data.get("goal", "")),
            preconditions=_strings(data.get("preconditions")),
            output_checks=_strings(data.get("output_checks")),
            downstream_rule=str(data.get("downstream_rule", "")),
            recovery=_strings(data.get("recovery")),
            experience=str(data.get("experience", "")),
            q=float(data.get("q", 0.0) or 0.0),
            n=int(data.get("n", 0) or 0),
            risk=float(data.get("risk", 0.0) or 0.0),
        )


@dataclass
class ToolPolicy:
    tool: str
    required_input_roles: list[str]
    parameter_binding_rules: list[str]
    output_contract: list[str]
    post_checks: list[str]
    downstream_rule: str
    recovery: list[str]
    experience: str
    q: float
    n: int
    risk: float
    source_event_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolPolicy":
        return cls(
            tool=str(data.get("tool", "")),
            required_input_roles=_strings(data.get("required_input_roles")),
            parameter_binding_rules=_strings(data.get("parameter_binding_rules")),
            output_contract=_strings(data.get("output_contract")),
            post_checks=_strings(data.get("post_checks")),
            downstream_rule=str(data.get("downstream_rule", "")),
            recovery=_strings(data.get("recovery")),
            experience=str(data.get("experience", "")),
            q=float(data.get("q", 0.0) or 0.0),
            n=int(data.get("n", 0) or 0),
            risk=float(data.get("risk", 0.0) or 0.0),
            source_event_ids=_strings(data.get("source_event_ids")),
        )


@dataclass
class TransitionFamily:
    family_id: str
    schema_version: int
    task_type: str
    intent_signature: str
    input_product_state: list[str]
    target_product_state: list[str]
    product_experience: ProductExperience
    tool_policies: list[ToolPolicy]
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionFamily":
        return cls(
            family_id=str(data.get("family_id", "")),
            schema_version=int(data.get("schema_version", 2) or 2),
            task_type=str(data.get("task_type", "general")),
            intent_signature=str(data.get("intent_signature", "general")),
            input_product_state=_strings(data.get("input_product_state")),
            target_product_state=_strings(data.get("target_product_state")),
            product_experience=ProductExperience.from_dict(
                dict(data.get("product_experience") or {})
            ),
            tool_policies=[
                ToolPolicy.from_dict(dict(item))
                for item in (data.get("tool_policies") or [])
                if isinstance(item, dict)
            ],
            provenance_summary=dict(data.get("provenance_summary") or {}),
            status=str(data.get("status", "candidate")),
        )

    @property
    def search_text(self) -> str:
        parts = [
            self.task_type,
            self.intent_signature,
            " ".join(self.input_product_state),
            " ".join(self.target_product_state),
            self.product_experience.goal,
            self.product_experience.experience,
            " ".join(self.product_experience.output_checks),
        ]
        return "\n".join(part for part in parts if part)
