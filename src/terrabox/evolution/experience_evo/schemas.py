"""Schemas for artifact-transition experiences."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TransitionRecord:
    """One observed tool-level state transition from a historical rollout."""

    transition_id: str
    task_id: str
    source: str
    task_type: str
    question: str
    step_index: int
    tool: str
    input_signature: str
    output_signature: str
    before_state: list[str]
    after_state: list[str]
    args_summary: dict[str, Any] = field(default_factory=dict)
    observation_summary: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    reward: float = 0.0
    risk: float = 0.0
    infra_error: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionRecord":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            task_id=str(data.get("task_id", "")),
            source=str(data.get("source", "")),
            task_type=str(data.get("task_type", "general")),
            question=str(data.get("question", "")),
            step_index=int(data.get("step_index", 0) or 0),
            tool=str(data.get("tool", "")),
            input_signature=str(data.get("input_signature", "")),
            output_signature=str(data.get("output_signature", "")),
            before_state=list(data.get("before_state") or []),
            after_state=list(data.get("after_state") or []),
            args_summary=dict(data.get("args_summary") or {}),
            observation_summary=dict(data.get("observation_summary") or {}),
            status=str(data.get("status", "success")),
            reward=float(data.get("reward", 0.0) or 0.0),
            risk=float(data.get("risk", 0.0) or 0.0),
            infra_error=bool(data.get("infra_error", False)),
            issues=[str(item) for item in data.get("issues", []) or []],
        )


@dataclass
class ExperienceEntry:
    """Distilled reusable experience for one signature/tool bucket."""

    experience_id: str
    level: str
    task_type: str
    input_signature: str
    output_signature: str
    tool: str | None
    q: float
    n: int
    risk: float
    status: str
    next_artifact: str
    input_constraints: list[str]
    output_checks: list[str]
    downstream_use: str
    failure_modes: list[str]
    recovery: list[str]
    tool_parameter_notes: list[str]
    experience: str
    source_task_ids: list[str] = field(default_factory=list)
    transition_ids: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceEntry":
        return cls(
            experience_id=str(data.get("experience_id", "")),
            level=str(data.get("level", "signature")),
            task_type=str(data.get("task_type", "general")),
            input_signature=str(data.get("input_signature", "")),
            output_signature=str(data.get("output_signature", "")),
            tool=str(data["tool"]) if data.get("tool") else None,
            q=float(data.get("q", 0.0) or 0.0),
            n=int(data.get("n", 0) or 0),
            risk=float(data.get("risk", 0.0) or 0.0),
            status=str(data.get("status", "candidate")),
            next_artifact=str(data.get("next_artifact", "")),
            input_constraints=[str(item) for item in data.get("input_constraints", []) or []],
            output_checks=[str(item) for item in data.get("output_checks", []) or []],
            downstream_use=str(data.get("downstream_use", "")),
            failure_modes=[str(item) for item in data.get("failure_modes", []) or []],
            recovery=[str(item) for item in data.get("recovery", []) or []],
            tool_parameter_notes=[str(item) for item in data.get("tool_parameter_notes", []) or []],
            experience=str(data.get("experience", "")),
            source_task_ids=[str(item) for item in data.get("source_task_ids", []) or []],
            transition_ids=[str(item) for item in data.get("transition_ids", []) or []],
            examples=list(data.get("examples", []) or []),
        )

    @property
    def search_text(self) -> str:
        example_terms: list[str] = []
        for example in self.examples:
            if isinstance(example, dict):
                example_terms.append(str(example.get("task_hint", "")))
                args = example.get("args")
                if isinstance(args, dict):
                    example_terms.extend(str(value) for value in args.values() if not isinstance(value, (dict, list)))
        parts = [
            self.task_type,
            self.input_signature,
            self.output_signature,
            self.tool or "",
            self.next_artifact,
            self.downstream_use,
            self.experience,
            " ".join(self.input_constraints),
            " ".join(self.output_checks),
            " ".join(self.failure_modes),
            " ".join(self.recovery),
            " ".join(self.tool_parameter_notes),
            " ".join(example_terms),
        ]
        return "\n".join(part for part in parts if part)
