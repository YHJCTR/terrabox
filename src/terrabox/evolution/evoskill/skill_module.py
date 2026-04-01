"""SkillModule: structured reusable skill definition for EvoSkill.

From: EvoSkill — Automated Skill Discovery for Multi-Agent Systems (arXiv 2603.02766)

Skills operate at the abstraction level — not low-level prompts or code,
but structured capability descriptions that can be injected as context
and transferred across domains.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillModule:
    """A structured, reusable geospatial skill.

    Fields mirror the EvoSkill paper's skill abstraction format,
    adapted for the Terrabox tool system.
    """
    name: str
    description: str
    trigger_condition: str          # when to apply this skill
    tool_sequence: list[str]        # ordered Terrabox slugs
    parameter_hints: dict[str, str] # slug → parameter guidance
    preconditions: list[str]        # conditions that must hold before applying
    domain: str                     # "geo_perception" | "spatial" | "raster" | "code" | "multi"

    # Evaluation metadata
    validation_f1_delta: float = 0.0   # improvement in tool-match F1 vs baseline
    generality_score: float = 0.0      # fraction of eval cases where skill is relevant
    use_count: int = 0
    created_at: float = field(default_factory=time.time)
    id: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "tool_sequence": self.tool_sequence,
            "parameter_hints": self.parameter_hints,
            "preconditions": self.preconditions,
            "domain": self.domain,
            "validation_f1_delta": self.validation_f1_delta,
            "generality_score": self.generality_score,
            "use_count": self.use_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillModule":
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            trigger_condition=d.get("trigger_condition", ""),
            tool_sequence=d.get("tool_sequence", []),
            parameter_hints=d.get("parameter_hints", {}),
            preconditions=d.get("preconditions", []),
            domain=d.get("domain", "multi"),
            validation_f1_delta=d.get("validation_f1_delta", 0.0),
            generality_score=d.get("generality_score", 0.0),
            use_count=d.get("use_count", 0),
            created_at=d.get("created_at", time.time()),
            id=d.get("id", ""),
        )

    def to_prompt_text(self) -> str:
        """Format skill as injection-ready text."""
        lines = [f"**{self.name}**: {self.description}"]
        if self.trigger_condition:
            lines.append(f"  When to use: {self.trigger_condition}")
        if self.tool_sequence:
            lines.append(f"  Tool sequence: {' → '.join(self.tool_sequence)}")
        if self.parameter_hints:
            hints = "; ".join(f"{k}: {v}" for k, v in self.parameter_hints.items())
            lines.append(f"  Parameter hints: {hints}")
        if self.preconditions:
            lines.append(f"  Preconditions: {', '.join(self.preconditions)}")
        return "\n".join(lines)
