"""Runtime for the independent counterexample-boundary ExperienceEvo mode."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any

from ..v4_clean.runtime import ExperienceEvoV4CleanRuntime
from ..v2.models import TransitionFamily
from .transforms import boundary_allows


class ExperienceEvoBoundaryRuntime(ExperienceEvoV4CleanRuntime):
    """v4-clean retrieval plus conservative applicability-boundary checks."""
    strict_augmentation = True

    def __init__(self, store_dir: str | Path, **kwargs: Any):
        super().__init__(store_dir, **kwargs)
        self._boundary_disabled = os.getenv("TERRABOX_EXPEVO_BOUNDARY_DISABLE", "0") == "1"
        self.boundary_trace: list[dict[str, Any]] = []
        path = Path(store_dir) / "boundary_rules.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"boundary store missing boundary_rules.jsonl: {path}")
        self._boundary_rules: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self._boundary_rules[str(row["family_id"])] = row
        manifest = json.loads((Path(store_dir) / "boundary_manifest.json").read_text())
        expected = manifest.get("rules_sha256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Boundary rule hash mismatch; refusing changed frozen store")
        self._selected_boundary_rules = []

    def retrieve(self, user_query: str, **kwargs: Any) -> list[TransitionFamily]:
        current_state = kwargs.get("current_product_state")
        if not isinstance(current_state, list):
            current_state = []
        families = super().retrieve(user_query, **kwargs)
        if self._boundary_disabled:
            return families
        self.boundary_trace = []
        accepted: list[TransitionFamily] = []
        for family in families:
            rule = self._boundary_rules.get(str(family.family_id))
            if rule is None:
                continue
            ok, _reason = boundary_allows(rule, [str(item) for item in current_state])
            self.boundary_trace.append({"family_id": family.family_id, "version": rule.get("version", 1),
                                        "allowed": ok, "reason": _reason})
            if ok:
                accepted.append(family)
        self._selected_boundary_rules = [self._boundary_rules[f.family_id] for f in accepted]
        return accepted

    def _condition_hint(self) -> str:
        conditions = {c["condition_id"]: c for r in self._selected_boundary_rules for c in r.get("conditions", [])}
        semantic = [r["semantic_condition"] for r in self._selected_boundary_rules if r.get("semantic_condition")]
        text = "\nValidated training-side applicability conditions (unknown metadata is not a failure):\n" + "\n".join(
            f"- {c['tool']}: require {c['predicate']}; if violated, repair the input binding or choose another action."
            for c in conditions.values()) if conditions else ""
        if semantic:
            text += "\nApplicability notes from paired tool execution and model audit (soft evidence):\n" + "\n".join(
                "- " + c + " If unmet, rebind current artifacts or fall back to ordinary tool reasoning."
                for c in semantic)
        return text

    def step_hint(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        text = super().step_hint(user_query, task_type=task_type, **kwargs)
        return text if self._boundary_disabled or not text else text + self._condition_hint()

    def guard_tool_call(self, user_query: str, selected_tool: str, task_type: str = "unknown", **kwargs: Any) -> str:
        base = super().guard_tool_call(user_query, selected_tool, task_type=task_type, **kwargs)
        if base or self._boundary_disabled:
            return base
        from .learning import check_condition
        tool = selected_tool.replace("__", ".")
        for rule in self._selected_boundary_rules:
            for condition in rule.get("conditions", []):
                if condition["tool"] != tool:
                    continue
                value = check_condition(condition, kwargs.get("selected_args") or {}, kwargs.get("artifact_state") or {})
                if value is False:
                    self.boundary_trace.append({"family_id": rule["family_id"], "version": rule["version"],
                        "condition_id": condition["condition_id"], "allowed": False, "reason": "validated_condition_violation"})
                    return f"ExperienceEvo boundary: {tool} violates {condition['predicate']}. Repair this input or select another action using current-run evidence."
        return ""

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs: Any) -> str:
        text = super().augment(user_query, task_type=task_type, **kwargs)
        if self._boundary_disabled:
            return text
        if not text:
            return text
        return text + self._condition_hint()
