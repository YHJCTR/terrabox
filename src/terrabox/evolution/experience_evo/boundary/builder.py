"""Build an isolated counterexample-boundary store from v4-clean."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..v2.store import ExperienceEvoV2Store
from .transforms import make_boundary_rule
from .auditor import apply_audit, audit_rule


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_boundary_store(parent_store: str | Path, output_store: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    parent = Path(parent_store).resolve()
    output = Path(output_store).resolve()
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("parent and output stores must be disjoint")
    if not (parent / "families_v2.jsonl").exists():
        raise FileNotFoundError(f"parent store has no families_v2.jsonl: {parent}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output store already exists: {output}")
        raise ValueError("Boundary stores are immutable; choose a new output directory")
    output.mkdir(parents=True)

    # Copy the parent store first.  The new runtime only reads the copied
    # families/events and uses the sidecar to add boundary checks.
    for source in parent.iterdir():
        if source.is_file():
            shutil.copy2(source, output / source.name)
    store = ExperienceEvoV2Store(parent)
    families = store.load_families()
    rules = []
    audits = []
    for family in families:
        rule = make_boundary_rule(family)
        audit = audit_rule(rule)
        rules.append(apply_audit(rule, audit))
        audits.append(audit)
    sidecar = output / "boundary_rules.jsonl"
    with sidecar.open("w", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(json.dumps(rule, ensure_ascii=False) + "\n")
    audit_path = output / "boundary_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    parent_manifest = store.manifest()
    manifest = {
        "method": "experience_evo_boundary",
        "parent_method": parent_manifest.get("method", "experience_evo_v4_clean"),
        "parent_store": str(parent),
        "parent_families_sha256": _sha256(parent / "families_v2.jsonl"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "strict_rollout_only": True,
        "boundary_mode": "deterministic_counterexample_v1",
        "boundary_rules": "boundary_rules.jsonl",
        "boundary_audit": "boundary_audit.jsonl",
        "family_count": len(families),
        "active_rule_count": len(rules),
        "audit_counts": {
            "keep": sum(a.get("decision") == "keep" for a in audits),
            "quarantine": sum(a.get("decision") == "quarantine" for a in audits),
        },
        "counterexample_types": ["semantic_preserving", "semantic_changing", "precondition_breaking"],
        "gold_free": True,
        "parent_store_unchanged": True,
    }
    (output / "boundary_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep the ordinary manifest usable by existing tooling while documenting
    # that this is an independent method.
    manifest_path = output / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    current.update({"method": "experience_evo_boundary", "boundary_manifest": "boundary_manifest.json", "parent_store": str(parent)})
    manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
