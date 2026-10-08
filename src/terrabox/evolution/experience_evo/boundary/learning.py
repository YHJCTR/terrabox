"""Learn conditional applicability from executed, training-only tool traces.

No gold fields are read. A counterexample is an attributable execution failure,
not a fabricated observation. Discovery and validation are split by public
question hash, keeping every turn of a task on the same side.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from ....agent.artifacts import product_state_tokens, update_artifact_state
from ....agent.artifacts.extractors import parse_tool_observation
from ....agent.artifacts.state import OUTPUT_PARAM_NAMES, looks_like_path
from ..transition_extractor import iter_tool_observations
from ..v2.extractor import _initial_state, _INFRA_PATTERNS, _ATTRIBUTABLE_PATTERNS
from .builder import _sha256, build_boundary_store


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def features(args: dict, state: dict) -> dict[str, bool]:
    """Only predicates observable both before execution and at inference time."""
    result = {}
    artifacts = state.get("artifacts", [])
    paths = {os.path.normpath(str(a["path"])) for a in artifacts if a.get("path")}
    layers = state.get("layers", [])
    for key, value in args.items():
        result[f"present:{key}"] = value is not None
        if key in OUTPUT_PARAM_NAMES or key in {"output_layer", "diff_layer_name"}:
            continue
        if isinstance(value, str) and looks_like_path(value):
            # Historical traces often omit the task's input file list. Absence
            # in that incomplete state is unknown, not evidence of wrong binding.
            if paths:
                result[f"known_artifact:{key}"] = os.path.normpath(value) in paths
        # A layer lookup must be scoped to its container; output names are not inputs.
        if key in {"layer", "source_layer", "target_layer", "layer1", "layer2"} and layers:
            gpkg = args.get("gpkg")
            result[f"known_layer:{key}"] = any(
                a.get("name") == value and (not gpkg or a.get("container") == gpkg)
                for a in layers
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in {"buffer_m", "resolution", "gsd", "pixel_size", "width", "height"}:
                result[f"positive:{key}"] = value > 0
    # Metadata equality is evaluated only when both references expose that field.
    bound = [(key, next((a for a in artifacts if a.get("path") == value), None))
             for key, value in args.items() if isinstance(value, str) and key not in OUTPUT_PARAM_NAMES]
    bound = [(key, a) for key, a in bound if a is not None]
    for i, (left, a) in enumerate(bound):
        for right, b in bound[i + 1:]:
            for slot in ("crs", "unit", "resolution"):
                if a.get(slot) is not None and b.get(slot) is not None:
                    result[f"same_metadata:{left}:{right}:{slot}"] = a[slot] == b[slot]
    return result


def check_condition(condition: dict, args: dict, state: dict) -> bool | None:
    predicate = condition["predicate"]
    if predicate.startswith("present:"):
        return args.get(predicate.split(":", 1)[1]) is not None
    return features(args, state).get(predicate)


def evidence_from_row(row: dict) -> list[dict]:
    # Deliberate allowlist: do not forward the raw benchmark record to extraction.
    visible = {k: row.get(k) for k in ("question", "images", "data_files", "conversation_history")}
    question = str(visible.get("question") or "")
    state = _initial_state(visible, question)
    split = "validation" if int(digest(question)[:8], 16) % 3 == 0 else "discovery"
    output = []
    for step, tool, args, observation in iter_tool_observations(visible.get("conversation_history") or []):
        parsed, error = parse_tool_observation(observation)
        lowered = observation.lower()
        infra = any(p in lowered for p in (*_INFRA_PATTERNS, "quota", "billing", "payment", "unauthorized", "402"))
        attributable = error and any(p in lowered for p in _ATTRIBUTABLE_PATTERNS)
        outcome = "unknown" if infra else "failure" if attributable else "unknown" if error else "success"
        output.append({
            "evidence_id": digest([question, step, tool, args, observation]),
            "group_id": digest(question), "split": split, "tool": tool,
            "state": product_state_tokens(state), "features": features(args, state),
            "outcome": outcome, "infra": infra,
            "observation_sha256": digest(observation),
            "execution": "recorded_real_tool_execution",
        })
        update_artifact_state(state, tool, args, observation)
    return output


def learn_conditions(rows: list[dict], *, min_support: int = 2) -> tuple[list[dict], list[dict]]:
    """Propose on discovery, accept only with independent validation support.

    Conditions are associative applicability evidence, not a causal success claim.
    Every retained negative region must contain failures and no observed success
    in BOTH partitions. Unknowns never become counterexamples.
    """
    proposals, accepted = [], []
    by_tool = defaultdict(list)
    for row in rows:
        if row["outcome"] in {"success", "failure"}:
            by_tool[row["tool"]].append(row)
    for tool, events in sorted(by_tool.items()):
        keys = sorted({k for e in events if e["split"] == "discovery" for k in e["features"]})
        for key in keys:
            # A predicate violation can be rejected only after observing both sides.
            stats = {}
            ids = []
            for split in ("discovery", "validation"):
                counts = Counter()
                groups = defaultdict(set)
                for e in events:
                    if e["split"] != split or key not in e["features"]:
                        continue
                    bucket = ("pass" if e["features"][key] else "fail") + "_" + e["outcome"]
                    groups[bucket].add(e["group_id"])
                    if not e["features"][key] and e["outcome"] == "failure":
                        ids.append(e["evidence_id"])
                counts.update({k: len(v) for k, v in groups.items()})
                stats[split] = dict(counts)
            def supported(c):
                return c.get("fail_failure", 0) >= min_support and c.get("pass_success", 0) >= min_support and not c.get("fail_success", 0)
            valid = all(supported(stats[s]) for s in ("discovery", "validation"))
            record = {"condition_id": digest([tool, key])[:20], "tool": tool,
                      "predicate": key, "required_value": True, "stats": stats,
                      "counterexample_ids": sorted(set(ids)),
                      "decision": "add_condition" if valid else "defer_unknown",
                      "evidence_scope": "train_only_heldout_execution_audit"}
            proposals.append(record)
            if valid:
                accepted.append(record)
    return accepted, proposals


def evolve_rules(rules: list[dict], families: list[dict], conditions: list[dict]) -> list[dict]:
    """Split applicability by tool, preserving each parent rule and its lineage."""
    output = []
    by_family = {str(f["family_id"]): f for f in families}
    for original in rules:
        rule = deepcopy(original)
        rule.update(version=2, conditions=[], children=[], update_history=[])
        tools = {p["tool"] for p in by_family[rule["family_id"]].get("tool_policies", [])}
        for condition in conditions:
            if condition["tool"] not in tools:
                continue
            rule["conditions"].append(condition)
            rule["children"].append({
                "rule_id": rule["family_id"] + ":" + condition["condition_id"],
                "parent_rule": rule["family_id"], "tool": condition["tool"],
                "predicate": condition["predicate"], "pass_status": "active",
                "fail_status": "quarantined", "unknown_status": "eligible",
            })
            rule["update_history"].append({"action": "split_applicability", "condition_id": condition["condition_id"]})
        output.append(rule)
    return output


def build_learned_boundary(parent_store: str, output_store: str, *, train_results: str,
                           eval_tasks: str, min_support: int = 2, replay_dir: str | None = None,
                           research_dir: str | None = None, proposer_url: str = "http://127.0.0.1:9102",
                           auditor_url: str = "http://127.0.0.1:9113") -> dict:
    parent, output = Path(parent_store).resolve(), Path(output_store).resolve()
    destination = output
    if destination.exists():
        raise FileExistsError("Use a new immutable boundary version directory")
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("Parent and learned store must be disjoint")
    source = Path(train_results).resolve()
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    allowed = {Path(p).resolve() for p in parent_manifest.get("source", [])}
    if source not in allowed:
        raise ValueError("Training source must match the parent store manifest")
    if min_support < 2:
        raise ValueError("min_support must be at least 2 independent questions per partition")
    tests = json.loads(Path(eval_tasks).read_text())
    if isinstance(tests, dict):
        tests = tests.get("tasks", tests.get("data", []))
    test_questions = {str(r.get("question") or r.get("query") or "").strip() for r in tests}
    rows, source_hashes = [], []
    metadata_probes = []
    for file in sorted(source.glob("*.json")):
        file_hash = _sha256(file)
        raw = json.loads(file.read_text())
        raw = raw.get("result", raw)
        if str(raw.get("question") or "").strip() in test_questions:
            raise ValueError("Train/eval question overlap; refusing boundary learning")
        rows.extend(evidence_from_row(raw))
        from .probes import audit_metadata_pairs
        state = _initial_state({k: raw.get(k) for k in ("images", "data_files")}, str(raw.get("question") or ""))
        for step, tool, args, obs in iter_tool_observations(raw.get("conversation_history") or []):
            for probe in audit_metadata_pairs(state, args):
                metadata_probes.append({**probe, "probe_id": digest([file_hash, step, tool, probe])})
            update_artifact_state(state, tool, args, obs)
        source_hashes.append(file_hash)
    if not rows:
        raise ValueError("No executed training evidence")
    replay_evidence = []
    if replay_dir:
        from .probes import replay_probes
        replay_evidence = replay_probes(str(source), replay_dir, per_tool_partition=min_support)
        rows.extend(replay_evidence)
    conditions, proposals = learn_conditions(rows, min_support=min_support)
    reviews = None
    if research_dir:
        from .research import review_families
        reviews = review_families(parent, source, Path(research_dir), proposer_url, auditor_url)
    output = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex[:8])
    manifest = build_boundary_store(parent, output)
    families = [json.loads(s) for s in (parent / "families_v2.jsonl").read_text().splitlines() if s]
    rules = [json.loads(s) for s in (output / "boundary_rules.jsonl").read_text().splitlines() if s]
    rules = evolve_rules(rules, families, conditions)
    if reviews:
        lookup = {r["family_id"]: r for r in reviews["records"]}
        for rule in rules:
            review = lookup[rule["family_id"]]
            audit = review["audit"]
            rule["semantic_audit"] = {"decision": audit["decision"], "judgement_type": "llm_estimate",
                                       "executed_pairs": sum(p["status"] == "executed" for p in review["probes"])}
            if audit["decision"] in {"add_condition", "split_family"}:
                rule["semantic_condition"] = audit["condition"]
                rule["update_history"].append({"action": audit["decision"], "evidence": "paired_execution_llm_audit"})
                if audit["decision"] == "split_family":
                    rule["semantic_branches"] = [
                        {"scope": "condition_satisfied", "policy": "use_parent_experience"},
                        {"scope": "condition_violated", "policy": "repair_binding_or_fallback"}]
            elif audit["decision"] == "quarantine":
                rule["status"] = "quarantined"
    for name, data in (("boundary_evidence.jsonl", rows), ("boundary_proposals.jsonl", proposals),
                       ("boundary_metadata_probes.jsonl", metadata_probes), ("boundary_rules.jsonl", rules)):
        with (output / name).open("w") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.update(schema_version=2, boundary_mode="train_counterexample_condition_learning",
                    active_rule_count=sum(r.get("status") == "active" for r in rules),
                    audit_counts=dict(Counter((r.get("semantic_audit") or {}).get("decision", "keep") for r in rules)),
                    evidence_count=len(rows), outcome_counts=dict(Counter(r["outcome"] for r in rows)),
                    proposal_count=len(proposals), accepted_condition_count=len(conditions),
                    conditioned_family_count=sum(bool(r["conditions"]) for r in rules),
                    source_digest=digest(source_hashes), source_count=len(source_hashes),
                    min_support=min_support, eval_frozen=True,
                    validation="disjoint question-hash train partitions; no eval feedback",
                    real_counterfactual_tool_replays=len(replay_evidence) // 2,
                    metadata_probe_counts=dict(Counter(p["status"] for p in metadata_probes)),
                    semantic_review=reviews["summary"] if reviews else None,
                    rules_sha256=_sha256(output / "boundary_rules.jsonl"))
    (output / "boundary_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    output.rename(destination)
    return manifest
