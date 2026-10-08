"""Paired real-tool probes with isolated artifacts and resumable evidence.

The executor sees recorded actor calls, never dataset gold calls. Semantic
equivalence is not inferred from a tool returning success. Unknown metadata
stays unvalidated. Historical inputs missing on disk are explicitly skipped.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from collections import defaultdict
from contextlib import contextmanager

from ....agent.artifacts.state import OUTPUT_PARAM_NAMES, looks_like_path
from ....agent.artifacts.extractors import parse_tool_observation
from ..transition_extractor import iter_tool_observations
from ..v2.extractor import _INFRA_PATTERNS, _ATTRIBUTABLE_PATTERNS
from .learning import digest


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def isolated_environment(folder: Path):
    keys = ("TERRABOX_ARTIFACT_OUTPUT_DIR", "TERRABOX_GPKG_OUTPUT_DIR", "TERRABOX_TASK_DATA_DIR", "TERRABOX_TASK_DATA_FILES")
    old = {k: os.environ.get(k) for k in keys}
    from terrabox.agent import tool_executor
    previous_gpkg = tool_executor._get_current_gpkg()
    try:
        os.environ["TERRABOX_ARTIFACT_OUTPUT_DIR"] = str(folder)
        os.environ["TERRABOX_GPKG_OUTPUT_DIR"] = str(folder / "gpkg")
        os.environ.pop("TERRABOX_TASK_DATA_DIR", None)
        os.environ["TERRABOX_TASK_DATA_FILES"] = "[]"
        tool_executor._set_current_gpkg(None)
        yield
    finally:
        tool_executor._set_current_gpkg(previous_gpkg)
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _isolated_args(args: dict, folder: Path) -> dict:
    """Copy mutable inputs; never replay a recorded call against original files."""
    folder.mkdir(parents=True, exist_ok=True)
    def visit(key, value):
        if isinstance(value, dict):
            return {k: visit(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(key, v) for v in value]
        if not isinstance(value, str):
            return value
        if key in OUTPUT_PARAM_NAMES:
            return str(folder / (key + Path(value).suffix))
        if looks_like_path(value):
            original = Path(value).resolve()
            if not original.is_file():
                raise FileNotFoundError("recorded input artifact unavailable")
            target = folder / (digest(str(original))[:16] + original.suffix)
            shutil.copy2(original, target)
            return str(target)
        return value
    return {key: visit(key, value) for key, value in args.items()}


def replay_probes(train_results: str, output_dir: str, *, per_tool_partition: int = 2,
                  execute=None, catalog=None) -> list[dict]:
    """Audit every observed tool with successful controls in both train partitions.

    The budget limits probes, not training records or the later full test eval.
    A deletion is accepted as a counterexample only if its paired control works
    now and the mutation returns an attributable error now.
    """
    if execute is None or catalog is None:
        from terrabox.extensions import load_builtin_toolkits
        from terrabox.core.registry import registry
        from terrabox.agent.tool_executor import AgentToolExecutor
        load_builtin_toolkits()
        catalog = {s.slug: s.parameters for s in registry.list_tools()}
        execute = lambda tool, args: AgentToolExecutor.execute(tool, args, user=None)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(set)
    evidence = []
    for source in sorted(Path(train_results).glob("*.json")):
        raw = json.loads(source.read_text())
        raw = raw.get("result", raw)
        question = str(raw.get("question") or "")
        group = digest(question)
        split = "validation" if int(group[:8], 16) % 3 == 0 else "discovery"
        for step, tool, args, observation in iter_tool_observations(raw.get("conversation_history") or []):
            if not tool.startswith(("geo_perception.", "osm_gis.", "compute.", "bing_search.")):
                continue
            if parse_tool_observation(observation)[1]:
                continue
            for key in sorted(catalog.get(tool, {}).get("required", [])):
                bucket = (tool, key, split)
                if key not in args or len(counts[bucket]) >= per_tool_partition or group in counts[bucket]:
                    continue
                probe_id = digest([question, step, tool, key, args])
                path = output / "results" / (probe_id + ".json")
                if path.exists() and json.loads(path.read_text()).get("status") != "planned":
                    result = json.loads(path.read_text())
                else:
                    result = {"probe_id": probe_id, "tool": tool, "predicate": f"present:{key}",
                              "group_id": group, "split": split, "type": "precondition_breaking",
                              "status": "planned", "execution": "real_tool_replay", "evidence": []}
                    atomic_json(path, result)
                    try:
                        # Both branches get fresh copies, preventing mutation of the control state.
                        control = _isolated_args(args, output / "artifacts" / probe_id / "control")
                        mutated = _isolated_args(args, output / "artifacts" / probe_id / "counterexample")
                        mutated.pop(key, None)
                        with isolated_environment(output / "artifacts" / probe_id / "control"):
                            control_obs = execute(tool, control)
                        control_bad = parse_tool_observation(control_obs)[1]
                        if control_bad:
                            result.update(status="unknown", reason="control_failed", control_observation=control_obs)
                        else:
                            with isolated_environment(output / "artifacts" / probe_id / "counterexample"):
                                mutated_obs = execute(tool, mutated)
                            low = mutated_obs.lower()
                            infra = any(p in low for p in (*_INFRA_PATTERNS, "quota", "billing", "payment", "402"))
                            bad = parse_tool_observation(mutated_obs)[1]
                            attributable = bad and any(p in low for p in (*_ATTRIBUTABLE_PATTERNS, "required positional", "field required"))
                            result.update(status="validated" if attributable and not infra else "unknown",
                                          control_observation=control_obs, mutation_observation=mutated_obs,
                                          reason="paired_missing_input" if attributable and not infra else "mutation_not_attributable")
                            if result["status"] == "validated":
                                for flag, obs, outcome in ((True, control_obs, "success"), (False, mutated_obs, "failure")):
                                    result["evidence"].append({"evidence_id": digest([probe_id, flag]),
                                        "group_id": group, "split": split, "tool": tool, "state": [],
                                        "features": {f"present:{key}": flag}, "outcome": outcome,
                                        "infra": False, "execution": "real_paired_tool_replay",
                                        "observation_sha256": digest(obs)})
                    except FileNotFoundError:
                        result.update(status="unknown", reason="input_artifact_unavailable")
                    except Exception as exc:
                        result.update(status="unknown", reason="execution_exception", exception_type=type(exc).__name__)
                    atomic_json(path, result)
                # Count attempted distinct questions to bound cost even with unavailable inputs.
                counts[bucket].add(group)
                evidence.extend(result.get("evidence", []))
    atomic_json(output / "summary.json", {"probe_count": len(list((output / "results").glob("*.json"))) if (output / "results").exists() else 0,
                "validated_evidence": len(evidence), "evidence": evidence,
                "semantic_changing": "unknown_without_independent_semantic_oracle",
                "semantic_preserving": "not_executed", "status": "complete"})
    return evidence


def audit_metadata_pairs(state: dict, args: dict) -> list[dict]:
    """Executable relational probes; these validate binding, not answer quality.

    Renaming preserves the artifact's immutable identity. A semantic mutation
    changes a known metadata value while keeping identity fixed. Neither
    operation is labelled a real tool replay.
    """
    from .learning import features
    result = []
    original = features(args, state)
    for item in state.get("artifacts", []):
        path = item.get("path")
        keys = [k for k, v in args.items() if v == path and k not in OUTPUT_PARAM_NAMES]
        if not path or not keys:
            continue
        renamed_state, renamed_args = deepcopy(state), deepcopy(args)
        alias = "tmp/boundary_alias/" + digest(path)[:12] + Path(path).suffix
        for a in renamed_state.get("artifacts", []):
            if a.get("path") == path:
                a["path"] = alias
        for key in keys:
            renamed_args[key] = alias
        result.append({"type": "semantic_preserving", "execution": "deterministic_metadata_probe",
                       "status": "validated" if features(renamed_args, renamed_state) == original else "failed"})
        for slot in ("crs", "unit", "resolution", "object", "time"):
            if item.get(slot) is None:
                result.append({"type": "semantic_changing", "slot": slot, "status": "unknown",
                               "reason": "metadata_not_observed", "execution": "not_executed"})
                continue
            changed = deepcopy(state)
            for a in changed["artifacts"]:
                if a.get("path") == path:
                    a[slot] = {"counterfactual": True, "original_type": type(item[slot]).__name__}
            updated = features(args, changed)
            affected = [k for k, v in original.items() if k.endswith(":" + slot) and updated.get(k) != v]
            result.append({"type": "semantic_changing", "slot": slot,
                           "execution": "deterministic_metadata_probe", "changed_predicates": affected,
                           "status": "validated" if affected else "unknown",
                           "reason": "binding_relation_changed" if affected else "no_observed_comparison_relation"})
    return result
