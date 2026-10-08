import json
from pathlib import Path

import pytest

from terrabox.evolution.experience_evo.boundary.builder import build_boundary_store
from terrabox.evolution.experience_evo.boundary.learning import (
    check_condition, digest, evidence_from_row, evolve_rules, learn_conditions,
)
from terrabox.evolution.experience_evo.boundary.probes import replay_probes
from terrabox.evolution.experience_evo.boundary.probes import audit_metadata_pairs
from terrabox.evolution.experience_evo.boundary.transforms import boundary_allows


def test_parent_cannot_be_deleted(tmp_path):
    (tmp_path / "families_v2.jsonl").write_text("")
    for output in (tmp_path, tmp_path.parent, tmp_path / "child"):
        with pytest.raises(ValueError):
            build_boundary_store(tmp_path, output, overwrite=True)
    assert (tmp_path / "families_v2.jsonl").exists()


def test_evidence_ignores_labels_and_filters_infra():
    row = {"question": "calculate", "ground_truth": "secret", "task_type": "secret",
           "metrics": {"f1": 1}, "expected_tools": ["secret"], "conversation_history": [
               {"type": "AIMessage", "tool_calls": [{"name": "compute__calculator", "args": {"expression": "1+1"}}]},
               {"type": "ToolMessage", "content": 'Tool execution error: connection timeout'},
           ]}
    result = evidence_from_row(row)
    assert len(result) == 1 and result[0]["outcome"] == "unknown"
    assert "secret" not in json.dumps(result)
    row["metrics"]["f1"] = 0
    assert evidence_from_row(row) == result


def examples():
    return [{"tool": "compute.solver", "features": {"present:query": flag},
             "outcome": "success" if flag else "failure", "split": split,
             "group_id": f"{split}-{i}", "evidence_id": f"{split}-{i}-{flag}"}
            for split in ("discovery", "validation") for i in range(2) for flag in (False, True)]


def test_independent_validation_and_success_counterexamples():
    rows = examples()
    conditions, _ = learn_conditions(rows)
    assert len(conditions) == 1
    # Repeating a single task never supplies independent support.
    assert not learn_conditions([dict(r, group_id="same") for r in rows])[0]
    assert not learn_conditions([r for r in rows if r["split"] == "discovery"])[0]
    rows.append(dict(rows[0], outcome="success", evidence_id="contradiction"))
    assert not learn_conditions(rows)[0]


def test_split_and_unknown():
    conditions = learn_conditions(examples())[0]
    rules = [{"family_id": "a", "version": 1, "status": "active"}]
    result = evolve_rules(rules, [{"family_id": "a", "tool_policies": [{"tool": "compute.solver"}]}], conditions)
    assert result[0]["children"][0]["fail_status"] == "quarantined"
    assert rules[0]["version"] == 1
    assert check_condition(conditions[0], {}, {}) is False
    assert check_condition({"predicate": "same_metadata:a:b:crs"}, {}, {}) is None
    assert boundary_allows({"required_product_state": ["task_request"]}, ["task_request"])[0]
    assert not boundary_allows({"required_product_state": ["input:image"]}, ["task_request"])[0]
    assert not boundary_allows({"required_product_state": ["input:image", "input:image"]}, ["input:image"])[0]


def test_metadata_preserving_and_changing():
    state = {"artifacts": [{"path": "tmp/a.tif", "crs": "EPSG:4326"},
                            {"path": "tmp/b.tif", "crs": "EPSG:4326"}]}
    probes = audit_metadata_pairs(state, {"first": "tmp/a.tif", "second": "tmp/b.tif"})
    assert all(p["status"] == "validated" for p in probes if p["type"] == "semantic_preserving")
    assert all(p["status"] == "validated" for p in probes if p.get("slot") == "crs")
    assert all(p["status"] == "unknown" for p in probes if p.get("slot") == "time")


def test_real_probe_resume_and_input_isolation(tmp_path):
    source = tmp_path / "train"
    source.mkdir()
    original = tmp_path / "input.txt"
    original.write_text("original")
    for i in range(20):
        row = {"question": f"task {i}", "conversation_history": [
            {"type": "AIMessage", "tool_calls": [{"name": "compute__test", "args": {"file": str(original), "value": 1}}]},
            {"type": "ToolMessage", "content": '{"status":"success"}'},
        ]}
        (source / f"{i}.json").write_text(json.dumps(row))
    calls = []
    def execute(tool, args):
        calls.append(args)
        Path(args["file"]).write_text("changed")
        return '{"status":"success"}' if "value" in args else 'Tool execution error: missing required parameter value'
    catalog = {"compute.test": {"required": ["value"]}}
    result = replay_probes(str(source), str(tmp_path / "probes"), execute=execute, catalog=catalog)
    assert original.read_text() == "original"
    assert len(learn_conditions(result)[0]) == 1
    before = len(calls)
    assert replay_probes(str(source), str(tmp_path / "probes"), execute=execute, catalog=catalog) == result
    assert len(calls) == before


def test_disabled_runtime_exactly_matches_v4_and_freezes_hash(tmp_path, monkeypatch):
    from test_experience_evo_v4_clean import _family
    from terrabox.evolution.experience_evo.v2.store import ExperienceEvoV2Store
    from terrabox.evolution.experience_evo.v4_clean.runtime import ExperienceEvoV4CleanRuntime
    from terrabox.evolution.experience_evo.boundary.runtime import ExperienceEvoBoundaryRuntime
    parent = tmp_path / "parent"
    store = ExperienceEvoV2Store(parent)
    store.write_families([_family("a", task_type="general", tool="compute.calculator", target="result:from:compute.calculator")])
    original = (parent / "families_v2.jsonl").read_bytes()
    build_boundary_store(parent, tmp_path / "boundary")
    monkeypatch.setenv("TERRABOX_EXPEVO_BOUNDARY_DISABLE", "1")
    base = ExperienceEvoV4CleanRuntime(parent)
    new = ExperienceEvoBoundaryRuntime(tmp_path / "boundary")
    kwargs = {"current_product_state": ["task_request"], "available_tools": {"compute.calculator"}}
    assert base.augment("Calculate a raster ratio", **kwargs) == new.augment("Calculate a raster ratio", **kwargs)
    assert (parent / "families_v2.jsonl").read_bytes() == original


def test_two_agent_audit_requires_execution_and_resumes(tmp_path, monkeypatch):
    from terrabox.evolution.experience_evo.boundary import research
    from terrabox.agent.tool_executor import AgentToolExecutor
    from terrabox.core.registry import registry
    import terrabox.extensions
    monkeypatch.setattr(terrabox.extensions, "load_builtin_toolkits", lambda: None)
    monkeypatch.setattr(registry, "list_tools", lambda: [])
    class FakeAgent:
        def __init__(self, **kwargs):
            self._use_docker = True
        def call_json(self, text, **kwargs):
            if "Boundary Proposer" in text:
                return {"condition": "Use compatible units", "probes": [{"case_index": 0, "argument": "expression", "value": "1/0", "kind": "semantic_changing", "reason": "zero denominator"}]}
            return {"decision": "add_condition", "condition": "The denominator must be nonzero", "confidence": 0.9}
    monkeypatch.setattr(research, "BoundaryAgent", FakeAgent)
    calls = []
    def execute(tool, args, user=None):
        calls.append(args)
        return '{"status":"success","value":2}' if args["expression"] == "1+1" else 'Tool execution error: invalid expression'
    monkeypatch.setattr(AgentToolExecutor, "execute", execute)
    parent, source = tmp_path / "parent", tmp_path / "source"
    parent.mkdir(); source.mkdir()
    family = {"family_id": "f", "input_product_state": ["task_request"], "target_product_state": ["result"],
              "product_experience": {}, "tool_policies": [{"tool": "compute.calculator"}]}
    (parent / "families_v2.jsonl").write_text(json.dumps(family) + "\n")
    row = {"question": "calculate", "ground_truth": "secret", "conversation_history": [
        {"type": "AIMessage", "tool_calls": [{"name": "compute__calculator", "args": {"expression": "1+1"}}]},
        {"type": "ToolMessage", "content": '{"status":"success","value":2}'}]}
    (source / "one.json").write_text(json.dumps(row))
    result = research.review_families(parent, source, tmp_path / "reviews", "p", "a")
    assert result["summary"]["executed_pairs"] == 1
    assert result["records"][0]["audit"]["decision"] == "add_condition"
    assert "secret" not in json.dumps(result)
    count = len(calls)
    assert research.review_families(parent, source, tmp_path / "reviews", "p", "a") == result
    assert len(calls) == count
