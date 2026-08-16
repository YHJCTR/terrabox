import json
import sys
from pathlib import Path

from terrabox.evolution.experience_evo.v2.models import (
    ProductExperience,
    ToolPolicy,
    TransitionFamily,
)
from terrabox.evolution.experience_evo.v2.store import ExperienceEvoV2Store
from terrabox.evolution.experience_evo.v4_clean.extractor import (
    extract_clean_transition_events_from_row,
)
from terrabox.evolution.experience_evo.v4_clean.runtime import ExperienceEvoV4CleanRuntime


def _row() -> dict:
    return {
        "task_id": "benchmark-secret-id",
        "task_type": "benchmark-secret-type",
        "expected_tools": ["secret.gold.tool"],
        "ground_truth": "secret answer",
        "metrics": {"f1": 1.0},
        "question": "Calculate the vegetation ratio from the current raster.",
        "status": "completed",
        "conversation_history": [
            {"type": "AIMessage", "tool_calls": [{"name": "compute__calculator", "args": {"expression": "3 / 4"}}]},
            {"type": "ToolMessage", "content": json.dumps({"status": "ok", "value": 0.75})},
        ],
    }


def _family(family_id: str, *, task_type: str, tool: str, target: str) -> TransitionFamily:
    product = ProductExperience(
        goal=f"Produce {target}.",
        preconditions=["Runtime state must contain task_request."],
        output_checks=[f"Check {target}."],
        downstream_rule="Use current-run values.",
        recovery=["Correct bindings before retrying."],
        experience=f"Advance to {target}.",
        q=0.9,
        n=10,
        risk=0.0,
    )
    policy = ToolPolicy(
        tool=tool,
        required_input_roles=["task_request"],
        parameter_binding_rules=["Use current-run values."],
        output_contract=[target],
        post_checks=[f"Check {target}."],
        downstream_rule="Use current-run values.",
        recovery=["Correct bindings before retrying."],
        experience=f"Use {tool}.",
        q=0.9,
        n=10,
        risk=0.0,
    )
    return TransitionFamily(
        family_id=family_id,
        schema_version=4,
        task_type=task_type,
        intent_signature="calculate+raster",
        input_product_state=["task_request"],
        target_product_state=[target],
        product_experience=product,
        tool_policies=[policy],
        provenance_summary={"strict_rollout_only": True},
        status="positive",
    )


def test_clean_extractor_removes_benchmark_labels_and_metrics():
    events = extract_clean_transition_events_from_row(_row(), source_name="rollout")
    assert len(events) == 1
    event = events[0].to_dict()
    serialized = json.dumps(event, ensure_ascii=True)
    assert event["task_id"] == "rollout"
    assert event["task_type"] == "general"
    assert "benchmark-secret" not in serialized
    assert "secret.gold.tool" not in serialized
    assert "secret answer" not in serialized
    assert "f1" not in serialized


def test_clean_runtime_ignores_task_type_for_retrieval(tmp_path: Path):
    store = ExperienceEvoV2Store(tmp_path / "store")
    store.write_families(
        [
            _family("a", task_type="private_train_label_a", tool="compute.calculator", target="result:from:compute.calculator"),
            _family("b", task_type="private_train_label_b", tool="compute.solver", target="result:from:compute.solver"),
        ]
    )
    runtime = ExperienceEvoV4CleanRuntime(store.store_dir, top_k=2)
    first = runtime.retrieve(
        "Calculate a raster ratio.",
        task_type="private_train_label_a",
        current_product_state=["task_request"],
        available_tools={"compute.calculator", "compute.solver"},
    )
    second = runtime.retrieve(
        "Calculate a raster ratio.",
        task_type="unseen_eval_label",
        current_product_state=["task_request"],
        available_tools={"compute.calculator", "compute.solver"},
    )
    assert [family.family_id for family in first] == [family.family_id for family in second]


def test_build_v4_clean_writes_strict_manifest(tmp_path: Path):
    from terrabox.evolution.experience_evo.runner import main

    source = tmp_path / "rollout.json"
    source.write_text(json.dumps(_row()), encoding="utf-8")
    store = tmp_path / "store"
    old_argv = sys.argv
    try:
        sys.argv = [
            "experience_evo",
            "build-v4-clean",
            "--source",
            str(source),
            "--store-dir",
            str(store),
            "--template-only",
        ]
        main()
    finally:
        sys.argv = old_argv
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["strict_rollout_only"] is True
    assert manifest["schema_version"] == 4
    event_text = (store / "events_v2.jsonl").read_text(encoding="utf-8")
    assert "benchmark-secret" not in event_text
