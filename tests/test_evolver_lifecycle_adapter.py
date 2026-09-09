import json
from pathlib import Path


def _write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def test_evolver_parse_principle_output_uses_official_markers():
    from terrabox.evolution.evolver_lifecycle.source_adapter import parse_principle_output

    text = """[DESCRIPTION]:
Verify that a raster artifact exists before attempting map display.
[STRUCTURE]:
[["raster artifact", "must_exist_before", "map display"]]
"""
    principle = parse_principle_output(text, "principle-1", "guiding", "traj-1")

    assert principle.type == "guiding"
    assert "raster artifact" in principle.description
    assert principle.structure == [["raster artifact", "must_exist_before", "map display"]]
    assert principle.successful_trajectory_ids == ["traj-1"]
    assert principle.failed_trajectory_ids == []


def test_evolver_compact_trajectory_strips_strict_leakage():
    from terrabox.evolution.evolver_lifecycle.source_adapter import compact_trajectory

    row = {
        "task_id": "oea_train_42",
        "question": "Generate a map for San Bernardino National Forest using /data1/raw/file.tif",
        "status": "completed",
        "has_tool_error": False,
        "tool_calls_deduped": ["osm_gis.get_area_boundary", "osm_gis.display_on_map"],
        "conversation_history": [
            {"type": "AIMessage", "tool_calls": [{"id": "c1", "name": "osm_gis.display_on_map", "args": {"path": "/data1/raw/file.tif"}}]},
            {"type": "ToolMessage", "tool_call_id": "c1", "content": "Saved to /home/user/out.png for San Bernardino National Forest"},
        ],
    }

    traj = compact_trajectory(row, "traj-1")
    payload = json.dumps(traj, ensure_ascii=False)

    assert "oea_train_42" not in payload
    assert "/data1" not in payload
    assert "/home" not in payload
    assert "San Bernardino" not in payload
    assert traj["golden_answer"] == ""
    assert traj["tool_sequence"] == ["osm_gis.get_area_boundary", "osm_gis.display_on_map"]


def test_evolver_bank_retrieves_guiding_and_cautionary(tmp_path: Path):
    from terrabox.evolution.evolver_lifecycle.principle_bank import EvolveRPrincipleBank, ExperiencePrinciple
    from terrabox.evolution.evolver_lifecycle.prompt_injector import EvolveRLifecyclePromptInjector

    bank = EvolveRPrincipleBank(tmp_path)
    bank.manifest = {"strict_nolabel": False}
    bank.add_trajectory({"trajectory_id": "traj-1", "question": "", "log": "osm_gis.display_on_map succeeded after raster creation", "tool_sequence": ["osm_gis.add_index_layer", "osm_gis.display_on_map"], "final_outcome": "success", "golden_answer": ""})
    bank.add_principle(
        ExperiencePrinciple(
            principle_id="principle-1",
            type="guiding",
            description="For burn severity tasks, create the index layer before displaying the map.",
            structure=[["index layer", "precedes", "map display"]],
            metric_score=1.0,
            successful_trajectory_ids=["traj-1"],
        )
    )
    bank.save()

    injector = EvolveRLifecyclePromptInjector(tmp_path, top_k=1)
    prompt = injector.augment("Generate a burn severity map and summary.")

    assert "EvolveR-Style Retrieved Experience Principles" in prompt
    assert "create the index layer" in prompt
    assert "osm_gis.add_index_layer" in prompt


def test_evolver_strict_registration_requires_qwen_index(tmp_path: Path, monkeypatch):
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.evolver_lifecycle.principle_bank import EvolveRPrincipleBank

    bank = EvolveRPrincipleBank(tmp_path)
    bank.manifest = {"strict_nolabel": True}
    bank.save()
    monkeypatch.delenv("TERRABOX_EVOLVER_RETRIEVAL", raising=False)

    try:
        get_prompt_augmenter("evolver_lifecycle", store_dir=str(tmp_path))
    except RuntimeError as exc:
        assert "requires TERRABOX_EVOLVER_RETRIEVAL=qwen" in str(exc)
    else:
        raise AssertionError("strict EvolveR lifecycle store should reject lexical fallback")


def test_evolver_leak_scan_flags_forbidden_json(tmp_path: Path):
    from terrabox.evolution.evolver_lifecycle.source_adapter import leak_scan_path

    (tmp_path / "bad.json").write_text('{"task_id":"oea_train_1"}', encoding="utf-8")

    scan = leak_scan_path(tmp_path)

    assert not scan["ok"]
    assert any(hit["pattern"] == "task_id" for hit in scan["hits"])
