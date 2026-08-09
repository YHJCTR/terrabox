import json
from pathlib import Path


def _write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def test_ace_playbook_builds_and_injects_relevant_bullets(tmp_path: Path):
    from terrabox.evolution.ace_playbook.builder import build_playbook
    from terrabox.evolution.ace_playbook.prompt_injector import ACEPlaybookPromptInjector

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "train_1",
            "question": "Compute burn severity with NBR change for a forest area.",
            "task_type": "ind_nbr",
            "status": "completed",
            "success": True,
            "real_success": True,
            "metrics": {"f1": 0.9},
            "tool_calls_deduped": [
                "osm_gis.get_area_boundary",
                "osm_gis.add_index_layer",
                "osm_gis.compute_index_change",
                "osm_gis.display_on_map",
            ],
        },
    )
    _write_result(
        results / "task2.json",
        {
            "task_id": "train_2",
            "question": "Measure distance between two fields in an image.",
            "task_type": "distance_image",
            "status": "completed",
            "success": True,
            "metrics": {"f1": 0.8},
            "tool_calls_deduped": ["geo_perception.instructsam", "compute.calculator"],
        },
    )

    store = tmp_path / "store"
    playbook = build_playbook(results, store, max_bullets=10)
    assert (store / "playbook.json").exists()
    assert (store / "playbook.txt").exists()
    assert playbook.manifest["uses_gold"] is False

    prompt = ACEPlaybookPromptInjector(store, top_k=1).augment(
        "Create a NBR burn severity map.",
        task_type="ind_nbr",
        available_tools=["osm_gis.compute_index_change"],
    )
    assert "ACE-Style Evolved Playbook" in prompt
    assert "osm_gis.compute_index_change" in prompt
    assert "geo_perception.instructsam" not in prompt


def test_ace_playbook_registered_as_prompt_augmenter(tmp_path: Path):
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.ace_playbook.playbook import ACEPlaybook, PlaybookBullet

    playbook = ACEPlaybook(tmp_path)
    playbook.bullets = [
        PlaybookBullet(
            id="str-00001",
            text="For count tasks, call geo_perception.sam2_segment before compute.calculator.",
            helpful=2,
            support=2,
            task_types=["count"],
            tools=["geo_perception.sam2_segment", "compute.calculator"],
            keywords=["count", "segment"],
        )
    ]
    playbook.save()

    augmenter = get_prompt_augmenter("ace_playbook", store_dir=str(tmp_path), top_k=1)
    prompt = augmenter.augment("count all tanks", task_type="count")
    assert "ACE-Style Evolved Playbook" in prompt
    assert "sam2_segment" in prompt
