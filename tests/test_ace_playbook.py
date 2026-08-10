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


def test_ace_official_style_builder_tags_existing_bullets(tmp_path: Path, monkeypatch):
    from terrabox.evolution.ace_playbook.builder import build_playbook_ace_official_style

    results = tmp_path / "results"
    for idx, task_type in enumerate(["poi_distance", "poi_distance"], 1):
        _write_result(
            results / f"task{idx}.json",
            {
                "task_id": f"train_{idx}",
                "question": "Find nearby cafes and render a map for a city landmark.",
                "task_type": task_type,
                "status": "completed",
                "success": True,
                "real_success": True,
                "metrics": {"f1": 0.9},
                "tool_calls_deduped": [
                    "osm_gis.get_area_boundary",
                    "osm_gis.add_pois_layer",
                    "osm_gis.display_on_map",
                ],
                "final_answer_full": "Rendered current-run map artifact.",
            },
        )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def call_json(self, prompt: str, *, system: str, max_tokens: int):
            self.calls += 1
            if "ACE Reflector" in system:
                if "str-00001" in prompt:
                    return {
                        "reasoning": "Existing bullet matched the successful artifact handoff.",
                        "error_identification": "none",
                        "root_cause_analysis": "good handoff",
                        "correct_approach": "reuse the current gpkg artifact",
                        "key_insight": "pass current-run artifacts directly",
                        "bullet_tags": [{"id": "str-00001", "tag": "helpful"}],
                        "insights": [],
                    }
                return {
                    "reasoning": "Need a playbook entry for map artifact handoff.",
                    "error_identification": "none",
                    "root_cause_analysis": "missing playbook",
                    "correct_approach": "keep gpkg handoff intact",
                    "key_insight": "carry current artifact paths forward",
                    "bullet_tags": [],
                    "insights": [
                        {
                            "section": "tool_flow",
                            "content": "For POI map tasks, keep the current-run GeoPackage from boundary to POI layer to display without inserting unrelated tools.",
                            "task_types": ["poi_distance"],
                            "tools": ["osm_gis.get_area_boundary", "osm_gis.add_pois_layer", "osm_gis.display_on_map"],
                            "polarity": "helpful",
                            "evidence_strength": 0.9,
                        }
                    ],
                }
            return {
                "reasoning": "Add missing transferable handoff advice.",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "tool_flow",
                        "content": "For POI map tasks, keep the current-run GeoPackage from boundary to POI layer to display without inserting unrelated tools.",
                        "task_types": ["poi_distance"],
                        "tools": ["osm_gis.get_area_boundary", "osm_gis.add_pois_layer", "osm_gis.display_on_map"],
                        "polarity": "helpful",
                    }
                ],
            }

    fake = FakeLLM()
    monkeypatch.setattr("terrabox.agent.llm_provider.make_llm_client", lambda provider: fake)

    store = tmp_path / "store"
    playbook = build_playbook_ace_official_style(
        results,
        store,
        provider="longcat",
        batch_size=1,
        max_batches=2,
        force=True,
    )

    assert playbook.manifest["build_status"] == "complete"
    assert playbook.manifest["counter_updates"]["helpful"] == 1
    assert playbook.manifest["ace_mechanisms_reproduced"]
    assert (store / "ace_traces" / "batches.jsonl").exists()
    assert (store / "intermediate_playbooks" / "batch_0002_playbook.txt").exists()
    assert playbook.bullets[0].helpful >= 2
