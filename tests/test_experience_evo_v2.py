import json
import sys
from pathlib import Path


def _rollout_row() -> dict:
    return {
        "task_id": "oea_train_demo",
        "source": "openearth",
        "task_type": "type_calculate",
        "question": "Calculate the percentage of high temperature pixels in the scene.",
        "status": "completed",
        "images": [],
        "data_files": [],
        "conversation_history": [
            {
                "type": "AIMessage",
                "tool_calls": [
                    {
                        "name": "compute__solver",
                        "args": {"expression": "high_temperature_pixels / total_pixels"},
                    }
                ],
            },
            {
                "type": "ToolMessage",
                "content": json.dumps({"status": "ok", "value": 0.42}),
            },
        ],
    }


def test_experience_evo_v2_build_preview_and_augmenter(tmp_path: Path):
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.experience_evo.runner import main
    from terrabox.evolution.experience_evo.v2.store import ExperienceEvoV2Store

    source = tmp_path / "result.json"
    source.write_text(json.dumps(_rollout_row()), encoding="utf-8")
    store_dir = tmp_path / "store"

    old_argv = sys.argv
    try:
        sys.argv = [
            "experience_evo",
            "build-v2",
            "--source",
            str(source),
            "--store-dir",
            str(store_dir),
            "--template-only",
        ]
        main()
    finally:
        sys.argv = old_argv

    store = ExperienceEvoV2Store(store_dir)
    stats = store.stats()
    assert stats["events"] == 1
    assert stats["families"] == 1
    assert (store_dir / "events_v2.jsonl").exists()
    assert (store_dir / "families_v2.jsonl").exists()
    assert (store_dir / "experience_evo_v2.sqlite").exists()

    prompt = get_prompt_augmenter(
        "experience_evo_v2",
        store_dir=str(store_dir),
        top_k=1,
    ).augment("Need to calculate the high temperature percentage.")

    assert "Retrieved Product-Transition Experiences" in prompt
    assert "Product-State Transitions" in prompt
    assert "Tool Policies Ranked by Quse" in prompt
    assert "compute.solver" in prompt
    assert "Quse=" in prompt
