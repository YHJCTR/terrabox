import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_row(task_id: str = "sample_1") -> dict:
    return {
        "id": task_id,
        "source": "openearth",
        "task_type": "type_distance",
        "question": "Measure the distance between two baseball fields.",
        "images": ["/data/image.jpg"],
        "data_files": [],
        "ground_truth": "42 meters",
        "messages": [
            {"role": "system", "content": "Tool catalog: geo_perception.instructsam"},
            {"role": "user", "content": "Measure the distance between two baseball fields."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": "Detect baseball fields.",
                        "actions": [
                            {
                                "tool": "geo_perception.instructsam",
                                "function_name": "geo_perception__instructsam",
                                "arguments": {
                                    "image": "/data/image.jpg",
                                    "text_prompt": "baseballfield",
                                },
                            }
                        ],
                    }
                ),
            },
        ],
        "expected_tools": ["geo_perception.instructsam", "ipython.execute"],
        "gold_tool_calls": [
            {
                "tool": "geo_perception.instructsam",
                "function_name": "geo_perception__instructsam",
                "arguments": {"image": "/data/image.jpg", "text_prompt": "baseballfield"},
                "is_executable_under_current_schema": True,
                "argument_status": "adapted",
            },
            {
                "tool": "ipython.execute",
                "function_name": "ipython__execute",
                "arguments": {"action": "print(42)"},
                "is_executable_under_current_schema": True,
                "argument_status": "adapted",
            },
        ],
        "all_gold_calls_executable": True,
    }


def test_full_sft_loader_reads_jsonl_and_turns_gold_calls_into_trajectory(tmp_path):
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row()])

    samples = load_sft_samples(data_path)

    assert len(samples) == 1
    assert samples[0].task_id == "sample_1"
    assert samples[0].tool_sequence == ["geo_perception.instructsam", "ipython.execute"]
    assert samples[0].to_trajectory().tools_called == samples[0].tool_sequence


def test_skillrl_full_builds_skillbank_and_verl_jsonl(tmp_path):
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples
    from terrabox.evolution.skillrl_full.skillbank_builder import build_skillbank
    from terrabox.evolution.skillrl_full.verl_adapter import write_verl_jsonl

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row(), _sample_row("sample_2")])
    samples = load_sft_samples(data_path)

    skillbank = build_skillbank(samples, min_support=1)
    assert skillbank["general_skills"]
    assert "task_type:type_distance" in skillbank["task_specific_skills"]
    assert "geo_perception.instructsam" in skillbank["task_specific_skills"]["task_type:type_distance"][0]["principle"]

    out_path = tmp_path / "verl.jsonl"
    write_verl_jsonl(samples, out_path)
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["data_source"] == "terrabox_skillrl_full"
    assert row["reward_model"]["ground_truth"]
    assert row["extra_info"]["task_id"] == "sample_1"


def test_memrl_full_builds_episodes_and_sqlite_memory(tmp_path):
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples
    from terrabox.evolution.memrl_full.memory_builder import (
        build_memory_records,
        populate_sqlite_memory,
    )

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row()])
    samples = load_sft_samples(data_path)

    memories = build_memory_records(samples)
    assert memories[0]["intent"]["task_type"] == "type_distance"
    assert memories[0]["experience"]["tool_sequence"] == [
        "geo_perception.instructsam",
        "ipython.execute",
    ]
    assert memories[0]["utility"] == 1.0

    db_path = tmp_path / "memories.db"
    count = populate_sqlite_memory(memories, db_path, reset=True)
    assert count == 1
    assert db_path.exists()
