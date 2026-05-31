import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sft_row() -> dict:
    return {
        "id": "oea_train_1",
        "source": "openearth",
        "task_type": "type_distance",
        "question": "Measure the distance between two baseball fields.",
        "images": ["/data/image.jpg"],
        "data_files": [],
        "ground_truth": "42 meters",
        "messages": [
            {"role": "user", "content": "Measure the distance between two baseball fields."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": "Segment both fields.",
                        "actions": [
                            {
                                "tool": "geo_perception.remotesam",
                                "arguments": {"image": "/data/image.jpg", "prompt": "baseball field"},
                            }
                        ],
                    }
                ),
            },
        ],
        "gold_tool_calls": [
            {
                "tool": "geo_perception.remotesam",
                "arguments": {"image": "/data/image.jpg", "prompt": "baseball field"},
            }
        ],
    }


def test_sft_records_convert_to_memrl_source_records(tmp_path):
    from terrabox.evolution.memrl_full.source_adapter import load_sft_as_memrl_records

    path = tmp_path / "sft.jsonl"
    _write_jsonl(path, [_sft_row()])

    records = load_sft_as_memrl_records(path)

    assert len(records) == 1
    record = records[0]
    assert record.task_id == "oea_train_1"
    assert record.task_description == "Measure the distance between two baseball fields."
    assert record.success is True
    assert record.reward == 1.0
    assert "geo_perception.remotesam" in record.trajectory
    assert record.metadata["source_benchmark"] == "terrabox_openearth_sft"


def test_real_trajectory_records_keep_success_and_failure_metadata(tmp_path):
    from terrabox.evolution.memrl_full.source_adapter import load_trajectories_as_memrl_records

    path = tmp_path / "trajectories_full.jsonl"
    _write_jsonl(
        path,
        [
            {
                "task_id": "run_1",
                "source": "openearth",
                "question": "Count storage tanks.",
                "expected_tools": ["geo_perception.remotesam"],
                "tool_sequence": ["geo_perception.remotesam"],
                "metrics": {"f1": 1.0},
                "real_success": True,
                "status": "completed",
                "conversation_history": [{"role": "assistant", "content": "done"}],
            },
            {
                "task_id": "run_2",
                "source": "openearth",
                "question": "Find a missing image.",
                "metrics": {"f1": 0.0},
                "success": False,
                "status": "exception",
                "error": "file not found",
                "conversation_history": [],
            },
        ],
    )

    records = load_trajectories_as_memrl_records(path, include_empty_failures=True)

    assert [r.task_id for r in records] == ["run_1", "run_2"]
    assert records[0].success is True
    assert records[0].reward == 1.0
    assert records[1].success is False
    assert records[1].metadata["failure_type"] == "exception_empty_conversation"


def test_lite_source_service_populates_index_and_prompt_injector_reads_it(tmp_path):
    from terrabox.evolution.memrl_full.source_adapter import load_sft_as_memrl_records
    from terrabox.evolution.memrl_full.source_memory_service import create_memrl_source_service
    from terrabox.evolution.memrl_full.source_prompt_injector import MemRLSourcePromptInjector

    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_sft_row()])
    records = load_sft_as_memrl_records(data)

    service = create_memrl_source_service(
        store_dir=tmp_path / "store",
        backend="lite",
    )
    manifest = service.add_records(records)
    service.save_snapshot("smoke")

    assert manifest["backend"] == "lite"
    assert manifest["added"] == 1
    assert Path(manifest["memory_index"]).exists()

    injector = MemRLSourcePromptInjector(tmp_path / "store", top_k=1)
    prompt = injector.augment("How far apart are the baseball fields?")
    assert "Relevant MemRL Source Memories" in prompt
    assert "baseball fields" in prompt


def test_source_prompt_injector_uses_memrl_style_success_failure_context(tmp_path):
    from terrabox.evolution.memrl_full.source_adapter import load_sft_as_memrl_records
    from terrabox.evolution.memrl_full.source_memory_service import create_memrl_source_service
    from terrabox.evolution.memrl_full.source_prompt_injector import MemRLSourcePromptInjector

    row = _sft_row()
    row["messages"].append({"role": "user", "content": "OBSERVATION:\nraw model output"})
    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [row])
    records = load_sft_as_memrl_records(data)
    service = create_memrl_source_service(store_dir=tmp_path / "store", backend="lite")
    service.add_records(records)

    prompt = MemRLSourcePromptInjector(tmp_path / "store", top_k=1).augment(
        "How far apart are the baseball fields?"
    )

    assert "SUCCESSFUL MEMORIES" in prompt
    assert "Archived Trajectory" in prompt
    assert "Tool sequence: geo_perception.remotesam" in prompt
    assert "GOLD TOOL CALLS" in prompt
    assert "OBSERVATION" in prompt


def test_sft_records_can_align_tool_sequence_to_rollout_task_file(tmp_path):
    from terrabox.evolution.memrl_full.source_adapter import load_sft_as_memrl_records

    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_sft_row()])

    records = load_sft_as_memrl_records(
        data,
        expected_tool_overrides={"oea_train_1": ["geo_perception.remotesam", "ipython.execute"]},
    )

    assert records[0].metadata["tool_sequence"] == ["geo_perception.remotesam", "ipython.execute"]
    assert records[0].metadata["sft_tool_sequence"] == ["geo_perception.remotesam"]


def test_memrl_full_source_registered_as_prompt_augmenter(tmp_path):
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.memrl_full.source_adapter import load_sft_as_memrl_records
    from terrabox.evolution.memrl_full.source_memory_service import create_memrl_source_service

    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_sft_row()])
    records = load_sft_as_memrl_records(data)
    service = create_memrl_source_service(store_dir=tmp_path / "store", backend="lite")
    service.add_records(records)

    augmenter = get_prompt_augmenter(
        "memrl_full_source",
        store_dir=str(tmp_path / "store"),
        top_k=1,
    )

    assert "Relevant MemRL Source Memories" in augmenter.augment("baseball field distance")


def test_source_runner_populate_source_smoke(tmp_path):
    from terrabox.evolution.memrl_full.source_runner import main

    data = tmp_path / "sft.jsonl"
    _write_jsonl(data, [_sft_row()])
    store = tmp_path / "store"

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "source_runner",
            "populate-source",
            "--sft",
            str(data),
            "--store-dir",
            str(store),
            "--backend",
            "lite",
        ]
        main()
    finally:
        sys.argv = old_argv

    assert (store / "manifest.json").exists()
    assert (store / "memory_index.jsonl").exists()


def test_source_runner_exposes_online_source_command():
    from terrabox.evolution.memrl_full.source_runner import build_parser

    help_text = build_parser().format_help()

    assert "online-source" in help_text


def test_source_runner_rollout_command_forwards_runtime_controls():
    import argparse

    from terrabox.evolution.memrl_full.source_runner import _rollout_command

    args = argparse.Namespace(
        python_bin="python",
        task_file="tasks.json",
        experiment="exp",
        mode="standard",
        port=9100,
        store_dir="store",
        limit=2,
        start_index=None,
        end_index=None,
        max_iterations=7,
        resume=False,
        no_restrict_tools=True,
        no_skip_mock=False,
        no_skip_bing=False,
        no_skip_osm=True,
        no_skip_vlm=False,
        no_skip_changeos=False,
        output_dir="",
    )

    cmd = _rollout_command(args)

    assert "--max-iterations" in cmd
    assert "7" in cmd
    assert "--resume" not in cmd
    assert "--no-restrict-tools" in cmd
    assert "--no-skip-osm" in cmd
