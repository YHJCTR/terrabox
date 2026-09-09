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


def test_strict_trajectory_records_omit_labels_metrics_and_task_ids(tmp_path):
    from terrabox.evolution.memrl_full.source_adapter import (
        load_trajectories_as_memrl_records,
        write_memrl_records_jsonl,
    )
    from terrabox.evolution.memrl_full.source_memory_service import create_memrl_source_service

    path = tmp_path / "trajectories_full.jsonl"
    _write_jsonl(
        path,
        [
            {
                "task_id": "openearth_train_7",
                "source": "openearth",
                "task_type": "secret_type",
                "question": "Measure distance near Grand Central Station from /data/images/a.jpg.",
                "expected_tools": ["gold.secret"],
                "tool_sequence": ["geo_perception.remotesam"],
                "metrics": {"f1": 0.0},
                "real_success": False,
                "status": "completed",
                "final_answer_full": "The answer is 42 meters.",
                "conversation_history": [{"role": "assistant", "content": "used /tmp/x.png near Grand Central Station"}],
            },
            {
                "task_id": "openearth_train_8",
                "source": "openearth",
                "question": "Broken provider call.",
                "status": "exception",
                "error": "provider timeout",
                "conversation_history": [{"role": "assistant", "content": "retry"}],
            },
        ],
    )

    records = load_trajectories_as_memrl_records(path, strict_nolabel=True)
    out = tmp_path / "records.jsonl"
    write_memrl_records_jsonl(records, out, strict_nolabel=True)
    serialized = out.read_text(encoding="utf-8")

    assert len(records) == 1
    assert records[0].success is True
    assert records[0].reward == 0.85
    assert "EXPECTED TOOLS" not in records[0].trajectory
    assert "FINAL ANSWER" not in records[0].trajectory
    assert "f1" not in records[0].trajectory.lower()
    assert "task_id" not in serialized
    assert "task_type" not in serialized
    assert "expected_tools" not in serialized
    assert "metrics" not in serialized
    assert "openearth_train_7" not in serialized
    assert "/data/images/a.jpg" not in serialized

    service = create_memrl_source_service(
        store_dir=tmp_path / "store",
        backend="lite",
        strict_nolabel=True,
    )
    manifest = service.add_records(records)
    index_text = (tmp_path / "store" / "memory_index.jsonl").read_text(encoding="utf-8")
    assert manifest["strict_nolabel"] is True
    assert "task_id" not in index_text
    assert "expected_tools" not in index_text


def test_strict_memrl_disables_auto_fallback_and_f1_filter(tmp_path):
    import pytest

    from terrabox.evolution.memrl_full.source_adapter import load_trajectories_as_memrl_records
    from terrabox.evolution.memrl_full.source_memory_service import create_memrl_source_service

    data = tmp_path / "rows.jsonl"
    _write_jsonl(data, [])

    with pytest.raises(ValueError, match="metrics/F1"):
        load_trajectories_as_memrl_records(data, strict_nolabel=True, min_f1=0.8)
    with pytest.raises(ValueError, match="auto fallback"):
        create_memrl_source_service(store_dir=tmp_path / "store", backend="auto", strict_nolabel=True)


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


def test_external_source_service_retrieve_uses_memrl_retrieve_query(tmp_path):
    from terrabox.evolution.memrl_full.source_memory_service import ExternalMemRLSourceService

    calls = []

    class FakeExternalService:
        def retrieve_query(self, task_description, k=5, threshold=0.0):
            calls.append((task_description, k, threshold))
            return {
                "selected": [
                    {
                        "memory_id": "mem-1",
                        "content": "Task: measure baseball field distance\nUse remotesam then compute distance.",
                        "similarity": 0.91,
                        "q_estimate": 0.7,
                        "metadata": {"tool_sequence": ["geo_perception.remotesam"]},
                    }
                ],
                "candidates": [],
                "actions": ["mem-1"],
                "simmax": 0.91,
            }

    service = object.__new__(ExternalMemRLSourceService)
    service.store_dir = tmp_path / "store"
    service.backend = "external_memrl"
    service._service = FakeExternalService()

    retrieval = service.retrieve_for_prompt("How far apart are the baseball fields?", top_k=3, threshold=0.2)

    assert calls == [("How far apart are the baseball fields?", 3, 0.2)]
    assert retrieval["retrieved_ids"] == ["mem-1"]
    assert retrieval["prompt"]
    assert "Use remotesam then compute distance" in retrieval["prompt"]


def test_external_closed_loop_updates_then_adds_rollout_memory(tmp_path, monkeypatch):
    from terrabox.evolution.memrl_full import source_runner

    events = []

    class FakeService:
        backend = "external_memrl"

        def retrieve_for_prompt(self, query, *, top_k=5, threshold=0.0):
            events.append(("retrieve", query, top_k, threshold))
            return {
                "prompt": "## Relevant MemRL Source Memories\nprior distance solution",
                "retrieved_ids": ["mem-1"],
                "retrieved_queries": [("previous distance task", 0.8)],
                "selected": [{"memory_id": "mem-1"}],
            }

        def update_values(self, successes, retrieved_ids_list, **kwargs):
            events.append(("update", list(successes), list(retrieved_ids_list)))
            return {"mem-1": 0.9}

        def add_records_with_retrieval(self, records, retrieved_queries_list=None, retrieved_ids_list=None):
            events.append(
                (
                    "add",
                    [r.task_id for r in records],
                    retrieved_queries_list,
                    retrieved_ids_list,
                )
            )
            return {"backend": "external_memrl", "added": len(records)}

        def save_snapshot(self, snapshot_id):
            events.append(("snapshot", snapshot_id))
            return {"backend": "external_memrl", "checkpoint_id": snapshot_id}

        def manifest(self, added=0):
            return {"backend": "external_memrl", "added": added, "count": 1}

    monkeypatch.setattr(source_runner, "_build_external_rollout_runtime", lambda args: (FakeService(), {}, None, None))
    monkeypatch.setattr(source_runner, "run_single_task", lambda **kwargs: {
        "task_id": kwargs["task"]["task_id"],
        "source": "openearth",
        "question": kwargs["task"]["question"],
        "expected_tools": ["geo_perception.remotesam"],
        "tool_calls": ["geo_perception.remotesam"],
        "tool_calls_deduped": ["geo_perception.remotesam"],
        "metrics": {"f1": 1.0, "precision": 1.0, "recall": 1.0, "exact_match": True},
        "status": "completed",
        "success": True,
        "real_success": True,
        "conversation_history": [{"role": "assistant", "content": "done"}],
    })

    task_file = tmp_path / "tasks.json"
    task_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "source": "openearth",
                    "question": "Measure the distance between two fields.",
                    "expected_tools": ["geo_perception.remotesam"],
                }
            ]
        ),
        encoding="utf-8",
    )

    args = source_runner.build_parser().parse_args(
        [
            "train-external",
            "--task-file",
            str(task_file),
            "--store-dir",
            str(tmp_path / "store"),
            "--output-dir",
            str(tmp_path / "out"),
            "--limit",
            "1",
            "--backend",
            "external_memrl",
            "--snapshot-id",
            "trained",
        ]
    )
    args.func(args)

    assert events[0] == ("retrieve", "Measure the distance between two fields.", 5, 0.0)
    assert events[1] == ("update", [True], [["mem-1"]])
    assert events[2] == ("add", ["task-1"], [[("previous distance task", 0.8)]], [["mem-1"]])
    assert events[3] == ("snapshot", "trained")
    summary = json.loads((tmp_path / "store" / "results" / "memrl_full_external_train_summary.json").read_text())
    assert summary["rollout_summary"]["avg_f1"] == 1.0
