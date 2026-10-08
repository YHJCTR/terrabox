import json


def test_cold_tool_server_wrapper_handles_child_exception(tmp_path):
    import builtins
    import io
    import types
    from contextlib import redirect_stdout
    from unittest.mock import Mock, patch

    from terrabox.evolution.promptevo.adapters.toolbench import pipeline

    profile = pipeline.StableToolBenchPipelineProfile(agent_provider="longcat")
    process = Mock()
    process.poll.return_value = None
    with (
        patch.object(pipeline, "TOOL_SERVER_LOG_DIR", tmp_path),
        patch.object(pipeline, "_tool_server_runtime_config", return_value=tmp_path / "config.yml"),
        patch.object(pipeline, "resolve_provider", return_value=types.SimpleNamespace(api_key="test-secret")),
        patch.object(pipeline, "_http_ok", side_effect=[False, True]),
        patch.object(pipeline.subprocess, "Popen", return_value=process) as popen,
    ):
        assert pipeline._ensure_tool_server(profile)[0] is process
    wrapper = popen.call_args.args[0][2]
    compile(wrapper, "tool_server_wrapper", "exec")
    fake_main = types.SimpleNamespace(
        fake_response_function_chat=Mock(side_effect=TypeError("test-secret")),
        json=json, app=object(), CONFIG={"port": 8081},
        uvicorn=types.SimpleNamespace(run=Mock()),
    )
    namespace = {}
    output = io.StringIO()
    with (
        patch("os.chdir"),
        patch("sys.path", []),
        patch("importlib.import_module", return_value=fake_main),
        patch.object(builtins, "print", builtins.print),
        redirect_stdout(output),
    ):
        exec(wrapper, namespace)
        result = fake_main.fake_response_function_chat({}, {}, {})
    assert json.loads(result)["error"] == "Failed to generate fake response"
    assert "TypeError" in output.getvalue()
    assert "test-secret" not in output.getvalue()
    assert "<redacted>" in output.getvalue()


def _write_result(root, name, data):
    path = root / f"{name}_DFS.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_tool_calls_and_win_override_valid_data(tmp_path):
    from terrabox.evolution.promptevo.adapters.toolbench.core import (
        ToolBenchMetricProvider,
        ToolBenchTrajectorySource,
    )

    _write_result(
        tmp_path,
        "123",
        {
            "win": False,
            "answer_generation": {
                "valid_data": True,
                "train_messages": [[
                    {"role": "user", "content": "do task"},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }]},
                    {"role": "tool", "content": '{"error":"request invalid"}'},
                ],],
            },
        },
    )

    trace = next(iter(ToolBenchTrajectorySource(lambda _exp: str(tmp_path)).traces("x")))
    assert trace.success is False
    assert trace.steps[1].tool == "lookup"
    assert trace.steps[2].errored is True

    metric = ToolBenchMetricProvider(lambda _exp: str(tmp_path)).per_task("x")["123"]
    assert metric.success is False
    assert metric.tool_f1 == 0.0
    assert metric.extra["n_tool_calls"] == 1


def test_dfs_diagnosis_selects_one_branch_but_metrics_count_all(tmp_path):
    from terrabox.evolution.promptevo.adapters.toolbench.core import (
        ToolBenchMetricProvider,
        ToolBenchTrajectorySource,
    )

    def action(name, code=0, observation="ok"):
        return {
            "node_type": "Action",
            "description": name,
            "children": [{
                "node_type": "Action Input",
                "description": "{}",
                "observation": observation,
                "observation_code": code,
                "children": [],
            }],
        }

    root = {
        "node_type": "Action Input",
        "children": [action("first", code=12, observation="500"), action("second")],
    }
    _write_result(
        tmp_path,
        "456",
        {"win": True, "tree": {"size": 5, "max_length": 3, "tree": root}, "answer_generation": {}},
    )

    trace = next(iter(ToolBenchTrajectorySource(lambda _exp: str(tmp_path)).traces("x")))
    called = [step.tool for step in trace.steps if step.tool]
    assert len(called) == 1
    assert called[0] in {"first", "second"}

    metric = ToolBenchMetricProvider(lambda _exp: str(tmp_path)).per_task("x")["456"]
    assert metric.extra["n_tool_calls"] == 2
    assert metric.failure_flags["transient_error"] is True


def test_linear_candidate_fallback_is_preserved(tmp_path):
    from terrabox.evolution.promptevo.adapters.toolbench.core import ToolBenchTrajectorySource

    _write_result(
        tmp_path,
        "789",
        {
            "win": True,
            "compare_candidates": [[
                {"node_type": "Action", "description": "lookup"},
                {"node_type": "Action Input", "description": "{}", "observation_code": 0},
            ]],
        },
    )
    trace = next(iter(ToolBenchTrajectorySource(lambda _exp: str(tmp_path)).traces("x")))
    assert [step.tool for step in trace.steps if step.tool] == ["lookup"]


def test_validation_query_ids_do_not_match_row_indexes(tmp_path):
    from terrabox.evolution.promptevo.adapters.toolbench.runner import _prepare_query_file

    input_path = tmp_path / "G1_instruction.json"
    input_path.write_text(
        json.dumps([
            {"query_id": "100", "query": "first"},
            {"query_id": "200", "query": "second"},
        ]),
        encoding="utf-8",
    )
    output = _prepare_query_file(
        str(input_path), str(tmp_path / "out"), task_ids=["100"], allow_index_ids=False
    )
    rows = json.loads((tmp_path / "out" / "inputs" / input_path.name).read_text(encoding="utf-8"))
    assert [row["query_id"] for row in rows] == ["100"]
    assert output.endswith("G1_instruction.json")


def test_split_manifest_is_group_stratified_and_disjoint(tmp_path):
    from terrabox.evolution.promptevo.adapters.toolbench.pipeline import (
        StableToolBenchPipelineProfile,
        _profile_for_split,
        create_split_manifest,
    )

    groups = ("G1_instruction", "G1_category", "G1_tool")
    input_dir = tmp_path / "solvable_queries" / "test_instruction"
    input_dir.mkdir(parents=True)
    for group in groups:
        rows = [{"query_id": f"{group}-{index}", "query": str(index)} for index in range(10)]
        (input_dir / f"{group}.json").write_text(json.dumps(rows), encoding="utf-8")

    profile = StableToolBenchPipelineProfile(stable_root=str(tmp_path), stable_groups=groups)
    manifest = create_split_manifest(profile, tmp_path / "split_manifest.json", seed=7)
    assert set(manifest["splits"]) == {"evolution", "dev", "test"}
    for group in groups:
        pieces = [set(manifest["splits"][split][group]) for split in ("evolution", "dev", "test")]
        assert len(pieces[0]) == 6
        assert len(pieces[1]) == 2
        assert len(pieces[2]) == 2
        assert not (pieces[0] & pieces[1] or pieces[0] & pieces[2] or pieces[1] & pieces[2])
    dev = _profile_for_split(profile, manifest, "dev")
    assert len(dev.task_ids_by_group["G1_instruction"]) == 2
