import json


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
