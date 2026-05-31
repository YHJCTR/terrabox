import types

import scripts.run_trajectory_experiment as rte


def test_classifies_tool_oom_acknowledgement_as_system_limited():
    messages = [
        types.SimpleNamespace(
            content='{"status":"error","error_type":"tool_oom","message":"CUDA out of memory"}'
        )
    ]

    status = rte.classify_rollout_status(
        "The task cannot be completed because the tool hit CUDA out of memory.",
        messages,
    )

    assert status["status"] == "system_limited"
    assert status["has_tool_error"] is True
    assert status["has_tool_oom"] is True
    assert status["system_limitation_acknowledged"] is True


def test_classifies_normal_final_as_completed():
    status = rte.classify_rollout_status("Final answer: 42", [])

    assert status["status"] == "completed"
    assert status["has_tool_error"] is False
    assert status["has_tool_oom"] is False
    assert status["system_limitation_acknowledged"] is False
