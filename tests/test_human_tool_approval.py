from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from terrabox.agent.human_approval import ConfigurableToolApprovalPolicy
from terrabox.agent.modes import standard
from terrabox.managers.gpu_status import _parse_nvidia_smi_gpu_status


def test_human_tool_approval_policy_matches_slug_and_prefix():
    config = SimpleNamespace(
        human_tool_approval={
            "enabled": True,
            "include_gpu_snapshot": True,
            "rules": [
                {
                    "name": "gpu_model_tools",
                    "match": {
                        "slugs": ["codegen.generate_and_run"],
                        "prefixes": ["geo_perception."],
                    },
                    "action": "require_approval",
                }
            ],
        }
    )

    policy = ConfigurableToolApprovalPolicy.from_config(config)

    codegen = policy.requirement_for("codegen.generate_and_run", {}, bucket="default")
    perception = policy.requirement_for("geo_perception.vlm_analyze", {}, bucket="perception")
    raster = policy.requirement_for("geo_raster.raster_diff", {}, bucket="compute")

    assert codegen is not None
    assert codegen.policy_name == "gpu_model_tools"
    assert codegen.include_gpu_snapshot is True
    assert perception is not None
    assert raster is None


def test_human_tool_approval_policy_can_be_disabled():
    config = SimpleNamespace(
        human_tool_approval={
            "enabled": False,
            "rules": [
                {
                    "name": "gpu_model_tools",
                    "match": {"slugs": ["codegen.generate_and_run"]},
                    "action": "require_approval",
                }
            ],
        }
    )

    policy = ConfigurableToolApprovalPolicy.from_config(config)

    assert policy.requirement_for("codegen.generate_and_run", {}, bucket="default") is None


def test_gpu_status_parser_returns_concise_free_memory_and_utilization():
    parsed = _parse_nvidia_smi_gpu_status("0, 18231, 4\n1, 9210, 0\n")

    assert parsed == [
        {"id": "0", "free_mib": 18231, "utilization_pct": 4},
        {"id": "1", "free_mib": 9210, "utilization_pct": 0},
    ]


def test_standard_dynamic_route_sends_matching_tool_calls_to_approval_gate():
    config = SimpleNamespace(
        human_tool_approval={
            "enabled": True,
            "rules": [
                {
                    "name": "gpu_model_tools",
                    "match": {"slugs": ["codegen.generate_and_run"]},
                    "action": "require_approval",
                }
            ],
        }
    )
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "codegen__generate_and_run",
                        "args": {"description": "test"},
                    }
                ],
            )
        ]
    }

    assert standard._route_after_standard_agent(state, config) == "approval_gate"


def test_standard_dynamic_route_bypasses_approval_for_unmatched_tools():
    config = SimpleNamespace(
        human_tool_approval={
            "enabled": True,
            "rules": [
                {
                    "name": "gpu_model_tools",
                    "match": {"slugs": ["codegen.generate_and_run"]},
                    "action": "require_approval",
                }
            ],
        }
    )
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "geo_raster__raster_info",
                        "args": {"path": "a.tif"},
                    }
                ],
            )
        ]
    }

    assert standard._route_after_standard_agent(state, config) == "tools"
