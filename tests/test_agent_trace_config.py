from __future__ import annotations

from types import SimpleNamespace

from terrabox.agent.trace_config import build_langchain_config


def test_build_langchain_config_includes_tags_metadata_and_recursion_limit():
    cfg = SimpleNamespace(agent_mode="standard")

    result = build_langchain_config(
        cfg,
        recursion_limit=15,
        metadata={"intent": "tool_call", "image_count": 2},
    )

    assert result["recursion_limit"] == 15
    assert result["tags"] == ["terrabox-agent", "standard"]
    assert result["metadata"]["agent_mode"] == "standard"
    assert result["metadata"]["intent"] == "tool_call"
    assert result["metadata"]["image_count"] == 2
