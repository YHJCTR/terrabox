from __future__ import annotations

from types import SimpleNamespace

from terrabox.agent import tools
from terrabox.agent.tool_contracts import ToolContractValidator


def test_standard_contract_validator_rejects_png_raster_diff_inputs(tmp_path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"\x89PNG\r\n\x1a\n")
    after.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = ToolContractValidator().validate(
        "geo_raster.raster_diff",
        {
            "path_a": str(before),
            "path_b": str(after),
            "output_path": str(tmp_path / "diff.png"),
        },
    )

    assert not result.ok
    assert "raster_diff_requires_geospatial_raster" in result.issue_codes
    assert "raster_diff_output_must_be_geotiff" in result.issue_codes
    assert "Tool contract validation failed" in result.to_tool_message()
    assert "path_a" in result.to_tool_message()


def test_langchain_tool_wrapper_short_circuits_when_contract_fails(monkeypatch):
    calls = []

    class _FailingValidator:
        def validate(self, slug, arguments):
            calls.append((slug, arguments))
            return SimpleNamespace(ok=False, to_tool_message=lambda: "contract failed before execution")

    monkeypatch.setattr(
        tools.registry,
        "list_tools",
        lambda: [
            SimpleNamespace(
                slug="example.echo",
                description="Echo input",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ],
    )
    monkeypatch.setattr(tools.registry, "get_handler", lambda _slug: object())
    monkeypatch.setattr(
        tools.AgentToolExecutor,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("executor should not run")),
    )

    [tool] = tools.build_langchain_tools(
        user=SimpleNamespace(id="user-1"),
        pre_execute_validator=_FailingValidator(),
    )

    assert tool.func(text="hello") == "contract failed before execution"
    assert calls == [("example.echo", {"text": "hello"})]
