from copy import deepcopy

from terrabox.core.utils.tool_frontend_schema import prepare_gui_tool_parameters


def test_gui_schema_hides_auto_output_for_new_tool_without_mutating_original():
    original = {
        "type": "object",
        "properties": {
            "dem_path": {"type": "string"},
            "output_path": {"type": "string"},
            "unit": {"type": "string"},
        },
        "required": ["dem_path", "output_path"],
    }
    before = deepcopy(original)

    gui_schema, metadata = prepare_gui_tool_parameters(
        "disaster_response.calc_slope",
        original,
    )

    assert gui_schema["required"] == ["dem_path"]
    assert "output_path" not in gui_schema["properties"]
    assert metadata["auto_output_parameters"] == ["output_path"]
    assert original == before


def test_gui_schema_leaves_main_tool_unchanged():
    original = {
        "type": "object",
        "properties": {
            "output_path": {"type": "string"},
            "value": {"type": "number"},
        },
        "required": ["output_path", "value"],
    }

    gui_schema, metadata = prepare_gui_tool_parameters("example.math_add", original)

    assert gui_schema == original
    assert metadata == {}


def test_gui_tool_response_uses_prepared_schema():
    from terrabox.core.schemas import ToolSpecOut
    from terrabox.routers.tools import _prepare_tool_for_gui

    tool = ToolSpecOut(
        slug="disaster_response.calc_slope",
        name="Calculate Terrain Slope",
        description="Calculate terrain slope.",
        requires_connection=False,
        status="available",
        toolkit_slug="disaster_response",
        parameters={
            "type": "object",
            "properties": {
                "dem_path": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["dem_path", "output_path"],
        },
        metadata={"existing": True},
    )

    prepared = _prepare_tool_for_gui(tool)

    assert prepared.parameters["required"] == ["dem_path"]
    assert "output_path" not in prepared.parameters["properties"]
    assert prepared.metadata["existing"] is True
    assert prepared.metadata["auto_output_parameters"] == ["output_path"]
    assert tool.parameters["required"] == ["dem_path", "output_path"]
