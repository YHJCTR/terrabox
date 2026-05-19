from terrabox.core.registry import get_tool, list_tools, registry
from terrabox.extensions import load_builtin_toolkits


def test_drone_video_tools_are_not_registered():
    registry.clear()
    load_builtin_toolkits()

    assert list_tools("drone_video") == []
    assert get_tool("drone_video.extract_frames") is None
    assert get_tool("drone_video.pixel_to_gps") is None
    assert get_tool("drone_video.parse_telemetry") is None
