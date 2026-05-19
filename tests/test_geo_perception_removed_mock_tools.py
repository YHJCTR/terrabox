from terrabox.core.registry import get_tool
from terrabox.extensions import load_builtin_toolkits


def test_geo_perception_mock_tools_are_not_registered():
    load_builtin_toolkits()

    removed_slugs = [
        "geo_perception.mscn_classify",
        "geo_perception.sm3det_detect",
        "geo_perception.change_os_detect",
    ]

    for slug in removed_slugs:
        assert get_tool(slug) is None
