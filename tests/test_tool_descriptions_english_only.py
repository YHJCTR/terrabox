from pathlib import Path

from terrabox.core.registry import list_tools, registry
from terrabox.extensions import load_builtin_toolkits


ROOT = Path(__file__).resolve().parents[1]


def _has_han(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def test_temporary_chinese_description_helper_is_removed():
    assert not (ROOT / "src/terrabox/core/utils/tool_chinese_descriptions.py").exists()
    assert not (ROOT / "tests/test_tool_chinese_descriptions.py").exists()


def test_registered_tool_descriptions_do_not_include_chinese_text():
    registry.clear()
    load_builtin_toolkits()

    for tool in list_tools():
        assert "\u4e2d\u6587\uff1a" not in tool.description
        assert not _has_han(tool.description), tool.slug
