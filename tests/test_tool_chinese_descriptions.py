from terrabox.core.utils.tool_chinese_descriptions import append_chinese_description


def test_appends_chinese_description_for_new_tool():
    description = append_chinese_description(
        "disaster_response.calc_slope",
        "Calculate terrain slope from a DEM.",
    )

    assert "中文：" in description
    assert "坡度" in description


def test_unknown_tool_description_is_unchanged():
    description = append_chinese_description("github.list_user_repos", "List repositories.")

    assert description == "List repositories."
