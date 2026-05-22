from terrabox.core.utils.tool_input_coercion import coerce_tool_inputs


def test_coerces_frontend_array_lines_to_nested_values():
    parameters = {
        "properties": {
            "diff_pairs": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        },
    }

    inputs = {"diff_pairs": ["[0, 1]", "[1, 2]"]}

    assert coerce_tool_inputs(inputs, parameters)["diff_pairs"] == [[0, 1], [1, 2]]


def test_coerces_frontend_object_lines_to_objects():
    parameters = {
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            },
        },
    }

    inputs = {"annotations": ['{"text":"sample target","x":"28","y":"24"}']}

    assert coerce_tool_inputs(inputs, parameters)["annotations"] == [
        {"text": "sample target", "x": 28, "y": 24}
    ]


def test_preserves_string_array_file_paths():
    parameters = {
        "properties": {
            "bt_paths": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }
    inputs = {"bt_paths": ["/tmp/a.tif", "/tmp/b.tif"]}

    assert coerce_tool_inputs(inputs, parameters) == inputs
