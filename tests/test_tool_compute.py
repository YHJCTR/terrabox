from terrabox.toolkits.compute import calculator_handler


def test_calculator_supports_math_namespace():
    assert calculator_handler({"expression": "math.floor(11.39 / 8.74)"}) == "1"
    assert calculator_handler({"expression": "floor(11.39 / 8.74)"}) == "1"
