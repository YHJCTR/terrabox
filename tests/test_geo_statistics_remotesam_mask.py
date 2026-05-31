import json

from terrabox.toolkits import geo_statistics


def test_read_valid_pixels_accepts_remotesam_mask_json(tmp_path):
    artifact = tmp_path / "geo_perception_remotesam.json"
    artifact.write_text(json.dumps({"mask": [[1, 0, 1], [0, 1, 0]]}), encoding="utf-8")

    valid = geo_statistics._read_valid_pixels(str(artifact))

    assert valid.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_count_pixels_condition_can_count_remotesam_mask(tmp_path):
    artifact = tmp_path / "geo_perception_remotesam.json"
    artifact.write_text(json.dumps({"mask": [[1, 0], [1, 1]]}), encoding="utf-8")

    result = geo_statistics.count_pixels_condition_handler(
        {"input_path": str(artifact), "lower": 1, "upper": 1},
        {},
        None,
    )

    assert result == {"count": 3, "total": 4, "ratio": 0.75}
