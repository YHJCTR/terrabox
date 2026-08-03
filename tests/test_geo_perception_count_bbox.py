from PIL import Image

from terrabox.toolkits import geo_perception


def test_count_given_object_normalizes_swapped_full_image_bbox(monkeypatch, tmp_path):
    image_path = tmp_path / "source.jpg"
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (4592, 3072), "white").save(image_path)

    def fake_instructsam(args, context, account):
        with Image.open(args["image"]) as crop:
            assert crop.size == (4592, 3072)
        return {
            "status": "success",
            "count": 2,
            "objects": [{"bbox": [10, 20, 30, 40]}],
            "detections": [{"bbox": [50, 60, 70, 80]}],
        }

    monkeypatch.setattr(geo_perception, "_auto_output_path", lambda prefix, ext="png": str(crop_path))
    monkeypatch.setattr(geo_perception, "instructsam_handler", fake_instructsam)

    result = geo_perception.count_given_object_handler(
        {
            "image": str(image_path),
            "text": "flooded house",
            "bbox": "(0,0,3072,4592)",
        },
        {},
        None,
    )

    assert result["status"] == "success"
    assert result["count"] == 2
    assert result["bbox_normalization"] == "swapped_full_image_bbox"
    assert result["objects"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
