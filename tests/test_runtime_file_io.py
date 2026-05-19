from pathlib import Path

import pytest


def test_map_uploaded_files_to_declared_file_parameters():
    from terrabox.routers.tools import _map_uploaded_files_to_inputs

    inputs = {"threshold": 0.5}
    saved_paths = ["/tmp/runtime/uploads/a.tif", "/tmp/runtime/uploads/zones.geojson"]
    metadata = {"file_param_names": ["raster_path", "zones_geojson"]}

    mapped = _map_uploaded_files_to_inputs(inputs, saved_paths, metadata)

    assert mapped["threshold"] == 0.5
    assert mapped["raster_path"] == "/tmp/runtime/uploads/a.tif"
    assert mapped["zones_geojson"] == "/tmp/runtime/uploads/zones.geojson"
    assert "image" not in mapped


def test_map_uploaded_files_to_nested_file_parameters():
    from terrabox.routers.tools import _map_uploaded_files_to_inputs

    saved_paths = [
        "/tmp/runtime/uploads/19v.tif",
        "/tmp/runtime/uploads/19h.tif",
        "/tmp/runtime/uploads/37v.tif",
        "/tmp/runtime/uploads/37h.tif",
    ]
    metadata = {"file_param_names": ["bt_paths.19V", "bt_paths.19H", "bt_paths.37V", "bt_paths.37H"]}

    mapped = _map_uploaded_files_to_inputs({}, saved_paths, metadata)

    assert mapped["bt_paths"] == {
        "19V": "/tmp/runtime/uploads/19v.tif",
        "19H": "/tmp/runtime/uploads/19h.tif",
        "37V": "/tmp/runtime/uploads/37v.tif",
        "37H": "/tmp/runtime/uploads/37h.tif",
    }
    assert "image" not in mapped


def test_nested_file_parameters_replace_empty_root_placeholder():
    from terrabox.routers.tools import _map_uploaded_files_to_inputs

    saved_paths = ["/tmp/runtime/uploads/h.tif", "/tmp/runtime/uploads/v.tif"]
    metadata = {"file_param_names": ["bt_paths.H", "bt_paths.V"]}

    mapped = _map_uploaded_files_to_inputs({"bt_paths": []}, saved_paths, metadata)

    assert mapped["bt_paths"] == {
        "H": "/tmp/runtime/uploads/h.tif",
        "V": "/tmp/runtime/uploads/v.tif",
    }


def test_single_upload_without_mapping_keeps_image_compatibility():
    from terrabox.routers.tools import _map_uploaded_files_to_inputs

    mapped = _map_uploaded_files_to_inputs({}, ["/tmp/runtime/uploads/image.png"], {})

    assert mapped["image"] == "/tmp/runtime/uploads/image.png"
    assert mapped["images"] == ["/tmp/runtime/uploads/image.png"]
    assert mapped["image_path"] == "/tmp/runtime/uploads/image.png"
    assert mapped["image_paths"] == ["/tmp/runtime/uploads/image.png"]


def test_runtime_file_path_must_stay_inside_runtime_root(tmp_path, monkeypatch):
    from terrabox.routers.tools import _resolve_runtime_file_path

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    output = runtime_root / "outputs" / "u" / "exec" / "geo_raster" / "result.tif"
    output.parent.mkdir(parents=True)
    output.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("TERRABOX_RUNTIME_DIR", str(runtime_root))

    assert _resolve_runtime_file_path(str(output)) == output.resolve()

    with pytest.raises(ValueError):
        _resolve_runtime_file_path(str(tmp_path / "outside.txt"))


def test_default_output_path_uses_tool_timestamp_and_execution_id(tmp_path, monkeypatch):
    from terrabox.core.utils.runtime_paths import apply_default_output_paths, prepare_runtime_context

    monkeypatch.setenv("TERRABOX_RUNTIME_DIR", str(tmp_path / "runtime"))
    runtime = prepare_runtime_context(
        "user-1",
        "earth_sci.calculate_lst_sc",
        execution_id="exec_20260518_174008_76f727ed",
    )
    parameters = {
        "type": "object",
        "properties": {
            "output_path": {"type": "string", "description": "Output GeoTIFF path."}
        },
    }

    updated = apply_default_output_paths("earth_sci.calculate_lst_sc", {}, parameters, runtime)

    assert Path(updated["output_path"]).name == "calculate_lst_sc_20260518_174008_76f727ed.tif"
    assert Path(updated["output_path"]).parent == Path(runtime["output_dir"])
