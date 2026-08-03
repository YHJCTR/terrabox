import json
import os

from terrabox.agent import tool_executor


def test_inject_gpkg_skips_get_bbox_raster_mode(tmp_path):
    gpkg = tmp_path / "boundary.gpkg"
    gpkg.write_bytes(b"gpkg")
    tool_executor._gpkg_by_scope.clear()
    tool_executor._set_current_gpkg(str(gpkg))

    prepared = tool_executor._inject_gpkg(
        "osm_gis.get_bbox_from_raster",
        {"geotiff": "/tmp/source.tif"},
    )

    assert prepared == {"geotiff": "/tmp/source.tif"}


def test_inject_gpkg_supports_get_bbox_vector_mode(tmp_path):
    gpkg = tmp_path / "boundary.gpkg"
    gpkg.write_bytes(b"gpkg")
    tool_executor._gpkg_by_scope.clear()
    tool_executor._set_current_gpkg(str(gpkg))

    prepared = tool_executor._inject_gpkg(
        "osm_gis.get_bbox_from_raster",
        {"layer": "area_boundary"},
    )

    assert prepared == {"layer": "area_boundary", "gpkg": str(gpkg)}


def test_prepare_artifact_paths_resolves_task_data_file_basename(monkeypatch, tmp_path):
    data_dir = tmp_path / "question45"
    data_dir.mkdir()
    raster = data_dir / "2022_08_01_1925_BT_31_Day.tif"
    raster.write_bytes(b"placeholder")

    monkeypatch.delenv("TERRABOX_ARTIFACT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TERRABOX_TOKEN_ARTIFACT_DIR", raising=False)
    monkeypatch.setenv("TERRABOX_TASK_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TERRABOX_TASK_DATA_FILES", json.dumps([str(raster)]))

    prepared, resolutions = tool_executor._prepare_artifact_paths(
        "geo_statistics.batch_raster_stats",
        {
            "input_paths": ["2022_08_01_1925_BT_31_Day.tif"],
            "output_path": "stats.json",
        },
    )

    assert prepared["input_paths"] == [str(raster)]
    assert prepared["output_path"] == "stats.json"
    assert resolutions == [
        {
            "param": "input_paths[0]",
            "requested": "2022_08_01_1925_BT_31_Day.tif",
            "resolved": str(raster),
            "kind": "task_data_file_resolution",
        }
    ]


def test_prepare_artifact_paths_repairs_duplicate_question_data_segment(monkeypatch, tmp_path):
    monkeypatch.delenv("TERRABOX_ARTIFACT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TERRABOX_TOKEN_ARTIFACT_DIR", raising=False)
    real = tmp_path / "data" / "earthbench" / "question_data" / "question3" / "B02.tif"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"placeholder")
    bad = str(real).replace("/question_data/", "/question_data/question_data/")

    prepared, resolutions = tool_executor._prepare_artifact_paths(
        "earth_sci.calculate_pwv",
        {"b02": bad},
    )

    assert prepared["b02"] == str(real)
    assert resolutions[0]["kind"] == "path_normalization"
