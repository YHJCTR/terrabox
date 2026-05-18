"""Tests for the raster_viewer toolkit."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

try:
    from terrabox.toolkits import raster_viewer
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from terrabox.toolkits import raster_viewer


class MockRegistrar:
    def __init__(self):
        self.toolkits = {}
        self.tools = {}
        self.handlers = {}

    def toolkit(self, name: str, description: str, version: str):
        self.toolkits[name] = {"description": description, "version": version}

    def tool(self, toolspec, handler):
        self.tools[toolspec.slug] = toolspec
        self.handlers[toolspec.slug] = handler

    def call(self, slug: str, arguments: dict, context: dict | None = None, account=None):
        handler = self.handlers[slug]
        return handler(arguments, context or {}, account)


def _create_multiband_geotiff(path: str) -> None:
    import rasterio
    from rasterio.transform import from_origin

    band1 = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype="float32")
    band2 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype="float32")
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 3,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(100, 30, 0.1, 0.1),
        "nodata": np.nan,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(band1, 1)
        dst.write(band2, 2)
        dst.set_band_description(1, "temperature")
        dst.set_band_description(2, "quality")


def test_inspect_geotiff_reports_metadata_and_band_stats():
    with tempfile.TemporaryDirectory() as tmp:
        raster_path = os.path.join(tmp, "sample.tif")
        _create_multiband_geotiff(raster_path)

        reg = MockRegistrar()
        raster_viewer.setup(reg)
        result = reg.call("raster_viewer.inspect_geotiff", {"raster_path": raster_path})

    assert result["status"] == "success"
    assert result["filename"] == "sample.tif"
    assert result["driver"] == "GTiff"
    assert result["width"] == 3
    assert result["height"] == 2
    assert result["band_count"] == 2
    assert result["crs"] == "EPSG:4326"
    assert result["bounds"] == [100.0, 29.8, 100.3, 30.0]

    band1 = result["bands"][0]
    assert band1["index"] == 1
    assert band1["description"] == "temperature"
    assert band1["valid_pixels"] == 5
    assert band1["nodata_pixels"] == 1
    assert band1["min"] == 1.0
    assert band1["max"] == 6.0
    assert band1["mean"] == 3.6

    band2 = result["bands"][1]
    assert band2["description"] == "quality"
    assert band2["valid_pixels"] == 6
    assert band2["min"] == 10.0
    assert band2["max"] == 60.0


if __name__ == "__main__":
    test_inspect_geotiff_reports_metadata_and_band_stats()
    print("PASS: raster_viewer.inspect_geotiff")
