"""
Minimal tests for geo_basic toolkit.

This file:
1) Mocks a Registrar to capture tools and call handlers (aligned with your example).
2) Registers the toolkit via setup(registrar).
3) Runs a few smoke tests for each tool to validate basic behavior.

Run:
  python -m tests.test_tool_geobasic
"""

import json
import math

from conftest import MockRegistrar

# Import the toolkit setup and handlers indirectly by calling setup(registrar)
from terrabox.toolkits.geobasic import setup as geo_basic_setup


# -----------------------------
# Helpers / sample inputs
# -----------------------------
SQUARE_POLY = {
    "type": "Polygon",
    "coordinates": [
        [
            [11.54, 48.14],
            [11.60, 48.14],
            [11.60, 48.18],
            [11.54, 48.18],
            [11.54, 48.14]
        ]
    ]
}

LINE = {
    "type": "LineString",
    "coordinates": [
        [11.54, 48.14],
        [11.60, 48.18],
        [11.62, 48.20]
    ]
}


# -----------------------------
# Smoke tests
# -----------------------------
def main():
    reg = MockRegistrar()
    geo_basic_setup(reg)

    # 1) AOI validate
    out = reg.call("geo_basic.aoi_validate", {"geojson": SQUARE_POLY}, context={"user_id": "u1"})
    assert out["geometry_type"] == "Polygon"
    assert "bbox" in out
    assert out["total_area_m2"] > 0
    print("aoi_validate OK:", out["bbox"], out["total_area_km2"])

    # 2) AOI area with GSD=10 m
    ar = reg.call("geo_basic.area", {"geojson": SQUARE_POLY, "gsd_m": 10})
    assert ar["area_m2"] > 0 and ar["pixel_estimate"] > 0
    print("area OK:", ar["area_km2"], "px_est:", int(ar["pixel_estimate"]))

    # 3) Distance
    dist = reg.call("geo_basic.distance", {
        "lon1": 11.54, "lat1": 48.14,
        "lon2": 11.60, "lat2": 48.18
    })
    assert dist["distance_m"] > 0
    print("distance OK:", round(dist["distance_m"], 1), "m")

    # 4) Pixel -> Area (10 m)
    pa = reg.call("geo_basic.pixel_area", {"pixels": 12345, "gsd_m": 10})
    assert math.isclose(pa["area_m2"], 12345 * 100, rel_tol=1e-9)
    print("pixel_area OK:", pa["area_m2"], "m²")

    # 5) Area -> Pixels (ceil)
    pf = reg.call("geo_basic.pixels_from_area", {"area_m2": 2500000, "gsd_m": 10, "rounding": "ceil"})
    assert pf["pixels_int"] >= 2500000 / 100
    print("pixels_from_area OK:", pf["pixels_int"], "px")

    # 6) Gridify (size mode, 100 m, centroid clipping ON)
    grid = reg.call("geo_basic.gridify", {"geojson": SQUARE_POLY, "mode": "size", "cell_width_m": 100, "clip": True})
    assert "features" in grid and len(grid["features"]) > 0
    print("gridify(size) OK: cells:", len(grid["features"]))

    # 7) Line length
    ll = reg.call("geo_basic.line_length", {"geojson": LINE})
    assert ll["length_m"] > 0
    print("line_length OK:", round(ll["length_m"], 1), "m")

    # 8) BBox Expansion
    bboxes = [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]]
    res = reg.call("geo_basic.bbox_expansion", {"bboxes": bboxes, "radius": 5.0, "gsd": 0.5})
    exp = res["expanded_bboxes"]
    # Each side expands by 5/0.5 = 10 pixels
    assert exp[0] == [0.0, 0.0, 30.0, 30.0], f"Expected [0,0,30,30], got {exp[0]}"
    print("bbox_expansion OK:", exp[0])

    # 9) BBoxes to Centroids
    bboxes2 = [[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 15.0, 15.0]]
    res = reg.call("geo_basic.bboxes_to_centroids", {"bboxes": bboxes2})
    c = res["centroids"]
    assert c[0] == [5.0, 10.0], f"Expected [5.0, 10.0], got {c[0]}"
    assert c[1] == [10.0, 10.0], f"Expected [10.0, 10.0], got {c[1]}"
    print("bboxes_to_centroids OK:", c)

    # 10) Centroid Distance Extremes
    centroids = [[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]]
    res = reg.call("geo_basic.centroid_distance_extremes", {"centroids": centroids})
    # Distance(0,1)=5, Distance(0,2)=10, Distance(1,2)=sqrt(49+16)~8.06
    assert math.isclose(res["min"]["distance"], 5.0, abs_tol=1e-6), \
        f"Min distance should be 5.0, got {res['min']['distance']}"
    assert math.isclose(res["max"]["distance"], 10.0, abs_tol=1e-6), \
        f"Max distance should be 10.0, got {res['max']['distance']}"
    print("centroid_distance_extremes OK: min=", res["min"]["distance"],
          "max=", res["max"]["distance"])

    # 11) BBox Area
    bboxes3 = [[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 15.0, 10.0]]
    res = reg.call("geo_basic.bbox_area", {"bboxes": bboxes3})
    # 10*20 + 15*10 = 200 + 150 = 350
    assert math.isclose(res["total_area_px2"], 350.0, abs_tol=1e-6), \
        f"Expected 350 px², got {res['total_area_px2']}"
    # With GSD=0.5m: 350 * 0.25 = 87.5 m²
    res2 = reg.call("geo_basic.bbox_area", {"bboxes": bboxes3, "gsd": 0.5})
    assert math.isclose(res2["total_area_m2"], 87.5, abs_tol=1e-6), \
        f"Expected 87.5 m², got {res2['total_area_m2']}"
    print("bbox_area OK: px²=", res["total_area_px2"], "m²=", res2["total_area_m2"])

    print("\nAll geo_basic smoke tests passed.")


if __name__ == "__main__":
    main()
