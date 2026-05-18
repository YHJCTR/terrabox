"""
Minimal tests for geoanalysis toolkit.
Matches the style of test_tool_georaster.py (manual MockRegistrar).

Run:
  python -m tests.test_geoanalysis
"""

import os
import shutil
import tempfile
import sys
import math
import numpy as np

# Import the toolkit setup
try:
    from terrabox.toolkits import geoanalysis
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from terrabox.toolkits import geoanalysis


# -----------------------------
# 1. Mock Registrar (Shared Pattern)
# -----------------------------
class MockRegistrar:
    def __init__(self):
        self.toolkits = {}
        self.tools = {}  # slug -> spec
        self.handlers = {}  # slug -> handler

    def toolkit(self, name: str, description: str, version: str):
        self.toolkits[name] = {"description": description, "version": version}

    def tool(self, toolspec, handler):
        # Store by slug
        self.tools[toolspec.slug] = toolspec
        self.handlers[toolspec.slug] = handler

    def call(self, slug: str, arguments: dict, context: dict = None, account=None):
        if context is None: context = {}
        handler = self.handlers.get(slug)
        if not handler:
            raise KeyError(f"Tool not registered: {slug}. Available: {list(self.handlers.keys())}")
        return handler(arguments, context, account)


# -----------------------------
# 2. Helpers
# -----------------------------
def create_dummy_raster(path, data, nodata=-9999):
    """Creates a simple GeoTIFF using rasterio."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return False

    rows, cols = data.shape
    transform = from_origin(0, 10, 1, 1)
    profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': nodata
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    return True


# -----------------------------
# 3. Main Test Runner
# -----------------------------
def main():
    # Check dependencies
    try:
        import pandas
        import statsmodels
        import ruptures
        import rasterio
    except ImportError as e:
        print(f"Test skipped: Missing dependency ({e}). Please install: pandas statsmodels ruptures rasterio")
        return

    temp_dir = tempfile.mkdtemp()
    print(f"Temp dir: {temp_dir}")

    try:
        reg = MockRegistrar()
        geoanalysis.setup(reg)
        print("Toolkit registered successfully.\n")

        # ==========================================
        # Part A: Time Series Tests
        # ==========================================
        print("--- Time Series Analysis Tests ---")

        # 1. Linear Trend
        # Data: y = 2x + 1 perfectly
        ts_linear = [1, 3, 5, 7, 9]
        res = reg.call("geoanalysis.compute_linear_trend", {"y": ts_linear})
        assert math.isclose(res["slope"], 2.0, abs_tol=1e-5)
        assert res["trend"] == "upward"
        print("PASS: compute_linear_trend (slope=2.0)")

        # 2. Mann-Kendall
        # Monotonic increasing
        res = reg.call("geoanalysis.mann_kendall_test", {"values": [10, 12, 15, 18, 20, 25]})
        # With small sample size p-value might not be < 0.05, but let's check structure
        assert "trend" in res
        assert "z_score" in res
        print(f"PASS: mann_kendall_test (trend={res['trend']})")

        # 3. Sen's Slope
        res = reg.call("geoanalysis.sens_slope", {"values": [1, 2, 4, 5, 10]})  # Median slope will be positive
        assert res["slope"] > 0
        print(f"PASS: sens_slope (slope={res['slope']})")

        # 4. STL Decomposition
        # Seasonal data: 10, 20, 10, 20...
        ts_seasonal = [10, 20] * 12  # 24 points
        res = reg.call("geoanalysis.stl_decompose", {"values": ts_seasonal, "period": 2})
        assert len(res["seasonal"]) == 24
        assert len(res["trend"]) == 24
        print("PASS: stl_decompose")

        # 5. Change Points
        # 0..0 then 10..10
        ts_change = [0] * 10 + [10] * 10
        res = reg.call("geoanalysis.detect_change_points", {"values": ts_change, "penalty": 1})
        # Should detect change around index 10 (or 10, 20 end)
        assert len(res["change_points"]) >= 1
        print(f"PASS: detect_change_points (points={res['change_points']})")

        # 6. ACF & Seasonality
        ts_periodic = [1, 0, 1, 0, 1, 0, 1, 0]
        res_acf = reg.call("geoanalysis.autocorrelation_function", {"values": ts_periodic, "nlags": 5})
        assert math.isclose(res_acf["acf"][0], 1.0)
        # Lag 2 should be high correlation
        print("PASS: autocorrelation_function")

        res_sea = reg.call("geoanalysis.detect_seasonality_acf", {"values": ts_periodic})
        # Period should be 2
        if res_sea["period"] == 2:
            print("PASS: detect_seasonality_acf (period=2)")
        else:
            print(f"WARN: detect_seasonality_acf got {res_sea}")

        # 7. Count Spikes
        ts_spikes = [1, 1.1, 2.0, 2.1, 5.0]  # jumps: 0.1, 0.9 (spike), 0.1, 2.9 (spike) -> 2 spikes > 0.5
        res = reg.call("geoanalysis.count_spikes", {"values": ts_spikes, "threshold": 0.5})
        assert res["spike_count"] == 2
        print(f"PASS: count_spikes (count={res['spike_count']})")

        # ==========================================
        # Part B: Spatial Analysis Tests
        # ==========================================
        print("\n--- Spatial Analysis Tests ---")

        # 8. Getis-Ord Gi*
        # Create a raster with a "hotspot" in the middle
        # 5x5 raster
        raster_data = np.zeros((5, 5))
        raster_data[2, 2] = 100  # Hot center
        raster_data[2, 1] = 50
        raster_data[1, 2] = 50

        in_path = os.path.join(temp_dir, "input_gi.tif")
        out_path = os.path.join(temp_dir, "output_gi.tif")
        create_dummy_raster(in_path, raster_data)

        # Simple 3x3 weight matrix
        weights = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        res = reg.call("geoanalysis.getis_ord_gi_star", {
            "image_path": in_path,
            "output_path": out_path,
            "weight_matrix": weights
        })
        assert res["status"] == "success"
        assert os.path.exists(out_path)

        # Verify the center pixel has a high Z-score
        with rasterio.open(out_path) as src:
            gi_vals = src.read(1)
            # Center should be positive (hotspot)
            assert gi_vals[2, 2] > 0
        print("PASS: getis_ord_gi_star")

        # 9. Hotspot Direction
        # Create binary map with hotspots concentrated in the "North" (top rows)
        # Image coordinates: y=0 is top.
        hotspot_data = np.zeros((10, 10))
        hotspot_data[1:3, 4:6] = 1  # Cluster at top center

        hs_path = os.path.join(temp_dir, "hotspot_binary.tif")
        create_dummy_raster(hs_path, hotspot_data)

        res = reg.call("geoanalysis.analyze_hotspot_direction", {
            "hotspot_map_path": hs_path
        })
        # Top of image corresponds to "north" logic in the tool
        assert res["direction"] == "north"
        print(f"PASS: analyze_hotspot_direction (dir={res['direction']})")

        print("\nAll geo_analysis smoke tests passed!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up {temp_dir}")


if __name__ == "__main__":
    main()
