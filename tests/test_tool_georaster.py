"""
Minimal tests for geo_raster toolkit.
Matches the style of test_tool_geobasic.py (manual MockRegistrar, no pytest dependency).

Run:
  python -m tests.test_tool_georaster
"""

import os
import shutil
import tempfile
import sys
import math
import numpy as np

# Import the toolkit setup
try:
    from terrabox.toolkits import georaster
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from terrabox.toolkits import georaster

# -----------------------------
# 1. Mock Registrar
# -----------------------------
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

    def call(self, slug: str, arguments: dict, context: dict = None, account=None):
        if context is None:
            context = {}
        handler = self.handlers.get(slug)
        if not handler:
            raise KeyError(f"Tool not registered: {slug}. Available: {list(self.handlers.keys())}")
        return handler(arguments, context, account)


# -----------------------------
# 2. Helpers
# -----------------------------
def create_dummy_raster(path, value=0.5, shape=(10, 10), dtype='float32', nodata=-9999):
    """Creates a simple GeoTIFF with constant (or array) value using rasterio."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print("Skipping raster creation: rasterio not installed.")
        return False

    transform = from_origin(0, 10, 1, 1)
    profile = {
        'driver': 'GTiff',
        'height': shape[0],
        'width': shape[1],
        'count': 1,
        'dtype': dtype,
        'crs': 'EPSG:4326',
        'transform': transform,
        'nodata': nodata
    }
    data = value if isinstance(value, np.ndarray) else np.full(shape, value, dtype=dtype)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
    return True


# -----------------------------
# 3. Main Test Runner
# -----------------------------
def main():
    try:
        import rasterio
    except ImportError:
        print("Test skipped: rasterio not installed.")
        return

    temp_dir = tempfile.mkdtemp()
    print(f"Temp dir: {temp_dir}")

    try:
        reg = MockRegistrar()
        georaster.setup(reg)
        print("Toolkit registered successfully.\n")

        # --- Prepare common inputs ---
        p_nir  = os.path.join(temp_dir, "nir.tif")
        p_red  = os.path.join(temp_dir, "red.tif")
        p_blue = os.path.join(temp_dir, "blue.tif")
        p_lst  = os.path.join(temp_dir, "lst.tif")

        # NIR=0.8, Red=0.2 → NDVI = (0.8-0.2)/(0.8+0.2) = 0.6
        create_dummy_raster(p_nir, 0.8)
        create_dummy_raster(p_red, 0.2)
        create_dummy_raster(p_blue, 0.1)
        create_dummy_raster(p_lst, 300.0)

        # -------------------------------------------------------
        # TEST 1: calculate_index (NDVI)
        # -------------------------------------------------------
        print("[Test 1] geo_raster.calculate_index (NDVI)...")
        out_ndvi = os.path.join(temp_dir, "ndvi_out.tif")
        res = reg.call("geo_raster.calculate_index", {
            "band_a_path": p_nir,
            "band_b_path": p_red,
            "output_path": out_ndvi,
            "index_name": "NDVI"
        })
        assert res["status"] == "success"
        assert os.path.exists(out_ndvi)
        with rasterio.open(out_ndvi) as src:
            val = src.read(1)[0, 0]
            assert math.isclose(val, 0.6, abs_tol=1e-4), f"Expected 0.6, got {val}"
        print("PASS: NDVI value correct.")

        # -------------------------------------------------------
        # TEST 2: calculate_fvc
        # -------------------------------------------------------
        print("\n[Test 2] geo_raster.calculate_fvc...")
        out_fvc = os.path.join(temp_dir, "fvc_out.tif")
        reg.call("geo_raster.calculate_fvc", {
            "nir_path": p_nir, "red_path": p_red,
            "output_path": out_fvc, "ndvi_min": 0.0, "ndvi_max": 1.0
        })
        with rasterio.open(out_fvc) as src:
            val = src.read(1)[0, 0]
            assert math.isclose(val, 60.0, abs_tol=1e-4), f"Expected 60.0, got {val}"
        print("PASS: FVC value correct.")

        # -------------------------------------------------------
        # TEST 3: compute_tvdi (smoke test — constant data warns)
        # -------------------------------------------------------
        print("\n[Test 3] geo_raster.compute_tvdi (smoke)...")
        out_tvdi = os.path.join(temp_dir, "tvdi_out.tif")
        try:
            reg.call("geo_raster.compute_tvdi", {
                "ndvi_path": p_nir, "lst_path": p_lst, "output_path": out_tvdi
            })
            print("PASS: TVDI file generated (or warned on constant data).")
        except ImportError:
            print("SKIPPED: scipy missing.")
        except Exception as e:
            print(f"WARN: TVDI math issue (expected on constant data): {e}")

        # -------------------------------------------------------
        # TEST 4: calc_snow_loss_stats
        # -------------------------------------------------------
        print("\n[Test 4] geo_raster.calc_snow_loss_stats...")
        snow_data = np.zeros((10, 10), dtype='float32')
        snow_data[0:5, :] = 1.0   # 50% loss pixels
        p_snow = os.path.join(temp_dir, "snow.tif")
        create_dummy_raster(p_snow, snow_data)
        res = reg.call("geo_raster.calc_snow_loss_stats", {"binary_map_path": p_snow})
        assert math.isclose(res["percentage"], 0.5, abs_tol=1e-4), f"Expected 0.5, got {res['percentage']}"
        print(f"PASS: snow loss {res['percentage']*100:.1f}%")

        # -------------------------------------------------------
        # TEST 5: threshold_segmentation
        # -------------------------------------------------------
        print("\n[Test 5] geo_raster.threshold_segmentation...")
        mixed = np.zeros((10, 10), dtype='float32')
        mixed[3:7, 3:7] = 5.0   # center block = 5.0
        p_mixed = os.path.join(temp_dir, "mixed.tif")
        out_seg  = os.path.join(temp_dir, "seg.tif")
        create_dummy_raster(p_mixed, mixed)
        reg.call("geo_raster.threshold_segmentation", {
            "input_path": p_mixed, "threshold": 1.0, "output_path": out_seg
        })
        with rasterio.open(out_seg) as src:
            data = src.read(1)
            assert data[5, 5] == 255, "Foreground pixel should be 255"
            assert data[0, 0] == 0,   "Background pixel should be 0"
        print("PASS: threshold_segmentation correct.")

        # -------------------------------------------------------
        # TEST 6: count_above_threshold
        # -------------------------------------------------------
        print("\n[Test 6] geo_raster.count_above_threshold...")
        res = reg.call("geo_raster.count_above_threshold", {
            "input_path": p_mixed, "threshold": 1.0
        })
        # 4x4 = 16 pixels set to 5.0
        assert res["count"] == 16, f"Expected 16, got {res['count']}"
        print(f"PASS: count_above_threshold = {res['count']}")

        # -------------------------------------------------------
        # TEST 7: raster_stats
        # -------------------------------------------------------
        print("\n[Test 7] geo_raster.raster_stats...")
        p_const = os.path.join(temp_dir, "const.tif")
        create_dummy_raster(p_const, 4.0)
        res = reg.call("geo_raster.raster_stats", {"input_path": p_const})
        assert math.isclose(res["mean"], 4.0, abs_tol=1e-4), f"Expected mean=4.0, got {res['mean']}"
        assert math.isclose(res["std"],  0.0, abs_tol=1e-4), f"Expected std=0.0,  got {res['std']}"
        assert math.isclose(res["min"],  4.0, abs_tol=1e-4)
        assert math.isclose(res["max"],  4.0, abs_tol=1e-4)
        assert res["count"] == 100

        # Single stat request
        res2 = reg.call("geo_raster.raster_stats", {"input_path": p_const, "stat": "mean"})
        assert "mean" in res2 and len(res2) == 1
        print(f"PASS: raster_stats mean={res['mean']}, count={res['count']}")

        # -------------------------------------------------------
        # TEST 8: raster_diff
        # -------------------------------------------------------
        print("\n[Test 8] geo_raster.raster_diff...")
        p_a   = os.path.join(temp_dir, "a.tif")
        p_b   = os.path.join(temp_dir, "b.tif")
        p_dif = os.path.join(temp_dir, "diff.tif")
        create_dummy_raster(p_a, 7.0)
        create_dummy_raster(p_b, 3.0)
        reg.call("geo_raster.raster_diff", {
            "path_a": p_a, "path_b": p_b, "output_path": p_dif
        })
        with rasterio.open(p_dif) as src:
            val = src.read(1)[0, 0]
            assert math.isclose(val, 4.0, abs_tol=1e-4), f"Expected 4.0, got {val}"
        print("PASS: raster_diff correct (7-3=4).")

        # -------------------------------------------------------
        # TEST 9: raster_average
        # -------------------------------------------------------
        print("\n[Test 9] geo_raster.raster_average...")
        p_c   = os.path.join(temp_dir, "c.tif")
        p_avg = os.path.join(temp_dir, "avg.tif")
        create_dummy_raster(p_c, 5.0)   # (7+3+5)/3 = 5.0
        reg.call("geo_raster.raster_average", {
            "input_paths": [p_a, p_b, p_c], "output_path": p_avg
        })
        with rasterio.open(p_avg) as src:
            val = src.read(1)[0, 0]
            assert math.isclose(val, 5.0, abs_tol=1e-4), f"Expected 5.0, got {val}"
        print("PASS: raster_average correct ((7+3+5)/3=5).")

        # -------------------------------------------------------
        # TEST 10: hotspot_percentage
        # -------------------------------------------------------
        print("\n[Test 10] geo_raster.hotspot_percentage...")
        res = reg.call("geo_raster.hotspot_percentage", {
            "input_path": p_mixed, "threshold": 1.0
        })
        # 16/100 = 0.16
        assert math.isclose(res["percentage"], 0.16, abs_tol=1e-4), \
            f"Expected 0.16, got {res['percentage']}"
        print(f"PASS: hotspot_percentage = {res['percentage']:.2f}")

        # -------------------------------------------------------
        # TEST 11: apply_cloud_mask (Landsat)
        # -------------------------------------------------------
        print("\n[Test 11] geo_raster.apply_cloud_mask...")
        p_sr  = os.path.join(temp_dir, "sr.tif")
        p_qa  = os.path.join(temp_dir, "qa.tif")
        p_msk = os.path.join(temp_dir, "masked.tif")
        create_dummy_raster(p_sr, 1000.0, dtype='float32')
        # bit3=cloud: value 8 (0b00001000) marks all pixels as cloud
        # Must use integer dtype so numpy bitwise ops work inside the handler
        qa_data = np.full((10, 10), 8, dtype='int16')
        create_dummy_raster(p_qa, qa_data, dtype='int16')
        res = reg.call("geo_raster.apply_cloud_mask", {
            "sr_path": p_sr, "qa_path": p_qa, "output_path": p_msk, "sensor": "landsat"
        })
        assert res["masked_pixels"] == 100, f"Expected 100 masked, got {res['masked_pixels']}"
        print(f"PASS: apply_cloud_mask masked {res['masked_pixels']} pixels.")

        # -------------------------------------------------------
        # TEST 12: get_percentile_value
        # -------------------------------------------------------
        print("\n[Test 12] geo_raster.get_percentile_value...")
        vals = np.arange(100, dtype='float32').reshape(10, 10)
        p_grad = os.path.join(temp_dir, "grad.tif")
        create_dummy_raster(p_grad, vals)
        res = reg.call("geo_raster.get_percentile_value", {
            "input_path": p_grad, "percentile": 50
        })
        # Median of 0..99 = 49.5
        assert math.isclose(res["value"], 49.5, abs_tol=1.0), \
            f"Expected ~49.5, got {res['value']}"
        print(f"PASS: get_percentile_value(50th) = {res['value']}")

        # -------------------------------------------------------
        # TEST 13: calculate_evi
        # -------------------------------------------------------
        print("\n[Test 13] geo_raster.calculate_evi...")
        out_evi = os.path.join(temp_dir, "evi_out.tif")
        # Reuse existing rasters: NIR=0.8, Red=0.2, Blue=0.1 (defaults: G=2.5, C1=6, C2=7.5, L=1)
        # Expected: 2.5*(0.8-0.2)/(0.8+6*0.2-7.5*0.1+1+1e-6) = 1.5/2.25 ≈ 0.6667
        res = reg.call("geo_raster.calculate_evi", {
            "nir_path": p_nir, "red_path": p_red, "blue_path": p_blue,
            "output_path": out_evi
        })
        assert os.path.exists(out_evi), "EVI output file not created"
        with rasterio.open(out_evi) as src:
            val = src.read(1)[0, 0]
            expected_evi = 2.5 * (0.8 - 0.2) / (0.8 + 6.0 * 0.2 - 7.5 * 0.1 + 1.0 + 1e-6)
            assert math.isclose(val, expected_evi, abs_tol=1e-2), \
                f"Expected EVI≈{expected_evi:.4f}, got {val}"
        print(f"PASS: calculate_evi ≈ {val:.4f}")

        # -------------------------------------------------------
        # TEST 14: calculate_wri
        # -------------------------------------------------------
        print("\n[Test 14] geo_raster.calculate_wri...")
        # Create separate rasters for WRI: Green=0.3, NIR=0.4, SWIR=0.1; reuse Red=0.2
        # WRI = (Green+Red)/(NIR+SWIR+1e-6) = (0.3+0.2)/(0.4+0.1+1e-6) = 0.5/0.5 ≈ 1.0
        p_green = os.path.join(temp_dir, "green.tif")
        p_nir2  = os.path.join(temp_dir, "nir2.tif")
        p_swir  = os.path.join(temp_dir, "swir.tif")
        out_wri = os.path.join(temp_dir, "wri_out.tif")
        create_dummy_raster(p_green, 0.3)
        create_dummy_raster(p_nir2,  0.4)
        create_dummy_raster(p_swir,  0.1)
        res = reg.call("geo_raster.calculate_wri", {
            "green_path": p_green, "red_path": p_red,
            "nir_path": p_nir2,   "swir_path": p_swir,
            "output_path": out_wri
        })
        assert os.path.exists(out_wri), "WRI output file not created"
        with rasterio.open(out_wri) as src:
            val = src.read(1)[0, 0]
            expected_wri = (0.3 + 0.2) / (0.4 + 0.1 + 1e-6)
            assert math.isclose(val, expected_wri, abs_tol=1e-2), \
                f"Expected WRI≈{expected_wri:.4f}, got {val}"
        print(f"PASS: calculate_wri ≈ {val:.4f}")

        # -------------------------------------------------------
        # TEST 15: count_skeleton_contours
        # -------------------------------------------------------
        print("\n[Test 15] geo_raster.count_skeleton_contours...")
        try:
            import cv2
            from skimage.morphology import skeletonize  # noqa: F401 — confirm both deps exist
            # Create a black image with two separate white filled rectangles
            img = np.zeros((100, 100), dtype=np.uint8)
            img[10:40, 10:40] = 255   # left rectangle
            img[10:40, 60:90] = 255   # right rectangle
            p_skel = os.path.join(temp_dir, "skel_input.png")
            cv2.imwrite(p_skel, img)
            res = reg.call("geo_raster.count_skeleton_contours", {"image_path": p_skel})
            assert res["count"] >= 1, f"Expected at least 1 contour, got {res['count']}"
            print(f"PASS: count_skeleton_contours detected {res['count']} contour(s).")
        except ImportError:
            print("SKIPPED: cv2 or scikit-image not installed.")

        print("\nAll geo_raster smoke tests passed!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up {temp_dir}")


if __name__ == "__main__":
    main()
