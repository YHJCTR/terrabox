"""
Comprehensive tests for earth_sci toolkit.
Covers all 15 registered tools including LST, PWV, ATI, Microwave, and Water analysis.

Run:
  python -m tests.test_tool_earth_sci
"""

import os
import shutil
import tempfile
import sys
import numpy as np

# Import the toolkit setup
try:
    from terrabox.toolkits import earth_sci
except ImportError:
    # Allow running from root directory without install
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    from terrabox.toolkits import earth_sci

# -----------------------------
# 1. Mock Registrar
# -----------------------------
class MockRegistrar:
    def __init__(self):
        self.handlers = {}

    def toolkit(self, *args, **kwargs):
        pass

    def tool(self, spec, handler):
        # Register handler by slug
        self.handlers[spec.slug] = handler

    def call(self, slug, arguments):
        if slug not in self.handlers:
            raise KeyError(f"Tool not found: {slug}. Available: {list(self.handlers.keys())}")
        return self.handlers[slug](arguments, {}, {})

# -----------------------------
# 2. Helpers
# -----------------------------
def create_dummy_tif(path, val=100.0, size=(10, 10), bands=1):
    """Creates a dummy GeoTIFF with specific value."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)
    transform = from_origin(0, 10, 1, 1) # Top-left at (0,10), pixel size 1.0

    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=size[0], width=size[1],
        count=bands, dtype='float32',
        crs='EPSG:4326', transform=transform
    ) as dst:
        for b in range(1, bands + 1):
            # Write value (can be broadcasted)
            dst.write(np.full(size, val, dtype=np.float32), b)
    return True

# -----------------------------
# 3. Main Test Suite
# -----------------------------
def main():
    try:
        import rasterio
    except ImportError:
        print("Skipping tests: rasterio not installed")
        return

    temp_dir = tempfile.mkdtemp()
    print(f"Temp dir: {temp_dir}")

    try:
        reg = MockRegistrar()
        earth_sci.setup(reg)
        print("Toolkit registered successfully.\n")

        # ==========================================
        # 1. Atmospheric Parameters (PWV)
        # ==========================================
        print("[Test 1] PWV (Band Ratio)...")
        # Needs b02, b05, b17, b18, b19
        pwv_inputs = {}
        for b in ["b02", "b05", "b17", "b18", "b19"]:
            p = os.path.join(temp_dir, f"modis_{b}.tif")
            create_dummy_tif(p, 0.5) # Reflectance 0.5
            pwv_inputs[b] = p

        out_pwv = os.path.join(temp_dir, "pwv_out.tif")
        pwv_inputs["output_path"] = out_pwv

        reg.call("earth_sci.calculate_pwv", pwv_inputs)
        assert os.path.exists(out_pwv)
        print("PASS")

        # ==========================================
        # 2. Land Surface Temperature (LST)
        # ==========================================

        # 2.1 Single Channel
        print("[Test 2.1] LST Single Channel...")
        p_bt = os.path.join(temp_dir, "lst_bt.tif"); create_dummy_tif(p_bt, 300.0) # 300K
        p_red = os.path.join(temp_dir, "lst_red.tif"); create_dummy_tif(p_red, 0.1)
        p_nir = os.path.join(temp_dir, "lst_nir.tif"); create_dummy_tif(p_nir, 0.6) # NDVI ~ 0.71
        out_sc = os.path.join(temp_dir, "lst_sc.tif")

        reg.call("earth_sci.calculate_lst_sc", {
            "bt_path": p_bt, "red_path": p_red, "nir_path": p_nir,
            "output_path": out_sc
        })
        assert os.path.exists(out_sc)
        print("PASS")

        # 2.2 Multi Channel
        print("[Test 2.2] LST Multi Channel...")
        p_b31 = os.path.join(temp_dir, "b31.tif"); create_dummy_tif(p_b31, 300.0)
        p_b32 = os.path.join(temp_dir, "b32.tif"); create_dummy_tif(p_b32, 298.0)
        out_mc = os.path.join(temp_dir, "lst_mc.tif")

        reg.call("earth_sci.calculate_lst_mc", {
            "band31_path": p_b31, "band32_path": p_b32,
            "output_path": out_mc
        })
        assert os.path.exists(out_mc)
        print("PASS")

        # 2.3 Split Window
        print("[Test 2.3] Split Window (LST)...")
        p_e31 = os.path.join(temp_dir, "e31.tif"); create_dummy_tif(p_e31, 0.98)
        p_e32 = os.path.join(temp_dir, "e32.tif"); create_dummy_tif(p_e32, 0.97)
        out_sw = os.path.join(temp_dir, "lst_sw.tif")

        reg.call("earth_sci.calculate_split_window", {
            "band31_path": p_b31, "band32_path": p_b32,
            "emissivity31_path": p_e31, "emissivity32_path": p_e32,
            "parameter": "LST", "output_path": out_sw
        })
        assert os.path.exists(out_sw)
        print("PASS")

        # 2.4 TES
        print("[Test 2.4] TES (Temperature Emissivity Separation)...")
        tes_paths = []
        for i in range(5):
            p = os.path.join(temp_dir, f"tes_b{i}.tif")
            create_dummy_tif(p, 300.0 + i) # Slightly different values
            tes_paths.append(p)
        out_tes = os.path.join(temp_dir, "lst_tes.tif")

        reg.call("earth_sci.calculate_tes", {
            "tir_band_paths": tes_paths, "output_path": out_tes
        })
        assert os.path.exists(out_tes)
        print("PASS")

        # 2.5 Day/Night
        print("[Test 2.5] LST Day/Night...")
        p_day_bt = os.path.join(temp_dir, "dn_bt_day.tif"); create_dummy_tif(p_day_bt, 310.0)
        p_night_bt = os.path.join(temp_dir, "dn_bt_night.tif"); create_dummy_tif(p_night_bt, 290.0)
        p_day_e = os.path.join(temp_dir, "dn_e_day.tif"); create_dummy_tif(p_day_e, 200) # Raw value to be scaled
        p_night_e = os.path.join(temp_dir, "dn_e_night.tif"); create_dummy_tif(p_night_e, 200)
        out_dn = os.path.join(temp_dir, "lst_dn.tif")

        reg.call("earth_sci.calculate_modis_day_night", {
            "bt_day": p_day_bt, "bt_night": p_night_bt,
            "emis_day": p_day_e, "emis_night": p_night_e,
            "output_path": out_dn
        })
        assert os.path.exists(out_dn)
        print("PASS")

        # 2.6 TTM (Three-Temperature Method)
        print("[Test 2.6] TTM LST...")
        ttm_paths = tes_paths[:3] # Reuse 3 bands
        out_ttm = os.path.join(temp_dir, "lst_ttm.tif")
        reg.call("earth_sci.calculate_ttm", {
            "tir_band_paths": ttm_paths, "output_path": out_ttm
        })
        assert os.path.exists(out_ttm)
        print("PASS")

        # ==========================================
        # 3. LST Statistics
        # ==========================================
        print("[Test 3] LST Stats by NDVI...")
        res = reg.call("earth_sci.stats_lst_ndvi", {
            "red_path": p_red, "nir_path": p_nir, "lst_path": p_bt, # Reuse BT as LST
            "threshold": 0.5, "mode": "above", "stat": "mean"
        })
        # NIR=0.6, Red=0.1 -> NDVI=0.71 > 0.5 -> Should include all pixels (300K)
        assert res["result"] is not None
        assert np.isclose(res["result"], 300.0)
        print("PASS")

        # ==========================================
        # 4. Thermal Inertia (ATI)
        # ==========================================
        print("[Test 4] ATI...")
        p_alb = os.path.join(temp_dir, "albedo.tif"); create_dummy_tif(p_alb, 0.2)
        out_ati = os.path.join(temp_dir, "ati.tif")

        reg.call("earth_sci.calculate_ati", {
            "day_temp_path": p_day_bt, # 310
            "night_temp_path": p_night_bt, # 290
            "albedo_path": p_alb, # 0.2
            "output_path": out_ati
        })
        # ATI = (1 - 0.2) / (310 - 290) = 0.8 / 20 = 0.04
        with rasterio.open(out_ati) as src:
            val = src.read(1)[0,0]
            assert np.isclose(val, 0.04)
        print("PASS")

        # ==========================================
        # 5. Microwave Inversion
        # ==========================================
        print("[Test 5.1] Microwave DPDM (Soil Moisture)...")
        p_pol1 = os.path.join(temp_dir, "pol1.tif"); create_dummy_tif(p_pol1, -10.0) # dB
        p_pol2 = os.path.join(temp_dir, "pol2.tif"); create_dummy_tif(p_pol2, -15.0) # dB
        out_dpdm = os.path.join(temp_dir, "mw_dpdm.tif")

        reg.call("earth_sci.microwave_dpdm", {
            "pol1_path": p_pol1, "pol2_path": p_pol2,
            "parameter": "soil_moisture", "output_path": out_dpdm
        })
        assert os.path.exists(out_dpdm)
        print("PASS")

        print("[Test 5.2] Microwave DDM...")
        p_b1 = os.path.join(temp_dir, "mw_b1.tif"); create_dummy_tif(p_b1, 0.5)
        p_b2 = os.path.join(temp_dir, "mw_b2.tif"); create_dummy_tif(p_b2, 0.3)
        out_ddm = os.path.join(temp_dir, "mw_ddm.tif")

        reg.call("earth_sci.microwave_ddm", {
            "band1_path": p_b1, "band2_path": p_b2, "output_path": out_ddm
        })
        assert os.path.exists(out_ddm)
        print("PASS")

        print("[Test 5.3] Microwave Multi-Freq...")
        mw_paths = [p_b1, p_b2, p_b1] # Reuse files
        out_mf = os.path.join(temp_dir, "mw_mf.tif")

        reg.call("earth_sci.microwave_multi_freq", {
            "bt_paths": mw_paths,
            "diff_pairs": [[0,1], [1,2]],
            "output_path": out_mf
        })
        assert os.path.exists(out_mf)
        print("PASS")

        print("[Test 5.4] Microwave PRM...")
        out_prm = os.path.join(temp_dir, "mw_prm.tif")
        reg.call("earth_sci.microwave_prm", {
            "bt_paths": {"V": p_pol1, "H": p_pol2}, # Can reuse any raster path
            "output_path": out_prm
        })
        assert os.path.exists(out_prm)
        print("PASS")

        # ==========================================
        # 6. Cryosphere & Water
        # ==========================================
        print("[Test 6.1] Sea Ice Concentration...")
        p_19v = os.path.join(temp_dir, "19v.tif"); create_dummy_tif(p_19v, 250)
        p_19h = os.path.join(temp_dir, "19h.tif"); create_dummy_tif(p_19h, 240)
        p_37v = os.path.join(temp_dir, "37v.tif"); create_dummy_tif(p_37v, 260)
        p_37h = os.path.join(temp_dir, "37h.tif"); create_dummy_tif(p_37h, 255)
        out_ice = os.path.join(temp_dir, "ice.tif")

        reg.call("earth_sci.calculate_sea_ice", {
            "bt_paths": {"19V":p_19v, "19H":p_19h, "37V":p_37v, "37H":p_37h},
            "output_path": out_ice
        })
        assert os.path.exists(out_ice)
        print("PASS")

        print("[Test 6.2] Water Turbidity...")
        p_turb = os.path.join(temp_dir, "turb_red.tif"); create_dummy_tif(p_turb, 0.2)
        out_turb = os.path.join(temp_dir, "turbidity.tif")

        reg.call("earth_sci.calculate_turbidity", {
            "input_red_path": p_turb,
            "output_path": out_turb,
            "method": "linear", "a": 100, "b": 5
        })
        # 100 * 0.2 + 5 = 25
        with rasterio.open(out_turb) as src:
            val = src.read(1)[0,0]
            assert np.isclose(val, 25.0)
        print("PASS")

        print("\nAll earth_sci smoke tests passed!")

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
