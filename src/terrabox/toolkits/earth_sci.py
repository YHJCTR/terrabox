"""
Earth Science Toolkit
---------------------
A comprehensive toolkit for physical parameter inversion from Earth Observation data.
Ported from Earth-Agent (Inversion.py) and adapted for Terrabox.

Key features:
1. Atmosphere: Precipitable Water Vapor (PWV).
2. Land Surface Temperature (LST): Single-Channel, Multi-Channel, Split-Window, TES, TTM, Day/Night.
3. Thermal Inertia: Apparent Thermal Inertia (ATI).
4. Microwave Inversion: Soil Moisture, Vegetation Index, VWC (DPDM, DDM, PRM, Chang, Multi-Freq).
5. Cryosphere & Water: Sea Ice Concentration, Water Turbidity.
"""

import os
from typing import Any, Dict, List, Optional, Union
from ..core.registry import ToolSpec

# ------------------------------------------------------------------------------
# Lazy Imports & Helpers
# ------------------------------------------------------------------------------

def _lazy_imports():
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        import numpy as np
        return rasterio, np
    except ImportError:
        raise ImportError("Missing dependencies. Please install: pip install rasterio numpy")

def _resample_to_reference(src_path: str, ref_path: str) -> "np.ndarray":
    """Helper to resample source raster to match reference raster."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    import numpy as np

    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)

    with rasterio.open(src_path) as src:
        source_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

        destination_data = np.zeros(dst_shape, dtype=np.float32)

        reproject(
            source=source_data,
            destination=destination_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        return destination_data

def _save_raster(output_path: str, data: "np.ndarray", profile: dict, count: int = 1):
    """Helper to save raster data."""
    import rasterio
    import os
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    profile.update(dtype=rasterio.float32, count=count, compress='lzw', nodata=np.nan)

    with rasterio.open(output_path, 'w', **profile) as dst:
        if count == 1:
            dst.write(data.astype(rasterio.float32), 1)
        else:
            for i in range(count):
                dst.write(data[i].astype(rasterio.float32), i + 1)

# ------------------------------------------------------------------------------
# 1. Atmospheric Parameters (PWV)
# ------------------------------------------------------------------------------

def calculate_pwv_band_ratio_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute Precipitable Water Vapor (PWV) using MODIS band ratio method."""
    rasterio, np = _lazy_imports()

    paths = [arguments[k] for k in ["b02", "b05", "b17", "b18", "b19"]]
    output_path = arguments["output_path"]

    data = []
    profile = None
    for p in paths:
        with rasterio.open(p) as src:
            data.append(src.read(1).astype(np.float32))
            if profile is None: profile = src.profile

    b02, b05, b17, b18, b19 = data

    # Wavelengths (um)
    l2, l5 = 0.865, 1.240
    l17, l18, l19 = 0.905, 0.936, 0.940

    # Interpolation
    a = (b05 - b02) / (l5 - l2)
    b = b02 - a * l2

    rho17 = a * l17 + b
    rho18 = a * l18 + b
    rho19 = a * l19 + b

    # Transmittance
    with np.errstate(divide='ignore', invalid='ignore'):
        T17 = np.where(rho17 != 0, b17 / rho17, 0)
        T18 = np.where(rho18 != 0, b18 / rho18, 0)
        T19 = np.where(rho19 != 0, b19 / rho19, 0)

        # PWV (using T18)
        k = 0.03
        PWV = -np.log(T18) / k
        PWV = np.nan_to_num(PWV, nan=0.0, posinf=0.0, neginf=0.0)
        PWV[PWV < 0] = 0

    out_stack = np.stack([PWV, T17, T18, T19])
    _save_raster(output_path, out_stack, profile, count=4)

    return {"output_path": output_path, "bands": ["PWV", "T17", "T18", "T19"]}

# ------------------------------------------------------------------------------
# 2. Land Surface Temperature (LST) Algorithms
# ------------------------------------------------------------------------------

def calculate_lst_single_channel_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate LST using Single-Channel method with NDVI-based emissivity."""
    rasterio, np = _lazy_imports()

    bt_path = arguments["bt_path"]
    red_path = arguments["red_path"]
    nir_path = arguments["nir_path"]
    output_path = arguments["output_path"]

    with rasterio.open(bt_path) as src:
        bt = src.read(1).astype(np.float32)
        profile = src.profile

    # Read/Align Red/NIR
    red = _resample_to_reference(red_path, bt_path)
    nir = _resample_to_reference(nir_path, bt_path)

    # NDVI & Emissivity
    denom = nir + red
    ndvi = np.divide(nir - red, denom, out=np.zeros_like(nir), where=denom!=0)

    emissivity = np.where(ndvi > 0.7, 0.99,
                 np.where(ndvi < 0.2, 0.96,
                 0.97 + 0.003 * ndvi)) # 0.2-0.7 mixed

    # LST Calculation
    w = 10.9 # Landsat 8 B10 center
    c2 = 14387.7

    lst = bt / (1 + (w * bt / c2) * np.log(emissivity))

    _save_raster(output_path, lst, profile)
    return {"output_path": output_path}

def calculate_lst_multi_channel_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate LST using Multi-Channel (Split-Window) empirical linear algorithm."""
    rasterio, np = _lazy_imports()

    b31_path = arguments["band31_path"]
    b32_path = arguments["band32_path"]
    output_path = arguments["output_path"]

    with rasterio.open(b31_path) as src:
        b31 = src.read(1).astype(np.float32)
        profile = src.profile
    b32 = _resample_to_reference(b32_path, b31_path)

    # Empirical coeffs
    a, b_coef, c = 1.022, 0.47, 0.43
    lst = a * b31 + b_coef * (b31 - b32) + c

    _save_raster(output_path, lst, profile)
    return {"output_path": output_path}

def calculate_split_window_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate LST or PWV using physical Split-Window algorithm."""
    rasterio, np = _lazy_imports()

    p31, p32 = arguments["band31_path"], arguments["band32_path"]
    e31p, e32p = arguments["emissivity31_path"], arguments["emissivity32_path"]
    param = arguments.get("parameter", "LST").upper()
    output_path = arguments["output_path"]

    with rasterio.open(p31) as src:
        b31 = src.read(1).astype(np.float32)
        profile = src.profile

    b32 = _resample_to_reference(p32, p31)
    e31 = _resample_to_reference(e31p, p31) * 0.002 + 0.49 # Scaling from original code
    e32 = _resample_to_reference(e32p, p31) * 0.002 + 0.49

    delta_T = b31 - b32
    eps_mean = (e31 + e32) / 2
    delta_eps = e31 - e32
    eps_mean = np.clip(eps_mean, 0.8, 1.0)

    if param == "LST":
        C0, C1, C2, C3, C4 = 0.268, 1.378, 0.183, 54.3, -2.238
        t31_c = b31 - 273.15
        term4 = (C3 + C4 * delta_T) * (1 - eps_mean)
        term5 = (C3 + C4 * delta_T) * delta_eps
        lst = C0 + C1*t31_c + C2*(t31_c**2)/1000 + term4 + term5 + 273.15
        result = np.where((lst < 200) | (lst > 350), np.nan, lst)
    elif param == "PWV":
        result = delta_T / (b31 * eps_mean) * 100
    else:
        raise ValueError("Unknown parameter")

    _save_raster(output_path, result, profile)
    return {"output_path": output_path}

def calculate_tes_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate LST using Temperature Emissivity Separation (TES)."""
    rasterio, np = _lazy_imports()

    paths = arguments["tir_band_paths"]
    ref_idx = arguments.get("representative_band_index", 0)
    output_path = arguments["output_path"]

    # Read all
    data = []
    profile = None
    for p in paths:
        with rasterio.open(p) as src:
            d = src.read(1).astype(np.float32)
            if profile is None: profile = src.profile
            data.append(d)

    stack = np.stack(data) # (B, H, W)

    # Delta Epsilon
    b_max = np.nanmax(stack, axis=0)
    b_min = np.nanmin(stack, axis=0)
    delta_eps = b_max - b_min # Simplification from original code logic

    # Estimate Emissivity
    emissivity = 0.982 - 0.072 * delta_eps
    emissivity = np.clip(emissivity, 0.85, 0.999)

    # Calculate LST from Ref Band
    Tb = stack[ref_idx]
    w = 10.6
    c2 = 14387.7
    lst = Tb / (1 + (w * Tb / c2) * np.log(emissivity))

    out_stack = np.stack([lst, emissivity, delta_eps])
    _save_raster(output_path, out_stack, profile, count=3)
    return {"output_path": output_path}

def calculate_modis_day_night_lst_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate LST from Day/Night MODIS pairs."""
    rasterio, np = _lazy_imports()

    paths = [arguments[k] for k in ["bt_day", "bt_night", "emis_day", "emis_night"]]
    output_path = arguments["output_path"]

    with rasterio.open(paths[0]) as src:
        bt_day = src.read(1).astype(np.float32)
        profile = src.profile

    bt_night = _resample_to_reference(paths[1], paths[0])
    e_day = _resample_to_reference(paths[2], paths[0]) * 0.002 + 0.49
    e_night = _resample_to_reference(paths[3], paths[0]) * 0.002 + 0.49

    e_day = np.clip(e_day, 0.5, 1.0)
    e_night = np.clip(e_night, 0.5, 1.0)

    w, c2 = 11.0, 14387.7
    lst_day = bt_day / (1 + (w*bt_day/c2)*np.log(e_day))
    lst_night = bt_night / (1 + (w*bt_night/c2)*np.log(e_night))

    stack = np.stack([lst_day, lst_night, bt_day, bt_night, e_day, e_night])
    _save_raster(output_path, stack, profile, count=6)
    return {"output_path": output_path}

def calculate_ttm_lst_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate LST using Three-Temperature Method (TTM)."""
    rasterio, np = _lazy_imports()

    paths = arguments["tir_band_paths"]
    output_path = arguments["output_path"]

    # Read 3 bands
    bands = []
    profile = None
    for p in paths[:3]:
        with rasterio.open(p) as src:
            bands.append(src.read(1).astype(np.float32))
            if profile is None: profile = src.profile

    B1, B2, B3 = bands
    # Weighted LST (Simplified model from source)
    lst = B1*0.3 + B2*0.3 + B3*0.4 + 2.0

    eps = np.full_like(lst, 0.95)
    stack = np.stack([lst, eps, eps])
    _save_raster(output_path, stack, profile, count=3)
    return {"output_path": output_path}

# ------------------------------------------------------------------------------
# 3. LST Statistics
# ------------------------------------------------------------------------------

def stats_lst_by_ndvi_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate Mean or Max LST for pixels within specific NDVI threshold."""
    rasterio, np = _lazy_imports()

    red_path = arguments["red_path"]
    nir_path = arguments["nir_path"]
    lst_path = arguments["lst_path"]
    thresh = arguments.get("threshold", 0.3)
    mode = arguments.get("mode", "above") # above/below
    stat = arguments.get("stat", "mean") # mean/max

    with rasterio.open(red_path) as src: red = src.read(1).astype(np.float32)
    with rasterio.open(nir_path) as src: nir = src.read(1).astype(np.float32)
    with rasterio.open(lst_path) as src: lst = src.read(1).astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-6)

    mask = (ndvi >= thresh) if mode == "above" else (ndvi < thresh)
    mask &= np.isfinite(lst)

    selected = lst[mask]
    if selected.size == 0:
        return {"result": None}

    val = np.mean(selected) if stat == "mean" else np.max(selected)
    return {"result": float(val), "pixel_count": int(selected.size)}

# ------------------------------------------------------------------------------
# 4. Thermal Inertia (ATI)
# ------------------------------------------------------------------------------

def calculate_ati_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Estimate Apparent Thermal Inertia (ATI)."""
    rasterio, np = _lazy_imports()

    day_path = arguments["day_temp_path"]
    night_path = arguments["night_temp_path"]
    albedo_path = arguments["albedo_path"]
    output_path = arguments["output_path"]

    with rasterio.open(day_path) as src:
        bt_day = src.read(1).astype(np.float32)
        profile = src.profile

    bt_night = _resample_to_reference(night_path, day_path)
    albedo = _resample_to_reference(albedo_path, day_path)

    delta_T = bt_day - bt_night
    ati = (1 - albedo) / (delta_T + 1e-6) # Avoid div0
    ati = np.clip(ati, 0, 10)

    _save_raster(output_path, ati, profile)
    return {"output_path": output_path}

# ------------------------------------------------------------------------------
# 5. Microwave Parameter Inversion (Soil Moisture / Vegetation)
# ------------------------------------------------------------------------------

def microwave_dpdm_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Dual-Polarization Differential Method (DPDM)."""
    rasterio, np = _lazy_imports()

    p1, p2 = arguments["pol1_path"], arguments["pol2_path"]
    param = arguments.get("parameter", "soil_moisture").lower()
    a, b = arguments.get("a", 0.3), arguments.get("b", 0.1)
    output_path = arguments["output_path"]

    with rasterio.open(p1) as src:
        b1 = src.read(1).astype(np.float32)
        profile = src.profile
    b2 = _resample_to_reference(p2, p1)

    # Assume input is dB, convert to linear? Original code has option.
    if arguments.get("input_unit", "dB") == "dB":
        b1 = 10**(b1/10)
        b2 = 10**(b2/10)

    if param == "soil_moisture":
        res = a * (b1 - b2) + b
    else: # vegetation_index
        res = (b1 - b2) / (b1 + b2 + 1e-6)

    _save_raster(output_path, res, profile)
    return {"output_path": output_path}

def microwave_ddm_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Dual-frequency Differential Method (DDM)."""
    rasterio, np = _lazy_imports()

    p1, p2 = arguments["band1_path"], arguments["band2_path"]
    alpha, beta = arguments.get("alpha", 0.7), arguments.get("beta", 0.1)
    output_path = arguments["output_path"]

    with rasterio.open(p1) as src: b1 = src.read(1).astype(np.float32); profile=src.profile
    b2 = _resample_to_reference(p2, p1)

    diff = b1 - b2
    param = alpha * diff + beta
    param = np.clip(param, 0, 1)

    out_stack = np.stack([diff, param])
    _save_raster(output_path, out_stack, profile, count=2)
    return {"output_path": output_path}

def microwave_multi_freq_bt_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Multi-frequency Brightness Temperature Method."""
    rasterio, np = _lazy_imports()

    paths = arguments["bt_paths"]
    pairs = arguments["diff_pairs"] # List[List[int]] e.g. [[0,1], [1,2]]
    param_type = arguments.get("parameter", "SM")
    output_path = arguments["output_path"]

    # Coeffs map (Simplified from original)
    defaults = {"SM": {"a": [0.6, 0.4], "b": 0.05}, "VWC": {"a":[0.5,0.5], "b":0.1}, "LAI": {"a":[0.7,0.3], "b":0.0}}
    coeffs = defaults.get(param_type, defaults["SM"])

    # Read all
    arrays = []
    profile = None
    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None: profile = src.profile

    # Calculate param
    result = np.full_like(arrays[0], coeffs["b"])
    alphas = coeffs["a"]

    for i, (idx1, idx2) in enumerate(pairs):
        if i < len(alphas):
            diff = arrays[idx1] - arrays[idx2]
            result += alphas[i] * diff

    result = np.clip(result, 0, 1)
    _save_raster(output_path, result, profile)
    return {"output_path": output_path}

def microwave_prm_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Dual-Polarization Ratio Method (PRM) for VWC or SM."""
    rasterio, np = _lazy_imports()

    paths = arguments["bt_paths"] # dict with 'V' and 'H'
    param_type = arguments.get("parameter", "VWC")
    output_path = arguments["output_path"]

    with rasterio.open(paths["V"]) as src: V = src.read(1).astype(np.float32); profile=src.profile
    H = _resample_to_reference(paths["H"], paths["V"])

    pr = (V - H) / (V + H + 1e-6)

    # Empirical
    if param_type == "VWC": a, b = 15.0, 5.0
    else: a, b = 0.4, 0.1 # SM

    res = a * pr + b

    _save_raster(output_path, np.stack([res, pr]), profile, count=2)
    return {"output_path": output_path}

# ------------------------------------------------------------------------------
# 6. Cryosphere & Water
# ------------------------------------------------------------------------------

def calculate_sea_ice_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """NASA Team Sea Ice Concentration."""
    rasterio, np = _lazy_imports()

    paths = arguments["bt_paths"] # dict 19V, 19H, 37V, 37H
    output_path = arguments["output_path"]

    # Read ref
    with rasterio.open(paths["19V"]) as src:
        b19v = src.read(1).astype(np.float32)
        profile = src.profile

    b19h = _resample_to_reference(paths["19H"], paths["19V"])
    b37v = _resample_to_reference(paths["37V"], paths["19V"])
    b37h = _resample_to_reference(paths["37H"], paths["19V"])

    ND = b19v - b19h
    S1 = b37v - b37h

    # NASA Team simple formula components (Coeffs default from original)
    term1 = (ND - 0.0) / (50.0 - 0.0)
    term2 = (S1 - 0.0) / (20.0 - 0.0)
    Ci = (term1 + term2) / 2
    Ci = np.clip(Ci, 0, 1)

    _save_raster(output_path, Ci, profile)
    return {"output_path": output_path}

def calculate_turbidity_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate Water Turbidity (NTU) from Red Band."""
    rasterio, np = _lazy_imports()

    path = arguments["input_red_path"]
    output_path = arguments["output_path"]
    method = arguments.get("method", "linear")
    a, b, n = arguments.get("a", 1.0), arguments.get("b", 0.0), arguments.get("n", 1.0)

    with rasterio.open(path) as src:
        red = src.read(1).astype(np.float32)
        profile = src.profile

    if method == "linear":
        ntu = a * red + b
    elif method == "power":
        ntu = a * (np.maximum(red, 1e-6)**n) + b
    elif method == "log":
        ntu = a * np.log(np.maximum(red, 1e-6)) + b
    else:
        raise ValueError("Unknown method")

    ntu = np.maximum(ntu, 0)
    _save_raster(output_path, ntu, profile)
    return {"output_path": output_path}

# ------------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    registrar.toolkit("earth_sci", "Physical parameter inversion tools (Atmosphere, LST, Microwave, Water)", "0.2.0")

    # 1. PWV
    registrar.tool(ToolSpec(slug="earth_sci.calculate_pwv", name="Calculate PWV",
                            description="Precipitable Water Vapor from MODIS bands.",
                            parameters={"type":"object", "required":["b02","b05","b17","b18","b19","output_path"],
                                        "properties":{"b02":{"type":"string"},"output_path":{"type":"string"}}},
                            requires_connection=False), calculate_pwv_band_ratio_handler)

    # 2. LST
    registrar.tool(ToolSpec(slug="earth_sci.calculate_lst_sc", name="LST Single Channel",
                            description="LST using Single-Channel method.",
                            parameters={"type":"object", "required":["bt_path","output_path"],
                                        "properties":{"bt_path":{"type":"string"},"red_path":{"type":"string"},"output_path":{"type":"string"}}},
                            requires_connection=False), calculate_lst_single_channel_handler)

    registrar.tool(ToolSpec(slug="earth_sci.calculate_lst_mc", name="LST Multi Channel",
                            description="LST using Multi-Channel (Split-Window) empirical method.",
                            parameters={"type":"object", "required":["band31_path","band32_path","output_path"],
                                        "properties":{"band31_path":{"type":"string"},"output_path":{"type":"string"}}},
                            requires_connection=False), calculate_lst_multi_channel_handler)

    registrar.tool(ToolSpec(slug="earth_sci.calculate_split_window", name="Split Window LST/PWV",
                            description="Physical Split-Window for LST or PWV.",
                            parameters={"type":"object", "required":["band31_path","band32_path","emissivity31_path","output_path"],
                                        "properties":{"parameter":{"type":"string"}}},
                            requires_connection=False), calculate_split_window_handler)

    registrar.tool(ToolSpec(slug="earth_sci.calculate_tes", name="LST TES",
                            description="Temperature Emissivity Separation.",
                            parameters={"type":"object", "required":["tir_band_paths","output_path"],
                                        "properties":{"tir_band_paths":{"type":"array"}}},
                            requires_connection=False), calculate_tes_handler)

    registrar.tool(ToolSpec(slug="earth_sci.calculate_modis_day_night", name="LST Day/Night",
                            description="LST from MODIS Day/Night pairs.",
                            parameters={"type":"object", "required":["bt_day","bt_night","output_path"],
                                        "properties":{"bt_day":{"type":"string"}}},
                            requires_connection=False), calculate_modis_day_night_lst_handler)

    registrar.tool(ToolSpec(slug="earth_sci.calculate_ttm", name="LST TTM",
                            description="Three-Temperature Method.",
                            parameters={"type":"object", "required":["tir_band_paths","output_path"],
                                        "properties":{"tir_band_paths":{"type":"array"}}},
                            requires_connection=False), calculate_ttm_lst_handler)

    # 3. Stats
    registrar.tool(ToolSpec(slug="earth_sci.stats_lst_ndvi", name="LST Stats by NDVI",
                            description="Mean/Max LST for specific NDVI ranges.",
                            parameters={"type":"object", "required":["red_path","nir_path","lst_path"],
                                        "properties":{"stat":{"type":"string"}}},
                            requires_connection=False), stats_lst_by_ndvi_handler)

    # 4. ATI
    registrar.tool(ToolSpec(slug="earth_sci.calculate_ati", name="Thermal Inertia",
                            description="Apparent Thermal Inertia.",
                            parameters={"type":"object", "required":["day_temp_path","night_temp_path","albedo_path","output_path"],
                                        "properties":{"day_temp_path":{"type":"string"}}},
                            requires_connection=False), calculate_ati_handler)

    # 5. Microwave
    registrar.tool(ToolSpec(slug="earth_sci.microwave_dpdm", name="Microwave DPDM",
                            description="Dual-Polarization Differential Method.",
                            parameters={"type":"object", "required":["pol1_path","pol2_path","output_path"],
                                        "properties":{"parameter":{"type":"string"}}},
                            requires_connection=False), microwave_dpdm_handler)

    registrar.tool(ToolSpec(slug="earth_sci.microwave_ddm", name="Microwave DDM",
                            description="Dual-Frequency Differential Method.",
                            parameters={"type":"object", "required":["band1_path","band2_path","output_path"],
                                        "properties":{"band1_path":{"type":"string"}}},
                            requires_connection=False), microwave_ddm_handler)

    registrar.tool(ToolSpec(slug="earth_sci.microwave_multi_freq", name="Microwave Multi Freq",
                            description="Multi-frequency Brightness Temperature Method.",
                            parameters={"type":"object", "required":["bt_paths","diff_pairs","output_path"],
                                        "properties":{"bt_paths":{"type":"array"}}},
                            requires_connection=False), microwave_multi_freq_bt_handler)

    registrar.tool(ToolSpec(slug="earth_sci.microwave_prm", name="Microwave PRM",
                            description="Dual-Polarization Ratio Method.",
                            parameters={"type":"object", "required":["bt_paths","output_path"],
                                        "properties":{"bt_paths":{"type":"object"}}},
                            requires_connection=False), microwave_prm_handler)

    # 6. Water/Ice
    registrar.tool(ToolSpec(slug="earth_sci.calculate_sea_ice", name="Sea Ice Concentration",
                            description="NASA Team Sea Ice Algorithm.",
                            parameters={"type":"object", "required":["bt_paths","output_path"],
                                        "properties":{"bt_paths":{"type":"object"}}},
                            requires_connection=False), calculate_sea_ice_handler)

    registrar.tool(ToolSpec(slug="earth_sci.calculate_turbidity", name="Water Turbidity",
                            description="Turbidity (NTU) from Red Band.",
                            parameters={"type":"object", "required":["input_red_path","output_path"],
                                        "properties":{"method":{"type":"string"}}},
                            requires_connection=False), calculate_turbidity_handler)