"""
Geo Raster Toolkit
------------------
A production-ready toolkit for raster-based spectral index calculations and analysis.

Key features:
1. Universal Normalized Difference Calculator (NDVI, NDWI, NDBI, NBR, NDTI, NDSI).
2. Specialized Indices (EVI, WRI, FVC).
3. Advanced Analysis (TVDI, FRP, Snow Loss Statistics).
4. Zero-dependency initialization (lazy imports).
"""

import os
from typing import Any, Dict
from ..core.registry import ToolSpec


# ------------------------------------------------------------------------------
# Lazy Imports & Helpers
# ------------------------------------------------------------------------------

def _lazy_imports():
    try:
        import rasterio
        import numpy as np
        return rasterio, np
    except ImportError:
        raise ImportError("Missing dependencies for geo_raster. Please install: pip install rasterio numpy")


def _lazy_scipy_imports():
    try:
        from scipy.stats import linregress
        from scipy.ndimage import zoom
        return linregress, zoom
    except ImportError:
        raise ImportError("Missing dependencies for advanced analysis. Please install: pip install scipy")


def _save_raster(output_path: str, data: Any, profile: Dict[str, Any], dtype=None, nodata=None):
    """Helper to save raster data to disk."""
    import rasterio

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Update profile
    new_profile = profile.copy()
    new_profile.update(
        count=1,
        compress='lzw'
    )
    if dtype:
        new_profile['dtype'] = dtype
    if nodata is not None:
        new_profile['nodata'] = nodata

    with rasterio.open(output_path, 'w', **new_profile) as dst:
        dst.write(data, 1)


# ------------------------------------------------------------------------------
# Universal Normalized Difference Calculator
# ------------------------------------------------------------------------------

def calculate_normalized_difference_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Calculate generic Normalized Difference Indices.
    Formula: (Band_A - Band_B) / (Band_A + Band_B)
    """
    rasterio, np = _lazy_imports()
    scipy_zoom = None  # Only needed for NDSI resizing if sizes differ

    band_a_path = arguments.get("band_a_path")
    band_b_path = arguments.get("band_b_path")
    output_path = arguments.get("output_path")
    index_name = arguments.get("index_name", "index").upper()
    scale_factor = float(arguments.get("scale_factor", 1.0))  # E.g. 0.0001 for MODIS

    if not (band_a_path and band_b_path and output_path):
        raise ValueError("Missing required paths: band_a_path, band_b_path, or output_path")

    with rasterio.open(band_a_path) as src_a:
        band_a = src_a.read(1).astype(np.float32)
        profile = src_a.profile

    with rasterio.open(band_b_path) as src_b:
        band_b = src_b.read(1).astype(np.float32)

    # Apply scaling if needed (e.g. for MODIS surface reflectance)
    if scale_factor != 1.0:
        band_a *= scale_factor
        band_b *= scale_factor

    # Handle size mismatch (logic ported from calculate_ndsi in Index.py)
    if band_a.shape != band_b.shape:
        _, zoom = _lazy_scipy_imports()
        target_h = min(band_a.shape[0], band_b.shape[0])
        target_w = min(band_a.shape[1], band_b.shape[1])

        # Helper to resize
        def resize_band(b, target_shape):
            if b.shape == target_shape: return b
            factors = (target_shape[0] / b.shape[0], target_shape[1] / b.shape[1])
            return zoom(b, factors, order=1)

        band_a = resize_band(band_a, (target_h, target_w))
        band_b = resize_band(band_b, (target_h, target_w))
        # Update profile height/width
        profile.update(height=target_h, width=target_w)

    # Filter invalid values
    if scale_factor != 1.0:
        band_a = np.where((band_a < 0) | (band_a > 1), np.nan, band_a)
        band_b = np.where((band_b < 0) | (band_b > 1), np.nan, band_b)

    # Calculation
    denominator = band_a + band_b + 1e-6
    result = (band_a - band_b) / denominator
    result = np.clip(result, -1, 1)

    _save_raster(output_path, result.astype(rasterio.float32), profile, dtype=rasterio.float32, nodata=-9999)

    return {
        "status": "success",
        "message": f"{index_name} calculated.",
        "output_path": output_path
    }


# ------------------------------------------------------------------------------
# Specialized Indices (EVI, FVC, WRI)
# ------------------------------------------------------------------------------

def calculate_evi_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate Enhanced Vegetation Index (EVI)."""
    rasterio, np = _lazy_imports()

    nir_path = arguments["nir_path"]
    red_path = arguments["red_path"]
    blue_path = arguments["blue_path"]
    output_path = arguments["output_path"]

    G = float(arguments.get("G", 2.5))
    C1 = float(arguments.get("C1", 6.0))
    C2 = float(arguments.get("C2", 7.5))
    L = float(arguments.get("L", 1.0))

    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(np.float32)
        profile = src.profile
    with rasterio.open(red_path) as src:
        red = src.read(1).astype(np.float32)
    with rasterio.open(blue_path) as src:
        blue = src.read(1).astype(np.float32)

    denominator = nir + C1 * red - C2 * blue + L + 1e-6
    evi = G * (nir - red) / denominator

    _save_raster(output_path, evi.astype(rasterio.float32), profile, dtype=rasterio.float32, nodata=-9999)
    return {"output_path": output_path}


def calculate_fvc_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate Fractional Vegetation Cover (FVC) from NIR and Red."""
    rasterio, np = _lazy_imports()

    nir_path = arguments["nir_path"]
    red_path = arguments["red_path"]
    output_path = arguments["output_path"]
    ndvi_min = float(arguments.get("ndvi_min", 0.1))
    ndvi_max = float(arguments.get("ndvi_max", 0.9))

    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(np.float32)
        profile = src.profile
    with rasterio.open(red_path) as src:
        red = src.read(1).astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-6)

    # FVC formula
    fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) * 100.0
    fvc = np.clip(fvc, 0, 100)

    _save_raster(output_path, fvc.astype(rasterio.float32), profile, dtype=rasterio.float32, nodata=-9999)
    return {"output_path": output_path}


def calculate_wri_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate Water Ratio Index (WRI): (Green + Red) / (NIR + SWIR)."""
    rasterio, np = _lazy_imports()

    green_path = arguments["green_path"]
    red_path = arguments["red_path"]
    nir_path = arguments["nir_path"]
    swir_path = arguments["swir_path"]
    output_path = arguments["output_path"]

    with rasterio.open(green_path) as src:
        green = src.read(1).astype(np.float32)
        profile = src.profile
    with rasterio.open(red_path) as src:
        red = src.read(1).astype(np.float32)
    with rasterio.open(nir_path) as src:
        nir = src.read(1).astype(np.float32)
    with rasterio.open(swir_path) as src:
        swir = src.read(1).astype(np.float32)

    denominator = nir + swir + 1e-6
    wri = (green + red) / denominator

    _save_raster(output_path, wri.astype(rasterio.float32), profile, dtype=rasterio.float32, nodata=-9999)
    return {"output_path": output_path}


# ------------------------------------------------------------------------------
# Advanced Analysis (FRP, TVDI, Snow)
# ------------------------------------------------------------------------------

def calculate_frp_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate Fire Radiative Power (FRP) mask based on threshold."""
    rasterio, np = _lazy_imports()

    input_path = arguments["input_path"]
    output_path = arguments["output_path"]
    threshold = float(arguments.get("threshold", 0.0))

    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile

    fire_mask = (data > threshold).astype(np.uint8) * 255

    _save_raster(output_path, fire_mask, profile, dtype=rasterio.uint8, nodata=0)
    return {"output_path": output_path}


def compute_tvdi_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute Temperature Vegetation Dryness Index (TVDI).
    Uses trapezoidal method with linear regression on LST/NDVI scatter space.
    """
    rasterio, np = _lazy_imports()
    linregress, _ = _lazy_scipy_imports()

    ndvi_path = arguments["ndvi_path"]
    lst_path = arguments["lst_path"]
    output_path = arguments["output_path"]

    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1).astype(np.float32) * 0.0001
        profile = src.profile
    with rasterio.open(lst_path) as src:
        lst = src.read(1).astype(np.float32) * 0.02

    # Mask valid data
    valid_mask = (ndvi >= 0) & (ndvi <= 1) & (lst > 0)

    if not np.any(valid_mask) or np.sum(valid_mask) < 100:
        tvdi = np.full_like(ndvi, np.nan)
        _save_raster(output_path, tvdi, profile, dtype=rasterio.float32, nodata=-9999)
        return {"output_path": output_path, "warning": "Insufficient valid pixels for TVDI."}

    ndvi_valid = ndvi[valid_mask]
    lst_valid = lst[valid_mask]

    # Binning logic
    n_bins = 100
    bins = np.linspace(ndvi_valid.min(), ndvi_valid.max(), n_bins + 1)
    bin_centers, lst_max_list, lst_min_list = [], [], []

    for i in range(n_bins):
        mask_bin = (ndvi_valid >= bins[i]) & (ndvi_valid < bins[i + 1])
        if np.any(mask_bin):
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            lst_subset = lst_valid[mask_bin]
            lst_max_list.append(np.max(lst_subset))
            lst_min_list.append(np.min(lst_subset))

    if len(bin_centers) < 2:
        tvdi = np.full_like(ndvi, np.nan)
        _save_raster(output_path, tvdi, profile, dtype=rasterio.float32, nodata=-9999)
        return {"output_path": output_path, "warning": "Insufficient bins for TVDI regression."}

    # Regression lines
    slope_max, int_max, _, _, _ = linregress(bin_centers, lst_max_list)
    slope_min, int_min, _, _, _ = linregress(bin_centers, lst_min_list)

    lst_dry = ndvi * slope_max + int_max
    lst_wet = ndvi * slope_min + int_min

    denom = lst_dry - lst_wet
    denom[denom == 0] = 1e-6
    tvdi = (lst - lst_wet) / denom

    tvdi = np.clip(tvdi, 0, 1)
    tvdi[~valid_mask] = np.nan

    _save_raster(output_path, tvdi.astype(rasterio.float32), profile, dtype=rasterio.float32, nodata=-9999)
    return {"output_path": output_path}


def calc_snow_loss_stats_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate percentage of extreme snow loss from a binary map (1=Loss)."""
    rasterio, np = _lazy_imports()

    binary_map_path = arguments["binary_map_path"]

    with rasterio.open(binary_map_path) as src:
        data = src.read(1)

    if data.size == 0:
        return {"percentage": 0.0}

    flat = data.flatten().astype(np.float32)
    valid = flat[np.isfinite(flat)]

    if valid.size == 0:
        return {"percentage": 0.0}

    loss_count = np.sum(valid == 1.0)
    percentage = float(loss_count / valid.size)

    return {"percentage": percentage, "total_pixels": int(valid.size), "loss_pixels": int(loss_count)}


# ------------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    """Register all raster calculation tools."""
    registrar.toolkit(
        name="geo_raster",
        description="Raster analysis: Spectral indices (NDVI/NDWI/etc), EVI, FVC, TVDI, and physical parameter masking.",
        version="0.2.0"
    )

    # 1. Universal Index Calculator
    registrar.tool(
        ToolSpec(
            slug="geo_raster.calculate_index",
            name="Calculate Spectral Index",
            description="Calculate generic normalized difference indices like NDVI, NDWI, NDBI, NBR, NDTI, NDSI.",
            parameters={
                "type": "object",
                "properties": {
                    "band_a_path": {"type": "string", "description": "Path to the first band (e.g., NIR for NDVI)."},
                    "band_b_path": {"type": "string", "description": "Path to the second band (e.g., Red for NDVI)."},
                    "output_path": {"type": "string", "description": "Output GeoTIFF path."},
                    "index_name": {"type": "string", "description": "Name of index for logging (default: INDEX)."},
                    "scale_factor": {"type": "number", "description": "Optional scaling factor (default: 1.0)."}
                },
                "required": ["band_a_path", "band_b_path", "output_path"]
            },
            requires_connection=False
        ),
        calculate_normalized_difference_handler,
    )

    # 2. EVI
    registrar.tool(
        ToolSpec(
            slug="geo_raster.calculate_evi",
            name="Calculate EVI",
            description="Calculate Enhanced Vegetation Index (EVI). Requires NIR, Red, Blue bands.",
            parameters={
                "type": "object",
                "properties": {
                    "nir_path": {"type": "string", "description": "Path to NIR band."},
                    "red_path": {"type": "string", "description": "Path to Red band."},
                    "blue_path": {"type": "string", "description": "Path to Blue band."},
                    "output_path": {"type": "string", "description": "Output GeoTIFF path."},
                    "G": {"type": "number", "default": 2.5},
                    "C1": {"type": "number", "default": 6.0},
                    "C2": {"type": "number", "default": 7.5},
                    "L": {"type": "number", "default": 1.0}
                },
                "required": ["nir_path", "red_path", "blue_path", "output_path"]
            },
            requires_connection=False
        ),
        calculate_evi_handler,
    )

    # 3. FVC
    registrar.tool(
        ToolSpec(
            slug="geo_raster.calculate_fvc",
            name="Calculate FVC",
            description="Calculate Fractional Vegetation Cover (FVC) from NIR and Red bands.",
            parameters={
                "type": "object",
                "properties": {
                    "nir_path": {"type": "string", "description": "Path to NIR band."},
                    "red_path": {"type": "string", "description": "Path to Red band."},
                    "output_path": {"type": "string", "description": "Output GeoTIFF path."},
                    "ndvi_min": {"type": "number", "default": 0.1},
                    "ndvi_max": {"type": "number", "default": 0.9}
                },
                "required": ["nir_path", "red_path", "output_path"]
            },
            requires_connection=False
        ),
        calculate_fvc_handler,
    )

    # 4. WRI
    registrar.tool(
        ToolSpec(
            slug="geo_raster.calculate_wri",
            name="Calculate WRI",
            description="Calculate Water Ratio Index (WRI). Requires Green, Red, NIR, SWIR bands.",
            parameters={
                "type": "object",
                "properties": {
                    "green_path": {"type": "string"},
                    "red_path": {"type": "string"},
                    "nir_path": {"type": "string"},
                    "swir_path": {"type": "string"},
                    "output_path": {"type": "string"}
                },
                "required": ["green_path", "red_path", "nir_path", "swir_path", "output_path"]
            },
            requires_connection=False
        ),
        calculate_wri_handler,
    )

    # 5. FRP
    registrar.tool(
        ToolSpec(
            slug="geo_raster.calculate_frp_mask",
            name="Calculate FRP Mask",
            description="Generate Fire Radiative Power (FRP) binary mask based on threshold.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to FRP raster."},
                    "output_path": {"type": "string", "description": "Output mask path."},
                    "threshold": {"type": "number", "default": 0.0}
                },
                "required": ["input_path", "output_path"]
            },
            requires_connection=False
        ),
        calculate_frp_handler,
    )

    # 6. TVDI
    registrar.tool(
        ToolSpec(
            slug="geo_raster.compute_tvdi",
            name="Compute TVDI",
            description="Compute Temperature Vegetation Dryness Index (TVDI) from NDVI and LST.",
            parameters={
                "type": "object",
                "properties": {
                    "ndvi_path": {"type": "string", "description": "Path to NDVI raster."},
                    "lst_path": {"type": "string", "description": "Path to LST raster."},
                    "output_path": {"type": "string", "description": "Output GeoTIFF path."}
                },
                "required": ["ndvi_path", "lst_path", "output_path"]
            },
            requires_connection=False
        ),
        compute_tvdi_handler,
    )

    # 7. Snow Stats
    registrar.tool(
        ToolSpec(
            slug="geo_raster.calc_snow_loss_stats",
            name="Calculate Snow Loss Stats",
            description="Calculate percentage of extreme snow loss from a binary map.",
            parameters={
                "type": "object",
                "properties": {
                    "binary_map_path": {"type": "string", "description": "Path to binary raster (1=loss)."}
                },
                "required": ["binary_map_path"]
            },
            requires_connection=False
        ),
        calc_snow_loss_stats_handler,
    )