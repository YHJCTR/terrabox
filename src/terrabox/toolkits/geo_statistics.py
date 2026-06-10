"""
Geo Statistics Toolkit
----------------------
Comprehensive batch raster statistics and scalar arithmetic tools.
Ported from Earth-Agent Statistics Kit and adapted for Terrabox.

Key features:
1. Batch raster statistics: mean, std, median, min, max, sum per image or across images.
2. Higher-order statistics: CV, skewness, kurtosis.
3. Aggregation: mean-of-means, threshold ratios, conditional counts.
4. Scalar arithmetic and unit conversion utilities.
"""

import json
import os
from typing import Any, Dict, List, Optional
from ..core.registry import ToolSpec


# ------------------------------------------------------------------------------
# Lazy Imports
# ------------------------------------------------------------------------------

def _lazy_imports():
    try:
        import rasterio
        import numpy as np
        return rasterio, np
    except ImportError:
        raise ImportError("Missing dependencies. Install: pip install rasterio numpy")


def _lazy_scipy():
    try:
        from scipy.stats import skew, kurtosis
        return skew, kurtosis
    except ImportError:
        raise ImportError("Missing scipy. Install: pip install scipy")


def _read_valid_pixels(path: str, band: int = 1) -> "np.ndarray":
    """Read a single-band raster and return valid (finite, non-nodata) pixels as 1D array."""
    import rasterio, numpy as np
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        mask = payload.get("mask") if isinstance(payload, dict) else None
        if mask is not None:
            data = np.asarray(mask, dtype=np.float32)
            flat = data.flatten()
            return flat[np.isfinite(flat)]
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        nodata = src.nodata
    flat = data.flatten()
    if nodata is not None:
        flat = flat[flat != nodata]
    return flat[np.isfinite(flat)]


# ------------------------------------------------------------------------------
# 1. Batch Raster Statistics
# ------------------------------------------------------------------------------

def batch_raster_stats_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute per-image and aggregate statistics for a list of single-band rasters.
    Returns mean, std, median, min, max, sum for each image and overall aggregates.
    """
    rasterio, np = _lazy_imports()

    input_paths = arguments.get("input_paths", [])
    stat = arguments.get("stat")  # optional: one of mean/std/median/min/max/sum

    if not input_paths:
        raise ValueError("input_paths must contain at least one file path")

    per_image = []
    all_values = []
    for p in input_paths:
        valid = _read_valid_pixels(p)
        if valid.size == 0:
            per_image.append({"path": p, "error": "no valid pixels"})
            continue
        stats = {
            "path": p,
            "mean":   float(np.mean(valid)),
            "std":    float(np.std(valid, ddof=1)),
            "median": float(np.median(valid)),
            "min":    float(np.min(valid)),
            "max":    float(np.max(valid)),
            "sum":    float(np.sum(valid)),
            "count":  int(valid.size),
        }
        if stat:
            per_image.append({"path": p, stat: stats[stat]})
        else:
            per_image.append(stats)
        all_values.append(valid)

    if not all_values:
        return {"per_image": per_image, "aggregate": None}

    combined = np.concatenate(all_values)
    aggregate = {
        "mean":   float(np.mean(combined)),
        "std":    float(np.std(combined, ddof=1)),
        "median": float(np.median(combined)),
        "min":    float(np.min(combined)),
        "max":    float(np.max(combined)),
        "sum":    float(np.sum(combined)),
        "count":  int(combined.size),
    }
    if stat:
        aggregate = {stat: aggregate[stat]}

    return {"per_image": per_image, "aggregate": aggregate}


# ------------------------------------------------------------------------------
# 2. Higher-Order Statistics
# ------------------------------------------------------------------------------

def calc_cv_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Coefficient of Variation = std / mean (dimensionless). Returns NaN if mean == 0."""
    _, np = _lazy_imports()
    input_path = arguments["input_path"]
    valid = _read_valid_pixels(input_path)
    if valid.size == 0:
        return {"error": "no valid pixels"}
    m = float(np.mean(valid))
    s = float(np.std(valid, ddof=1))
    cv = s / m if m != 0 else float("nan")
    return {"cv": cv, "mean": m, "std": s}


def calc_skewness_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Fisher-Pearson skewness of pixel values in a single-band raster."""
    _, np = _lazy_imports()
    skew, _ = _lazy_scipy()
    input_path = arguments["input_path"]
    valid = _read_valid_pixels(input_path)
    if valid.size == 0:
        return {"error": "no valid pixels"}
    return {"skewness": float(skew(valid, bias=True)), "count": int(valid.size)}


def calc_kurtosis_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Excess kurtosis (Fisher) of pixel values in a single-band raster."""
    _, np = _lazy_imports()
    _, kurtosis = _lazy_scipy()
    input_path = arguments["input_path"]
    valid = _read_valid_pixels(input_path)
    if valid.size == 0:
        return {"error": "no valid pixels"}
    return {"kurtosis": float(kurtosis(valid, fisher=True, bias=True)), "count": int(valid.size)}


def calc_percentile_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Return pixel value at a given percentile (0–100) in a raster, ignoring nodata/NaN."""
    _, np = _lazy_imports()
    input_path = arguments["input_path"]
    percentile = float(arguments["percentile"])
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be in [0, 100]")
    valid = _read_valid_pixels(input_path)
    if valid.size == 0:
        return {"error": "no valid pixels"}
    return {"percentile": percentile, "value": float(np.percentile(valid, percentile))}


# ------------------------------------------------------------------------------
# 3. Cross-Image Aggregation
# ------------------------------------------------------------------------------

def mean_of_means_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute the mean of per-image means across multiple rasters (temporal average)."""
    _, np = _lazy_imports()
    input_paths = arguments.get("input_paths", [])
    if not input_paths:
        raise ValueError("input_paths must contain at least one file path")
    means = []
    for p in input_paths:
        valid = _read_valid_pixels(p)
        if valid.size > 0:
            means.append(float(np.mean(valid)))
    if not means:
        return {"error": "no valid data in any image"}
    return {
        "mean_of_means": float(np.mean(means)),
        "per_image_means": means,
        "count": len(means),
    }


def count_images_exceeding_threshold_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    For each image, compute the ratio of pixels exceeding `pixel_threshold`.
    Count how many images have a ratio >= `ratio_threshold`.
    """
    _, np = _lazy_imports()
    input_paths = arguments.get("input_paths", [])
    pixel_threshold = float(arguments["pixel_threshold"])
    ratio_threshold = float(arguments.get("ratio_threshold", 0.1))

    count = 0
    ratios = []
    for p in input_paths:
        valid = _read_valid_pixels(p)
        if valid.size == 0:
            ratios.append(0.0)
            continue
        ratio = float(np.sum(valid > pixel_threshold) / valid.size)
        ratios.append(ratio)
        if ratio >= ratio_threshold:
            count += 1

    return {
        "images_exceeding": count,
        "total_images": len(input_paths),
        "per_image_ratios": ratios,
    }


# ------------------------------------------------------------------------------
# 4. Threshold & Conditional Analysis
# ------------------------------------------------------------------------------

def threshold_ratio_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Fraction of valid pixels above (or below) a threshold.
    Returns ratio in [0.0, 1.0].
    """
    _, np = _lazy_imports()
    input_path = arguments["input_path"]
    threshold = float(arguments["threshold"])
    mode = arguments.get("mode", "above")  # above / below

    valid = _read_valid_pixels(input_path)
    if valid.size == 0:
        return {"ratio": 0.0, "count": 0, "total": 0}

    if mode == "above":
        count = int(np.sum(valid > threshold))
    else:
        count = int(np.sum(valid < threshold))

    return {"ratio": float(count / valid.size), "count": count, "total": int(valid.size)}


def count_pixels_condition_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Count pixels satisfying: lower <= pixel <= upper.
    Either bound can be omitted (open interval).
    """
    _, np = _lazy_imports()
    input_path = arguments["input_path"]
    lower = arguments.get("lower")
    upper = arguments.get("upper")

    valid = _read_valid_pixels(input_path)
    mask = np.ones(valid.size, dtype=bool)
    if lower is not None:
        mask &= (valid >= float(lower))
    if upper is not None:
        mask &= (valid <= float(upper))

    count = int(np.sum(mask))
    return {"count": count, "total": int(valid.size), "ratio": float(count / valid.size) if valid.size > 0 else 0.0}


def intersection_percentage_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Co-occurrence of two threshold conditions across two co-registered rasters.
    Returns the fraction of pixels where raster_a > threshold_a AND raster_b > threshold_b.
    """
    _, np = _lazy_imports()
    import rasterio

    path_a = arguments["path_a"]
    threshold_a = float(arguments["threshold_a"])
    path_b = arguments["path_b"]
    threshold_b = float(arguments["threshold_b"])

    with rasterio.open(path_a) as src:
        da = src.read(1).astype(np.float32)
        nodata_a = src.nodata
    with rasterio.open(path_b) as src:
        db = src.read(1).astype(np.float32)
        nodata_b = src.nodata

    valid = np.ones(da.shape, dtype=bool)
    valid &= np.isfinite(da) & np.isfinite(db)
    if nodata_a is not None:
        valid &= (da != nodata_a)
    if nodata_b is not None:
        valid &= (db != nodata_b)

    total = int(np.sum(valid))
    if total == 0:
        return {"intersection_ratio": 0.0, "count": 0, "total": 0}

    both = int(np.sum(valid & (da > threshold_a) & (db > threshold_b)))
    return {"intersection_ratio": float(both / total), "count": both, "total": total}


def multi_band_threshold_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Count pixels where ALL bands satisfy their respective thresholds simultaneously.
    `band_thresholds`: list of {path, threshold} dicts.
    """
    _, np = _lazy_imports()
    import rasterio

    band_thresholds = arguments.get("band_thresholds", [])
    if len(band_thresholds) < 2:
        raise ValueError("Provide at least 2 band_thresholds entries")

    arrays = []
    thresholds = []
    shape = None
    for bt in band_thresholds:
        with rasterio.open(bt["path"]) as src:
            d = src.read(1).astype(np.float32)
            nd = src.nodata
        if nd is not None:
            d[d == nd] = np.nan
        if shape is None:
            shape = d.shape
        arrays.append(d)
        thresholds.append(float(bt["threshold"]))

    valid = np.all([np.isfinite(a) for a in arrays], axis=0)
    total = int(np.sum(valid))
    if total == 0:
        return {"count": 0, "total": 0, "ratio": 0.0}

    cond = valid.copy()
    for a, t in zip(arrays, thresholds):
        cond &= (a > t)

    count = int(np.sum(cond))
    return {"count": count, "total": total, "ratio": float(count / total)}


# ------------------------------------------------------------------------------
# 5. Scalar Arithmetic & Unit Conversion
# ------------------------------------------------------------------------------

def percentage_change_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Percentage change from a to b: (b - a) / |a| * 100. Returns NaN if a == 0."""
    a = float(arguments["a"])
    b = float(arguments["b"])
    if a == 0:
        return {"percentage_change": float("nan")}
    return {"percentage_change": (b - a) / abs(a) * 100.0}


def scalar_arithmetic_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Perform basic arithmetic on two scalar values.
    `operation`: add | subtract | multiply | divide
    """
    a = float(arguments["a"])
    b = float(arguments["b"])
    op = arguments.get("operation", "add")

    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "divide":
        if b == 0:
            return {"error": "division by zero"}
        result = a / b
    else:
        raise ValueError(f"Unknown operation: {op}. Choose from: add, subtract, multiply, divide")

    return {"result": result, "operation": op, "a": a, "b": b}


def kelvin_to_celsius_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Convert temperature from Kelvin to Celsius: °C = K - 273.15."""
    value = float(arguments["value"])
    return {"celsius": value - 273.15, "kelvin": value}


def celsius_to_kelvin_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Convert temperature from Celsius to Kelvin: K = °C + 273.15."""
    value = float(arguments["value"])
    return {"kelvin": value + 273.15, "celsius": value}


# ------------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    """Register all geo statistics tools."""
    registrar.toolkit(
        name="geo_statistics",
        description="Batch raster statistics, higher-order statistics, cross-image aggregation, threshold analysis, and unit conversions for EO analysis.",
        version="1.0.0"
    )

    # 1. Batch raster stats
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.batch_raster_stats",
            name="Batch Raster Statistics",
            description="Compute per-image and aggregate statistics (mean, std, median, min, max, sum) for a list of single-band rasters. Returns {per_image:[{mean,std,median,min,max,sum}], aggregate:{mean,std,...}}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of raster file paths to analyze."
                    },
                    "stat": {
                        "type": "string",
                        "enum": ["mean", "std", "median", "min", "max", "sum"],
                        "description": "Optional: return only this statistic for each image. Omit to return all."
                    }
                },
                "required": ["input_paths"]
            },
            requires_connection=False
        ),
        batch_raster_stats_handler,
    )

    # 2. CV
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.calc_cv",
            name="Coefficient of Variation",
            description="Compute CV = std / mean for a single-band raster. Useful for assessing spatial variability. Returns {cv, mean, std}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input raster."}
                },
                "required": ["input_path"]
            },
            requires_connection=False
        ),
        calc_cv_handler,
    )

    # 3. Skewness
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.calc_skewness",
            name="Skewness",
            description="Compute Fisher-Pearson skewness of pixel distribution in a single-band raster. Returns {skewness, count}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input raster."}
                },
                "required": ["input_path"]
            },
            requires_connection=False
        ),
        calc_skewness_handler,
    )

    # 4. Kurtosis
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.calc_kurtosis",
            name="Kurtosis",
            description="Compute excess kurtosis (Fisher) of pixel distribution in a single-band raster. Returns {kurtosis, count}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input raster."}
                },
                "required": ["input_path"]
            },
            requires_connection=False
        ),
        calc_kurtosis_handler,
    )

    # 5. Percentile
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.calc_percentile",
            name="Percentile Value",
            description="Return the pixel value at a specified percentile (0–100) in a raster, ignoring nodata and NaN. Returns {percentile, value}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input raster."},
                    "percentile": {"type": "number", "description": "Percentile in [0, 100], e.g. 95 for the 95th percentile."}
                },
                "required": ["input_path", "percentile"]
            },
            requires_connection=False
        ),
        calc_percentile_handler,
    )

    # 6. Mean of means
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.mean_of_means",
            name="Mean of Means",
            description="Compute the mean of per-image mean values across a time series of rasters. Useful for temporal averaging. Returns {mean_of_means, per_image_means:[...], count}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of raster file paths (e.g., monthly composites)."
                    }
                },
                "required": ["input_paths"]
            },
            requires_connection=False
        ),
        mean_of_means_handler,
    )

    # 7. Count images exceeding threshold
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.count_images_exceeding",
            name="Count Images Exceeding Threshold Ratio",
            description="Count how many images in a time series have a pixel-above-threshold ratio >= ratio_threshold. Returns {images_exceeding, total_images, per_image_ratios:[...]}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of raster file paths."
                    },
                    "pixel_threshold": {"type": "number", "description": "Per-pixel value threshold."},
                    "ratio_threshold": {"type": "number", "default": 0.1, "description": "Minimum ratio of exceeding pixels for an image to be counted (default: 0.1)."}
                },
                "required": ["input_paths", "pixel_threshold"]
            },
            requires_connection=False
        ),
        count_images_exceeding_threshold_handler,
    )

    # 8. Threshold ratio
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.threshold_ratio",
            name="Threshold Ratio",
            description="Compute the fraction of valid pixels above (or below) a threshold in a single-band raster. Returns {ratio (0-1), count, total}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input raster."},
                    "threshold": {"type": "number", "description": "Threshold value."},
                    "mode": {
                        "type": "string",
                        "enum": ["above", "below"],
                        "default": "above",
                        "description": "'above': count pixels > threshold; 'below': count pixels < threshold."
                    }
                },
                "required": ["input_path", "threshold"]
            },
            requires_connection=False
        ),
        threshold_ratio_handler,
    )

    # 9. Count pixels condition
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.count_pixels_condition",
            name="Count Pixels in Range",
            description="Count pixels satisfying lower <= value <= upper. Either bound may be omitted. Returns {count, total, ratio}.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input raster."},
                    "lower": {"type": "number", "description": "Lower bound (inclusive). Omit for open lower bound."},
                    "upper": {"type": "number", "description": "Upper bound (inclusive). Omit for open upper bound."}
                },
                "required": ["input_path"]
            },
            requires_connection=False
        ),
        count_pixels_condition_handler,
    )

    # 10. Intersection percentage
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.intersection_percentage",
            name="Threshold Intersection Percentage",
            description="Fraction of pixels where raster_a > threshold_a AND raster_b > threshold_b simultaneously. Both rasters must be co-registered. Returns {intersection_ratio, count, total}.",
            parameters={
                "type": "object",
                "properties": {
                    "path_a": {"type": "string", "description": "Path to first raster."},
                    "threshold_a": {"type": "number", "description": "Threshold for first raster."},
                    "path_b": {"type": "string", "description": "Path to second raster."},
                    "threshold_b": {"type": "number", "description": "Threshold for second raster."}
                },
                "required": ["path_a", "threshold_a", "path_b", "threshold_b"]
            },
            requires_connection=False
        ),
        intersection_percentage_handler,
    )

    # 11. Multi-band threshold
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.multi_band_threshold",
            name="Multi-Band Threshold Analysis",
            description="Count pixels where all specified rasters simultaneously exceed their respective thresholds. band_thresholds is a list of {path, threshold} (>=2). Returns {count, total, ratio}.",
            parameters={
                "type": "object",
                "properties": {
                    "band_thresholds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Raster file path."},
                                "threshold": {"type": "number", "description": "Threshold value for this band."}
                            },
                            "required": ["path", "threshold"]
                        },
                        "description": "List of {path, threshold} pairs (at least 2)."
                    }
                },
                "required": ["band_thresholds"]
            },
            requires_connection=False
        ),
        multi_band_threshold_handler,
    )

    # 12. Percentage change
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.percentage_change",
            name="Percentage Change",
            description="Compute percentage change from scalar a to b: (b - a) / |a| * 100. Returns {percentage_change} (percent).",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Initial value (baseline)."},
                    "b": {"type": "number", "description": "Final value."}
                },
                "required": ["a", "b"]
            },
            requires_connection=False
        ),
        percentage_change_handler,
    )

    # 13. Scalar arithmetic
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.scalar_arithmetic",
            name="Scalar Arithmetic",
            description="Perform basic arithmetic (add, subtract, multiply, divide) on two scalar values. Returns {result, operation, a, b}.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand."},
                    "b": {"type": "number", "description": "Second operand."},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "default": "add",
                        "description": "Arithmetic operation to perform."
                    }
                },
                "required": ["a", "b"]
            },
            requires_connection=False
        ),
        scalar_arithmetic_handler,
    )

    # 14. Kelvin to Celsius
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.kelvin_to_celsius",
            name="Kelvin to Celsius",
            description="Convert a temperature value from Kelvin to Celsius (°C = K - 273.15). Returns {celsius, kelvin}.",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Temperature in Kelvin."}
                },
                "required": ["value"]
            },
            requires_connection=False
        ),
        kelvin_to_celsius_handler,
    )

    # 15. Celsius to Kelvin
    registrar.tool(
        ToolSpec(
            slug="geo_statistics.celsius_to_kelvin",
            name="Celsius to Kelvin",
            description="Convert a temperature value from Celsius to Kelvin (K = °C + 273.15).",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Temperature in Celsius."}
                },
                "required": ["value"]
            },
            requires_connection=False
        ),
        celsius_to_kelvin_handler,
    )
