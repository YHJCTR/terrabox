"""
Geo Analysis Toolkit
--------------------
A toolkit for time-series and spatial statistical analysis of Earth Observation data.

Key features:
1. Time Series Analysis: Linear Trend, Mann-Kendall, Sen's Slope, STL Decomposition.
2. Anomaly Detection: Change Point Detection (Pelt), Spike Counting.
3. Spatial Statistics: Getis-Ord Gi* (Hotspot Analysis), Hotspot Direction.
4. Periodicity: Autocorrelation (ACF), Seasonality Detection.
"""

import os
from typing import Any, Dict, List, Optional, Union
from ..core.registry import ToolSpec
import numpy as np
import json

# ------------------------------------------------------------------------------
# Lazy Imports & Helpers
# ------------------------------------------------------------------------------

def _lazy_stats_deps():
    try:
        import numpy as np
        from scipy.stats import norm
        return np, norm
    except ImportError:
        raise ImportError("Missing dependencies. Please install: pip install numpy scipy")


def _lazy_ts_deps():
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.stattools import acf
        import ruptures as rpt
        return pd, STL, acf, rpt
    except ImportError:
        raise ImportError("Missing dependencies. Please install: pip install pandas statsmodels ruptures")


def _lazy_raster_deps():
    try:
        import rasterio
        import numpy as np
        from scipy.ndimage import convolve
        return rasterio, np, convolve
    except ImportError:
        raise ImportError("Missing dependencies. Please install: pip install rasterio scipy numpy")


# ------------------------------------------------------------------------------
# Time Series Trend Analysis
# ------------------------------------------------------------------------------

def _parse_input(input_data):
    “””
    Smart parser: handles List, String, JSON String, and even List[String].
    “””
    if input_data is None:
        return None

    # Unwrap the single-element-list-of-string pattern, e.g. ['[1,2,3]']
    if isinstance(input_data, list):
        if len(input_data) == 1 and isinstance(input_data[0], str):
            input_data = input_data[0]

    # 1. String handling (JSON or CSV)
    if isinstance(input_data, str):
        input_data = input_data.strip()
        # Case A: JSON format “[1, 2, 3]”
        if input_data.startswith(“[“) and input_data.endswith(“]”):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                # JSON parse failed; strip brackets and split manually
                input_data = input_data.strip(“[]”).split(“,”)
        # Case B: plain comma-separated “1, 2, 3”
        else:
            input_data = input_data.split(“,”)

    # 2. Convert to float array
    try:
        return np.asarray(input_data, dtype=float)
    except Exception as e:
        raise ValueError(f”Cannot convert data to numeric array: {input_data}. Error: {str(e)}”)

def compute_linear_trend_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    “””Computes the linear trend (slope and intercept) of a time series.”””
    try:
        y = _parse_input(arguments.get(“y”))
        x = _parse_input(arguments.get(“x”))
    except ValueError as e:
        return {“success”: False, “error”: str(e)}

    if x is None:
        x = np.arange(len(y), dtype=float)

    if len(x) != len(y):
        return {
            “success”: False,
            “error”: f”x and y must have the same length (x: {len(x)}, y: {len(y)})”
        }

    try:
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception as e:
        return {“success”: False, “error”: f”Fitting computation failed: {str(e)}”}

    trend_desc = "no trend"
    if a > 1e-6:
        trend_desc = "upward"
    elif a < -1e-6:
        trend_desc = "downward"

    return {
        "slope": float(a),
        "intercept": float(b),
        "trend": trend_desc
    }


def mann_kendall_test_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Perform the non-parametric Mann-Kendall trend test."""
    np, norm = _lazy_stats_deps()

    x = np.asarray(arguments["values"])
    n = len(x)
    if n < 2:
        return {"trend": "insufficient data"}

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))

    # Variance calculation with tie correction
    unique_x, counts = np.unique(x, return_counts=True)
    if n == len(unique_x):
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p_value = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    alpha = 0.05
    if p_value < alpha:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no trend"

    return {
        "trend": trend,
        "p_value": float(p_value),
        "z_score": float(z),
        "tau": float(tau)
    }


def sens_slope_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute Sen's Slope estimator (robust median slope)."""
    np, _ = _lazy_stats_deps()

    x = np.asarray(arguments["values"])
    n = len(x)
    if n < 2:
        raise ValueError("At least two data points are required.")

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope_ij = (x[j] - x[i]) / (j - i)
            slopes.append(slope_ij)

    median_slope = np.median(slopes)

    return {
        "slope": float(median_slope),
        "pairwise_slopes_count": len(slopes)
    }


# ------------------------------------------------------------------------------
# Decomposition & Anomaly Detection
# ------------------------------------------------------------------------------

def stl_decompose_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Apply Seasonal-Trend decomposition using LOESS (STL)."""
    pd, STL, _, _ = _lazy_ts_deps()

    values = arguments["values"]
    period = arguments["period"]
    robust = arguments.get("robust", True)

    series = pd.Series(values)
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()

    return {
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "resid": result.resid.tolist()
    }


def detect_change_points_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Detect structural change points using PELT algorithm."""
    _, _, _, rpt = _lazy_ts_deps()
    import numpy as np  # Ensure numpy is available

    signal = np.asarray(arguments["values"])
    model = arguments.get("model", "l2")  # "l1", "l2", "rbf"
    penalty = float(arguments.get("penalty", 10))

    algo = rpt.Pelt(model=model).fit(signal)
    change_points = algo.predict(pen=penalty)

    # Convert np.int to int for JSON serialization
    return {"change_points": [int(cp) for cp in change_points]}


def count_spikes_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Count upward spikes in a sequence greater than a threshold."""
    np, _ = _lazy_stats_deps()

    values = np.array(arguments["values"], dtype=np.float32)
    threshold = float(arguments.get("threshold", 0.1))

    valid_values = values[~np.isnan(values)]

    if len(valid_values) < 2:
        return {"spike_count": 0}

    diffs = valid_values[1:] - valid_values[:-1]
    spike_count = np.sum(diffs > threshold)

    return {"spike_count": int(spike_count)}


# ------------------------------------------------------------------------------
# Periodicity Analysis
# ------------------------------------------------------------------------------

def autocorrelation_function_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute Autocorrelation Function (ACF)."""
    np, _ = _lazy_stats_deps()

    x = np.asarray(arguments["values"])
    nlags = arguments.get("nlags", 20)

    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)

    acf_vals = []
    if var == 0:
        acf_vals = [1.0] + [0.0] * nlags
    else:
        for lag in range(nlags + 1):
            if lag == 0:
                acf_vals.append(1.0)
            elif n - lag > 0:
                # Optimized dot product for lag correlation
                val = np.dot(x[:-lag], x[lag:]) / (n - lag) / var
                acf_vals.append(float(val))
            else:
                acf_vals.append(0.0)

    return {"acf": acf_vals}


def detect_seasonality_acf_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Detect dominant seasonality period using ACF."""
    _, _, acf, _ = _lazy_ts_deps()
    import numpy as np

    values = np.asarray(arguments["values"])
    min_acf = float(arguments.get("min_acf", 0.3))
    n = len(values)

    nlags = min(n // 3, 40)
    if nlags < 2:
        return {"period": None, "message": "Data series too short"}

    try:
        acf_values = acf(values, nlags=nlags, fft=True)
    except Exception:
        return {"period": None, "message": "ACF computation failed"}

    # Find peak beyond lag 1
    max_val = 0
    best_lag = None

    for lag in range(2, len(acf_values)):
        if acf_values[lag] > max_val and acf_values[lag] > min_acf:
            max_val = acf_values[lag]
            best_lag = lag

    if best_lag:
        return {"period": int(best_lag), "acf_score": float(max_val)}
    else:
        return {"period": None, "message": "Data is not cyclical"}


# ------------------------------------------------------------------------------
# 4. Spatial Analysis (Raster Based)
# ------------------------------------------------------------------------------

def getis_ord_gi_star_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute Getis-Ord Gi* statistic for local spatial autocorrelation."""
    rasterio, np, convolve = _lazy_raster_deps()

    image_path = arguments["image_path"]
    output_path = arguments["output_path"]
    weight_matrix = arguments["weight_matrix"]  # List[List[float]]

    # Open and read image
    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float64)
        profile = src.profile
        # Treat nodata as NaN or 0? Earth-agent puts 0 for nan.
        img[np.isnan(img)] = 0

    W = np.array(weight_matrix, dtype=np.float64)
    W_sum = np.sum(W)
    if W_sum == 0:
        raise ValueError("Sum of weights must not be zero.")

    n = img.size
    x_bar = np.mean(img)
    s = np.std(img)

    if s == 0:
        raise ValueError("Image has zero standard deviation (constant value).")

    # Numerator: local sum
    numerator = convolve(img, W, mode='constant', cval=0) - (x_bar * W_sum)

    # Denominator
    W_sq_sum = np.sum(W ** 2)
    S_denom = s * np.sqrt(((n * W_sq_sum) - (W_sum ** 2)) / (n - 1))

    gi_star = numerator / (S_denom + 1e-9)

    # Save result
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=None)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(gi_star.astype(rasterio.float32), 1)

    return {"status": "success", "output_path": output_path}


def analyze_hotspot_direction_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Analyze the main directional concentration of hotspots (pixels=1)."""
    rasterio, np, _ = _lazy_raster_deps()

    image_path = arguments["hotspot_map_path"]

    with rasterio.open(image_path) as src:
        data = src.read(1)

    # Find hotspots (value == 1)
    # Note: earth-agent uses explicit == 1.
    y_indices, x_indices = np.where(data == 1)

    if len(y_indices) == 0:
        return {"direction": "no hotspots found"}

    center_y, center_x = data.shape[0] // 2, data.shape[1] // 2

    # Calculate directions
    # Image coords: y increases downwards.
    # North: y < center_y (so center_y - y > 0)
    # South: y > center_y

    dy = center_y - y_indices
    dx = x_indices - center_x

    counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}

    # Vectorized logic for speed
    # North/South dominance: |dy| > |dx|
    ns_mask = np.abs(dy) > np.abs(dx)
    ew_mask = ~ns_mask

    counts['north'] = np.sum((dy > 0) & ns_mask)
    counts['south'] = np.sum((dy < 0) & ns_mask)
    counts['east'] = np.sum((dx > 0) & ew_mask)
    counts['west'] = np.sum((dx < 0) & ew_mask)

    dominant = max(counts, key=counts.get)

    return {
        "direction": dominant,
        "counts": counts
    }


# ------------------------------------------------------------------------------
# Descriptive Statistics for Time Series
# ------------------------------------------------------------------------------

def coefficient_of_variation_handler(arguments: dict, context: dict, account=None):
    """
    Compute the Coefficient of Variation (CV = std / mean) for a numeric list.
    A normalized measure of dispersion: higher CV → more variable.
    Returns NaN if mean == 0.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required. Install: pip install numpy")

    x = arguments.get("x", [])
    ddof = int(arguments.get("ddof", 1))

    arr = np.asarray(x, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr, ddof=ddof)

    cv = float("nan") if mean == 0 else float(std / mean)
    return {"cv": cv}


def skewness_handler(arguments: dict, context: dict, account=None):
    """
    Compute the skewness (distribution asymmetry) of a numeric list.
    Positive → right tail; negative → left tail; ~0 → symmetric.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required. Install: pip install numpy")

    x = arguments.get("x", [])
    bias = bool(arguments.get("bias", True))

    arr = np.asarray(x, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=0 if bias else 1)

    if std == 0:
        return {"skewness": 0.0}

    m3 = np.mean((arr - mean) ** 3)
    skew = float(m3 / std ** 3)

    if not bias and n > 2:
        skew *= float(np.sqrt(n * (n - 1))) / (n - 2)

    return {"skewness": skew}


def kurtosis_handler(arguments: dict, context: dict, account=None):
    """
    Compute the kurtosis (tailedness) of a numeric list.
    With fisher=True (default) returns excess kurtosis (normal dist → 0).
    With fisher=False returns regular kurtosis (normal dist → 3).
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required. Install: pip install numpy")

    x = arguments.get("x", [])
    fisher = bool(arguments.get("fisher", True))

    arr = np.asarray(x, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)

    if std == 0:
        return {"kurtosis": 0.0}

    m4 = np.mean((arr - mean) ** 4)
    kurt = float(m4 / std ** 4)

    if fisher:
        kurt -= 3.0

    return {"kurtosis": kurt}


def percentage_change_handler(arguments: dict, context: dict, account=None):
    """
    Compute percentage change: (new - old) / old × 100.
    Positive → increase; negative → decrease.
    Returns +inf if old == 0.
    """
    old = float(arguments["old"])
    new = float(arguments["new"])

    if old == 0:
        return {"percentage_change": float("inf")}
    return {"percentage_change": float((new - old) / old * 100)}


# ------------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    """Register all geoanalysis tools."""
    registrar.toolkit(
        name="geoanalysis",
        description="Time-series and spatial statistical analysis tools (Trend, Seasonality, Hotspots).",
        version="0.1.0"
    )

    # 1. Linear Trend
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.compute_linear_trend",
            name="Compute Linear Trend",
            description="Compute slope and intercept of a time series using least squares.",
            parameters={
                "type": "object",
                "properties": {
                    "y": {"type": "array", "items": {"type": "number"},
                          "description": "Dependent variable (time series)."},
                    "x": {"type": "array", "items": {"type": "number"},
                          "description": "Independent variable (time indices). Optional."}
                },
                "required": ["y"]
            },
            requires_connection=False
        ),
        compute_linear_trend_handler
    )

    # 2. Mann-Kendall
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.mann_kendall_test",
            name="Mann-Kendall Trend Test",
            description="Perform non-parametric Mann-Kendall trend test.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Input time series values."}
                },
                "required": ["values"]
            },
            requires_connection=False
        ),
        mann_kendall_test_handler
    )

    # 3. Sen's Slope
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.sens_slope",
            name="Sen's Slope Estimator",
            description="Compute robust median slope of a time series.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["values"]
            },
            requires_connection=False
        ),
        sens_slope_handler
    )

    # 4. STL Decomposition
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.stl_decompose",
            name="STL Decomposition",
            description="Decompose time series into trend, seasonal, and residual components.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "period": {"type": "integer", "description": "Seasonality period (e.g. 12)."},
                    "robust": {"type": "boolean", "default": True}
                },
                "required": ["values", "period"]
            },
            requires_connection=False
        ),
        stl_decompose_handler
    )

    # 5. Change Points
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.detect_change_points",
            name="Detect Change Points",
            description="Detect structural change points using PELT algorithm.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "model": {"type": "string", "default": "l2", "description": "Cost model (l1, l2, rbf)."},
                    "penalty": {"type": "number", "default": 10}
                },
                "required": ["values"]
            },
            requires_connection=False
        ),
        detect_change_points_handler
    )

    # 6. ACF
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.autocorrelation_function",
            name="Compute ACF",
            description="Compute Autocorrelation Function.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "nlags": {"type": "integer", "default": 20}
                },
                "required": ["values"]
            },
            requires_connection=False
        ),
        autocorrelation_function_handler
    )

    # 7. Seasonality Detection
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.detect_seasonality_acf",
            name="Detect Seasonality",
            description="Detect dominant seasonal period using ACF.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "min_acf": {"type": "number", "default": 0.3}
                },
                "required": ["values"]
            },
            requires_connection=False
        ),
        detect_seasonality_acf_handler
    )

    # 8. Getis-Ord Gi*
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.getis_ord_gi_star",
            name="Getis-Ord Gi* Hotspot",
            description="Compute Getis-Ord Gi* statistic for spatial hotspots.",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Input raster path."},
                    "output_path": {"type": "string", "description": "Output raster path."},
                    "weight_matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Spatial weight kernel matrix (e.g. [[1,1,1],[1,0,1],[1,1,1]])."
                    }
                },
                "required": ["image_path", "output_path", "weight_matrix"]
            },
            requires_connection=False
        ),
        getis_ord_gi_star_handler
    )

    # 9. Hotspot Direction
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.analyze_hotspot_direction",
            name="Analyze Hotspot Direction",
            description="Determine the cardinal direction of hotspot concentration.",
            parameters={
                "type": "object",
                "properties": {
                    "hotspot_map_path": {"type": "string", "description": "Binary hotspot map (1=hotspot)."}
                },
                "required": ["hotspot_map_path"]
            },
            requires_connection=False
        ),
        analyze_hotspot_direction_handler
    )

    # 10. Count Spikes
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.count_spikes",
            name="Count Spikes",
            description="Count number of upward spikes exceeding a threshold.",
            parameters={
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                    "threshold": {"type": "number", "default": 0.1}
                },
                "required": ["values"]
            },
            requires_connection=False
        ),
        count_spikes_handler
    )

    # 11. Coefficient of Variation
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.coefficient_of_variation",
            name="Coefficient of Variation",
            description="Compute CV = std / mean for a numeric list. Normalized measure of dispersion useful for comparing variability across time series with different scales.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {"type": "number"}, "description": "Input data values."},
                    "ddof": {"type": "integer", "default": 1, "description": "Degrees of freedom for std (0=population, 1=sample). Default 1."}
                },
                "required": ["x"]
            },
            requires_connection=False
        ),
        coefficient_of_variation_handler
    )

    # 12. Skewness
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.skewness",
            name="Skewness",
            description="Compute the skewness (asymmetry) of a numeric list. Positive = right-tailed, negative = left-tailed, ~0 = symmetric.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {"type": "number"}, "description": "Input data values."},
                    "bias": {"type": "boolean", "default": True, "description": "If False, apply Fisher-Pearson bias correction. Default True."}
                },
                "required": ["x"]
            },
            requires_connection=False
        ),
        skewness_handler
    )

    # 13. Kurtosis
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.kurtosis",
            name="Kurtosis",
            description="Compute the kurtosis (tailedness) of a numeric list. With fisher=True (default), returns excess kurtosis where normal distribution = 0.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {"type": "number"}, "description": "Input data values."},
                    "fisher": {"type": "boolean", "default": True, "description": "If True (default), returns excess kurtosis (normal→0). If False, returns regular kurtosis (normal→3)."}
                },
                "required": ["x"]
            },
            requires_connection=False
        ),
        kurtosis_handler
    )

    # 14. Percentage Change
    registrar.tool(
        ToolSpec(
            slug="geoanalysis.percentage_change",
            name="Percentage Change",
            description="Compute percentage change between two values: (new - old) / old × 100. Useful for quantifying temporal change in index values, area measurements, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "old": {"type": "number", "description": "Original (baseline) value."},
                    "new": {"type": "number", "description": "New (comparison) value."}
                },
                "required": ["old", "new"]
            },
            requires_connection=False
        ),
        percentage_change_handler
    )