"""Granular (de-collapsed) tool aliases.

Terrabox ships generic, parameterized tools (e.g. ``geo_statistics.scalar_arithmetic``
with an ``operation`` arg, ``geo_raster.calculate_index`` for any normalized
difference). The OpenEarth/EarthBench source datasets instead used many specific
named tools (``multiply``, ``calculate_ndwi`` …). This toolkit registers the
specific names that actually appear in the dataset as thin aliases over the
generic handlers, so a "de-collapsed" dataset can exercise fine-grained tool
selection (the discriminating choice lives in the tool name, not an argument).

These tools are always registered (harmless extras). The COLLAPSED dataset never
references them, so collapsed experiments are unaffected; the DE-COLLAPSED dataset
(`data/fixdata_decollapse`) references them and a rollout binds exactly these.
Only the granular tools that occur in the dataset are defined here (data-driven).
"""
from __future__ import annotations

from typing import Any, Dict

from ..core.registry import ToolSpec
from ._batch_util import auto_batch
from .georaster import calculate_normalized_difference_handler
from .geo_statistics import (
    scalar_arithmetic_handler,
    batch_raster_stats_handler,
    mean_of_means_handler,
)
from .earth_sci import stats_lst_by_ndvi_handler


# ── parameter schema fragments (aligned with the generic handlers) ────────────
_IDX_PARAMS = {
    "type": "object",
    "required": ["band_a_path", "band_b_path", "output_path"],
    "properties": {
        "band_a_path": {"type": "string", "description": "Path to the first band."},
        "band_b_path": {"type": "string", "description": "Path to the second band."},
        "output_path": {"type": "string", "description": "Output GeoTIFF path."},
    },
}
_SCALAR_PARAMS = {
    "type": "object",
    "required": ["a", "b"],
    "properties": {
        "a": {"type": "number", "description": "First operand."},
        "b": {"type": "number", "description": "Second operand."},
    },
}
_BATCH_PARAMS = {
    "type": "object",
    "required": ["input_paths"],
    "properties": {
        "input_paths": {"type": "array", "items": {"type": "string"},
                         "description": "List of single-band raster paths."},
    },
}
# Numeric list-reduction tools (mean / argmax / argmin) operate on a list of
# NUMBERS (e.g. the per-image means from a previous step), NOT raster paths.
_NUMLIST_PARAMS = {
    "type": "object",
    "required": ["values"],
    "properties": {
        "values": {"type": "array", "items": {"type": "number"},
                   "description": "List of numeric values to reduce."},
    },
}


def _num_values(arguments):
    raw = (arguments or {}).get("values")
    if raw is None:
        raw = (arguments or {}).get("input_paths") or (arguments or {}).get("x") or []
    out = []
    for v in raw:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _mean_list_handler(arguments, context, account):
    xs = _num_values(arguments)
    if not xs:
        return {"status": "error", "message": "no numeric values"}
    return {"mean": sum(xs) / len(xs), "count": len(xs)}


def _max_value_and_index_handler(arguments, context, account):
    xs = _num_values(arguments)
    if not xs:
        return {"status": "error", "message": "no numeric values"}
    i = max(range(len(xs)), key=lambda k: xs[k])
    return {"max": xs[i], "index": i}


def _min_value_and_index_handler(arguments, context, account):
    xs = _num_values(arguments)
    if not xs:
        return {"status": "error", "message": "no numeric values"}
    i = min(range(len(xs)), key=lambda k: xs[k])
    return {"min": xs[i], "index": i}
_LST_PARAMS = {
    "type": "object",
    "required": ["red_path", "nir_path", "lst_path"],
    "properties": {
        "red_path": {"type": "string", "description": "Red band raster for NDVI."},
        "nir_path": {"type": "string", "description": "NIR band raster for NDVI."},
        "lst_path": {"type": "string", "description": "Land surface temperature raster (Kelvin)."},
        "threshold": {"type": "number", "description": "NDVI threshold (default 0.3)."},
        "mode": {"type": "string", "description": "'above' (NDVI>=thr) or 'below'."},
    },
}


def _with(handler, **preset):
    """Return a handler that injects preset args then delegates to ``handler``."""
    def _h(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
        merged = dict(arguments or {})
        for k, v in preset.items():
            merged.setdefault(k, v)
        return handler(merged, context, account)
    return _h


def setup(registrar) -> None:
    # ── geo_raster: specific spectral indices (reuse the generic ND handler) ──
    _idx = [
        ("geo_raster.calculate_ndwi", "Calculate NDWI",
         "Normalized Difference Water Index NDWI=(Green-NIR)/(Green+NIR); band_a=Green, band_b=NIR. Returns {output_path} (single-band NDWI GeoTIFF). Path arguments also accept a parallel list for batch processing (returns {output_paths, results})."),
        ("geo_raster.calculate_ndti", "Calculate NDTI",
         "Normalized Difference Turbidity Index NDTI=(Red-Green)/(Red+Green); band_a=Red, band_b=Green. Returns {output_path} (single-band NDTI GeoTIFF). Path arguments also accept a parallel list for batch processing (returns {output_paths, results})."),
        ("geo_raster.calculate_ndsi", "Calculate NDSI",
         "Normalized Difference Snow Index NDSI=(Green-SWIR)/(Green+SWIR); band_a=Green, band_b=SWIR. Returns {output_path} (single-band NDSI GeoTIFF). Path arguments also accept a parallel list for batch processing (returns {output_paths, results})."),
    ]
    for slug, name, desc in _idx:
        registrar.tool(
            ToolSpec(slug=slug, name=name, description=desc, parameters=_IDX_PARAMS,
                     requires_connection=False),
            auto_batch(_with(calculate_normalized_difference_handler,
                  index_name=slug.rsplit(".", 1)[-1].replace("calculate_", "").upper())),
        )

    # ── geo_statistics: specific scalar arithmetic ops ───────────────────────
    _scalar = [
        ("geo_statistics.subtract", "Subtract", "subtract",
         "Subtract two scalars (a - b). Returns {result, operation, a, b}."),
        ("geo_statistics.divide", "Divide", "divide",
         "Divide two scalars (a / b). Returns {result, operation, a, b}."),
        ("geo_statistics.multiply", "Multiply", "multiply",
         "Multiply two scalars (a * b). Returns {result, operation, a, b}."),
    ]
    for slug, name, op, desc in _scalar:
        registrar.tool(
            ToolSpec(slug=slug, name=name, description=desc, parameters=_SCALAR_PARAMS,
                     requires_connection=False),
            _with(scalar_arithmetic_handler, operation=op),
        )

    # ── geo_statistics: specific batch-image statistics ──────────────────────
    _batch = [
        ("geo_statistics.batch_image_mean", "Batch Image Mean", {"stat": "mean"},
         "Per-image mean for a list of rasters. Returns {per_image:[{mean,...}], aggregate}."),
        ("geo_statistics.batch_image_max", "Batch Image Max", {"stat": "max"},
         "Per-image max for a list of rasters. Returns {per_image:[{max,...}], aggregate}."),
        ("geo_statistics.batch_image_sum", "Batch Image Sum", {"stat": "sum"},
         "Per-image sum for a list of rasters. Returns {per_image:[{sum,...}], aggregate}."),
        ("geo_statistics.batch_image_mean_max_min", "Batch Image Mean/Max/Min", {},
         "Per-image mean, max and min for a list of rasters. Returns {per_image:[{mean,max,min,...}], aggregate}."),
    ]
    for slug, name, preset, desc in _batch:
        registrar.tool(
            ToolSpec(slug=slug, name=name, description=desc, parameters=_BATCH_PARAMS,
                     requires_connection=False),
            _with(batch_raster_stats_handler, **preset),
        )

    # ── geo_statistics: numeric list-reduction tools (operate on numbers) ─────
    registrar.tool(
        ToolSpec(slug="geo_statistics.mean", name="Mean",
                 description="Arithmetic mean of a list of numeric values. Returns {mean, count}.",
                 parameters=_NUMLIST_PARAMS, requires_connection=False),
        _mean_list_handler,
    )
    registrar.tool(
        ToolSpec(slug="geo_statistics.max_value_and_index", name="Max Value And Index",
                 description="Largest value in a numeric list and its index. Returns {max, index}.",
                 parameters=_NUMLIST_PARAMS, requires_connection=False),
        _max_value_and_index_handler,
    )
    registrar.tool(
        ToolSpec(slug="geo_statistics.min_value_and_index", name="Min Value And Index",
                 description="Smallest value in a numeric list and its index. Returns {min, index}.",
                 parameters=_NUMLIST_PARAMS, requires_connection=False),
        _min_value_and_index_handler,
    )

    # ── earth_sci: specific LST-by-NDVI statistics ───────────────────────────
    registrar.tool(
        ToolSpec(slug="earth_sci.mean_lst_by_ndvi", name="Mean LST by NDVI",
                 description="Mean Land Surface Temperature over pixels in an NDVI range. Returns {result: mean LST (float, Kelvin; null if none), pixel_count}.",
                 parameters=_LST_PARAMS, requires_connection=False),
        _with(stats_lst_by_ndvi_handler, stat="mean"),
    )
    registrar.tool(
        ToolSpec(slug="earth_sci.max_lst_by_ndvi", name="Max LST by NDVI",
                 description="Maximum Land Surface Temperature over pixels in an NDVI range. Returns {result: max LST (float, Kelvin; null if none), pixel_count}.",
                 parameters=_LST_PARAMS, requires_connection=False),
        _with(stats_lst_by_ndvi_handler, stat="max"),
    )
