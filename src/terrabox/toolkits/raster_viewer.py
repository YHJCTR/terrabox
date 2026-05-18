"""Raster viewer toolkit for inspecting GeoTIFF and TIFF outputs."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable

from ..core.registry import ToolSpec


DEFAULT_PERCENTILES = (0, 2, 5, 25, 50, 75, 95, 98, 100)


def _round_float(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _round_list(values: Iterable[Any], digits: int = 6) -> list[float | None]:
    return [_round_float(value, digits) for value in values]


def _band_stats(src: Any, band_index: int, percentiles: Iterable[float], histogram_bins: int) -> Dict[str, Any]:
    import numpy as np

    data = src.read(band_index, masked=True)
    if np.ma.isMaskedArray(data):
        mask = np.ma.getmaskarray(data)
        values = np.asarray(data.compressed(), dtype="float64")
    else:
        array = np.asarray(data)
        mask = np.zeros(array.shape, dtype=bool)
        values = array.reshape(-1).astype("float64")

    finite_values = values[np.isfinite(values)]
    total_pixels = int(data.size)
    valid_pixels = int(finite_values.size)
    nodata_pixels = total_pixels - valid_pixels

    stats: Dict[str, Any] = {
        "index": band_index,
        "description": src.descriptions[band_index - 1] or None,
        "dtype": str(src.dtypes[band_index - 1]),
        "colorinterp": str(src.colorinterp[band_index - 1]) if src.colorinterp else None,
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nodata_pixels": nodata_pixels,
        "masked_pixels": int(mask.sum()) if mask is not None else 0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "percentiles": {},
        "histogram": None,
    }

    if valid_pixels == 0:
        return stats

    stats.update(
        {
            "min": _round_float(np.min(finite_values)),
            "max": _round_float(np.max(finite_values)),
            "mean": _round_float(np.mean(finite_values)),
            "std": _round_float(np.std(finite_values)),
            "percentiles": {
                str(_round_float(p, 3)).rstrip("0").rstrip("."): _round_float(v)
                for p, v in zip(percentiles, np.percentile(finite_values, list(percentiles)))
            },
        }
    )

    if histogram_bins > 0 and stats["min"] is not None and stats["max"] is not None:
        if stats["min"] == stats["max"]:
            stats["histogram"] = {
                "bins": [stats["min"], stats["max"]],
                "counts": [valid_pixels],
            }
        else:
            counts, bins = np.histogram(finite_values, bins=histogram_bins)
            stats["histogram"] = {
                "bins": _round_list(bins),
                "counts": [int(count) for count in counts],
            }

    return stats


def inspect_geotiff_handler(arguments: Dict[str, Any], context: Any = None, account: Any = None) -> Dict[str, Any]:
    """Inspect a GeoTIFF/TIFF file and return metadata plus per-band statistics."""
    import rasterio

    raster_path = arguments.get("raster_path") or arguments.get("path")
    if not raster_path:
        return {"status": "error", "message": "Missing required parameter: raster_path"}

    max_bands = int(arguments.get("max_bands", 20))
    histogram_bins = int(arguments.get("histogram_bins", 16))
    include_tags = bool(arguments.get("include_tags", True))
    include_histogram = bool(arguments.get("include_histogram", True))
    percentiles = arguments.get("percentiles") or list(DEFAULT_PERCENTILES)
    percentiles = [float(p) for p in percentiles]
    if not include_histogram:
        histogram_bins = 0

    try:
        with rasterio.open(raster_path) as src:
            band_limit = min(src.count, max_bands)
            bands = [
                _band_stats(src, band_index, percentiles, histogram_bins)
                for band_index in range(1, band_limit + 1)
            ]

            transform = src.transform
            return {
                "status": "success",
                "filename": os.path.basename(raster_path),
                "driver": src.driver,
                "width": int(src.width),
                "height": int(src.height),
                "band_count": int(src.count),
                "inspected_band_count": int(band_limit),
                "dtypes": [str(dtype) for dtype in src.dtypes],
                "crs": str(src.crs) if src.crs else None,
                "bounds": _round_list([src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]),
                "resolution": _round_list(src.res),
                "transform": _round_list(transform.to_gdal()),
                "nodata": _round_float(src.nodata),
                "units": list(src.units or []),
                "tags": src.tags() if include_tags else {},
                "bands": bands,
            }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def setup(registrar):
    registrar.toolkit(
        "raster_viewer",
        "Inspect GeoTIFF/TIFF outputs and summarize raster content.",
        version="1.0",
    )

    registrar.tool(
        ToolSpec(
            slug="raster_viewer.inspect_geotiff",
            name="Inspect GeoTIFF",
            description=(
                "Inspect a GeoTIFF or TIFF raster file and return metadata, bounds, "
                "resolution, data types, per-band valid/nodata pixel counts, summary "
                "statistics, percentiles, and optional histograms."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "raster_path": {
                        "type": "string",
                        "description": "Path to the GeoTIFF or TIFF file to inspect.",
                    },
                    "max_bands": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "description": "Maximum number of bands to inspect.",
                    },
                    "include_histogram": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include per-band histograms.",
                    },
                    "histogram_bins": {
                        "type": "integer",
                        "default": 16,
                        "minimum": 0,
                        "description": "Number of bins for each band histogram.",
                    },
                    "include_tags": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include dataset-level raster tags.",
                    },
                    "percentiles": {
                        "type": "array",
                        "items": {"type": "number"},
                        "default": list(DEFAULT_PERCENTILES),
                        "description": "Percentiles to compute for each band.",
                    },
                },
                "required": ["raster_path"],
            },
            requires_connection=False,
        ),
        inspect_geotiff_handler,
    )
