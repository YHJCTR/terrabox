"""Shared file-upload utilities."""
from __future__ import annotations

import os
import json
import mimetypes
import warnings
import uuid
from pathlib import Path
from typing import Any, List

from fastapi import UploadFile

def _upload_dir(default_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the upload directory, keeping TERRABOX_UPLOAD_DIR as an override."""
    return Path(os.getenv("TERRABOX_UPLOAD_DIR") or default_dir or "./terrabox_uploads")


async def save_upload_files(files: List[UploadFile], upload_dir: str | os.PathLike[str] | None = None) -> List[str]:
    """Save UploadFile objects and return their paths."""
    target_dir = _upload_dir(upload_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for f in files:
        suffix = Path(f.filename or "upload").suffix or ".bin"
        dest = target_dir / f"{uuid.uuid4().hex}{suffix}"
        dest.write_bytes(await f.read())
        saved.append(str(dest))
    return saved


def _file_family(extension: str, mime_type: str | None) -> str:
    if mime_type and mime_type.startswith("image/"):
        return "image"
    if extension in {".tif", ".tiff", ".geotiff", ".img"}:
        return "raster"
    if extension in {".geojson", ".json", ".gpkg", ".shp", ".kml"}:
        return "vector"
    return "unknown"


def _raster_metadata(path: str) -> dict[str, Any]:
    try:
        import rasterio
        from affine import Affine
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as src:
                transform = src.transform
                is_georeferenced = bool(src.crs) or (transform not in {Affine.identity(), Affine.scale(1, -1)})
                return {
                    "readable": True,
                    "driver": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "band_count": src.count,
                    "dtypes": list(src.dtypes),
                    "crs": str(src.crs) if src.crs else None,
                    "is_georeferenced": bool(is_georeferenced),
                }
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def inspect_uploaded_file(path: str, original_name: str | None = None) -> dict[str, Any]:
    """Return dynamic, tool-agnostic facts about an uploaded local file."""
    file_path = Path(path)
    extension = file_path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(original_name or str(file_path))
    raster = _raster_metadata(str(file_path))
    family = _file_family(extension, mime_type)

    capabilities: list[str] = []
    if family == "image":
        capabilities.append("visual_image")
    if raster.get("readable"):
        capabilities.append("raster_readable")
        if raster.get("is_georeferenced"):
            capabilities.append("georeferenced")

    semantic_type = family
    if raster.get("readable") and raster.get("is_georeferenced"):
        semantic_type = "geospatial_raster"
    elif raster.get("readable") and family == "image":
        semantic_type = "raster_readable_image"

    return {
        "path": str(file_path),
        "original_name": original_name or file_path.name,
        "extension": extension,
        "mime_type": mime_type,
        "size_bytes": file_path.stat().st_size if file_path.exists() else None,
        "semantic_type": semantic_type,
        "capabilities": capabilities,
        "raster": raster,
    }


def inspect_uploaded_files(paths: list[str]) -> list[dict[str, Any]]:
    return [inspect_uploaded_file(path) for path in paths]


def uploaded_files_metadata_text(paths: list[str]) -> str:
    metadata = inspect_uploaded_files(paths)
    return json.dumps(metadata, ensure_ascii=False, indent=2)
