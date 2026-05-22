"""Runtime path helpers for uploaded files, tool outputs, and manifests."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RUNTIME_DIR = "tmp/terrabox_runtime"
OUTPUT_KEYS = {"output_path", "output_dir"}
PLACEHOLDER_MARKERS = ("<", ">", "path/to/", "/path/to/", "path_to_")


def generate_execution_id() -> str:
    """Return a stable, sortable execution id for one tool invocation."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"exec_{stamp}_{uuid.uuid4().hex[:8]}"


def runtime_root() -> Path:
    """Return the root runtime directory, creating it if needed."""
    root = Path(os.getenv("TERRABOX_RUNTIME_DIR", DEFAULT_RUNTIME_DIR))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_part(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def toolkit_from_slug(tool_slug: str) -> str:
    return _safe_part(str(tool_slug).split(".", 1)[0], "tool")


def tool_name_from_slug(tool_slug: str) -> str:
    parts = str(tool_slug).split(".", 1)
    return _safe_part(parts[1] if len(parts) > 1 else parts[0], "result")


def prepare_runtime_context(user_id: str, tool_slug: str, execution_id: str | None = None) -> dict[str, str]:
    """Create runtime folders and return metadata for one tool invocation."""
    execution_id = execution_id or generate_execution_id()
    safe_user = _safe_part(user_id, "anonymous")
    toolkit = toolkit_from_slug(tool_slug)
    root = runtime_root()

    upload_dir = root / "uploads" / safe_user / execution_id
    output_dir = root / "outputs" / safe_user / execution_id / toolkit
    artifact_dir = root / "artifacts" / safe_user / execution_id / toolkit
    scratch_dir = root / "scratch" / safe_user / execution_id
    manifest_dir = root / "manifests"

    for path in (upload_dir, output_dir, artifact_dir, scratch_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "execution_id": execution_id,
        "runtime_dir": str(root),
        "upload_dir": str(upload_dir),
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "scratch_dir": str(scratch_dir),
        "manifest_path": str(manifest_dir / f"{execution_id}.json"),
    }


def is_placeholder_path(value: Any) -> bool:
    if not isinstance(value, str):
        return value is None
    text = value.strip()
    if not text or text.lower() in {"none", "null"}:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _extension_for_output(tool_slug: str, param_name: str, prop: Mapping[str, Any] | None = None) -> str:
    description = str((prop or {}).get("description") or "").lower()
    toolkit = toolkit_from_slug(tool_slug)
    if "gpkg" in description or "geopackage" in description:
        return ".gpkg"
    if "json" in description or "track" in description:
        return ".json"
    if any(word in description for word in ("png", "jpg", "jpeg", "image", "annotated")):
        return ".png"
    if "csv" in description:
        return ".csv"
    if "npy" in description:
        return ".npy"
    if "geotiff" in description or "tiff" in description:
        return ".tif"
    if toolkit in {"geo_raster", "earth_sci", "disaster_response", "geoanalysis"}:
        return ".tif"
    if param_name.endswith("_dir"):
        return ""
    return ".out"


def allocate_upload_path(user_id: str, execution_id: str, filename: str | None = None) -> str:
    """Allocate a path under the runtime upload directory."""
    metadata = prepare_runtime_context(user_id, "upload", execution_id=execution_id)
    name = _safe_part(Path(filename or "upload.bin").stem, "upload")
    suffix = Path(filename or "upload.bin").suffix or ".bin"
    return str(Path(metadata["upload_dir"]) / f"{uuid.uuid4().hex}_{name}{suffix}")


def allocate_output_path(
    user_id: str,
    execution_id: str,
    toolkit: str,
    filename: str | None = None,
    suffix: str | None = None,
) -> str:
    """Allocate a tool output file path under the runtime output directory."""
    metadata = prepare_runtime_context(user_id, toolkit, execution_id=execution_id)
    name = _safe_part(Path(filename or "result").stem, "result")
    ext = suffix if suffix is not None else (Path(filename or "").suffix or ".out")
    return str(Path(metadata["output_dir"]) / f"{name}{ext}")


def apply_default_output_paths(
    tool_slug: str,
    inputs: Mapping[str, Any] | None,
    parameters: Mapping[str, Any] | None,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    """Fill missing or placeholder output_path/output_dir inputs from runtime metadata."""
    updated = dict(inputs or {})
    properties = dict((parameters or {}).get("properties") or {})
    tool_name = tool_name_from_slug(tool_slug)
    for key in OUTPUT_KEYS:
        if key not in properties:
            continue
        current = updated.get(key)
        if current is not None and not is_placeholder_path(current):
            continue
        if key == "output_dir":
            out_dir = Path(runtime["output_dir"]) / tool_name
            out_dir.mkdir(parents=True, exist_ok=True)
            updated[key] = str(out_dir)
        else:
            ext = _extension_for_output(tool_slug, key, properties.get(key))
            updated[key] = str(Path(runtime["output_dir"]) / f"{tool_name}{ext}")
            Path(updated[key]).parent.mkdir(parents=True, exist_ok=True)
    return updated


def _path_entry(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
    }


def _collect_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if isinstance(item, str) and any(token in lowered for token in ("path", "dir", "gpkg", "artifact")):
                paths.append(item)
            else:
                paths.extend(_collect_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_collect_paths(item))
    return list(dict.fromkeys(paths))


def write_manifest(
    *,
    execution_id: str,
    user_id: str,
    tool_slug: str,
    runtime: Mapping[str, str],
    inputs: Mapping[str, Any] | None,
    outputs: Mapping[str, Any] | None,
    status: str,
    error: str | None = None,
) -> str:
    """Write a JSON manifest describing one tool execution."""
    manifest_path = Path(runtime["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    input_entries = [_path_entry(path) for path in _collect_paths(inputs or {})]
    output_entries = [_path_entry(path) for path in _collect_paths(outputs or {})]
    total_bytes = sum((entry.get("size_bytes") or 0) for entry in [*input_entries, *output_entries])
    payload = {
        "execution_id": execution_id,
        "user_id": user_id,
        "tool_slug": tool_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "runtime": dict(runtime),
        "inputs": input_entries,
        "outputs": output_entries,
        "total_bytes": total_bytes,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)
