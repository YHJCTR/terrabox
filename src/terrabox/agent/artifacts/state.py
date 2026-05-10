"""Runtime artifact state used by artifact-aware tool disclosure."""

from __future__ import annotations

import re
from typing import Any


FILE_PARAM_MARKERS = ("path", "gpkg", "image", "raster", "file")
OUTPUT_PARAM_NAMES = {"output_path", "result_path", "preview_path", "artifact_path"}
_KNOWN_FILE_SUFFIX_RE = re.compile(
    r"\.(gpkg|tif|tiff|png|jpg|jpeg|json|geojson|csv|txt)$",
    re.IGNORECASE,
)


def extract_paths_from_text(text: str) -> list[str]:
    candidates = re.findall(
        r"(?<![A-Za-z0-9])(?:(?:/|\.{1,2}/|tmp/)[^\s,;:'\")\]]+)",
        text or "",
    )
    return [p for p in dict.fromkeys(candidates) if looks_like_path(p)]


def looks_like_path(value: str) -> bool:
    if not value or any(ch.isspace() for ch in value):
        return False
    if _KNOWN_FILE_SUFFIX_RE.search(value):
        return True
    if value.startswith("/"):
        return "/" in value[1:]
    return (
        value.startswith(("./", "../", "tmp/"))
        or bool(_KNOWN_FILE_SUFFIX_RE.search(value))
    )


def infer_artifact_kind(path: str, key: str = "") -> str:
    lowered = f"{key} {path}".lower()
    if lowered.endswith(".gpkg") or "gpkg" in key.lower():
        return "gpkg"
    if lowered.endswith((".tif", ".tiff")) or any(token in lowered for token in ("raster", "band_")):
        return "raster"
    if lowered.endswith((".png", ".jpg", ".jpeg")) or "image" in lowered:
        return "image"
    if lowered.endswith((".json", ".geojson", ".csv")) or any(token in lowered for token in ("table", "data")):
        return "table"
    return "file"


def initial_artifact_state(question: str, image_paths: list[str] | None = None) -> dict[str, Any]:
    """Build the artifact state from explicit user inputs.

    This state is intentionally dataset-agnostic. It only records artifacts the
    user actually supplied, such as image/raster paths embedded in the request.
    """
    artifacts: list[dict[str, Any]] = []
    for path in image_paths or []:
        artifacts.append({"kind": "image", "path": path, "source": "task_image"})
    for path in extract_paths_from_text(question):
        artifacts.append({"kind": infer_artifact_kind(path), "path": path, "source": "question"})
    return {
        "artifacts": artifacts,
        "layers": [],
        "results": [],
        "failed_calls": [],
        "successful_calls": [],
        "last_error": None,
    }


def artifact_paths(state: dict[str, Any], kind: str | None = None) -> list[str]:
    paths: list[str] = []
    for artifact in state.get("artifacts", []):
        if kind and artifact.get("kind") != kind:
            continue
        path = artifact.get("path")
        if path:
            paths.append(path)
    return list(dict.fromkeys(paths))


def artifact_paths_for_param(state: dict[str, Any], name: str) -> list[str]:
    lowered = name.lower()
    if lowered == "gpkg":
        return artifact_paths(state, "gpkg")
    if "image" in lowered:
        return artifact_paths(state, "image")
    if "raster" in lowered or "band" in lowered:
        return artifact_paths(state, "raster")
    return artifact_paths(state)


def current_gpkg(state: dict[str, Any]) -> str | None:
    gpkg_paths = artifact_paths(state, "gpkg")
    return gpkg_paths[-1] if gpkg_paths else None


def record_artifact_once(state: dict[str, Any], artifact: dict[str, Any]) -> None:
    path = artifact.get("path")
    if not path:
        return
    key = (artifact.get("kind"), path)
    existing = {(a.get("kind"), a.get("path")) for a in state.get("artifacts", [])}
    if key not in existing:
        state.setdefault("artifacts", []).append(artifact)


def artifact_state_text(state: dict[str, Any]) -> str:
    """Render compact state text injected into each artifact-progressive turn."""
    artifacts = state.get("artifacts", [])
    layers = state.get("layers", [])
    results = state.get("results", [])[-3:]
    lines = ["Current runtime state:"]
    if artifacts:
        lines.append("Artifacts:")
        for artifact in artifacts[-8:]:
            lines.append(
                f"- {artifact.get('kind', 'file')}: {artifact.get('path')} "
                f"(from {artifact.get('source', 'unknown')})"
            )
    else:
        lines.append("Artifacts: none")
    if layers:
        lines.append("Vector/raster layers:")
        for layer in layers[-8:]:
            lines.append(
                f"- {layer.get('name')} in {layer.get('container', 'unknown')} "
                f"(from {layer.get('source', 'unknown')})"
            )
    else:
        lines.append("Vector/raster layers: none")
    if state.get("last_error"):
        lines.append(f"Last tool error: {state['last_error']}")
    failed = state.get("failed_calls", [])[-3:]
    if failed:
        lines.append("Recent failed calls:")
        for item in failed:
            lines.append(f"- {item.get('tool')}: {item.get('args')} -> {item.get('error')}")
    if results:
        lines.append("Recent tool results:")
        for result in results:
            lines.append(f"- {result.get('tool')}: {result.get('summary')}")
    return "\n".join(lines)


def progress_signature(state: dict[str, Any]) -> tuple[int, int, int]:
    """Return coarse state counts used to detect whether a tool made progress."""
    return (
        len(state.get("artifacts", [])),
        len(state.get("layers", [])),
        len(state.get("successful_calls", [])),
    )
