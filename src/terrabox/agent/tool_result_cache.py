"""Content-addressed result cache for slow, deterministic tools.

Targets the only genuinely slow tools in the OEA toolset:
  * GPU perception models (instructsam / sam2 / strip_rcnn / vlm_analyze / ...):
    each call otherwise pays docker cold-start + model load + inference.
  * osm_gis.* network calls (Overpass / Nominatim / OSRM / STAC): network-bound.
(compute.* and the CPU drawing tools are sub-second → not cached; bing_search has
its own .db cache → left as-is.)

A cache HIT returns the stored text AND restores any produced artifact files to
their original paths, and is checked at the OUTERMOST entry (AgentToolExecutor.
execute) BEFORE any service/model is started — so a hit pays ~0 GPU/network cost.

Key = sha256( slug + every non-output argument, with file-path args replaced by a
sha256 of their CONTENT ). So a "hit" requires the tool, the input image/raster
CONTENT, and ALL parameters to match — not merely the tool name. Filename aliasing
(same name, different bytes) cannot collide; identical bytes under different names
still hit. The eval set is fixed, so base/reflection/SFT runs share hits via one
persistent dir.

Only SUCCESSFUL results are stored (error/timeout/OOM outputs are never cached, so
the transient-retry layer is not poisoned). Output is content-deterministic for
these tools, so caching is methodologically neutral (returns what the tool would
have returned, just faster) and does not change tool-F1.

Env:
  TERRABOX_TOOL_RESULT_CACHE       "0"/"false"/"off" disables (default ON)
  TERRABOX_TOOL_RESULT_CACHE_DIR   default ~/.verl_cache/tool_result_cache
  TERRABOX_TOOL_RESULT_CACHE_SLUGS comma list to override the default allowlist
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Args that say WHERE to write output, not WHAT to compute → excluded from the key.
_OUTPUT_KEYS = {"artifact_path", "output_path", "result_path", "preview_path", "gpkg"}

_PERCEPTION = {
    "geo_perception.ocr_extract", "geo_perception.vlm_analyze",
    "geo_perception.region_attribute_description", "geo_perception.instructsam",
    "geo_perception.change_os_detect", "geo_perception.strip_rcnn_detect",
    "geo_perception.sam2_segment", "geo_perception.count_given_object",
}

# Markers that mean the result is a failure → never cache (else retry hits a frozen error).
_ERROR_MARKERS = (
    "tool execution error", "tool blocked by runtime guard",
    '"status": "error"', '"status":"error"', "out of memory", "cuda out of memory",
    "timed out", "timeout", "no valid connection", "no such container",
    "container exited", "traceback (most recent", "connection error",
    "connection refused", "connection reset", "max retries exceeded",
    "service unavailable", "bad gateway", "gateway timeout",
)

_LOCK = threading.Lock()
_HASH_CACHE: dict[str, tuple[float, int, str]] = {}


def _enabled() -> bool:
    return os.environ.get("TERRABOX_TOOL_RESULT_CACHE", "").strip().lower() not in ("0", "false", "off")


def _cache_dir() -> Path:
    v = os.environ.get("TERRABOX_TOOL_RESULT_CACHE_DIR", "").strip()
    return Path(os.path.expanduser(v)) if v else (Path.home() / ".verl_cache" / "tool_result_cache")


def _slug_allowed(slug: str) -> bool:
    override = os.environ.get("TERRABOX_TOOL_RESULT_CACHE_SLUGS", "").strip()
    if override:
        return slug in {s.strip() for s in override.split(",") if s.strip()}
    return slug in _PERCEPTION or slug.startswith("osm_gis.")


def _file_hash(path: str) -> str | None:
    try:
        st = os.stat(path)
        cached = _HASH_CACHE.get(path)
        if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
            return cached[2]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        digest = h.hexdigest()
        _HASH_CACHE[path] = (st.st_mtime, st.st_size, digest)
        return digest
    except Exception:
        return None


def _normalize(value: Any) -> Any:
    """Replace input-file paths by a content hash; keep everything else verbatim."""
    if isinstance(value, str):
        if os.path.isfile(value):
            h = _file_hash(value)
            return f"file:sha256:{h}" if h else value
        return value
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items()) if k not in _OUTPUT_KEYS}
    return value


def cache_key(slug: str, arguments: dict[str, Any] | None) -> str | None:
    """Deterministic content key, or None if caching is off / slug not eligible."""
    if not _enabled() or not _slug_allowed(slug):
        return None
    norm = {k: _normalize(v) for k, v in sorted((arguments or {}).items()) if k not in _OUTPUT_KEYS}
    blob = json.dumps({"slug": slug, "args": norm}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _entry_dir(key: str) -> Path:
    return _cache_dir() / key[:2] / key


def _is_error_text(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _ERROR_MARKERS)


def lookup(key: str | None) -> str | None:
    """Return cached result text (and restore artifacts), or None on miss."""
    if not key:
        return None
    d = _entry_dir(key)
    meta = d / "meta.json"
    if not meta.is_file():
        return None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    for a in m.get("artifacts", []):
        src = d / "artifacts" / a.get("stored", "")
        dest = a.get("orig", "")
        if not dest or not src.is_file():
            # Incomplete/corrupt entry: cannot reproduce the tool's output files →
            # treat as a MISS so the caller re-runs the tool (correctness over speed).
            logger.debug("cache entry missing artifact %s; treating as miss", src)
            return None
        try:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            if not (os.path.isfile(dest) and os.path.getsize(dest) == src.stat().st_size):
                shutil.copy2(src, dest)
        except Exception as exc:
            logger.debug("artifact restore failed (%s -> %s): %s", src, dest, exc)
            return None
    return m.get("result_text")


def store(key: str | None, slug: str, arguments: dict[str, Any] | None,
          result_text: str, artifact_paths: list[str] | None) -> None:
    """Persist a SUCCESSFUL result (text + produced files). No-op on error text."""
    if not key or result_text is None or _is_error_text(result_text):
        return
    d = _entry_dir(key)
    try:
        with _LOCK:
            if (d / "meta.json").is_file():
                return  # first writer wins; identical content anyway
            (d / "artifacts").mkdir(parents=True, exist_ok=True)
            arts: list[dict[str, str]] = []
            used: set[str] = set()
            for p in artifact_paths or []:
                ap = os.path.abspath(p)
                if not os.path.isfile(ap):
                    continue
                name = os.path.basename(ap) or "artifact"
                stored, i = name, 1
                while stored in used:
                    stored, i = f"{i}_{name}", i + 1
                used.add(stored)
                shutil.copy2(ap, d / "artifacts" / stored)
                arts.append({"orig": ap, "stored": stored})
            meta = {
                "slug": slug,
                "result_text": result_text,
                "artifacts": arts,
                "args_preview": {k: str(v)[:160] for k, v in (arguments or {}).items()},
            }
            tmp = d / "meta.json.tmp"
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, d / "meta.json")
    except Exception as exc:
        logger.debug("tool result cache store failed for %s: %s", slug, exc)
