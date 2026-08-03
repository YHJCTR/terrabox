"""Unified tool executor for agent harness mode."""
from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import contextmanager
import fcntl
import inspect
import json
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Bounded thread pool for running async tool handlers synchronously.
# Prevents unbounded daemon thread creation under concurrent load.
_EXECUTOR = ThreadPoolExecutor(max_workers=32, thread_name_prefix="tool_async")
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=32, thread_name_prefix="tool_exec")

from ..core.registry import get_handler, get_tool, list_toolkits, list_tools
from ..core.services.agent_run_service import AgentRunService
from ..core.services.connection_service import ConnectionService
from ..core.services.tool_override_service import ToolOverrideService
from ..core.services.tool_service import ToolService
from ..core.utils.runtime_paths import apply_default_output_paths, prepare_runtime_context, write_manifest
from . import tool_result_cache as _trc
from ..core.utils.tool_output_serialization import make_json_safe
from .harness import create_approval, current_context, emit_event, record_artifact, record_step, update_step
from .runtime import get_runtime

logger = logging.getLogger(__name__)

_PATH_KEYS = {"artifact_path", "output_path", "result_path", "path", "image_path", "preview_path", "out_file", "gpkg"}
_OUTPUT_PATH_KEYS = {"artifact_path", "output_path", "result_path", "preview_path", "out_file"}
_INPUT_PATH_KEYS = _PATH_KEYS - _OUTPUT_PATH_KEYS
_ARTIFACT_INDEX_LOCK = threading.Lock()
_PATH_SUFFIXES = (
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".json",
    ".geojson",
    ".gpkg",
    ".csv",
    ".npy",
)


class ToolExecutionTimeout(TimeoutError):
    """Raised when a tool call exceeds its configured execution timeout."""

    def __init__(self, slug: str, timeout_seconds: int, elapsed_seconds: float, arguments: dict[str, Any]):
        self.slug = slug
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.arguments = arguments
        super().__init__(f"Tool {slug} timed out after {timeout_seconds}s")


def _toolkit_for_slug(slug: str) -> str | None:
    for toolkit in list_toolkits():
        if any(t.slug == slug for t in list_tools(toolkit.name)):
            return toolkit.name
    return None


def _is_risky(slug: str) -> bool:
    return (
        slug.startswith("bash.")
        or slug.startswith("ipython.")
        or slug.startswith("ipython_code.")
        or slug.startswith("github.")
    )


def _bucket_for(slug: str) -> str:
    if slug.startswith("geo_perception."):
        return "perception"
    if (
        slug.startswith("bash.")
        or slug.startswith("ipython.")
        or slug.startswith("ipython_code.")
        or slug.startswith("github.")
    ):
        return "risky"
    if slug.startswith("bing_search."):
        return "network"
    if slug.startswith("geo_raster.") or slug.startswith("georaster.") or slug.startswith("disaster_response."):
        return "compute"
    return "default"


def _display_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        text = json.dumps(result, ensure_ascii=False)
    except Exception:
        text = str(result)
    return text[:4000]


def _tool_timeout_env_name(slug: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in slug.upper())
    return f"TERRABOX_TOOL_TIMEOUT_{safe}"


def _timeout_for_tool(slug: str, config: Any | None = None) -> int:
    specific = os.environ.get(_tool_timeout_env_name(slug))
    if specific:
        return int(specific)
    generic = os.environ.get("TERRABOX_TOOL_TIMEOUT_SECONDS")
    if generic:
        return int(generic)
    if config is not None:
        return int(getattr(config, "default_tool_timeout_seconds", 120))
    return 120


def _make_tool_context(slug: str, timeout_seconds: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    now = time.time()
    ctx = {
        "timeout": timeout_seconds,
        "deadline": now + timeout_seconds if timeout_seconds > 0 else None,
        "tool_slug": slug,
    }
    if extra:
        ctx.update(extra)
    return ctx


def _format_tool_timeout(exc: ToolExecutionTimeout) -> str:
    payload = {
        "status": "error",
        "error_type": "tool_timeout",
        "tool": exc.slug,
        "timeout_seconds": exc.timeout_seconds,
        "elapsed_seconds": round(exc.elapsed_seconds, 2),
        "args": exc.arguments,
        "message": f"Tool timed out after {exc.timeout_seconds} seconds.",
        "likely_causes": [
            "The input is too large or the selected tool is doing expensive computation.",
            "The requested parameters may create too many pairwise operations or external requests.",
            "The tool may be waiting on slow IO, network, or model inference.",
        ],
        "recovery_suggestions": [
            "Reduce the input size or narrow the area of interest.",
            "Lower parameters such as top/limit if the tool supports them.",
            "Use a cheaper prerequisite/filtering tool before retrying.",
            "Do not repeat the same tool call with the same arguments.",
        ],
    }
    return "Tool execution error: " + json.dumps(payload, ensure_ascii=False)


def _run_handler_call(
    handler,
    arguments: dict[str, Any],
    context: dict[str, Any],
    connection_or_user,
    accepts_connection: bool,
) -> Any:
    if inspect.iscoroutinefunction(handler):
        if accepts_connection:
            return _run_coro_sync(handler(arguments, context, connection_or_user))
        return _run_coro_sync(handler(arguments, context))
    if accepts_connection:
        return handler(arguments, context, connection_or_user)
    return handler(arguments, context)


def _execute_with_timeout(
    slug: str,
    handler,
    arguments: dict[str, Any],
    context: dict[str, Any],
    connection_or_user,
    accepts_connection: bool,
    timeout_seconds: int,
    started: float,
) -> Any:
    if timeout_seconds <= 0:
        return _run_handler_call(handler, arguments, context, connection_or_user, accepts_connection)
    future = _TOOL_EXECUTOR.submit(
        _run_handler_call,
        handler,
        arguments,
        context,
        connection_or_user,
        accepts_connection,
    )
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise ToolExecutionTimeout(slug, timeout_seconds, time.time() - started, arguments) from exc


def _artifact_paths(result: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(result, str) and os.path.exists(result):
        return [result]
    if isinstance(result, dict):
        for key, value in result.items():
            if key in _PATH_KEYS and isinstance(value, str) and value:
                paths.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and os.path.exists(item):
                        paths.append(item)
    return [p for p in dict.fromkeys(paths) if p]


def _managed_artifact_dir() -> str | None:
    value = (
        os.environ.get("TERRABOX_ARTIFACT_OUTPUT_DIR", "").strip()
        or os.environ.get("TERRABOX_TOKEN_ARTIFACT_DIR", "").strip()
    )
    if not value:
        return None
    return value if os.path.isabs(value) else os.path.abspath(value)


def _artifact_index_path() -> str | None:
    value = os.environ.get("TERRABOX_ARTIFACT_INDEX_PATH", "").strip()
    if value:
        return value if os.path.isabs(value) else os.path.abspath(value)
    artifact_dir = _managed_artifact_dir()
    if not artifact_dir:
        return None
    return os.path.join(artifact_dir, "artifact_index.json")


@contextmanager
def _artifact_index_file_lock(path: str):
    """Cross-process lock for shared artifact indexes written by parallel flows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = f"{path}.lock"
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _is_url_or_placeholder(value: str) -> bool:
    lowered = value.lower()
    return (
        "://" in value
        or lowered.startswith(("s3://", "gs://"))
        or value.startswith("<")
        or value in {"", "none", "null"}
    )


def _safe_relative_path(value: str) -> str:
    normalized = os.path.normpath(value.strip())
    parts = [
        part
        for part in normalized.split(os.sep)
        if part and part not in {".", ".."}
    ]
    if not parts:
        base = os.path.basename(value.strip()) or "artifact"
        return base
    return os.path.join(*parts)


def _artifact_subdir_for(key: str, path: str) -> str:
    lowered = f"{key} {path}".lower()
    if lowered.endswith(".gpkg") or "gpkg" in lowered:
        return "gpkg"
    if lowered.endswith((".tif", ".tiff")) or "raster" in lowered:
        return "rasters"
    if lowered.endswith((".png", ".jpg", ".jpeg")) or "image" in lowered:
        return "images"
    if lowered.endswith((".json", ".geojson", ".csv", ".txt")):
        return "data"
    return "outputs"


def _load_artifact_index() -> dict[str, Any]:
    path = _artifact_index_path()
    if not path or not os.path.exists(path):
        return {"aliases": {}, "artifacts": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("aliases", {})
            data.setdefault("artifacts", [])
            return data
    except Exception as exc:
        logger.debug("Failed to load artifact index %s: %s", path, exc)
    return {"aliases": {}, "artifacts": []}


def _save_artifact_index(data: dict[str, Any]) -> None:
    path = _artifact_index_path()
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _record_artifact_alias(
    *,
    requested: str | None,
    resolved: str,
    key: str,
    source: str,
    slug: str,
) -> None:
    if not resolved:
        return
    resolved_abs = resolved if os.path.isabs(resolved) else os.path.abspath(resolved)
    path = _artifact_index_path()
    if not path:
        return
    with _ARTIFACT_INDEX_LOCK:
        with _artifact_index_file_lock(path):
            index = _load_artifact_index()
            aliases = index.setdefault("aliases", {})
            if requested:
                aliases[requested] = resolved_abs
                aliases[os.path.normpath(requested)] = resolved_abs
                aliases[os.path.basename(requested)] = resolved_abs
            aliases[resolved_abs] = resolved_abs
            aliases[os.path.basename(resolved_abs)] = resolved_abs
            artifacts = index.setdefault("artifacts", [])
            record = {
                "tool": slug,
                "key": key,
                "source": source,
                "path": resolved_abs,
                "requested_path": requested,
                "exists": os.path.exists(resolved_abs),
                "size_bytes": os.path.getsize(resolved_abs) if os.path.exists(resolved_abs) else None,
                "recorded_at": time.time(),
            }
            artifacts.append(record)
            _save_artifact_index(index)


def _lookup_artifact_alias(value: str) -> str | None:
    if not value:
        return None
    path = _artifact_index_path()
    if not path:
        return None
    with _ARTIFACT_INDEX_LOCK:
        with _artifact_index_file_lock(path):
            aliases = _load_artifact_index().get("aliases", {})
    for key in (value, os.path.normpath(value), os.path.basename(value)):
        resolved = aliases.get(key)
        if isinstance(resolved, str) and os.path.exists(resolved):
            return resolved
    return None


def _repo_tmp_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, "tmp")


def _is_under_dir(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _redirect_output_path(slug: str, key: str, value: str) -> tuple[str, dict[str, Any] | None]:
    artifact_dir = _managed_artifact_dir()
    if not artifact_dir or not value or _is_url_or_placeholder(value):
        return value, None

    if os.path.isabs(value):
        allowed_roots = (artifact_dir, _repo_tmp_dir())
        if any(_is_under_dir(value, root) for root in allowed_roots):
            return value, None
        safe_rel = _safe_relative_path(value.lstrip(os.sep))
        kind = "absolute_output_redirect"
    else:
        safe_rel = _safe_relative_path(value)
        kind = "output_redirect"

    subdir = _artifact_subdir_for(key, safe_rel)
    resolved = os.path.join(artifact_dir, subdir, safe_rel)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    _record_artifact_alias(requested=value, resolved=resolved, key=key, source=kind, slug=slug)
    return resolved, {
        "param": key,
        "requested": value,
        "resolved": resolved,
        "kind": kind,
    }


def _resolve_input_artifact_alias(slug: str, key: str, value: str) -> tuple[str, dict[str, Any] | None]:
    if (
        not value
        or _is_url_or_placeholder(value)
        or os.path.exists(value)
    ):
        return value, None
    resolved = _lookup_artifact_alias(value)
    if not resolved:
        return value, None
    return resolved, {
        "param": key,
        "requested": value,
        "resolved": resolved,
        "kind": "input_alias_resolution",
    }


def _load_task_data_files() -> list[str]:
    raw = os.environ.get("TERRABOX_TASK_DATA_FILES", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str) and item]


def _task_data_candidates(value: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    normalized = os.path.normpath(value)
    if "/question_data/question_data/" in normalized:
        candidates.append((
            normalized.replace("/question_data/question_data/", "/question_data/", 1),
            "path_normalization",
        ))

    data_files = _load_task_data_files()
    basename = os.path.basename(normalized)
    for path in data_files:
        path_norm = os.path.normpath(path)
        if os.path.basename(path_norm) == basename:
            candidates.append((path_norm, "task_data_file_resolution"))
        elif path_norm.endswith(normalized.lstrip(os.sep)):
            candidates.append((path_norm, "task_data_file_resolution"))

    data_dir = os.environ.get("TERRABOX_TASK_DATA_DIR", "").strip()
    if data_dir:
        safe_rel = _safe_relative_path(value)
        candidates.append((os.path.join(data_dir, safe_rel), "task_data_dir_resolution"))
        if basename:
            candidates.append((os.path.join(data_dir, basename), "task_data_dir_resolution"))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for path, kind in candidates:
        path_abs = path if os.path.isabs(path) else os.path.abspath(path)
        if path_abs in seen:
            continue
        seen.add(path_abs)
        unique.append((path_abs, kind))
    return unique


def _resolve_task_data_path(slug: str, key: str, value: str) -> tuple[str, dict[str, Any] | None]:
    if not value or _is_url_or_placeholder(value) or os.path.exists(value):
        return value, None
    for candidate, kind in _task_data_candidates(value):
        if os.path.exists(candidate):
            _record_artifact_alias(
                requested=value,
                resolved=candidate,
                key=key,
                source=kind,
                slug=slug,
            )
            return candidate, {
                "param": key,
                "requested": value,
                "resolved": candidate,
                "kind": kind,
            }
    return value, None


def _is_input_path_like(key: str, value: str) -> bool:
    lowered_key = key.lower()
    lowered_value = value.lower()
    if lowered_key in _INPUT_PATH_KEYS:
        return True
    if lowered_key.endswith(("_path", "_paths", "_file", "_files")):
        return True
    if any(token in lowered_key for token in ("path", "file", "image", "raster", "gpkg")):
        return True
    if lowered_key.startswith("band") or (
        len(lowered_key) == 3 and lowered_key[0] == "b" and lowered_key[1:].isdigit()
    ):
        return True
    return lowered_value.endswith(_PATH_SUFFIXES) or "/" in value or "\\" in value


def _prepare_artifact_paths(slug: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Redirect output paths and resolve known artifact aliases for tool calls.

    Experiments or batch jobs can set TERRABOX_ARTIFACT_OUTPUT_DIR to keep
    generated files out of the repo root. If the LLM later refers to its
    originally requested relative path, the artifact index maps that alias back
    to the real path.
    """
    if not isinstance(arguments, dict):
        return arguments, []
    resolutions: list[dict[str, Any]] = []

    def prepare_value(key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {child_key: prepare_value(child_key, child_value) for child_key, child_value in value.items()}
        if isinstance(value, list):
            return [prepare_value(f"{key}[{idx}]", item) for idx, item in enumerate(value)]
        if not isinstance(value, str):
            return value

        lowered = key.lower()
        if lowered in _OUTPUT_PATH_KEYS:
            new_value, resolution = _redirect_output_path(slug, key, value)
        elif _is_input_path_like(key, value):
            new_value, resolution = _resolve_input_artifact_alias(slug, key, value)
            if not resolution:
                new_value, resolution = _resolve_task_data_path(slug, key, value)
        else:
            return value
        if resolution:
            resolutions.append(resolution)
        return new_value

    prepared = {key: prepare_value(key, value) for key, value in arguments.items()}
    return prepared, resolutions


def _annotate_path_resolution(result: Any, resolutions: list[dict[str, Any]]) -> Any:
    if not resolutions or not isinstance(result, dict):
        return result
    annotated = dict(result)
    existing = annotated.get("_path_resolution")
    if isinstance(existing, list):
        annotated["_path_resolution"] = [*existing, *resolutions]
    else:
        annotated["_path_resolution"] = resolutions
    index_path = _artifact_index_path()
    if index_path:
        annotated["_artifact_index_path"] = index_path
    return annotated


def _record_result_artifacts(slug: str, result: Any) -> None:
    if not _artifact_index_path():
        return
    if isinstance(result, dict):
        for key, value in result.items():
            if key in _PATH_KEYS and isinstance(value, str) and value:
                _record_artifact_alias(requested=value, resolved=value, key=key, source="tool_result", slug=slug)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and os.path.exists(item):
                        _record_artifact_alias(requested=item, resolved=item, key=key, source="tool_result", slug=slug)
    elif isinstance(result, str) and os.path.exists(result):
        _record_artifact_alias(requested=result, resolved=result, key="result", source="tool_result", slug=slug)


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}
    err: dict[str, BaseException] = {}

    def _runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - passthrough
            err["error"] = exc

    future = _EXECUTOR.submit(_runner)
    future.result()
    if "error" in err:
        raise err["error"]
    return box.get("value")


# ---------------------------------------------------------------------------
# GeoPackage session state: auto-inject gpkg path for chained tool calls
# ---------------------------------------------------------------------------

_gpkg_state_lock = threading.Lock()
_gpkg_by_scope: dict[str, str] = {}

GPKG_REQUIRED_SLUGS = frozenset({
    "osm_gis.add_pois_layer",
    "osm_gis.compute_route_dist",
    "osm_gis.add_index_layer",
    "osm_gis.compute_index_change",
    "osm_gis.show_index_layer",
    "osm_gis.display_on_map",
    "osm_gis.display_on_geotiff",
    "osm_gis.get_bbox_from_raster",
})


def _gpkg_scope_key() -> str:
    ctx = current_context()
    if ctx is not None:
        return f"run:{ctx.run_id}"
    return "legacy"


def _get_current_gpkg() -> str | None:
    scope_key = _gpkg_scope_key()
    with _gpkg_state_lock:
        return _gpkg_by_scope.get(scope_key)


def _set_current_gpkg(gpkg: str) -> None:
    scope_key = _gpkg_scope_key()
    with _gpkg_state_lock:
        _gpkg_by_scope[scope_key] = gpkg


def _inject_gpkg(slug: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Auto-inject current_gpkg for tools that need it (like OpenEarthAgent)."""
    needs_gpkg = slug in GPKG_REQUIRED_SLUGS
    if slug == "osm_gis.get_bbox_from_raster":
        has_raster_input = bool(arguments.get("input_path") or arguments.get("geotiff"))
        needs_gpkg = bool(arguments.get("layer")) and not has_raster_input
    if needs_gpkg:
        gpkg_val = arguments.get("gpkg")
        # Inject if missing, placeholder (e.g. "gpkg_1"), or not an existing file
        if not gpkg_val or (isinstance(gpkg_val, str) and not os.path.exists(gpkg_val)):
            current = _get_current_gpkg()
            if current and os.path.exists(current):
                arguments = {**arguments, "gpkg": current}
                logger.info("gpkg auto-injected: %s → %s", slug, current)
    return arguments


def _capture_gpkg(result: Any) -> None:
    """Capture gpkg path from tool response for downstream injection."""
    if isinstance(result, dict):
        gpkg = result.get("gpkg")
        if gpkg and isinstance(gpkg, str) and os.path.exists(gpkg):
            _set_current_gpkg(gpkg)
            logger.info("gpkg captured: %s", gpkg)


class AgentToolExecutor:
    @staticmethod
    def execute(slug: str, arguments: dict[str, Any], user) -> str:
        # Content-addressed result cache: checked at the OUTERMOST point so a hit
        # returns WITHOUT starting the tool's docker service / model load / network
        # call. Key covers tool + input-file content + all non-output args (see
        # tool_result_cache). Only successful results are stored.
        cache_key = _trc.cache_key(slug, arguments)
        if cache_key is not None:
            hit = _trc.lookup(cache_key)
            if hit is not None:
                logger.info("Tool result cache HIT: %s", slug)
                return hit
        ctx = current_context()
        if ctx is None:
            return AgentToolExecutor._execute_legacy(slug, arguments, user, _cache_key=cache_key)
        return AgentToolExecutor._execute_harness(slug, arguments, user, _cache_key=cache_key)

    @staticmethod
    def _execute_legacy(slug: str, arguments: dict[str, Any], user, _cache_key: str | None = None) -> str:
        handler = get_handler(slug)
        if handler is None:
            return f"Tool execution error: Tool not found: {slug}"
        timeout_seconds = _timeout_for_tool(slug)
        started = time.time()
        trace_id = str(uuid.uuid4())
        user_id = getattr(user, "user_id", getattr(user, "id", None))
        runtime_metadata = prepare_runtime_context(str(user_id or "anonymous"), slug, execution_id=trace_id)
        context = _make_tool_context(
            slug,
            timeout_seconds,
            {
                "user_id": user_id,
                "connection_id": None,
                "trace_id": trace_id,
                "run_id": None,
                **runtime_metadata,
            },
        )
        try:
            spec = get_tool(slug)
            arguments = apply_default_output_paths(
                slug,
                arguments,
                spec.parameters if spec else {},
                runtime_metadata,
            )
            arguments, path_resolutions = _prepare_artifact_paths(slug, arguments)
            arguments = _inject_gpkg(slug, arguments)
            sig = inspect.signature(handler)
            accepts_connection = len(sig.parameters) >= 3
            result = _execute_with_timeout(
                slug,
                handler,
                arguments,
                context,
                user,
                accepts_connection,
                timeout_seconds,
                started,
            )
            result = make_json_safe(result)
            result = _annotate_path_resolution(result, path_resolutions)
            _capture_gpkg(result)
            _record_result_artifacts(slug, result)
            write_manifest(
                execution_id=trace_id,
                user_id=str(user_id or "anonymous"),
                tool_slug=slug,
                runtime=runtime_metadata,
                inputs=arguments,
                outputs=result if isinstance(result, dict) else {"result": result},
                status="success",
            )
            display_text = _display_text(result)
            _trc.store(_cache_key, slug, arguments, display_text, _artifact_paths(result))
            return display_text
        except ToolExecutionTimeout as exc:
            logger.warning("Tool execution timeout: %s", exc)
            write_manifest(
                execution_id=trace_id,
                user_id=str(user_id or "anonymous"),
                tool_slug=slug,
                runtime=runtime_metadata,
                inputs=arguments,
                outputs=None,
                status="error",
                error=_format_tool_timeout(exc),
            )
            return _format_tool_timeout(exc)
        except Exception as exc:
            logger.warning("Tool execution error: %s", exc)
            write_manifest(
                execution_id=trace_id,
                user_id=str(user_id or "anonymous"),
                tool_slug=slug,
                runtime=runtime_metadata,
                inputs=arguments,
                outputs=None,
                status="error",
                error=str(exc),
            )
            return f"Tool execution error: {exc}"

    @staticmethod
    def _execute_harness(slug: str, arguments: dict[str, Any], user, _cache_key: str | None = None) -> str:
        ctx = current_context()
        if ctx is None:
            return AgentToolExecutor._execute_legacy(slug, arguments, user, _cache_key=_cache_key)

        config = ctx.config
        db = ctx.db
        trace_id = str(uuid.uuid4())
        bucket = _bucket_for(slug)
        toolkit_name = _toolkit_for_slug(slug)
        runtime = get_runtime()
        user_id = getattr(user, "user_id", user.id)
        runtime_metadata = prepare_runtime_context(str(user_id), slug, execution_id=trace_id)

        ok, runtime_reason = runtime.start_tool(bucket)
        if not ok:
            step = record_step(
                "tool",
                title=f"Tool blocked: {slug}",
                content=runtime_reason or "runtime_guard",
                status="failed",
                tool_slug=slug,
                trace_id=trace_id,
                input_data=arguments,
                error_type=runtime_reason,
                error_message=f"Tool blocked by runtime guard: {runtime_reason}",
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=False,
                trace_id=trace_id,
                error_type=runtime_reason,
                message=f"Tool blocked by runtime guard: {runtime_reason}",
            )
            return f"Tool execution error: Tool blocked by runtime guard: {runtime_reason}"

        spec = get_tool(slug)
        arguments = apply_default_output_paths(
            slug,
            arguments,
            spec.parameters if spec else {},
            runtime_metadata,
        )
        arguments, path_resolutions = _prepare_artifact_paths(slug, arguments)

        step = record_step(
            "tool",
            title=f"Execute tool {slug}",
            content="tool_start",
            status="running",
            tool_slug=slug,
            trace_id=trace_id,
            input_data=arguments,
            metadata={"bucket": bucket, "path_resolutions": path_resolutions},
        )
        emit_event(
            "tool_start",
            step_id=step.id if step else None,
            tool_slug=slug,
            trace_id=trace_id,
            args=arguments,
        )

        started = time.time()
        connection = None
        approval = None
        try:
            spec = get_tool(slug)
            if spec is None:
                raise RuntimeError(f"Tool not found: {slug}")

            if getattr(config, "record_risky_tool_approvals", True) and _is_risky(slug):
                approval_status = "pending" if config.require_approval_for_risky_tools else "auto_approved"
                approval = create_approval(
                    slug,
                    reason="Manual approval required for risky tool" if approval_status == "pending" else "Risky tool category logged for audit",
                    step_id=step.id if step else None,
                    status=approval_status,
                )
                if approval_status == "pending":
                    raise PermissionError(f"Approval required for risky tool: {slug}")

            arguments = _inject_gpkg(slug, arguments)

            if spec.requires_connection and toolkit_name:
                connection = ConnectionService.select_connection(db, user.id, toolkit_name)
                if not connection:
                    raise RuntimeError("No valid connection found for this tool")

            if connection:
                effective = ToolOverrideService.get_effective_tools(db, str(connection.id), include_disabled=False)
                enabled = {item["tool_key"] for item in effective["tools"]}
                if slug not in enabled:
                    raise RuntimeError("Tool is disabled for this connection")

            handler = get_handler(slug)
            if handler is None:
                raise RuntimeError(f"No handler registered for tool: {slug}")

            context = {
                "user_id": user_id,
                "connection_id": str(connection.id) if connection else None,
                "trace_id": trace_id,
                "run_id": ctx.run_id,
                **runtime_metadata,
            }
            timeout_seconds = _timeout_for_tool(slug, config)
            context = _make_tool_context(slug, timeout_seconds, context)
            sig = inspect.signature(handler)
            accepts_connection = len(sig.parameters) >= 3

            result = _execute_with_timeout(
                slug,
                handler,
                arguments,
                context,
                connection,
                accepts_connection,
                timeout_seconds,
                started,
            )

            result = make_json_safe(result)
            result = _annotate_path_resolution(result, path_resolutions)
            _capture_gpkg(result)
            _record_result_artifacts(slug, result)
            write_manifest(
                execution_id=trace_id,
                user_id=str(user_id),
                tool_slug=slug,
                runtime=runtime_metadata,
                inputs=arguments,
                outputs=result if isinstance(result, dict) else {"result": result},
                status="success",
            )
            duration_ms = int((time.time() - started) * 1000)
            display_text = _display_text(result)
            artifact_paths = _artifact_paths(result)

            update_step(
                step.id if step else "",
                status="completed",
                content=display_text,
                output_data=result,
                duration_ms=duration_ms,
                metadata={"bucket": bucket, "artifact_count": len(artifact_paths)},
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=True,
                trace_id=trace_id,
                duration_ms=duration_ms,
                display_text=display_text,
            )
            for path in artifact_paths:
                if os.path.exists(path):
                    record_artifact(step.id if step else None, path=path, metadata={"tool_slug": slug})

            _trc.store(_cache_key, slug, arguments, display_text, artifact_paths)

            ToolService._log_tool_execution(
                db,
                user.id,
                slug,
                connection,
                trace_id,
                arguments,
                result,
                True,
                None,
                toolkit_name,
            )
            runtime.finish_tool(bucket, True)
            return display_text

        except ToolExecutionTimeout as exc:
            duration_ms = int((time.time() - started) * 1000)
            error_type = "ToolExecutionTimeout"
            error_message = _format_tool_timeout(exc)
            update_step(
                step.id if step else "",
                status="failed",
                content=error_message,
                output_data={},
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=error_message,
                metadata={"bucket": bucket, "approval_id": getattr(approval, "id", None)},
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=False,
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=error_type,
                message=error_message,
            )
            write_manifest(
                execution_id=trace_id,
                user_id=str(user_id),
                tool_slug=slug,
                runtime=runtime_metadata,
                inputs=arguments,
                outputs=None,
                status="error",
                error=error_message,
            )
            ToolService._log_tool_execution(
                db,
                user.id,
                slug,
                connection,
                trace_id,
                arguments,
                None,
                False,
                error_message,
                toolkit_name,
            )
            runtime.finish_tool(bucket, False, error_type=error_type)
            return error_message

        except Exception as exc:
            duration_ms = int((time.time() - started) * 1000)
            error_type = type(exc).__name__
            error_message = traceback.format_exc()
            update_step(
                step.id if step else "",
                status="failed",
                content=f"Tool execution error: {error_message}",
                output_data={},
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=error_message,
                metadata={"bucket": bucket, "approval_id": getattr(approval, "id", None)},
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=False,
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=error_type,
                message=error_message,
            )
            write_manifest(
                execution_id=trace_id,
                user_id=str(user_id),
                tool_slug=slug,
                runtime=runtime_metadata,
                inputs=arguments,
                outputs=None,
                status="error",
                error=error_message,
            )
            ToolService._log_tool_execution(
                db,
                user.id,
                slug,
                connection,
                trace_id,
                arguments,
                None,
                False,
                error_message,
                toolkit_name,
            )
            runtime.finish_tool(bucket, False, error_type=error_type)
            return f"Tool execution error: {error_message}"
