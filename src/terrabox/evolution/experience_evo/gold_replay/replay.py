"""Teacher-forced OEA gold trajectory replay.

This module executes ``gold_tool_calls`` against the current Terrabox tool
registry. It is a data/teacher audit path: the actor LLM is not called and does
not see gold tool sequences, arguments, or answers.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[5]
_ERROR_MARKERS = (
    '"status": "error"',
    '"status":"error"',
    '"status": "failed"',
    '"status":"failed"',
    "Tool execution error:",
    "Error in calculator:",
    "Error in Solver:",
    "Error in solver:",
    "Error in Plot:",
    "Error in plot:",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "out of memory",
    "timed out",
)
_INFRA_ERROR_MARKERS = (
    "connection",
    "remote end closed connection",
    "rate limit",
    "429",
    "overload",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "failed to start service",
    "failed to start",
    "container name",
    "docker:",
    "service is not healthy",
)
_PROVIDER_ACCOUNT_ERROR_MARKERS = (
    "not enough credits",
    "insufficient credits",
    "quota",
    "billing",
    "payment required",
    "402",
    "unauthorized",
    "authentication",
    "invalid api key",
)
_SERVICE_RUNTIME_ERROR_MARKERS = (
    "api error 500",
    "an image must be set with .set_image",
    "boolean index did not match indexed array",
    "inhomogeneous shape",
)
_TRANSIENT_REPLAY_FAILURE_TYPES = {"infra_or_provider", "timeout"}
_GPU_TOOL_PREFIXES = (
    "geo_perception.vlm_analyze",
    "geo_perception.instructsam",
    "geo_perception.remotesam",
    "geo_perception.strip_rcnn_detect",
    "geo_perception.sam2_segment",
    "geo_perception.change_os_detect",
    "geo_perception.region_attribute_description",
    "geo_perception.count_given_object",
)
_RASTER_SUFFIXES = (".tif", ".tiff", ".geotiff")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".geotiff")
_GOLD_ALIAS_RE = re.compile(r"^(gpkg|img|tif)_(\d+)$")
_NAMED_GPKG_ALIAS_RE = re.compile(r"^(?:.*_)?gpkg(?:_.*)?_(\d+)$")
_RESULT_PATH_KEYS = (
    "gpkg",
    "gpkg_path",
    "out_file",
    "output_file",
    "output_path",
    "output_gpkg",
    "image",
    "image_path",
    "path",
    "result",
)
_OUTPUT_ARGUMENT_KEYS = {"out_file", "output_file", "output_path", "output_gpkg"}
_FIXED_OUTPUT_HANDLES = {"out.tif", "out.tiff", "out.png", "out.jpg", "out.jpeg", "dummy_generated_image.jpg"}
_GOLD_REPLAY_ENV_DEFAULTS = {
    "no_proxy": "localhost,127.0.0.1",
    "NO_PROXY": "localhost,127.0.0.1",
    "TERRABOX_TOOL_SERVICE_SCOPE": "call",
    "TERRABOX_SERVICE_CALL_LOCKS": "1",
    "TERRABOX_SERVICE_LOCK_TIMEOUT_SECONDS": "1800",
    "TERRABOX_SERVICE_LOCK_DIR": str(_REPO_ROOT / "tmp" / "service_locks"),
    "TERRABOX_OCR_USE_GPU": "0",
    "TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_SAM2_SEGMENT": "420",
    "TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_REMOTESAM": "420",
    "TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_INSTRUCTSAM": "600",
    "TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_COUNT_GIVEN_OBJECT": "600",
    "TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_REGION_ATTRIBUTE_DESCRIPTION": "420",
    "TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_VLM_ANALYZE": "420",
    "REMOTESAM_TOOL_TIMEOUT": "360",
    "TERRABOX_TOOL_TIMEOUT_OSM_GIS_GET_AREA_BOUNDARY": "240",
    "TERRABOX_TOOL_TIMEOUT_OSM_GIS_ADD_POIS_LAYER": "240",
    "TERRABOX_ENABLE_SOURCE_SCHEMA_TOOL_ALIASES": "false",
    # Gold replay often runs with one VLM lane plus one or more lighter
    # perception lanes. These defaults intentionally override broader
    # agent_config.yaml settings such as vlm_tensor_parallel=2 when the replay
    # CLI is invoked directly without a launcher script.
    "VLM_TENSOR_PARALLEL_SIZE": "1",
    "VLM_MAX_MODEL_LEN": "8192",
    "VLM_MIN_IMAGE_MODEL_LEN": "8192",
    "VLM_GPU_MEMORY_UTILIZATION": "0.95",
    "VLM_MAX_NUM_SEQS": "1",
    "TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS": "4096",
}


def _read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"expected list or {{'tasks': [...]}} in {path}")
    return [row for row in rows if isinstance(row, dict)]


def _task_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("task_id") or row.get("id") or f"task_{index}")


def _load_subset_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    rows = _read_json_or_jsonl(path)
    return {_task_id(row, index) for index, row in enumerate(rows) if _task_id(row, index)}


def _expected_tools(row: dict[str, Any]) -> list[str]:
    expected = [str(tool) for tool in (row.get("expected_tools") or []) if tool != "final_answer"]
    if expected:
        return expected
    return [
        str(call.get("tool") or call.get("slug") or "")
        for call in (row.get("gold_tool_calls") or [])
        if isinstance(call, dict) and (call.get("tool") or call.get("slug"))
    ]


def _is_online(expected: list[str]) -> bool:
    return any(tool.startswith("osm_gis.") or tool == "bing_search.search" for tool in expected)


def _is_gpu(expected: list[str]) -> bool:
    return any(any(tool == prefix or tool.startswith(prefix + ".") for prefix in _GPU_TOOL_PREFIXES) for tool in expected)


def _keep_by_scope(row: dict[str, Any], *, scope: str, gpu_class: str) -> bool:
    expected = _expected_tools(row)
    online = _is_online(expected)
    gpu = _is_gpu(expected)
    if scope == "online" and not online:
        return False
    if scope == "offline" and online:
        return False
    if gpu_class == "gpu" and not gpu:
        return False
    if gpu_class == "nogpu" and gpu:
        return False
    return True


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _tool_metrics(called: list[str], expected: list[str]) -> dict[str, Any]:
    called_set = set(called)
    expected_set = set(expected)
    if not expected_set:
        empty = not called
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": not called_set,
            "multiset_precision": 1.0,
            "multiset_recall": 1.0,
            "multiset_f1": 1.0,
            "ordered_exact_match": empty,
            "lcs_ratio": 1.0,
        }

    tp = len(called_set & expected_set)
    precision = tp / len(called_set) if called_set else 0.0
    recall = tp / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    called_counter = Counter(called)
    expected_counter = Counter(expected)
    inter = sum((called_counter & expected_counter).values())
    m_prec = inter / len(called) if called else 0.0
    m_rec = inter / len(expected) if expected else 0.0
    m_f1 = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) else 0.0

    lcs = _lcs_len(called, expected)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": called_set == expected_set,
        "multiset_precision": m_prec,
        "multiset_recall": m_rec,
        "multiset_f1": m_f1,
        "ordered_exact_match": called == expected,
        "lcs_ratio": lcs / len(expected) if expected else 0.0,
    }


def _json_safe(value: Any, max_string: int = 4000) -> Any:
    if isinstance(value, str):
        text = value
        try:
            return json.loads(text)
        except Exception:
            return text[:max_string]
    if isinstance(value, dict):
        return {str(k): _json_safe(v, max_string=max_string) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item, max_string=max_string) for item in value[:80]]
    return value


def _path_suffix(path: str) -> str:
    return Path(str(path)).suffix.lower()


def _existing_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).resolve()) if os.path.exists(value) else None


def _add_numbered_alias(state: dict[str, Any], prefix: str, path: str) -> str:
    counters = state.setdefault("counters", {})
    counters[prefix] = int(counters.get(prefix, 0)) + 1
    alias = f"{prefix}_{counters[prefix]}"
    state.setdefault("aliases", {})[alias] = path
    state[f"latest_{prefix}"] = path
    return alias


def _alias_prefix_for_path(key: str, path: str) -> str | None:
    suffix = _path_suffix(path)
    if key == "gpkg" or suffix == ".gpkg":
        return "gpkg"
    if suffix in _RASTER_SUFFIXES:
        return "tif"
    if suffix in _IMAGE_SUFFIXES:
        return "img"
    return None


def _existing_alias_for_path(state: dict[str, Any], prefix: str, path: str) -> str | None:
    aliases: dict[str, str] = state.get("aliases", {})
    for alias, resolved in aliases.items():
        if alias.startswith(f"{prefix}_") and resolved == path:
            return alias
    return None


def _capture_path_alias(
    state: dict[str, Any],
    *,
    key: str,
    path: str,
    source_key: str,
) -> dict[str, Any] | None:
    prefix = _alias_prefix_for_path(key, path)
    if not prefix:
        return None
    alias = _existing_alias_for_path(state, prefix, path)
    if alias:
        state[f"latest_{prefix}"] = path
        return None
    alias = _add_numbered_alias(state, prefix, path)
    return {"alias": alias, "path": path, "source_key": source_key}


def _build_gold_alias_state(row: dict[str, Any]) -> dict[str, Any]:
    """Build source/output alias state for OEA teacher-forced replay.

    OEA gold traces use symbolic artifact names such as ``gpkg_1``, ``tif_1``
    and generated-map ``img_1``. These are not model decisions; they are source
    trajectory handles that must be rebound before calling Terrabox tools.
    """

    state: dict[str, Any] = {"aliases": {}, "counters": {"img": 0, "tif": 0, "gpkg": 0}}
    aliases: dict[str, str] = state["aliases"]
    images = [str(path) for path in (row.get("images") or []) if path]
    data_files = [str(path) for path in (row.get("data_files") or []) if path]

    for idx, path in enumerate(images, start=1):
        aliases[f"img_{idx}"] = path
        state["counters"]["img"] = max(state["counters"]["img"], idx)
        if _path_suffix(path) in _RASTER_SUFFIXES:
            _add_numbered_alias(state, "tif", path)

    for path in data_files:
        suffix = _path_suffix(path)
        if suffix in _RASTER_SUFFIXES:
            _add_numbered_alias(state, "tif", path)
        elif suffix in _IMAGE_SUFFIXES:
            _add_numbered_alias(state, "img", path)
    return state


def _resolve_gold_aliases(value: Any, state: dict[str, Any], resolutions: list[dict[str, Any]], *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {child_key: _resolve_gold_aliases(child_value, state, resolutions, key=child_key)
                for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_resolve_gold_aliases(item, state, resolutions, key=key) for item in value]
    if not isinstance(value, str):
        return value

    aliases: dict[str, str] = state.get("aliases", {})
    resolved = aliases.get(value)
    if resolved is None:
        match = _GOLD_ALIAS_RE.match(value)
        if match:
            prefix = match.group(1)
            alias = f"{prefix}_{match.group(2)}"
            resolved = aliases.get(alias) or state.get(f"latest_{prefix}")
    named_gpkg_match = _NAMED_GPKG_ALIAS_RE.match(value)
    if resolved is None and named_gpkg_match:
        # A few OEA gold traces use human-readable handles such as
        # marienplatz_gpkg_1, duomo_gpkg_1, or gpkg_jeronimos_1. They refer to
        # the same first GeoPackage artifact produced by GetAreaBoundary.
        index = named_gpkg_match.group(1)
        resolved = aliases.get(f"gpkg_{index}") or state.get("latest_gpkg")
    if resolved is None and key not in _OUTPUT_ARGUMENT_KEYS:
        resolved = _resolve_fixed_output_handle(value, state)
    if resolved is None:
        return value
    resolutions.append({"param": key, "requested": value, "resolved": resolved})
    return resolved


def _drop_null_arguments(value: Any) -> Any:
    """Remove JSON-null optional arguments from converted gold calls.

    OpenEarthAgent traces sometimes preserve optional placeholders as
    ``null`` (for example ``buffer_m: null``). Terrabox tools treat omitted
    optional arguments as defaults, so replay should not pass null through as
    an explicit value unless it is inside a list.
    """

    if isinstance(value, dict):
        return {k: _drop_null_arguments(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_null_arguments(item) for item in value]
    return value


def _capture_gold_result_aliases(result: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _json_safe(result)
    if not isinstance(parsed, dict):
        return []
    captured: list[dict[str, Any]] = []

    def visit(value: Any, source_key: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                child_source = f"{source_key}.{key_text}" if source_key else key_text
                if key_text in _RESULT_PATH_KEYS:
                    path = _existing_path(child)
                    if path:
                        item = _capture_path_alias(state, key=key_text, path=path, source_key=child_source)
                        if item:
                            captured.append(item)
                visit(child, child_source)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{source_key}[{idx}]" if source_key else f"[{idx}]")

    visit(parsed)
    return captured


def _resolve_fixed_output_handle(value: str, state: dict[str, Any]) -> str | None:
    """Map OEA fixed output placeholders to the latest captured artifact path.

    Some original OEA traces say a rendered artifact was saved as ``out.tif`` or
    ``dummy_generated_image.jpg`` in the natural-language observation, then use
    that fixed name as the next tool input. Terrabox writes timestamped artifact
    paths, so replay has to bind those handles to the latest captured output.
    """

    name = Path(value).name
    if name not in _FIXED_OUTPUT_HANDLES:
        return None
    suffix = _path_suffix(name)
    if suffix in _RASTER_SUFFIXES:
        return state.get("latest_tif")
    if suffix in _IMAGE_SUFFIXES:
        return state.get("latest_img") or state.get("latest_tif")
    return None


def _capture_artifact_index_aliases(artifact_dir: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Backfill aliases from AgentToolExecutor's artifact index.

    Most tools return artifact paths in their inline JSON, but cache hits or
    display truncation can make replay miss a path even though the executor
    recorded it in ``artifact_index.json``. Reading that index keeps gold
    placeholders like ``gpkg_1`` bound without changing the tool call itself.
    """

    index_path = artifact_dir / "artifact_index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    captured: list[dict[str, Any]] = []
    for record in data.get("artifacts") or []:
        if not isinstance(record, dict):
            continue
        path = _existing_path(record.get("path"))
        if not path:
            continue
        key = str(record.get("key") or "")
        item = _capture_path_alias(state, key=key, path=path, source_key=f"artifact_index:{key or 'path'}")
        if item:
            captured.append(item)
    return captured


def _looks_error(result: Any) -> bool:
    parsed = _json_safe(result)
    if isinstance(parsed, dict):
        status = str(parsed.get("status") or "").lower()
        if status in {"error", "failed", "mock"}:
            return True
        if "error" in parsed or "error_type" in parsed:
            return True
    text = result if isinstance(result, str) else json.dumps(parsed, ensure_ascii=False)
    return any(marker.lower() in text.lower() for marker in _ERROR_MARKERS)


def _error_type(result: Any) -> str:
    text = result if isinstance(result, str) else json.dumps(_json_safe(result), ensure_ascii=False)
    lowered = text.lower()
    if "tool not found" in lowered or "no handler" in lowered:
        return "tool_not_found"
    if "out of memory" in lowered or "cuda" in lowered or "oom" in lowered:
        return "oom"
    if any(marker in lowered for marker in _PROVIDER_ACCOUNT_ERROR_MARKERS):
        return "provider_account_error"
    if any(marker in lowered for marker in _INFRA_ERROR_MARKERS):
        return "infra_or_provider"
    if any(marker in lowered for marker in _SERVICE_RUNTIME_ERROR_MARKERS):
        return "infra_or_provider"
    if "missing" in lowered and ("required" in lowered or "parameter" in lowered):
        return "missing_required"
    if "error in calculator" in lowered or "error in solver" in lowered or "error in plot" in lowered:
        return "tool_error"
    if "invalid" in lowered or "bad request" in lowered or "validation" in lowered:
        return "invalid_argument"
    if (
        "not found" in lowered
        or "does not exist" in lowered
        or "no such file" in lowered
        or "no such layer" in lowered
        or "cannot find table" in lowered
    ):
        return "missing_artifact_or_layer"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    return "tool_error"


def _compact_text(text: str, limit: int = 1800) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
    return text[:limit] + ("\n...[truncated]" if len(text) > limit else "")


def _prediction_from_observations(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return ""
    # Keep the final observation first because OEA gold often ends with the
    # exact measurement/result needed for answer judging.
    selected = observations[-4:]
    parts = []
    for item in selected:
        parts.append(f"Tool {item['tool']} observation:\n{_compact_text(item.get('content', ''), 2600)}")
    return "\n\n".join(parts)


@contextmanager
def _task_environment(row: dict[str, Any], artifact_dir: Path):
    keys = (
        "TERRABOX_ARTIFACT_OUTPUT_DIR",
        "TERRABOX_GPKG_OUTPUT_DIR",
        "TERRABOX_TASK_DATA_DIR",
        "TERRABOX_TASK_DATA_FILES",
        *_GOLD_REPLAY_ENV_DEFAULTS.keys(),
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in _GOLD_REPLAY_ENV_DEFAULTS.items():
            os.environ.setdefault(key, value)
        os.environ["TERRABOX_ARTIFACT_OUTPUT_DIR"] = str(artifact_dir)
        os.environ["TERRABOX_GPKG_OUTPUT_DIR"] = str(artifact_dir / "gpkg")
        data_dir = str(row.get("data_dir") or "").strip()
        data_files = [str(path) for path in (row.get("data_files") or []) if path]
        if data_dir:
            os.environ["TERRABOX_TASK_DATA_DIR"] = data_dir
        else:
            os.environ.pop("TERRABOX_TASK_DATA_DIR", None)
        if data_files:
            os.environ["TERRABOX_TASK_DATA_FILES"] = json.dumps(data_files, ensure_ascii=False)
        else:
            os.environ.pop("TERRABOX_TASK_DATA_FILES", None)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "gpkg").mkdir(parents=True, exist_ok=True)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _reset_legacy_gpkg_state() -> None:
    try:
        from terrabox.agent import tool_executor

        tool_executor._gpkg_by_scope.pop("legacy", None)  # type: ignore[attr-defined]
    except Exception:
        pass


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _existing_replay_is_final(path: Path) -> bool:
    """Return whether an existing replay result should be kept on resume."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if str(data.get("status") or "") == "completed":
        if bool(data.get("has_tool_error")):
            return False
        replay_meta = data.get("replay_meta") or {}
        for observation in replay_meta.get("observations") or []:
            if isinstance(observation, dict) and (
                observation.get("is_error") or _looks_error(observation.get("content_preview", ""))
            ):
                return False
        return True
    replay_meta = data.get("replay_meta") or {}
    failure_type = str(replay_meta.get("failure_type") or "").lower()
    return bool(data.get("has_tool_oom") or failure_type == "oom")


def _is_transient_replay_failure(result: dict[str, Any]) -> bool:
    if str(result.get("status") or "") == "completed":
        return False
    replay_meta = result.get("replay_meta") or {}
    failure_type = str(replay_meta.get("failure_type") or "").lower()
    return failure_type in _TRANSIENT_REPLAY_FAILURE_TYPES


def _load_tool_runtime(*, use_docker: bool) -> None:
    if use_docker:
        os.environ.setdefault("TERRABOX_USE_DOCKER", "true")
    from terrabox.extensions import load_builtin_toolkits

    load_builtin_toolkits()


def replay_one_task(
    row: dict[str, Any],
    *,
    index: int,
    out_dir: str | Path,
    use_docker: bool = True,
) -> dict[str, Any]:
    """Execute one row's gold calls and return a rollout-compatible result."""

    del use_docker  # Runtime has already been loaded by the batch runner.
    from terrabox.agent.tool_executor import AgentToolExecutor

    started = time.time()
    task_id = _task_id(row, index)
    expected = _expected_tools(row)
    calls = [call for call in (row.get("gold_tool_calls") or []) if isinstance(call, dict)]
    results_root = Path(out_dir)
    artifact_dir = results_root / "artifacts" / task_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    observations: list[dict[str, Any]] = []
    conversation_history: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    alias_state = _build_gold_alias_state(row)
    alias_resolutions: list[dict[str, Any]] = []
    alias_captures: list[dict[str, Any]] = []
    failed_step: int | None = None
    failure_type: str | None = None
    failure_message = ""

    _reset_legacy_gpkg_state()
    with _task_environment(row, artifact_dir):
        for step_idx, call in enumerate(calls, start=1):
            slug = str(call.get("tool") or call.get("slug") or "").strip()
            arguments = copy.deepcopy(call.get("arguments") or {})
            if not slug:
                failed_step = step_idx
                failure_type = "missing_tool_name"
                failure_message = "gold call has no tool/slug"
                break
            step_resolutions: list[dict[str, Any]] = []
            arguments = _drop_null_arguments(arguments)
            arguments = _resolve_gold_aliases(arguments, alias_state, step_resolutions)
            if step_resolutions:
                alias_resolutions.extend({"step": step_idx, "tool": slug, **item} for item in step_resolutions)

            tool_calls.append(slug)
            conversation_history.append(
                {
                    "type": "AIMessage",
                    "content": json.dumps(
                        {
                            "thought": f"[gold replay] execute gold step {step_idx}",
                            "actions": [
                                {
                                    "tool": slug,
                                    "function_name": slug.replace(".", "__"),
                                    "arguments": arguments,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )[:2000],
                    "tool_calls": [
                        {
                            "name": slug,
                            "args": arguments,
                            "id": f"gold_{task_id}_{step_idx}",
                        }
                    ],
                }
            )

            try:
                result = AgentToolExecutor.execute(slug, arguments, user=None)
            except Exception as exc:  # Defensive; AgentToolExecutor usually returns error text.
                result = f"Tool execution error: {type(exc).__name__}: {exc}"

            content = result if isinstance(result, str) else json.dumps(_json_safe(result), ensure_ascii=False)
            step_captures = _capture_gold_result_aliases(content, alias_state)
            step_captures.extend(_capture_artifact_index_aliases(artifact_dir, alias_state))
            if step_captures:
                alias_captures.extend({"step": step_idx, "tool": slug, **item} for item in step_captures)
            observations.append(
                {
                    "step": step_idx,
                    "tool": slug,
                    "arguments": arguments,
                    "content": content,
                    "is_error": _looks_error(content),
                }
            )
            conversation_history.append(
                {
                    "type": "ToolMessage",
                    "content": content[:2000],
                    "tool_call_id": f"gold_{task_id}_{step_idx}",
                    "name": slug,
                }
            )

            if _looks_error(content):
                failed_step = step_idx
                failure_type = _error_type(content)
                failure_message = content[:2000]
                break

    status = "completed" if failed_step is None and bool(calls) else "failed"
    if not calls:
        failure_type = "missing_gold_tool_calls"
        failure_message = "row has no gold_tool_calls"
        status = "failed"

    final_answer = _prediction_from_observations(observations)
    if status != "completed" and failure_message:
        final_answer = f"Gold replay failed at step {failed_step}: {failure_message}"

    metrics = _tool_metrics(tool_calls, expected)
    has_tool_error = status != "completed" or any(item.get("is_error") for item in observations)
    result = {
        "task_id": task_id,
        "source": row.get("source", "openearth"),
        "question": row.get("question", ""),
        "expected_tools": expected,
        "allowed_slugs": None,
        "task_type": row.get("task_type") or row.get("type") or "general",
        "tool_calls": tool_calls,
        "tool_calls_deduped": list(dict.fromkeys(tool_calls)),
        "metrics": metrics,
        "status": status,
        "success": status == "completed" and metrics.get("f1", 0.0) >= 0.999,
        "real_success": status == "completed" and metrics.get("f1", 0.0) >= 0.999,
        "system_limitation_acknowledged": False,
        "llm_calls": 0,
        "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "time": time.time() - started,
        "final_answer_preview": final_answer[:500],
        "final_answer_full": final_answer,
        "has_tool_error": has_tool_error,
        "has_tool_oom": bool(failure_type == "oom"),
        "conversation_history": conversation_history,
        "replay_meta": {
            "method": "gold_teacher_forced_replay",
            "actor_llm_used": False,
            "gold_arguments_visible_to_actor": False,
            "num_gold_calls": len(calls),
            "failed_step": failed_step,
            "failure_type": failure_type,
            "failure_message": failure_message[:1000] if failure_message else "",
            "alias_resolutions": alias_resolutions,
            "alias_captures": alias_captures,
            "observations": [
                {
                    "step": item["step"],
                    "tool": item["tool"],
                    "is_error": item["is_error"],
                    "content_preview": item["content"][:500],
                }
                for item in observations
            ],
            "artifact_dir": str(artifact_dir),
        },
    }
    return result


def _write_derived_outputs(out_dir: Path, *, data_path: str | Path, total_tasks: int) -> dict[str, Any]:
    results_dir = out_dir / "results"
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(row, dict):
            row.setdefault("task_id", path.stem)
            rows.append(row)

    success_count = sum(1 for row in rows if row.get("success"))
    report = {
        "experiment": out_dir.name,
        "mode": "gold_teacher_forced_replay",
        "task_file": str(data_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_tasks": len(rows),
        "expected_total_tasks": total_tasks,
        "progress": {
            "done": len(rows),
            "total": total_tasks,
            "pct": (100 * len(rows) / total_tasks) if total_tasks else 0.0,
        },
        "success_count": success_count,
        "success_rate": success_count / len(rows) if rows else 0.0,
        "total_tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "avg_f1": sum(row.get("metrics", {}).get("f1", 0.0) for row in rows) / len(rows) if rows else 0.0,
        "status_distribution": dict(Counter(str(row.get("status") or "unknown") for row in rows)),
        "failure_type_distribution": dict(
            Counter(
                str((row.get("replay_meta") or {}).get("failure_type") or "none")
                for row in rows
                if str(row.get("status") or "") != "completed"
            )
        ),
        "unique_tools_called": sorted({tool for row in rows for tool in (row.get("tool_calls_deduped") or [])}),
        "derived_from_results": True,
        "gold_replay": {
            "actor_llm_used": False,
            "final_answer_full_is_replay_evidence": True,
            "answer_judge_required_for_correct_conclusion": True,
        },
    }

    traj_path = out_dir / "trajectories.jsonl"
    full_path = out_dir / "trajectories_full.jsonl"
    with traj_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "task_id": row.get("task_id"),
                        "source": row.get("source", "openearth"),
                        "query": row.get("question", ""),
                        "tool_sequence": row.get("tool_calls_deduped", []),
                        "tools_called": row.get("tool_calls", []),
                        "expected_tools": row.get("expected_tools", []),
                        "f1": row.get("metrics", {}).get("f1", 0.0),
                        "reward": 1.0 if row.get("success") else 0.0,
                        "task_type": row.get("task_type", "general"),
                        "status": row.get("status", "unknown"),
                        "real_success": row.get("real_success", row.get("success", False)),
                        "system_limitation_acknowledged": False,
                        "has_tool_error": row.get("has_tool_error", False),
                        "has_tool_oom": row.get("has_tool_oom", False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with full_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "task_id": row.get("task_id"),
                        "source": row.get("source", "openearth"),
                        "question": row.get("question", ""),
                        "expected_tools": row.get("expected_tools", []),
                        "tool_sequence": row.get("tool_calls_deduped", []),
                        "tools_called": row.get("tool_calls", []),
                        "metrics": row.get("metrics", {}),
                        "status": row.get("status", "unknown"),
                        "success": row.get("success", False),
                        "real_success": row.get("real_success", row.get("success", False)),
                        "tokens": row.get("tokens", {}),
                        "llm_calls": row.get("llm_calls", 0),
                        "time": row.get("time", 0.0),
                        "final_answer": row.get("final_answer_full", ""),
                        "conversation_history": row.get("conversation_history", []),
                        "replay_meta": row.get("replay_meta", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    _atomic_json(out_dir / "report.json", report)
    return report


def replay_gold_data(
    data_path: str | Path,
    *,
    out_dir: str | Path,
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
    subset_file: str | Path | None = None,
    task_ids: list[str] | None = None,
    resume: bool = True,
    use_docker: bool = True,
    scope: str = "all",
    gpu_class: str = "any",
    max_transient_retries: int = 3,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Replay a batch and write rollout-compatible results."""

    if scope not in {"all", "online", "offline"}:
        raise ValueError("scope must be all, online, or offline")
    if gpu_class not in {"any", "gpu", "nogpu"}:
        raise ValueError("gpu_class must be any, gpu, or nogpu")

    rows = _read_json_or_jsonl(data_path)
    task_id_filter = {str(tid) for tid in (task_ids or []) if str(tid)}
    task_id_filter.update(_load_subset_ids(subset_file))
    indexed = [(i, row) for i, row in enumerate(rows) if _keep_by_scope(row, scope=scope, gpu_class=gpu_class)]
    if task_id_filter:
        indexed = [(i, row) for i, row in indexed if _task_id(row, i) in task_id_filter]
    if end_index is None:
        end_index = len(indexed)
    selected = indexed[start_index:end_index]
    if limit is not None:
        selected = selected[:limit]

    out = Path(out_dir)
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    _load_tool_runtime(use_docker=use_docker)

    done = skipped = retried_existing_failed = 0
    for local_pos, (row_index, row) in enumerate(selected, start=1):
        tid = _task_id(row, row_index)
        result_path = results_dir / f"{tid}.json"
        if resume and result_path.exists() and _existing_replay_is_final(result_path):
            skipped += 1
            continue
        if resume and result_path.exists():
            retried_existing_failed += 1
        attempts: list[dict[str, Any]] = []
        retries_left = max(0, int(max_transient_retries))
        for attempt_index in range(retries_left + 1):
            result = replay_one_task(row, index=row_index, out_dir=out, use_docker=use_docker)
            replay_meta = result.setdefault("replay_meta", {})
            failure = replay_meta.get("failure_type")
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": result.get("status"),
                    "failure_type": failure,
                    "time": round(float(result.get("time") or 0.0), 3),
                }
            )
            if not _is_transient_replay_failure(result) or attempt_index >= retries_left:
                break
            wait_seconds = min(60, 5 * (attempt_index + 1))
            print(
                f"[gold-replay] transient failure for {tid} "
                f"(failure={failure}); retry {attempt_index + 1}/{retries_left} after {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        result.setdefault("replay_meta", {})["attempts"] = attempts
        result.setdefault("replay_meta", {})["transient_retries_used"] = max(0, len(attempts) - 1)
        _atomic_json(result_path, result)
        done += 1
        if progress_every and (done % progress_every == 0 or local_pos == len(selected)):
            status = result.get("status")
            failure = (result.get("replay_meta") or {}).get("failure_type")
            print(
                f"[gold-replay] {done} new / {len(selected)} selected "
                f"(skipped={skipped}) last={tid} status={status} failure={failure}",
                flush=True,
            )

    report = _write_derived_outputs(out, data_path=data_path, total_tasks=len(indexed))
    report["batch"] = {
        "data_path": str(data_path),
        "out_dir": str(out),
        "start_index": start_index,
        "end_index": end_index,
        "limit": limit,
        "subset_file": str(subset_file) if subset_file else None,
        "task_ids": sorted(task_id_filter) if task_id_filter else None,
        "scope": scope,
        "gpu_class": gpu_class,
        "max_transient_retries": max_transient_retries,
        "selected": len(selected),
        "newly_replayed": done,
        "skipped_existing": skipped,
        "retried_existing_failed": retried_existing_failed,
        "resume": resume,
    }
    _atomic_json(out / "report.json", report)
    return report
