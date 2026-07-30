"""Extract artifact transitions from historical Terrabox rollout results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from ...agent.artifacts import initial_artifact_state, update_artifact_state
from ...agent.artifacts.extractors import parse_tool_observation
from ...agent.artifacts.state import FILE_PARAM_MARKERS, infer_artifact_kind, looks_like_path
from .schemas import TransitionRecord


_OUTPUT_KEYS = {"output_path", "result_path", "preview_path", "out_file", "artifact_path", "gpkg"}
_SOURCE_KEYS = {"task_id", "question", "query", "source", "task_type", "status", "metrics", "success"}


def canonical_slug(name: str) -> str:
    return str(name or "").replace("__", ".")


def load_rollout_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
            continue
        if path.is_file() and path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(data.get("result", data) if isinstance(data, dict) else data)
            continue
        root = path
        if (root / "trajectories_full.jsonl").exists():
            with (root / "trajectories_full.jsonl").open(encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
            continue
        if root.name != "results" and (root / "results").exists():
            root = root / "results"
        if root.exists():
            for item in sorted(root.glob("*.json")):
                try:
                    data = json.loads(item.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    rows.append(data.get("result", data))
    return [row for row in rows if isinstance(row, dict)]


def iter_tool_observations(history: list[dict[str, Any]]) -> Iterable[tuple[int, str, dict[str, Any], str]]:
    pending: list[tuple[str, dict[str, Any]]] = []
    step_index = 0
    for msg in history:
        for call in msg.get("tool_calls") or []:
            pending.append((canonical_slug(call.get("name", "")), dict(call.get("args") or {})))
        content = msg.get("content")
        if not pending or not isinstance(content, str):
            continue
        content_stripped = content.strip()
        if not content_stripped:
            continue
        msg_type = str(msg.get("type") or "")
        is_observation = (
            msg_type == "ToolMessage"
            or content_stripped.startswith("OBSERVATION:")
            or content_stripped.startswith("Tool execution error:")
            or (msg_type not in {"AIMessage", "HumanMessage"} and content_stripped.startswith("{"))
        )
        if not is_observation:
            # Some malformed/invalid tool-call turns are followed by normal AI
            # text rather than a ToolMessage. Drop those pending calls instead
            # of treating the assistant's prose as a tool observation.
            if msg_type in {"AIMessage", "HumanMessage"}:
                pending.clear()
            continue
        if content_stripped.startswith("OBSERVATION:"):
            content = content_stripped.removeprefix("OBSERVATION:").strip()
        slug, args = pending.pop(0)
        yield step_index, slug, args, content
        step_index += 1


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _clean_label(value: str, limit: int = 40) -> str:
    value = re.sub(r"[^a-zA-Z0-9_./:-]+", "_", str(value).strip().lower()).strip("_")
    if not value:
        return ""
    if looks_like_path(value):
        suffix = Path(value).suffix.lower().lstrip(".") or "file"
        return f"{suffix}_file"
    return value[:limit]


def _state_signature(state: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for artifact in state.get("artifacts", []):
        kind = _clean_label(str(artifact.get("kind") or "file"))
        source = _clean_label(str(artifact.get("source") or ""), limit=28)
        parts.append(f"{kind}:{source}" if source and source != "question" else kind)
    for layer in state.get("layers", []):
        name = _clean_label(str(layer.get("name") or "layer"), limit=32)
        parts.append(f"vector_layer:{name}")
    if not parts:
        return ["task_request"]
    return list(dict.fromkeys(parts))


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"message": text[:1000]}


def _summarize_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        if looks_like_path(value):
            return f"<{infer_artifact_kind(value, key)}_path>"
        return value[:160]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_summarize_value(key, item) for item in value[:5]]
    if isinstance(value, dict):
        return {str(k): _summarize_value(str(k), v) for k, v in list(value.items())[:8]}
    return str(value)[:160]


def summarize_args(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in args.items():
        lowered = key.lower()
        if lowered in _OUTPUT_KEYS:
            out[key] = f"<{infer_artifact_kind(str(value), key)}_output>"
        elif any(marker in lowered for marker in FILE_PARAM_MARKERS):
            out[key] = _summarize_value(key, value)
        else:
            out[key] = _summarize_value(key, value)
    return out


def summarize_observation(parsed: dict[str, Any], text: str) -> dict[str, Any]:
    keys = [str(key) for key in parsed.keys()]
    summary: dict[str, Any] = {"keys": keys[:20]}
    for key, value in parsed.items():
        lowered = key.lower()
        if lowered in {"status", "error_type", "feature_count", "poi_count", "count", "distance", "distance_m"}:
            summary[key] = value
        elif lowered in _OUTPUT_KEYS or "path" in lowered or "gpkg" in lowered:
            summary[key] = _summarize_value(key, value)
        elif lowered in {"text", "message", "warning"}:
            summary[key] = str(value)[:300]
    if "text" not in summary and "message" not in summary:
        summary["preview"] = text[:300]
    return summary


def _input_signature(slug: str, args: dict[str, Any], before_parts: list[str]) -> str:
    file_params: list[str] = []
    for key, value in args.items():
        lowered = key.lower()
        if lowered in _OUTPUT_KEYS:
            continue
        if any(marker in lowered for marker in FILE_PARAM_MARKERS):
            kind = infer_artifact_kind(str(value), key)
            file_params.append(f"{kind}:{_clean_label(key, 32)}")
    if file_params:
        return " + ".join(sorted(dict.fromkeys(file_params)))
    if before_parts:
        return " + ".join(before_parts)
    return "task_request"


def _delta_parts(before_parts: list[str], after_parts: list[str]) -> list[str]:
    before_set = set(before_parts)
    return [part for part in after_parts if part not in before_set]


def _output_signature(slug: str, args: dict[str, Any], parsed: dict[str, Any],
                      before_parts: list[str], after_parts: list[str]) -> str:
    delta = _delta_parts(before_parts, after_parts)
    if delta:
        return " + ".join(delta)
    if slug == "osm_gis.get_area_boundary":
        return "gpkg:area_boundary"
    if slug == "osm_gis.add_pois_layer":
        layer = _clean_label(str(args.get("layer_name") or parsed.get("layer_name") or "poi_layer"), 40)
        return f"vector_layer:{layer}"
    if slug == "osm_gis.add_index_layer":
        layer = _clean_label(str(args.get("layer_name") or parsed.get("layer_name") or args.get("index_type") or "index_layer"), 40)
        return f"index_layer:{layer}"
    if slug == "osm_gis.compute_route_dist":
        return "route_distance_result"
    if slug == "osm_gis.compute_index_change":
        layer = _clean_label(str(args.get("output_layer") or args.get("layer_name") or parsed.get("layer_name") or "index_change"), 40)
        return f"index_change_result:{layer}"
    if slug == "osm_gis.show_index_layer":
        return "index_layer_preview"
    if slug == "osm_gis.get_bbox_from_raster":
        return "bbox_result"
    if slug == "osm_gis.display_on_map":
        return "map_visualization"
    if slug.startswith("bing_search."):
        return "search_results"
    if slug.endswith("vlm_analyze"):
        return "visual_text_answer"
    if any(token in slug for token in ("segment", "sam", "detect", "rcnn", "change_os")):
        if any(key in parsed for key in ("positive_pixels", "pixel_counts", "detections", "objects")):
            return "perception_counts"
        return "perception_result"
    if any(key in parsed for key in ("output_path", "result_path", "gpkg")):
        value = parsed.get("output_path") or parsed.get("result_path") or parsed.get("gpkg")
        return infer_artifact_kind(str(value), "output_path")
    numeric_keys = [key for key, value in parsed.items() if isinstance(value, (int, float))]
    if numeric_keys:
        return "numeric_result:" + "+".join(sorted(numeric_keys[:4]))
    if "text" in parsed or "message" in parsed:
        return "text_result"
    return "tool_observation"


_INFRA_PATTERNS = (
    # Provider / network jitter.
    "timeout",
    "timed out",
    "read timed out",
    "rate limit",
    "429",
    "provider overload",
    "overloaded",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "503",
    "504",
    "connection",
    "network",
    "proxy",
    "temporary failure",
    # Resource / host-service failures. These are explicitly not ExperienceEvo
    # risks because they do not tell us the product/tool experience is bad.
    "cuda out of memory",
    "out of memory",
    "oom",
    "maximum context",
    "context length",
    "docker",
    "container",
    "service is not healthy",
)

_RISK_PATTERNS = (
    # LLM/tool-use mistakes that should affect ExperienceEvo risk.
    "missing required",
    "required parameter",
    "invalid parameter",
    "invalid argument",
    "invalid value",
    "schema",
    "validation",
    "no such file",
    "does not exist",
    "missing artifact",
    "unknown layer",
    "layer not found",
    "file not found",
    "invalid geometry",
    "invalid bbox",
    "invalid expression",
)


def _issues(row: dict[str, Any], is_error: bool, parsed: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    text = json.dumps(parsed, ensure_ascii=False).lower()
    if is_error:
        issues.append("tool_error")
    if row.get("has_tool_oom") or any(pattern in text for pattern in _INFRA_PATTERNS):
        issues.append("infra_error")
    if parsed.get("warning"):
        issues.append("tool_warning")
    if any(pattern in text for pattern in _RISK_PATTERNS):
        issues.append("experience_risk")
    return sorted(set(issues))


def _is_infra_error(issues: list[str]) -> bool:
    return "infra_error" in issues


def _experience_risk(issues: list[str]) -> float:
    # Infra failures are filtered out before distillation and should not make a
    # reusable experience look risky. A deterministic infra failure plus a
    # parameter error is still excluded from Q/R/N updates by infra_error=True.
    if "infra_error" in issues:
        return 0.0
    return 1.0 if "experience_risk" in issues else 0.0


def extract_transitions_from_row(
    row: dict[str, Any],
    *,
    source_name: str = "",
) -> list[TransitionRecord]:
    task_id = str(row.get("task_id") or row.get("id") or _short_hash(str(row.get("question", ""))))
    question = str(row.get("question") or row.get("query") or "")
    state = initial_artifact_state(question, image_paths=row.get("images") or [])
    metrics = row.get("metrics") or {}
    try:
        reward = float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)
    except (TypeError, ValueError):
        reward = 1.0 if row.get("success") else 0.0
    records: list[TransitionRecord] = []
    for step_index, slug, args, observation in iter_tool_observations(row.get("conversation_history") or []):
        before_parts = _state_signature(state)
        parsed, is_error = parse_tool_observation(observation)
        update_artifact_state(state, slug, args, observation)
        after_parts = _state_signature(state)
        input_sig = _input_signature(slug, args, before_parts)
        output_sig = _output_signature(slug, args, parsed, before_parts, after_parts)
        issues = _issues(row, is_error, parsed)
        transition_key = "|".join([task_id, str(step_index), slug, input_sig, output_sig])
        records.append(
            TransitionRecord(
                transition_id=_short_hash(transition_key),
                task_id=task_id,
                source=source_name or str(row.get("source") or "rollout"),
                task_type=str(row.get("task_type") or row.get("type") or "general"),
                question=question,
                step_index=step_index,
                tool=slug,
                input_signature=input_sig,
                output_signature=output_sig,
                before_state=before_parts,
                after_state=after_parts,
                args_summary=summarize_args(args),
                observation_summary=summarize_observation(parsed, observation),
                status="failed" if is_error else "success",
                reward=reward,
                risk=_experience_risk(issues),
                infra_error=_is_infra_error(issues),
                issues=issues,
            )
        )
    return records


def extract_transitions(
    paths: Iterable[str | Path],
    *,
    source_name: str = "",
    completed_only: bool = True,
    min_reward: float = 0.0,
    max_tasks: int | None = None,
) -> list[TransitionRecord]:
    rows = load_rollout_rows(paths)
    transitions: list[TransitionRecord] = []
    used_tasks = 0
    for row in rows:
        status = str(row.get("status", "")).lower()
        if completed_only and status and status not in {"completed", "completed_with_recovery"}:
            continue
        metrics = row.get("metrics") or {}
        try:
            reward = float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)
        except (TypeError, ValueError):
            reward = 1.0 if row.get("success") else 0.0
        if reward < min_reward:
            continue
        task_records = extract_transitions_from_row(row, source_name=source_name)
        if not task_records:
            continue
        transitions.extend(task_records)
        used_tasks += 1
        if max_tasks is not None and used_tasks >= max_tasks:
            break
    return transitions
