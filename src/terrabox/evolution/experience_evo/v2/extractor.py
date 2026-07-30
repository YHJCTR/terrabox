"""Extract locally scored product-transition events from historical rollouts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from ....agent.artifacts import (
    initial_artifact_state,
    product_state_delta,
    product_state_tokens,
    update_artifact_state,
)
from ....agent.artifacts.contracts import get_contract
from ....agent.artifacts.extractors import parse_tool_observation
from ....agent.artifacts.state import artifact_paths, infer_artifact_kind, looks_like_path
from ..transition_extractor import iter_tool_observations, load_rollout_rows
from .models import LocalEvidence, TransitionEvent


_INFRA_PATTERNS = (
    "timeout",
    "timed out",
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
    "cuda out of memory",
    "out of memory",
    "oom",
    "maximum context",
    "context length",
    "docker",
    "container",
    "service is not healthy",
)

_ATTRIBUTABLE_PATTERNS = (
    "missing required",
    "required parameter",
    "invalid parameter",
    "invalid argument",
    "invalid value",
    "invalid syntax",
    "forbidden operation",
    "schema",
    "validation",
    "no such file",
    "does not exist",
    "missing artifact",
    "unknown layer",
    "layer not found",
    "file not found",
    "not found:",
    "could not open layer",
    "missing layers",
    "invalid geometry",
    "invalid bbox",
    "invalid expression",
    "returned none",
)

_INTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("count", r"\b(count|number of|how many)\b"),
    ("distance", r"\b(distance|nearest|closest|route|proximity)\b"),
    ("segment", r"\b(segment|mask|pixel area|pixels?)\b"),
    ("detect", r"\b(detect|locate|find|bounding box|bbox)\b"),
    ("attribute", r"\b(attribute|color|condition|orientation|heading|describe)\b"),
    ("change", r"\b(change|difference|before and after|pre-event|post-event)\b"),
    ("index", r"\b(ndvi|ndbi|nbr|spectral index|vegetation index)\b"),
    ("boundary", r"\b(boundary|study area|area boundary)\b"),
    ("poi", r"\b(poi|point of interest|hospital|school|station|restaurant|park)\b"),
    ("visualize", r"\b(display|visualize|render|draw|plot|map)\b"),
    ("search", r"\b(search|look up|web)\b"),
    ("calculate", r"\b(calculate|compute|solve|ratio|percentage|proportion|average)\b"),
    ("ocr", r"\b(ocr|read text|extract text)\b"),
    ("raster", r"\b(raster|geotiff|satellite image|aerial image)\b"),
)

_OUTPUT_ARTIFACT_ARG_KEYS = {
    "output_path",
    "result_path",
    "preview_path",
    "out_file",
    "artifact_path",
}

_OUTPUT_LAYER_ARG_KEYS = {
    "layer_name",
    "diff_layer_name",
    "output_layer",
}

_OUTPUT_ARG_KEYS = {
    *_OUTPUT_ARTIFACT_ARG_KEYS,
    *_OUTPUT_LAYER_ARG_KEYS,
}


def _short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def intent_signature(question: str, task_type: str = "general") -> tuple[str, str]:
    """Build a location-free intent signature from a fixed domain vocabulary."""
    lowered = (question or "").lower()
    terms = sorted(name for name, pattern in _INTENT_PATTERNS if re.search(pattern, lowered))
    normalized_type = re.sub(r"[^a-z0-9_]+", "_", (task_type or "general").lower()).strip("_")
    if not terms:
        terms = [normalized_type or "general"]
    signature = "+".join(terms)
    if normalized_type not in {"", "general", "unknown"}:
        signature = f"{normalized_type}:{signature}"
    return signature, " / ".join(terms)


def _sanitize_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if lowered in _OUTPUT_ARTIFACT_ARG_KEYS:
        return "<output_artifact_reference>"
    if lowered in _OUTPUT_LAYER_ARG_KEYS:
        return "<output_layer_name>"
    if any(token in lowered for token in ("path", "file", "gpkg", "image", "raster", "geotiff")):
        return "<artifact_reference>"
    if "layer" in lowered:
        return "<layer_name>"
    if lowered in {"area", "place", "location", "region", "address", "boundary"}:
        return "<named_area>"
    if lowered in {"text", "label", "object", "target", "prompt", "description", "query"}:
        return "<task_specific_value>"
    if lowered in {"expression", "command", "code"}:
        return "<computation>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value[:5]]
    if isinstance(value, dict):
        return {str(k): _sanitize_value(str(k), v) for k, v in list(value.items())[:8]}
    text = str(value).strip()
    if looks_like_path(text):
        return f"<{infer_artifact_kind(text, key)}_reference>"
    if len(text) <= 24 and re.fullmatch(r"[A-Z0-9_.:-]+", text):
        return text
    return "<task_specific_value>"


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_value(str(key), value) for key, value in args.items()}


def sanitize_observation(parsed: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {"keys": sorted(str(key) for key in parsed)[:30]}
    for key, value in parsed.items():
        lowered = str(key).lower()
        if lowered in {
            "status",
            "error_type",
            "feature_count",
            "poi_count",
            "count",
            "pair_count",
            "distance",
            "distance_m",
            "pixel_count",
            "positive_pixels",
            "layer_saved",
        }:
            safe[str(key)] = value
        elif lowered in _OUTPUT_ARG_KEYS or "path" in lowered or lowered == "gpkg":
            safe[str(key)] = "<artifact_reference>"
        elif "layer" in lowered:
            safe[str(key)] = "<layer_name>"
    return safe


def _initial_state(row: dict[str, Any], question: str) -> dict[str, Any]:
    state = initial_artifact_state(question, image_paths=list(row.get("images") or []))
    existing = {str(item.get("path")) for item in state.get("artifacts", [])}
    for raw_path in row.get("data_files") or []:
        path = str(raw_path)
        if path and path not in existing:
            state["artifacts"].append(
                {"kind": infer_artifact_kind(path), "path": path, "source": "task_file"}
            )
            existing.add(path)
    return state


def _kind_matches(token: str, kind: str) -> bool:
    return token == f"input:{kind}" or token.startswith(f"{kind}:from:")


def _required_product_state(
    slug: str,
    args: dict[str, Any],
    state: dict[str, Any],
    before_tokens: list[str],
) -> list[str]:
    contract = get_contract(slug)
    if slug == "osm_gis.get_bbox_from_raster" and args.get("gpkg"):
        required_kinds = (("gpkg", 1), ("vector_layer", 1))
    elif contract is not None:
        required_kinds = tuple((need.kind, need.count) for need in contract.inputs)
    else:
        required_kinds = ()

    selected: list[str] = []
    for kind, count in required_kinds:
        matches = [token for token in before_tokens if _kind_matches(token, kind)]
        selected.extend(matches[-count:])

    if selected:
        return sorted(selected)
    if contract is not None and not contract.inputs:
        return ["task_request"]

    referenced_paths = {
        str(value)
        for key, value in args.items()
        if key not in _OUTPUT_ARG_KEYS and isinstance(value, str) and looks_like_path(value)
    }
    if referenced_paths:
        mini_state = {
            "artifacts": [
                artifact
                for artifact in state.get("artifacts", [])
                if str(artifact.get("path")) in referenced_paths
            ],
            "layers": [],
            "successful_calls": [],
        }
        referenced = product_state_tokens(mini_state)
        if referenced != ["task_request"]:
            return referenced

    layer_values: set[str] = set()
    for key, value in args.items():
        if "layer" not in key.lower():
            continue
        if isinstance(value, list):
            layer_values.update(str(item) for item in value)
        elif value is not None:
            layer_values.add(str(value))
    if layer_values:
        mini_state = {
            "artifacts": [],
            "layers": [
                layer for layer in state.get("layers", []) if str(layer.get("name")) in layer_values
            ],
            "successful_calls": [],
        }
        referenced = product_state_tokens(mini_state)
        if referenced != ["task_request"]:
            return referenced

    return before_tokens if before_tokens else ["task_request"]


def _input_binding_score(
    slug: str,
    args: dict[str, Any],
    state: dict[str, Any],
    error_text: str,
) -> float:
    lowered_error = error_text.lower()
    if any(pattern in lowered_error for pattern in _ATTRIBUTABLE_PATTERNS):
        return 0.0
    contract = get_contract(slug)
    if contract is None or not contract.inputs:
        return 1.0

    checks: list[bool] = []
    layers = state.get("layers", [])
    for need in contract.inputs:
        if need.param:
            value = args.get(need.param)
            checks.append(value not in (None, "", []))
            if isinstance(value, str) and looks_like_path(value):
                checks.append(value in artifact_paths(state))
        if need.kind.endswith("_layer"):
            matching = [
                layer
                for layer in layers
                if str(layer.get("kind") or "vector_layer") == need.kind
            ]
            checks.append(len(matching) >= need.count)
        else:
            checks.append(len(artifact_paths(state, need.kind)) >= need.count)
    return sum(checks) / len(checks) if checks else 1.0


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return [text] if len(text) >= 2 else []
    return []


def _output_references(args: dict[str, Any], parsed: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key, value in parsed.items():
        lowered = str(key).lower()
        if (
            lowered in _OUTPUT_ARG_KEYS
            or "path" in lowered
            or "layer" in lowered
            or lowered == "gpkg"
            or isinstance(value, (int, float))
        ):
            refs.update(_flatten_values(value))
    for key in _OUTPUT_ARG_KEYS:
        if key in args:
            refs.update(_flatten_values(args[key]))
    return refs


def _downstream_use(
    index: int,
    raw_events: list[dict[str, Any]],
) -> tuple[float | None, float]:
    later = raw_events[index + 1 :]
    current = raw_events[index]
    if not later:
        return None, 1.0 if not current["is_error"] else 0.0
    refs = _output_references(current["args"], current["parsed"])
    later_args = json.dumps([item["args"] for item in later], ensure_ascii=False)
    if refs:
        return (1.0 if any(ref in later_args for ref in refs) else 0.0), 0.0
    return 0.5, 0.0


def _local_reward(
    input_binding: float,
    output_valid: float,
    target_completed: float,
    downstream: float | None,
    terminal_usable: float,
) -> float:
    continuation = terminal_usable if downstream is None else downstream
    value = (
        0.20 * input_binding
        + 0.35 * output_valid
        + 0.30 * target_completed
        + 0.15 * continuation
    )
    return max(0.0, min(1.0, value))


def _attribution(
    row: dict[str, Any], parsed: dict[str, Any], observation: str, is_error: bool
) -> tuple[bool, float, list[str]]:
    text = (json.dumps(parsed, ensure_ascii=False) + " " + observation).lower()
    infra = bool(row.get("has_tool_oom")) or any(pattern in text for pattern in _INFRA_PATTERNS)
    if infra:
        return True, 0.0, ["infra_error"]
    reasons = [pattern for pattern in _ATTRIBUTABLE_PATTERNS if pattern in text]
    risk = 1.0 if is_error and reasons else 0.0
    return False, risk, sorted(set(reasons))[:8]


def extract_transition_events_from_row(
    row: dict[str, Any], *, source_name: str = ""
) -> list[TransitionEvent]:
    question = str(row.get("question") or row.get("query") or "")
    task_id = str(row.get("task_id") or row.get("id") or _short_hash(question, 12))
    task_type = str(row.get("task_type") or row.get("type") or "general")
    intent, hint = intent_signature(question, task_type)
    state = _initial_state(row, question)

    raw_events: list[dict[str, Any]] = []
    for step_index, slug, args, observation in iter_tool_observations(
        row.get("conversation_history") or []
    ):
        before_tokens = product_state_tokens(state)
        required_state = _required_product_state(slug, args, state, before_tokens)
        input_binding = _input_binding_score(slug, args, state, observation)
        parsed, is_error = parse_tool_observation(observation)
        update_artifact_state(state, slug, args, observation)
        after_tokens = product_state_tokens(state)
        target_state = product_state_delta(before_tokens, after_tokens)
        if not target_state and not is_error:
            target_state = [f"result:from:{slug}"]
        raw_events.append(
            {
                "step_index": step_index,
                "slug": slug,
                "args": args,
                "observation": observation,
                "parsed": parsed,
                "is_error": is_error,
                "required_state": required_state,
                "target_state": target_state,
                "after_state": after_tokens,
                "input_binding": input_binding,
            }
        )

    events: list[TransitionEvent] = []
    for index, raw in enumerate(raw_events):
        downstream, terminal = _downstream_use(index, raw_events)
        infra, risk, reasons = _attribution(
            row, raw["parsed"], raw["observation"], raw["is_error"]
        )
        output_valid = 0.0 if raw["is_error"] else 1.0
        target_completed = 1.0 if output_valid and raw["target_state"] else 0.0
        reward = _local_reward(
            raw["input_binding"], output_valid, target_completed, downstream, terminal
        )
        if infra:
            reward = 0.0
        evidence = LocalEvidence(
            input_binding_ok=raw["input_binding"],
            output_valid=output_valid,
            target_completed=target_completed,
            downstream_consumed=downstream,
            terminal_usable=terminal,
            reward=reward,
            risk_observed=risk,
            risk_reasons=reasons,
            infra_error=infra,
        )
        key = "|".join(
            [task_id, str(raw["step_index"]), raw["slug"], intent, "+".join(raw["target_state"])]
        )
        events.append(
            TransitionEvent(
                event_id=_short_hash(key),
                task_id=task_id,
                source=source_name or str(row.get("source") or "rollout"),
                task_type=task_type,
                intent_signature=intent,
                task_hint=hint,
                step_index=raw["step_index"],
                tool=raw["slug"],
                input_product_state=raw["required_state"],
                target_product_state=raw["target_state"],
                output_product_state=raw["after_state"],
                args_summary=sanitize_args(raw["args"]),
                observation_summary=sanitize_observation(raw["parsed"]),
                evidence=evidence,
            )
        )
    return events


def extract_transition_events(
    paths: Iterable[str | Path],
    *,
    source_name: str = "",
    completed_only: bool = False,
    max_tasks: int | None = None,
) -> list[TransitionEvent]:
    events: list[TransitionEvent] = []
    used_tasks = 0
    for row in load_rollout_rows(paths):
        status = str(row.get("status") or "").lower()
        if completed_only and status not in {"completed", "completed_with_recovery"}:
            continue
        task_events = extract_transition_events_from_row(row, source_name=source_name)
        if not task_events:
            continue
        events.extend(task_events)
        used_tasks += 1
        if max_tasks is not None and used_tasks >= max_tasks:
            break
    return events
