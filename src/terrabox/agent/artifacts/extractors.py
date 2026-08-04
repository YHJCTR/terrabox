"""Extract artifact state updates from tool observations."""

from __future__ import annotations

import json
from typing import Any

from .contracts import get_contract
from .state import (
    FILE_PARAM_MARKERS,
    OUTPUT_PARAM_NAMES,
    current_gpkg,
    infer_artifact_kind,
    looks_like_path,
    record_artifact_once,
)


def safe_json_loads(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"status": "error", "message": text}


def parse_tool_observation(text: str) -> tuple[dict[str, Any], bool]:
    parsed = safe_json_loads(text)
    stripped = (text or "").lstrip()
    lowered_prefix = stripped[:300].lower()
    truncated_success = (
        parsed.get("status") == "error"
        and (
            '"status": "success"' in lowered_prefix
            or "'status': 'success'" in lowered_prefix
            or lowered_prefix.startswith('{"status":"success"')
        )
    )
    if truncated_success:
        return {"status": "success", "message": text[:1000], "truncated": True}, False
    is_error = (
        stripped.startswith("Tool execution error:")
        or stripped.startswith("Error in ")
        or stripped.startswith("ERROR:")
        or parsed.get("status") == "error"
        or "not a valid tool" in text
    )
    return parsed, is_error


def _record_contract_outputs(
    state: dict[str, Any],
    slug: str,
    args: dict[str, Any],
    parsed: dict[str, Any],
) -> None:
    contract = get_contract(slug)
    if contract is None:
        return
    for output in contract.outputs:
        if output.kind.endswith("_layer"):
            name = (
                parsed.get(output.name_field or "")
                or args.get(output.name_param or "layer_name")
            )
            container = (
                parsed.get(output.container_param or "")
                or args.get(output.container_param or "")
                or current_gpkg(state)
            )
            if not name:
                continue
            layer_key = (container, name)
            existing = {(l.get("container"), l.get("name")) for l in state.get("layers", [])}
            if layer_key not in existing:
                state.setdefault("layers", []).append(
                    {
                        "kind": output.kind,
                        "name": name,
                        "container": container,
                        "source": slug,
                        "query": args.get("query"),
                        "count": parsed.get("poi_count") or parsed.get("feature_count"),
                    }
                )
            continue

        value = None
        if output.field:
            value = parsed.get(output.field)
        if value is None and output.param:
            value = args.get(output.param)
        if isinstance(value, str) and value:
            record_artifact_once(state, {"kind": output.kind, "path": value, "source": slug})


def update_artifact_state(
    state: dict[str, Any],
    slug: str,
    args: dict[str, Any],
    observation_text: str,
) -> None:
    """Update runtime artifact state after a tool call.

    This is dataset-agnostic. Tool-specific behavior comes from contracts, while
    generic path extraction handles tools that do not have a contract yet.
    """
    parsed, is_error = parse_tool_observation(observation_text)
    if is_error:
        state["last_error"] = parsed.get("message") or observation_text[:500]
        state.setdefault("failed_calls", []).append(
            {"tool": slug, "args": args, "error": state["last_error"]}
        )
    else:
        state["last_error"] = None
        state.setdefault("successful_calls", []).append(slug)
        state.setdefault("successful_call_records", []).append(
            {"tool": slug, "args": args}
        )

    if not is_error:
        _record_contract_outputs(state, slug, args, parsed)

    for key, value in parsed.items():
        if not isinstance(value, str) or not value:
            continue
        lowered = key.lower()
        if not looks_like_path(value):
            continue
        if lowered == "gpkg" or value.endswith(".gpkg"):
            record_artifact_once(state, {"kind": "gpkg", "path": value, "source": slug})
        elif lowered in OUTPUT_PARAM_NAMES or any(marker in lowered for marker in FILE_PARAM_MARKERS):
            record_artifact_once(
                state,
                {"kind": infer_artifact_kind(value, lowered), "path": value, "source": slug},
            )

    summary = parsed.get("text") or parsed.get("message") or observation_text
    state.setdefault("results", []).append({"tool": slug, "summary": str(summary)[:1000]})
