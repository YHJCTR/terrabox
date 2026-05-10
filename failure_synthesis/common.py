"""Shared helpers for source-format failure trajectory generation."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "failure_synthesis" / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

DEFAULT_OPENEARTH_SKIP_TOOLS = {
    "ChangeDetection",
    "geo_perception.change_os_detect",
}

OPENEARTH_TO_SLUG: dict[str, str | None] = {
    "Calculator": "ipython.execute",
    "Solver": "ipython.execute",
    "Plot": "ipython.execute",
    "DisplayOnGeotiff": "ipython.execute",
    "DisplayOnMap": "ipython.execute",
    "ShowIndexLayer": "ipython.execute",
    "TextToBbox": "geo_perception.instructsam",
    "CountGivenObject": "geo_perception.remotesam",
    "ObjectDetection": "geo_perception.strip_rcnn_detect",
    "SegmentObjectPixels": "geo_perception.sam2_segment",
    "RegionAttributeDescription": "geo_perception.vlm_analyze",
    "ImageDescription": "geo_perception.vlm_analyze",
    "DrawBox": "geo_perception.draw_bboxes",
    "AddText": "geo_perception.add_text",
    "OCR": "geo_perception.ocr_extract",
    "ChangeDetection": "geo_perception.change_os_detect",
    "AddPoisLayer": "osm_gis.add_pois_layer",
    "GetAreaBoundary": "osm_gis.get_area_boundary",
    "ComputeDistance": "osm_gis.compute_route_dist",
    "ComputeRouteDist": "osm_gis.compute_route_dist",
    "GetBboxFromGeotiff": "osm_gis.get_bbox_from_raster",
    "AddIndexLayer": "geo_raster.calculate_index",
    "ComputeIndexChange": "geo_raster.raster_diff",
    "GoogleSearch": "bing_search.search",
    "BingSearch": "bing_search.search",
    "Terminate": None,
}

SLUG_ALIASES = {
    "ipython_code.execute": "ipython.execute",
    "geobasic.area": "geo_basic.area",
    "geobasic.distance": "geo_basic.distance",
    "geobasic.gridify": "geo_basic.gridify",
    "geobasic.aoi_validate": "geo_basic.aoi_validate",
}

INPUT_PATH_KEYS = {
    "input_path",
    "image",
    "image_path",
    "pre_image",
    "post_image",
    "band_a_path",
    "band_b_path",
    "raster_path",
    "mask_path",
}
INPUT_PATH_LIST_KEYS = {"images", "image_paths"}
OUTPUT_PATH_KEYS = {"output_path", "result_path", "preview_path", "gpkg"}
STEP_REF_RE = re.compile(r"^\$step(?P<step>\d+)(?:\.(?P<path>[A-Za-z0-9_.-]+))?$")

NUMERIC_KEYS = {
    "threshold",
    "lower",
    "upper",
    "radius",
    "radius_m",
    "buffer_m",
    "gsd",
    "gsd_m",
    "month",
    "year",
    "top",
}

TOOL_REPLACEMENTS = {
    "geo_raster.calculate_index": "geo_raster.raster_stats",
    "geo_raster.threshold_segmentation": "geo_raster.raster_diff",
    "geo_statistics.count_pixels_condition": "geo_raster.count_above_threshold",
    "geo_basic.pixel_area": "geo_basic.distance",
    "geo_perception.vlm_analyze": "geo_perception.remotesam",
    "geo_perception.remotesam": "geo_perception.vlm_analyze",
    "geo_perception.instructsam": "geo_perception.vlm_analyze",
    "geo_perception.sam2_segment": "geo_perception.vlm_analyze",
    "geo_perception.strip_rcnn_detect": "geo_perception.vlm_analyze",
    "osm_gis.add_pois_layer": "osm_gis.compute_route_dist",
    "osm_gis.compute_route_dist": "osm_gis.add_pois_layer",
    "osm_gis.get_area_boundary": "osm_gis.add_pois_layer",
    "bing_search.search": "ipython.execute",
    "ipython.execute": "geo_perception.vlm_analyze",
    "ipython_code.execute": "geo_perception.vlm_analyze",
}

SLUG_TO_OPENEARTH_ACTION = {
    "ipython.execute": "Calculator",
    "geo_perception.instructsam": "TextToBbox",
    "geo_perception.remotesam": "CountGivenObject",
    "geo_perception.strip_rcnn_detect": "ObjectDetection",
    "geo_perception.sam2_segment": "SegmentObjectPixels",
    "geo_perception.vlm_analyze": "ImageDescription",
    "geo_perception.draw_bboxes": "DrawBox",
    "geo_perception.add_text": "AddText",
    "geo_perception.ocr_extract": "OCR",
    "geo_perception.change_os_detect": "ChangeDetection",
    "osm_gis.add_pois_layer": "AddPoisLayer",
    "osm_gis.get_area_boundary": "GetAreaBoundary",
    "osm_gis.compute_route_dist": "ComputeDistance",
    "osm_gis.get_bbox_from_raster": "GetBboxFromGeotiff",
    "geo_raster.calculate_index": "AddIndexLayer",
    "geo_raster.raster_diff": "ComputeIndexChange",
    "bing_search.search": "GoogleSearch",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: str | Path) -> Any:
    with repo_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    out = repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def limited(items: list[Any], limit: int | None) -> list[Any]:
    return items[:limit] if limit is not None else items


def canonical_tool(raw_tool: str) -> str | None:
    if raw_tool in OPENEARTH_TO_SLUG:
        return OPENEARTH_TO_SLUG[raw_tool]
    return SLUG_ALIASES.get(raw_tool, raw_tool)


def normalize_skip_tools(values: Iterable[str] | None, dataset: str) -> set[str]:
    skips = set(values or [])
    if dataset == "openearth" and not skips:
        skips.update(DEFAULT_OPENEARTH_SKIP_TOOLS)
    expanded = set(skips)
    for item in skips:
        mapped = canonical_tool(item)
        if mapped:
            expanded.add(mapped)
    return expanded


def should_skip_tools(raw_tools: Iterable[str], skip_tools: set[str]) -> bool:
    for raw in raw_tools:
        mapped = canonical_tool(raw)
        if raw in skip_tools or (mapped and mapped in skip_tools):
            return True
    return False


def make_disaster_wrapper(source: dict[str, Any], samples: list[dict[str, Any]], description: str) -> dict[str, Any]:
    data = {
        "version": source.get("version", "failure_synthesis_v1"),
        "description": description,
        "created_from": source.get("description", ""),
        "total_samples": len(samples),
        "samples": samples,
    }
    for key in ("disaster_categories", "task_types", "validation"):
        if key in source:
            data[key] = source[key]
    return data


def build_image_index(mapping_data: Any) -> dict[str, dict[str, Any]]:
    entries = mapping_data.get("mappings", []) if isinstance(mapping_data, dict) else mapping_data
    index: dict[str, dict[str, Any]] = {}
    for entry in entries or []:
        sid = entry.get("sft_id") or entry.get("id")
        if sid:
            index[sid] = dict(entry)
    return index


def attach_disaster_images(sample: dict[str, Any], image_index: dict[str, dict[str, Any]]) -> list[str]:
    if sample.get("images"):
        return list(sample.get("images") or [])
    meta = image_index.get(sample.get("id", ""), {})
    if not meta:
        sid = str(sample.get("id", ""))
        if "_alt_" in sid:
            meta = image_index.get(sid.replace("_alt_", "_"), {})
    return [p for p in (meta.get("image_pre"), meta.get("image_post")) if p]


def parse_openearth_actions(record: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(record.get("conversation", [])):
        if turn.get("from") != "gpt":
            continue
        try:
            payload = json.loads(turn.get("value", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        for action_index, action in enumerate(payload.get("actions", []) or []):
            name = action.get("name")
            if not name:
                continue
            actions.append({
                "turn_index": turn_index,
                "action_index": action_index,
                "name": name,
                "arguments": copy.deepcopy(action.get("arguments", {})),
                "canonical": canonical_tool(name),
            })
    return actions


def openearth_action_names(record: dict[str, Any]) -> list[str]:
    return [a["name"] for a in parse_openearth_actions(record)]


def replace_openearth_action(record: dict[str, Any], target: dict[str, Any], new_action: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(record)
    turn = out["conversation"][target["turn_index"]]
    payload = json.loads(turn.get("value", "{}"))
    payload["actions"][target["action_index"]] = new_action
    turn["value"] = json.dumps(payload, ensure_ascii=False)
    return out


def delete_openearth_action(record: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(record)
    turn = out["conversation"][target["turn_index"]]
    payload = json.loads(turn.get("value", "{}"))
    actions = payload.get("actions", [])
    if 0 <= target["action_index"] < len(actions):
        actions.pop(target["action_index"])
    payload["actions"] = actions
    payload["thought"] = f"[FAILURE SYNTHESIS] Removed a key action. {payload.get('thought', '')}".strip()
    turn["value"] = json.dumps(payload, ensure_ascii=False)
    return out


def append_openearth_observation(record: dict[str, Any], message: str) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out.setdefault("conversation", []).append({
        "from": "human",
        "value": f"OBSERVATION:\nERROR: {message}\nPlease summarize the model outputs and answer my first question.",
    })
    return out


def result_to_jsonable(result: Any, max_string: int = 2000) -> Any:
    if isinstance(result, str):
        text = result
        try:
            return json.loads(text)
        except Exception:
            return text[:max_string]
    if isinstance(result, dict):
        return {k: result_to_jsonable(v, max_string=max_string) for k, v in result.items()}
    if isinstance(result, list):
        return [result_to_jsonable(v, max_string=max_string) for v in result[:50]]
    return result


def is_error_result(result: Any) -> bool:
    parsed = result_to_jsonable(result)
    if isinstance(parsed, str):
        lowered = parsed.lower()
        return lowered.startswith("tool execution error:") or "connection failed" in lowered
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "")).lower()
        return status in {"error", "failed", "mock"} or "error" in parsed
    return False


def collect_path_errors(args: dict[str, Any], produced_paths: set[str]) -> list[str]:
    errors: list[str] = []
    for key, value in args.items():
        if key in INPUT_PATH_KEYS and isinstance(value, str) and value and not value.startswith("$"):
            if value in produced_paths:
                continue
            path = repo_path(value)
            if not path.exists():
                errors.append(f"missing input path for {key}: {value}")
        elif key in INPUT_PATH_LIST_KEYS and isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item and item not in produced_paths and not repo_path(item).exists():
                    errors.append(f"missing input path in {key}: {item}")
    return errors


def collect_produced_paths(args: dict[str, Any], result: Any) -> set[str]:
    produced: set[str] = set()
    for key, value in args.items():
        if key in OUTPUT_PATH_KEYS and isinstance(value, str):
            produced.add(value)
    parsed = result_to_jsonable(result)
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if key in OUTPUT_PATH_KEYS and isinstance(value, str):
                produced.add(value)
    return produced


def _nested_get(value: Any, dotted_path: str | None) -> Any:
    if not dotted_path:
        return value
    cur = value
    for part in dotted_path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
    return cur


def resolve_step_references(value: Any, step_outputs: dict[int, Any]) -> Any:
    """Resolve SFT placeholders such as ``$step1.bboxes`` from prior outputs."""
    if isinstance(value, str):
        match = STEP_REF_RE.match(value.strip())
        if not match:
            return value
        step = int(match.group("step"))
        resolved = _nested_get(step_outputs.get(step), match.group("path"))
        return value if resolved is None else resolved
    if isinstance(value, list):
        return [resolve_step_references(item, step_outputs) for item in value]
    if isinstance(value, dict):
        return {key: resolve_step_references(item, step_outputs) for key, item in value.items()}
    return value


def resolve_disaster_image_placeholders(value: Any, images: list[str]) -> Any:
    """Map common Disaster SFT image placeholders to attached real image paths."""
    if isinstance(value, str):
        low = value.lower()
        pre = images[0] if images else ""
        post = images[-1] if images else ""
        if low in {"rgb.tif", "image.tif", "image.png", "post.tif", "post.png", "post_image.png"} and post:
            return post
        if low in {"pre.tif", "pre.png", "pre_image.png"} and pre:
            return pre
    if isinstance(value, list):
        return [resolve_disaster_image_placeholders(item, images) for item in value]
    if isinstance(value, dict):
        return {key: resolve_disaster_image_placeholders(item, images) for key, item in value.items()}
    return value


def safe_artifact_path(value: str, artifact_dir: Path) -> str:
    name = Path(value).name if value else "artifact"
    return str(artifact_dir / (name or "artifact"))


def redirect_output_args(args: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    out = copy.deepcopy(args)
    for key in OUTPUT_PATH_KEYS:
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = safe_artifact_path(value, artifact_dir)
    return out


def prepare_disaster_replay_args(
    args: dict[str, Any],
    *,
    images: list[str],
    step_outputs: dict[int, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    prepared = resolve_step_references(copy.deepcopy(args), step_outputs)
    prepared = resolve_disaster_image_placeholders(prepared, images)
    return redirect_output_args(prepared, artifact_dir)


def import_terrabox_runtime() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from terrabox.extensions import load_builtin_toolkits
    except Exception as exc:
        raise RuntimeError(f"failed to import Terrabox runtime: {exc}") from exc

    try:
        load_builtin_toolkits()
    except Exception as exc:
        raise RuntimeError(f"failed to load Terrabox toolkits: {exc}") from exc


def execute_registry_tool(slug: str, args: dict[str, Any]) -> Any:
    import_terrabox_runtime()
    from terrabox.agent.tool_executor import AgentToolExecutor

    return AgentToolExecutor.execute(slug, args, user=None)


def registry_has_tool(slug: str) -> bool:
    import_terrabox_runtime()
    from terrabox.core.registry import get_handler

    return get_handler(slug) is not None


def mutate_numeric_args(args: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    out = copy.deepcopy(args)
    for key in sorted(out):
        value = out[key]
        if key in NUMERIC_KEYS and isinstance(value, (int, float)):
            if key == "month":
                out[key] = 13 if int(value) != 13 else 0
            elif key == "year":
                out[key] = int(value) + 30
            elif key in {"threshold", "lower", "upper"}:
                out[key] = float(value) + 999.0
            elif key == "top":
                out[key] = 0
            else:
                out[key] = float(value) * 100.0 + 1.0
            return out, f"mutated numeric arg {key}"
    for key in sorted(out):
        value = out[key]
        if key in INPUT_PATH_KEYS and isinstance(value, str) and value:
            out[key] = f"missing__{Path(value).name}"
            return out, f"mutated path arg {key}"
    return None


def replacement_tool(tool: str) -> str:
    canonical = canonical_tool(tool) or tool
    return TOOL_REPLACEMENTS.get(canonical, "geo_perception.vlm_analyze")


def replacement_openearth_action(tool: str) -> str:
    replacement = replacement_tool(tool)
    return SLUG_TO_OPENEARTH_ACTION.get(replacement, "ImageDescription")


def write_summary(run: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"runs": [], "totals": {}}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            summary = {"runs": [], "totals": {}}
    runs = summary.setdefault("runs", [])
    run_key = (run.get("dataset"), run.get("method"), run.get("output"))
    replaced = False
    for idx, existing in enumerate(runs):
        existing_key = (existing.get("dataset"), existing.get("method"), existing.get("output"))
        if existing_key == run_key:
            runs[idx] = run
            replaced = True
            break
    if not replaced:
        runs.append(run)

    totals: dict[str, Any] = {
        "runs": len(summary["runs"]),
        "failures": 0,
        "success_replay": 0,
        "skipped_changeos": 0,
        "skipped_total": 0,
        "by_dataset": Counter(),
        "by_method": Counter(),
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
    }
    for item in summary["runs"]:
        totals["failures"] += int(item.get("failures", 0))
        totals["success_replay"] += int(item.get("success_replay", 0))
        totals["skipped_changeos"] += int(item.get("skipped_changeos", 0))
        totals["skipped_total"] += int(item.get("skipped_total", 0))
        for counter_key, total_key in (
            ("by_failure_type", "by_failure_type"),
            ("by_tool", "by_tool"),
            ("by_task_type", "by_task_type"),
        ):
            totals[total_key].update(item.get(counter_key, {}))
        if item.get("dataset"):
            totals["by_dataset"][item["dataset"]] += 1
        if item.get("method"):
            totals["by_method"][item["method"]] += 1

    summary["totals"] = {
        key: (dict(value) if isinstance(value, Counter) else value)
        for key, value in totals.items()
    }
    save_json(SUMMARY_PATH, summary)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["disaster", "openearth"], required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-tools", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=7)


def init_random(seed: int) -> None:
    random.seed(seed)
