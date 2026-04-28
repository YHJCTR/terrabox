#!/usr/bin/env python3
"""Convert OpenEarthAgent train.json → Evolution module trajectory formats.

OpenEarthAgent uses its own action names (GetAreaBoundary, Calculator, …).
This script maps them to Terrabox tool slugs and produces the same trajectory
files that the evolution methods expect.

OUTPUTS
-------
1. data/openearth/trajectories.json   — full Trajectory JSON list
   (AgentEvolver mine, MemRL populate, SkillRL distill, CausalEvo build)

2. data/openearth/agentevolver.jsonl  — compact JSONL per trajectory
   (AgentEvolver experience_pool seed; SeqGraphEvo build)

USAGE
-----
    # Default: read data/openearth/train.json, write to data/openearth/
    python scripts/convert_openearth_to_evolution.py

    # Custom paths
    python scripts/convert_openearth_to_evolution.py \\
        --train  data/openearth/train.json \\
        --traj   data/openearth/trajectories.json \\
        --ae     data/openearth/agentevolver.jsonl \\
        --limit  1000   # optional: only convert first N records
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Action → Terrabox slug mapping
# ──────────────────────────────────────────────────────────────────────────────

OEA_TO_SLUG: dict[str, Optional[str]] = {
    # Code / computation
    "Calculator":                "ipython_code.execute",
    "Solver":                    "ipython_code.execute",
    "Plot":                      "ipython_code.execute",
    "DisplayOnGeotiff":          "ipython_code.execute",
    "DisplayOnMap":              "ipython_code.execute",
    "ShowIndexLayer":            "ipython_code.execute",
    # Visual perception
    "TextToBbox":                "geo_perception.instructsam",
    "CountGivenObject":          "geo_perception.remotesam",
    "ObjectDetection":           "geo_perception.strip_rcnn_detect",
    "SegmentObjectPixels":       "geo_perception.sam2_segment",
    "RegionAttributeDescription":"geo_perception.vlm_analyze",
    "ImageDescription":          "geo_perception.vlm_analyze",
    "DrawBox":                   "geo_perception.draw_bboxes",
    "AddText":                   "geo_perception.add_text",
    "OCR":                       "geo_perception.ocr_extract",
    "ChangeDetection":           "geo_perception.change_os_detect",
    # GIS / OSM
    "AddPoisLayer":              "osm_gis.add_pois_layer",
    "GetAreaBoundary":           "osm_gis.get_area_boundary",
    "ComputeDistance":           "osm_gis.compute_route_dist",
    "GetBboxFromGeotiff":        "osm_gis.get_bbox_from_raster",
    # Raster
    "AddIndexLayer":             "geo_raster.calculate_index",
    "ComputeIndexChange":        "geo_raster.raster_diff",
    # Search
    "GoogleSearch":              "bing_search.search",
    # Terminal action — skip
    "Terminate":                 None,
}


def _map_action(name: str) -> Optional[str]:
    """Return terrabox slug for an OEA action name, or None to skip."""
    slug = OEA_TO_SLUG.get(name)
    if slug is None and name != "Terminate":
        # Unknown action: keep a best-effort slug so nothing is silently lost
        log.debug(f"Unknown OEA action '{name}' → fallback ipython_code.execute")
        return "ipython_code.execute"
    return slug


# ──────────────────────────────────────────────────────────────────────────────
# Task-type normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_task_type(oea_type: str) -> str:
    """Map OpenEarthAgent type strings to compact task_type labels."""
    oea_type = oea_type.strip().lower()
    if oea_type.startswith("ind_"):
        return "index_analysis"
    if oea_type.startswith("gis_type"):
        return "gis_analysis"
    if oea_type.startswith(("b1_", "b2_")):
        return "building_analysis"
    if oea_type.startswith("complex_"):
        return "complex_task"
    # Numeric types
    _type_map = {
        "type1": "land_cover",      "type4": "change_detection",
        "type5": "object_detection", "type8": "scene_classification",
        "type9": "building_extraction", "type11": "road_extraction",
        "type13": "water_extraction", "type14": "vegetation_analysis",
        "type15": "soil_analysis",   "type19": "urban_analysis",
        "type25": "sar_analysis",    "type30": "gis_routing",
        "type31": "gis_pois",        "type32": "gis_spatial",
        "type38": "remote_sensing",  "type39": "remote_sensing",
        "type40": "remote_sensing",  "type41": "remote_sensing",
        "type42": "remote_sensing",
    }
    # Strip _sar suffix for lookup then re-tag
    base = re.sub(r"_sar$", "", oea_type)
    label = _type_map.get(base, "geospatial_analysis")
    if oea_type.endswith("_sar"):
        label += "_sar"
    return label


# ──────────────────────────────────────────────────────────────────────────────
# Record → Trajectory dict
# ──────────────────────────────────────────────────────────────────────────────

def _extract_tools_from_conversation(conversation: list[dict]) -> list[str]:
    """Return ordered list of terrabox slugs from a conversation list."""
    tools_called: list[str] = []
    for turn in conversation:
        if turn.get("from") != "gpt":
            continue
        try:
            j = json.loads(turn["value"])
        except (json.JSONDecodeError, KeyError):
            continue
        for action in j.get("actions", []):
            slug = _map_action(action.get("name", ""))
            if slug is not None:
                tools_called.append(slug)
    return tools_called


def _record_to_trajectory(record: dict, idx: int) -> Optional[dict]:
    """Convert one train.json record to a Trajectory-compatible dict."""
    question = record.get("question", "").strip()
    if not question:
        return None

    oea_type = record.get("type", "unknown")
    task_type = _normalize_task_type(oea_type)
    images = record.get("images", [])
    conversation = record.get("conversation", [])

    tools_called = _extract_tools_from_conversation(conversation)
    if not tools_called:
        return None

    # Build minimal turns list (human question + tool turns)
    turns: list[dict] = [
        {
            "role": "human",
            "content": question,
            "tool_name": None,
            "tool_args": None,
            "tool_result": None,
            "is_error": False,
        }
    ]
    for slug in tools_called:
        turns.append({
            "role": "tool",
            "content": "",
            "tool_name": slug,
            "tool_args": {},
            "tool_result": "",
            "is_error": False,
        })

    task_id = f"oea_train_{idx:05d}"
    final_answer = record.get("label", "")

    return {
        "task_id": task_id,
        "question": question,
        "images": images,
        "turns": turns,
        "tools_called": tools_called,
        "expected_tools": tools_called,   # OEA train = expert demos
        "final_answer": final_answer,
        "success": True,
        "source": "openearth",
        "task_type": task_type,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert(
    train_path: str,
    traj_path: str,
    ae_path: str,
    limit: Optional[int] = None,
) -> None:
    log.info(f"Loading OpenEarthAgent train data from {train_path} ...")
    with open(train_path, encoding="utf-8") as f:
        records = json.load(f)

    if limit:
        records = records[:limit]
        log.info(f"Limiting to first {limit} records")

    log.info(f"Converting {len(records)} records ...")
    trajectories: list[dict] = []
    skipped_no_question = 0
    skipped_no_tools = 0
    tool_counter: Counter = Counter()

    for i, rec in enumerate(records):
        traj = _record_to_trajectory(rec, i)
        if traj is None:
            if not rec.get("question", "").strip():
                skipped_no_question += 1
            else:
                skipped_no_tools += 1
            continue
        trajectories.append(traj)
        tool_counter.update(traj["tools_called"])

    log.info(
        f"Converted {len(trajectories)} trajectories "
        f"(skipped: {skipped_no_question} no-question, {skipped_no_tools} no-tools)"
    )

    # Tool usage stats
    log.info("Top tool slugs:")
    for slug, cnt in tool_counter.most_common(10):
        log.info(f"  {slug}: {cnt}")

    # 1. Full Trajectory JSON
    Path(traj_path).parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    log.info(f"Written trajectories JSON ({len(trajectories)} records) → {traj_path}")

    # 2. Compact AgentEvolver / SeqGraphEvo JSONL
    with open(ae_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            record = {
                "task_id":       traj["task_id"],
                "task_type":     traj["task_type"],
                "question":      traj["question"],
                "tool_sequence": traj["tools_called"],
                "reward":        1.0,
                "f1":            1.0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Written AgentEvolver JSONL ({len(trajectories)} records) → {ae_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenEarthAgent train.json → evolution trajectory formats"
    )
    parser.add_argument(
        "--train",
        default=str(REPO_ROOT / "data" / "openearth" / "train.json"),
        help="Input: OpenEarthAgent train.json",
    )
    parser.add_argument(
        "--traj",
        default=str(REPO_ROOT / "data" / "openearth" / "trajectories.json"),
        help="Output: full Trajectory JSON list",
    )
    parser.add_argument(
        "--ae",
        default=str(REPO_ROOT / "data" / "openearth" / "agentevolver.jsonl"),
        help="Output: compact AgentEvolver / SeqGraphEvo JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N records (for quick testing)",
    )
    args = parser.parse_args()
    convert(
        train_path=args.train,
        traj_path=args.traj,
        ae_path=args.ae,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
