#!/usr/bin/env python
"""Build the FULL OpenEarth + EarthBench SFT dataset, de-collapsed to the real
Terrabox tools, split by source (openearth / earthbench) and split (train / test).

Why this exists
---------------
Earlier SFT data (data/newdata, data/fixdata_decollapse_v2) dropped every task
that used a tool which was not yet implemented in Terrabox (the OSM/GeoPackage
workflow, GoogleSearch, ChangeDetection, CountGivenObject, …) — that is why the
OpenEarth train set shrank 14538 -> 8234 -> 6587. Now that all 24 OpenEarthAgent
tools have real Terrabox implementations, we can map the COMPLETE raw OEA data.

OpenEarth: converted directly from the raw author split
(OpenEarthAgent/data/{train,test}.json) so it is genuinely full (14538 / 1169).
EarthBench: reused from the already-de-collapsed SFT records in
fixdata_decollapse_v2 (EB does not use the previously-missing tools), only the
system-prompt tool catalog is refreshed to the current registry.

Output (data/oea_full_sft/ by default):
  openearth/train.jsonl  openearth/test.jsonl
  earthbench/train.jsonl earthbench/test.jsonl
  tools_catalog.json  manifest.json

Each row matches the prior SFT schema (v2): id, source, task_type, question,
images, data_files, data_dir, ground_truth, messages, expected_tools,
gold_tool_calls, argument_status_counts, conversion_warnings.

Run with the unsloth env + PYTHONPATH=src (live registry needed for the catalog
and per-tool argument schemas).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OEA_ROOT = Path(os.environ.get("OEA_ROOT", "/data1/yuhongjie2/OpenEarthAgent"))
OEA_DATA = OEA_ROOT / "data"

# ── OpenEarthAgent tool name → real Terrabox slug (full de-collapse) ──────────
# Terminate is not a tool: it carries the final answer (-> ground_truth + an
# empty-actions closing turn), matching the prior SFT format.
OEA_TO_SLUG: dict[str, str | None] = {
    "Calculator": "compute.calculator",
    "Solver": "compute.solver",
    "Plot": "compute.plot",
    "TextToBbox": "geo_perception.instructsam",
    "CountGivenObject": "geo_perception.count_given_object",
    "ObjectDetection": "geo_perception.strip_rcnn_detect",
    "SegmentObjectPixels": "geo_perception.sam2_segment",
    "RegionAttributeDescription": "geo_perception.region_attribute_description",
    "ImageDescription": "geo_perception.vlm_analyze",
    "DrawBox": "geo_perception.draw_bboxes",
    "AddText": "geo_perception.add_text",
    "OCR": "geo_perception.ocr_extract",
    "ChangeDetection": "geo_perception.change_os_detect",
    "AddPoisLayer": "osm_gis.add_pois_layer",
    "GetAreaBoundary": "osm_gis.get_area_boundary",
    "ComputeDistance": "osm_gis.compute_route_dist",
    "GetBboxFromGeotiff": "osm_gis.get_bbox_from_raster",
    "AddIndexLayer": "osm_gis.add_index_layer",
    "ComputeIndexChange": "osm_gis.compute_index_change",
    "ShowIndexLayer": "osm_gis.show_index_layer",
    "DisplayOnMap": "osm_gis.display_on_map",
    "DisplayOnGeotiff": "osm_gis.display_on_geotiff",
    "GoogleSearch": "bing_search.search",
    "Terminate": None,
}

_IMG_REF = re.compile(r"^img_(\d+)$")


def _load(path: Path):
    d = json.load(open(path, encoding="utf-8"))
    return d if isinstance(d, list) else d.get("tasks", d)


def _resolve_image(value, images_abs: list[str]):
    """img_N -> the Nth resolved image path; leave gpkg_N / tif_N symbolic."""
    if isinstance(value, str):
        m = _IMG_REF.match(value)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(images_abs):
                return images_abs[i]
    return value


def _abs_image(rel: str, split: str = "train") -> str:
    """Resolve an OEA image reference to an absolute path. Train uses relative
    paths ('data/train_images/x.jpg'); test uses bare filenames that live under
    data/test/."""
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = OEA_ROOT / rel
    if cand.exists():
        return str(cand)
    search_dirs = (["test"] if split == "test" else ["train_images", "train_part2", "train"])
    for d in search_dirs + ["test", "train_images", "train_part2", "test_images"]:
        c = OEA_DATA / d / p.name
        if c.exists():
            return str(c)
    return str(cand)


def get_schema_props(slug: str) -> set[str]:
    from terrabox.core.registry import get_tool
    spec = get_tool(slug)
    if spec is None:
        return set()
    return set((spec.parameters or {}).get("properties", {}).keys())


def align_action(name: str, raw_args: dict, images_abs: list[str], schema_cache: dict):
    """Return (slug, function_name, args, status) or None to skip (Terminate).

    The Terrabox tools have been adapted to accept the OpenEarthAgent argument
    contracts natively, so we keep the gold arguments VERBATIM (only resolving
    img_N image references to absolute paths). No renaming, no dropping → the SFT
    targets match exactly what the tools execute at rollout."""
    slug = OEA_TO_SLUG.get(name, "ipython.execute")  # unknown → ipython fallback
    if slug is None:  # Terminate → final answer
        return None
    resolved = {k: _resolve_image(v, images_abs) for k, v in (raw_args or {}).items()}
    return slug, slug.replace(".", "__"), resolved, "adapted"


def extract_terminate_ans(value: str) -> str | None:
    try:
        o = json.loads(value)
    except Exception:
        return None
    for a in (o.get("actions") or []):
        if a.get("name") == "Terminate":
            return str((a.get("arguments") or {}).get("ans", ""))
    return None


def convert_oea_task(task: dict, idx: int, source_split: str, schema_cache: dict, system_msg: str):
    conv = task.get("conversation", [])
    images_abs = [_abs_image(i, source_split) for i in (task.get("images") or [])]
    messages = [{"role": "system", "content": system_msg}]
    gold_calls: list[dict] = []
    expected_tools: list[str] = []
    status_counts: Counter = Counter()
    warnings: list[str] = []
    ground_truth = ""
    question = ""
    first_human = True

    for turn in conv:
        role = turn.get("from")
        val = turn.get("value", "")
        if role == "human":
            content = val
            if first_human:
                first_human = False
                if "Question:" in val:
                    question = val.split("Question:", 1)[1].strip()
                    content = "Question: " + question
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            try:
                obj = json.loads(val)
            except Exception:
                messages.append({"role": "assistant", "content": val})
                warnings.append("unparseable_gpt_turn")
                continue
            thought = obj.get("thought", "")
            actions = obj.get("actions") or []
            # Terminate → close with empty actions, capture answer
            term = next((a for a in actions if a.get("name") == "Terminate"), None)
            if term is not None:
                ground_truth = str((term.get("arguments") or {}).get("ans", ""))
                messages.append({"role": "assistant",
                                 "content": json.dumps({"thought": thought, "actions": []}, ensure_ascii=False)})
                continue
            new_actions = []
            for a in actions:
                res = align_action(a.get("name", ""), a.get("arguments") or {}, images_abs, schema_cache)
                if res is None:
                    continue
                slug, fn, aligned, status = res
                new_actions.append({"tool": slug, "function_name": fn, "arguments": aligned})
                expected_tools.append(slug)
                status_counts[status] += 1
                gold_calls.append({
                    "raw_tool": a.get("name"), "tool": slug, "function_name": fn,
                    "raw_arguments": a.get("arguments") or {}, "arguments": aligned,
                    "argument_status": status,
                })
                if status == "raw_only_schema_mismatch":
                    warnings.append(f"{a.get('name')}_args_unmapped")
            messages.append({"role": "assistant",
                             "content": json.dumps({"thought": thought, "actions": new_actions}, ensure_ascii=False)})

    if not question:
        question = task.get("question", "")
    return {
        "id": f"oea_{source_split}_{task.get('idx', idx)}",
        "source": "openearth",
        "task_type": task.get("type", task.get("task_type")),
        "question": question,
        "images": images_abs,
        "data_files": [],
        "data_dir": "",
        "ground_truth": ground_truth,
        "messages": messages,
        "expected_tools": expected_tools,
        "gold_tool_calls": gold_calls,
        "argument_status_counts": dict(status_counts),
        "conversion_warnings": warnings,
    }


def refresh_system(rec: dict, system_msg: str) -> dict:
    if rec.get("messages") and rec["messages"][0].get("role") == "system":
        rec["messages"][0] = {"role": "system", "content": system_msg}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data" / "oea_full_sft"))
    ap.add_argument("--oea-train", default=str(OEA_DATA / "train.json"))
    ap.add_argument("--oea-test", default=str(OEA_DATA / "test.json"))
    ap.add_argument("--eb-fixdata-train", default=str(REPO / "data/fixdata_decollapse_v2/sft_train_strict.jsonl"))
    ap.add_argument("--eb-fixdata-test", default=str(REPO / "data/fixdata_decollapse_v2/_eb_test.jsonl"))
    ap.add_argument("--drop-test-overlap", action="store_true", default=True,
                    help="Drop OE test tasks whose question appears in OE train (no leakage).")
    args = ap.parse_args()

    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import list_tools
    load_builtin_toolkits()

    # Build the system message: reuse the prior prefix, refresh the catalog from
    # the live registry (so all 24 OEA tools + everything else are listed).
    catalog = [{
        "slug": s.slug, "function_name": s.slug.replace(".", "__"),
        "description": s.description, "parameters": s.parameters,
    } for s in sorted(list_tools(), key=lambda x: x.slug)]
    catalog_blob = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    prefix = (
        "You are a Terrabox geospatial tool-use agent. Solve Earth observation, "
        "remote sensing, GIS, and geospatial analysis tasks by planning and calling "
        "the provided Terrabox tools.\n\n"
        "Use only tools from the catalog below. Tool names are canonical Terrabox slugs, "
        "and function_name is the dot-to-double-underscore form used by the runtime tool binding.\n\n"
        "When a tool is needed, respond as a JSON object with this schema:\n"
        '{"thought": "...", "actions": [{"tool": "toolkit.tool", "function_name": "toolkit__tool", "arguments": {...}}]}\n'
        "When the task is complete, return a final answer in the same JSON object using an empty actions list and a final_answer field.\n\n"
        "Tool catalog:\n"
    )
    system_msg = prefix + catalog_blob

    out = Path(args.out_dir)
    (out / "openearth").mkdir(parents=True, exist_ok=True)
    (out / "earthbench").mkdir(parents=True, exist_ok=True)
    schema_cache: dict = {}

    def norm_q(q: str) -> str:
        return re.sub(r"\s+", " ", (q or "").strip().lower())

    # ── OpenEarth (from raw, full) ───────────────────────────────────────────
    oe_train = _load(Path(args.oea_train))
    oe_test = _load(Path(args.oea_test))
    train_recs = [convert_oea_task(t, i, "train", schema_cache, system_msg) for i, t in enumerate(oe_train)]
    test_recs = [convert_oea_task(t, i, "test", schema_cache, system_msg) for i, t in enumerate(oe_test)]

    dropped = 0
    if args.drop_test_overlap:
        train_q = {norm_q(r["question"]) for r in train_recs}
        kept = [r for r in test_recs if norm_q(r["question"]) not in train_q]
        dropped = len(test_recs) - len(kept)
        test_recs = kept

    def dump(path: Path, recs: list[dict]) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(recs)

    n_oe_tr = dump(out / "openearth" / "train.jsonl", train_recs)
    n_oe_te = dump(out / "openearth" / "test.jsonl", test_recs)

    # ── EarthBench (reuse de-collapsed SFT records, refresh catalog) ──────────
    eb_train = [refresh_system(json.loads(l), system_msg)
                for l in open(args.eb_fixdata_train, encoding="utf-8")
                if json.loads(l).get("source") == "earthbench"]
    eb_test = [refresh_system(json.loads(l), system_msg)
               for l in open(args.eb_fixdata_test, encoding="utf-8")]
    n_eb_tr = dump(out / "earthbench" / "train.jsonl", eb_train)
    n_eb_te = dump(out / "earthbench" / "test.jsonl", eb_test)

    json.dump(catalog, open(out / "tools_catalog.json", "w"), ensure_ascii=False, indent=2)
    manifest = {
        "format": "terrabox_oea_full_sft_v1",
        "counts": {
            "openearth": {"train": n_oe_tr, "test": n_oe_te},
            "earthbench": {"train": n_eb_tr, "test": n_eb_te},
        },
        "oe_test_overlap_dropped": dropped,
        "catalog_tools": len(catalog),
        "notes": "OpenEarth converted from raw author split (full); EarthBench reused from "
                 "fixdata_decollapse_v2 with refreshed catalog. The Terrabox tools accept the "
                 "OpenEarthAgent argument contracts natively, so gold arguments are kept verbatim "
                 "(only img_N references resolved to absolute paths) — every call's args match its "
                 "tool schema. add_index_layer's {year, month} are the imagery-fetch contract "
                 "(stac_basic backend); local band_a_path/band_b_path are also accepted.",
    }
    json.dump(manifest, open(out / "manifest.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
