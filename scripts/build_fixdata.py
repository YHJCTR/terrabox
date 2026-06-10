#!/usr/bin/env python
"""Build data/fixdata from data/newdata: restore merge-distorted tools, recover
ipython-routed tools that actually have a real Terrabox tool, and rebuild the
SFT tool catalog from the LIVE registry so SFT and rollout see the same tools.

Background
----------
The newdata conversion collapsed distinct tools into ``ipython.execute``:
  * OpenEarth ``Calculator / Solver / Plot`` → ipython (the big distortion);
  * EarthAgent helpers (``calculate_ndwi``, ``calculate_ndti``,
    ``calc_batch_image_mean_max_min``, ``argmax`` …) → ipython, even though a
    real Terrabox tool exists (e.g. ``geo_raster.calculate_index``).
Each gold call still records the original ``raw_tool`` / ``raw_arguments``, so we
can faithfully restore tool identity.

This script:
  1. remaps every ``ipython.execute`` gold call back to its real Terrabox tool
     (``COMBINED_REMAP``); any unmapped ipython call falls back to
     ``compute.solver`` (general Python compute) so the dataset contains NO
     ``ipython.execute`` at all;
  2. applies the same remap to the assistant ``messages`` action blocks and
     recomputes ``expected_tools``;
  3. rebuilds the system-prompt tool catalog from the LIVE CoreRegistry,
     restricted to the dataset's active tool set (the union of gold tools after
     remap). This guarantees: (a) the catalog carries the current tool
     descriptions, (b) the catalog == the tools a restricted rollout binds, and
     (c) no mock / osm / bing / vlm / ipython tool ever appears;
  4. writes data/fixdata/{sft_train_strict.jsonl, shuffled, tools_catalog.json,
     manifest.json, README.md}.

Run with PYTHONPATH=src so the live registry is importable.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# raw_tool -> (real Terrabox slug, function_name).  Applied only to gold calls
# whose current ``tool`` is ``ipython.execute`` (i.e. the collapsed placeholders).
# ---------------------------------------------------------------------------
COMBINED_REMAP: dict[str, tuple[str, str]] = {
    # OpenEarth compute tools collapsed into ipython.
    "Calculator": ("compute.calculator", "compute__calculator"),
    "Solver":     ("compute.solver",     "compute__solver"),
    "Plot":       ("compute.plot",        "compute__plot"),
    # EarthAgent single-image index variants → generic normalized-difference tool.
    "calculate_ndvi": ("geo_raster.calculate_index", "geo_raster__calculate_index"),
    "calculate_ndwi": ("geo_raster.calculate_index", "geo_raster__calculate_index"),
    "calculate_ndbi": ("geo_raster.calculate_index", "geo_raster__calculate_index"),
    "calculate_nbr":  ("geo_raster.calculate_index", "geo_raster__calculate_index"),
    "calculate_ndti": ("geo_raster.calculate_index", "geo_raster__calculate_index"),
    "calculate_ndsi": ("geo_raster.calculate_index", "geo_raster__calculate_index"),
    # EarthAgent batch stats variant whose name didn't match the mapping key.
    "calc_batch_image_mean_max_min": ("geo_statistics.batch_raster_stats", "geo_statistics__batch_raster_stats"),
    "calculate_batch_image_mean_max_min": ("geo_statistics.batch_raster_stats", "geo_statistics__batch_raster_stats"),
    # Pure-Python helper ops with no dedicated tool → general compute (solver).
    "argmax":                          ("compute.solver", "compute__solver"),
    "index_to_date_range":             ("compute.solver", "compute__solver"),
    "get_list_object_via_indexes":     ("compute.solver", "compute__solver"),
    "ceil_number":                     ("compute.solver", "compute__solver"),
    "calculate_area":                  ("compute.solver", "compute__solver"),
    "calculate_band_mean_by_condition":("compute.solver", "compute__solver"),
}

# Any ipython call whose raw_tool is not explicitly mapped falls back to this,
# so the dataset never contains ipython.execute.
IPY_FALLBACK = ("compute.solver", "compute__solver")

# Semantic correction (applied in BOTH modes): these EarthBench ops reduce a
# list of NUMBERS, not rasters. The original conversion wrongly mapped them to
# raster tools (mean_of_means / batch_raster_stats), which crash on numbers.
# Route them to the dedicated numeric tools (granular_tools.py).
SEMANTIC_FIX_REMAP: dict[str, tuple[str, str]] = {
    "mean":                ("geo_statistics.mean", "geo_statistics__mean"),
    "max_value_and_index": ("geo_statistics.max_value_and_index", "geo_statistics__max_value_and_index"),
    "min_value_and_index": ("geo_statistics.min_value_and_index", "geo_statistics__min_value_and_index"),
}

# ---------------------------------------------------------------------------
# De-collapse map (only in --mode decollapse): raw_tool -> specific granular
# tool, overriding the generic (collapsed) slug. Only the granular tools that
# actually occur in the dataset are listed (data-driven); they are registered by
# src/terrabox/toolkits/granular_tools.py. raw_tools not listed here keep their
# collapsed slug (e.g. Solver/argmax → compute.solver, TextToBbox → instructsam).
# ---------------------------------------------------------------------------
RAW_GRANULAR: dict[str, tuple[str, str]] = {
    "calculate_ndwi":       ("geo_raster.calculate_ndwi", "geo_raster__calculate_ndwi"),
    "calculate_ndti":       ("geo_raster.calculate_ndti", "geo_raster__calculate_ndti"),
    "calculate_ndsi":       ("geo_raster.calculate_ndsi", "geo_raster__calculate_ndsi"),
    "calculate_batch_ndsi": ("geo_raster.calculate_ndsi", "geo_raster__calculate_ndsi"),
    "difference":           ("geo_statistics.subtract", "geo_statistics__subtract"),
    "division":             ("geo_statistics.divide", "geo_statistics__divide"),
    "multiply":             ("geo_statistics.multiply", "geo_statistics__multiply"),
    "calc_batch_image_mean":          ("geo_statistics.batch_image_mean", "geo_statistics__batch_image_mean"),
    "calc_batch_image_max":           ("geo_statistics.batch_image_max", "geo_statistics__batch_image_max"),
    "calc_batch_image_sum":           ("geo_statistics.batch_image_sum", "geo_statistics__batch_image_sum"),
    "calc_batch_image_mean_max_min":  ("geo_statistics.batch_image_mean_max_min", "geo_statistics__batch_image_mean_max_min"),
    "calculate_batch_image_mean_max_min": ("geo_statistics.batch_image_mean_max_min", "geo_statistics__batch_image_mean_max_min"),
    "calculate_mean_lst_by_ndvi": ("earth_sci.mean_lst_by_ndvi", "earth_sci__mean_lst_by_ndvi"),
    "calculate_max_lst_by_ndvi":  ("earth_sci.max_lst_by_ndvi", "earth_sci__max_lst_by_ndvi"),
}

# Tool families that must never appear in the catalog or gold (goal 3).
EXCLUDE_SUBSTR = ("mscn", "sm3det", "change_os", "osm_gis.", "bing_search.",
                  "vlm_analyze", "ipython")

# ---------------------------------------------------------------------------
# Argument alignment: the source datasets kept the ORIGINAL EarthBench arg names
# (input_nir_path, file_list, …) which the real Terrabox tools don't accept.
# These per-slug key renames translate gold arguments to the tool's schema, so
# the dataset's (tool, args) match the real tool — required for SFT/RL/exec.
# ---------------------------------------------------------------------------
ALIGN_RENAME: dict[str, dict[str, str]] = {
    "geo_raster.calculate_ndwi": {"input_nir_path": "band_a_path", "input_swir_path": "band_b_path",
                                  "output_filename": "output_path"},
    "geo_raster.calculate_ndti": {"input_red_path": "band_a_path", "input_green_path": "band_b_path",
                                  "output_filename": "output_path"},
    "geo_raster.calculate_ndsi": {"input_green_path": "band_a_path", "input_swir_path": "band_b_path",
                                  "output_filename": "output_path"},
    # collapsed generic index tool: cover all source band names (band_a first, band_b second)
    "geo_raster.calculate_index": {"input_nir_path": "band_a_path", "input_red_path": "band_a_path",
                                   "input_green_path": "band_b_path", "input_swir_path": "band_b_path",
                                   "output_filename": "output_path"},
    "geo_statistics.batch_image_mean_max_min": {"file_list": "input_paths"},
    "geo_statistics.batch_raster_stats": {"file_list": "input_paths"},
    # numeric list-reduction tools: gold stored the number list under input_paths
    "geo_statistics.mean": {"input_paths": "values", "x": "values"},
    "geo_statistics.max_value_and_index": {"input_paths": "values", "x": "values"},
    "geo_statistics.min_value_and_index": {"input_paths": "values", "x": "values"},
}


def _synth_solver_command(args: dict) -> str | None:
    """Synthesize a runnable compute.solver `command` for EarthBench helper ops
    that have no dedicated tool (argmax / index_to_date_range /
    calculate_band_mean_by_condition). Returns None if not recognized."""
    a = args or {}
    if "x" in a and set(a) <= {"x"}:
        return ("def solution():\n"
                f"    x = {a['x']!r}\n"
                "    return str(x.index(max(x)))")
    if "dates" in a and "start_index" in a:
        return ("def solution():\n"
                f"    dates = {a['dates']!r}\n"
                f"    return str(dates[{int(a['start_index'])}:])")
    if "image_path" in a and "condition_band_index" in a:
        return ("def solution():\n"
                "    import rasterio, numpy as np\n"
                f"    with rasterio.open({a.get('image_path')!r}) as ds:\n"
                f"        cond = ds.read({int(a.get('condition_band_index',0))}+1).astype('float64')\n"
                f"        tgt = ds.read({int(a.get('target_band_index',0))}+1).astype('float64')\n"
                f"    thr = {float(a.get('condition_threshold',0))}\n"
                f"    mask = cond > thr if {a.get('condition_mode','above')!r} == 'above' else cond < thr\n"
                "    vals = tgt[mask]\n"
                "    return str(float(np.nanmean(vals)) if vals.size else 'no pixels match')")
    return None


def align_args(slug: str, args: dict, schema_props: set[str]) -> dict:
    """Translate gold arguments so their keys match the tool's schema:
    1) compute.solver helpers without `command` → synthesized command;
    2) per-slug key renames (EarthBench name → Terrabox schema name);
    3) drop any remaining keys the tool's schema does not declare."""
    args = dict(args or {})
    if slug == "compute.solver" and "command" not in args:
        cmd = _synth_solver_command(args)
        if cmd is not None:
            args = {"command": cmd}
    # input_pairs: a list of per-image dicts (batch of {red,green,output}) →
    # unroll into parallel lists so auto_batch can iterate them.
    if isinstance(args.get("input_pairs"), list):
        rename = ALIGN_RENAME.get(slug, {})
        collected: dict[str, list] = {}
        for pair in args.pop("input_pairs"):
            if not isinstance(pair, dict):
                continue
            for k, v in pair.items():
                collected.setdefault(rename.get(k, k), []).append(v)
        args.update(collected)
    for old, new in ALIGN_RENAME.get(slug, {}).items():
        if old in args and new not in args:
            args[new] = args.pop(old)
    if schema_props:
        args = {k: v for k, v in args.items() if k in schema_props}
    return args


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _target_for(call: dict, mode: str) -> tuple[str, str, dict | None] | None:
    """Return (slug, func, args) for the new tool identity, or None if unchanged.

    decollapse: a granular override (RAW_GRANULAR) wins; ipython placeholders
    still get restored. collapse: only ipython placeholders are remapped.
    """
    raw_tool = call.get("raw_tool")
    cur = call.get("tool")
    # Semantic correction first (both modes): numeric-list ops → numeric tools.
    if raw_tool in SEMANTIC_FIX_REMAP:
        slug, func = SEMANTIC_FIX_REMAP[raw_tool]
        args = None
        if cur == "ipython.execute":
            ra = call.get("raw_arguments")
            args = dict(ra) if isinstance(ra, dict) and ra else None
        return slug, func, args
    if mode == "decollapse" and raw_tool in RAW_GRANULAR:
        slug, func = RAW_GRANULAR[raw_tool]
        args = None
        if cur == "ipython.execute":
            raw_args = call.get("raw_arguments")
            args = dict(raw_args) if isinstance(raw_args, dict) and raw_args else None
        return slug, func, args
    if cur == "ipython.execute":
        slug, func = COMBINED_REMAP.get(raw_tool, IPY_FALLBACK)
        raw_args = call.get("raw_arguments")
        args = dict(raw_args) if isinstance(raw_args, dict) and raw_args else None
        return slug, func, args
    return None


def remap_gold_calls(gold: list[dict], mode: str) -> tuple[list[dict], Counter]:
    out: list[dict] = []
    by_tool: Counter = Counter()
    for call in gold:
        new = dict(call)
        tgt = _target_for(call, mode)
        if tgt is not None:
            slug, func, args = tgt
            new["tool"] = slug
            new["function_name"] = func
            if args is not None:
                new["arguments"] = args
            by_tool[call.get("raw_tool")] += 1
        out.append(new)
    return out, by_tool


def remap_messages(messages: list[dict], gold: list[dict], mode: str) -> tuple[list[dict], bool]:
    """Apply the same remap to assistant action blocks (1:1 ordered with gold)."""
    targets = [_target_for(c, mode) for c in gold]
    new_messages: list[dict] = []
    pos = 0
    aligned = True
    for msg in messages:
        if msg.get("role") != "assistant":
            new_messages.append(msg)
            continue
        try:
            payload = json.loads(msg.get("content", ""))
        except (json.JSONDecodeError, TypeError):
            new_messages.append(msg)
            continue
        if not isinstance(payload, dict):
            new_messages.append(msg)
            continue
        actions = payload.get("actions") or []
        changed = False
        for action in actions:
            if pos >= len(targets):
                aligned = False
                break
            tgt = targets[pos]
            pos += 1
            if tgt is None:
                continue
            slug, func, args = tgt
            # Overwrite by position (actions are 1:1 with gold). Works for both
            # ipython restoration and collapsed→granular renaming.
            action["tool"] = slug
            action["function_name"] = func
            if args is not None:
                action["arguments"] = args
            changed = True
        if changed:
            new_msg = dict(msg)
            new_msg["content"] = json.dumps(payload, ensure_ascii=False)
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)
    if pos != len(targets):
        aligned = False
    return new_messages, aligned


def build_registry_catalog(slugs: set[str]) -> list[dict]:
    """Build catalog entries for the given slugs from the LIVE registry."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import get_tool
    load_builtin_toolkits()
    entries: list[dict] = []
    missing: list[str] = []
    for slug in sorted(slugs):
        spec = get_tool(slug)
        if spec is None:
            missing.append(slug)
            continue
        entries.append({
            "slug": slug,
            "function_name": slug.replace(".", "__"),
            "description": spec.description,
            "parameters": spec.parameters,
        })
    if missing:
        raise SystemExit(f"Active tools missing from registry: {missing}")
    return entries


def set_system_catalog(messages: list[dict], catalog_blob: str) -> bool:
    """Replace the JSON array after 'Tool catalog:' with catalog_blob."""
    if not messages or messages[0].get("role") != "system":
        return False
    c = messages[0].get("content", "")
    i = c.find("Tool catalog:")
    if i < 0:
        return False
    jstart = c.find("[", i)
    if jstart < 0:
        return False
    messages[0] = dict(messages[0])
    messages[0]["content"] = c[:jstart] + catalog_blob
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data/fixdata with restored real tools + registry catalog")
    ap.add_argument("--in-dir", default="data/newdata")
    ap.add_argument("--out-dir", default=None,
                    help="default: data/fixdata (collapse) or data/fixdata_decollapse (decollapse)")
    ap.add_argument("--mode", choices=["collapse", "decollapse"], default="collapse",
                    help="collapse: generic tools (current). decollapse: specific granular tools.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = REPO / args.in_dir
    default_out = "data/fixdata" if args.mode == "collapse" else "data/fixdata_decollapse"
    out_dir = REPO / (args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = in_dir / "sft_train_strict.jsonl"
    stats = {"rows": 0, "remapped_calls": 0, "rows_with_remap": 0,
             "msg_align_fail": 0, "sys_patch_fail": 0}
    remap_by_tool: Counter = Counter()
    rows: list[dict] = []
    active_slugs: set[str] = set()

    # Pass 1: remap gold + messages, collect the active tool set.
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["rows"] += 1
            gold = row.get("gold_tool_calls") or []
            new_gold, by_tool = remap_gold_calls(gold, args.mode)
            if by_tool:
                stats["remapped_calls"] += sum(by_tool.values())
                stats["rows_with_remap"] += 1
                remap_by_tool.update(by_tool)
            new_messages, aligned = remap_messages(row.get("messages") or [], gold, args.mode)
            if not aligned and by_tool:
                stats["msg_align_fail"] += 1
            row["gold_tool_calls"] = new_gold
            row["messages"] = new_messages
            row["expected_tools"] = [c.get("tool") for c in new_gold if c.get("tool")]
            active_slugs.update(row["expected_tools"])
            rows.append(row)

    # Safety: active set must be clean (goal 3) and ipython-free.
    bad = sorted(s for s in active_slugs if any(k in s for k in EXCLUDE_SUBSTR))
    if bad:
        raise SystemExit(f"Active set still contains excluded tools: {bad}")

    # Build the catalog from the live registry, restricted to the active set.
    catalog = build_registry_catalog(active_slugs)
    catalog_blob = _compact(catalog)
    schema_props = {c["slug"]: set((c["parameters"].get("properties") or {}).keys()) for c in catalog}
    schema_req = {c["slug"]: set((c["parameters"].get("required") or [])) for c in catalog}

    def _best_aligned(slug, call):
        """Align from whichever source (arguments / raw_arguments) best covers
        the tool's required params — gold `arguments` is sometimes a wrapper
        like {'action': ...} while the real args live in raw_arguments."""
        props, req = schema_props[slug], schema_req.get(slug, set())
        cands = []
        for key in ("arguments", "raw_arguments"):
            v = call.get(key)
            if isinstance(v, dict):
                cands.append(align_args(slug, v, props))
        if not cands:
            return {}
        # prefer one with no missing required, else the one with most keys
        full = [a for a in cands if not (req - set(a))]
        return max(full or cands, key=len)

    # Pass 1.5: align every gold/message argument to the tool's real schema.
    stats["aligned_calls"] = 0
    for row in rows:
        for call in row.get("gold_tool_calls") or []:
            slug = call.get("tool")
            if slug not in schema_props:
                continue
            new_args = _best_aligned(slug, call)
            if new_args != (call.get("arguments") or {}):
                stats["aligned_calls"] += 1
            call["arguments"] = new_args
        for msg in row.get("messages") or []:
            if msg.get("role") != "assistant":
                continue
            try:
                payload = json.loads(msg.get("content", ""))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            changed = False
            for a in (payload.get("actions") or []):
                if not isinstance(a, dict):
                    continue
                slug = a.get("tool")
                if slug not in schema_props or not isinstance(a.get("arguments"), dict):
                    continue
                aligned = align_args(slug, a["arguments"], schema_props[slug])
                if aligned != a["arguments"]:
                    a["arguments"] = aligned
                    changed = True
            if changed:
                msg["content"] = json.dumps(payload, ensure_ascii=False)

    # Pass 2: overwrite every system-prompt catalog with the registry catalog.
    for row in rows:
        if not set_system_catalog(row.get("messages") or [], catalog_blob):
            stats["sys_patch_fail"] += 1

    # Write strict jsonl.
    dst = out_dir / "sft_train_strict.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Shuffled copy (reflection / ReAct default ordering).
    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    shuf_path = out_dir / f"sft_train_strict_shuffled_seed{args.seed}.jsonl"
    with shuf_path.open("w", encoding="utf-8") as f:
        for row in shuffled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (shuf_path.with_suffix(shuf_path.suffix + ".metadata.json")).write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(dst.relative_to(REPO)),
            "output_path": str(shuf_path.relative_to(REPO)),
            "shuffle_seed": args.seed,
            "num_samples": len(shuffled),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Catalog + manifest + README.
    (out_dir / "tools_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "format": "terrabox_openearth_earthbench_sft_v3_fixdata",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "derived_from": str(src.relative_to(REPO)),
        "description": (
            "Fixdata restores every ipython.execute placeholder to its real "
            "Terrabox tool (compute.* and EarthAgent index/stats helpers), with "
            "unmapped helpers routed to compute.solver so NO ipython remains. The "
            "SFT tool catalog is rebuilt from the live registry, restricted to the "
            "active tool set, so SFT and rollout see identical tools."
        ),
        "counts": {
            "strict_samples": stats["rows"],
            "remapped_calls": stats["remapped_calls"],
            "rows_with_remap": stats["rows_with_remap"],
            "remap_by_raw_tool": dict(remap_by_tool),
            "active_tools": len(active_slugs),
        },
        "active_tools": sorted(active_slugs),
        "integrity": {
            "msg_align_fail": stats["msg_align_fail"],
            "sys_patch_fail": stats["sys_patch_fail"],
        },
        "files": {
            "sft_train_strict": "sft_train_strict.jsonl",
            "sft_train_strict_shuffled": shuf_path.name,
            "tools_catalog": "tools_catalog.json",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "# Terrabox fixdata (real tools restored + registry catalog)\n\n"
        "Derived from `data/newdata/sft_train_strict.jsonl`. Every "
        "`ipython.execute` placeholder is remapped to a real Terrabox tool "
        "(OpenEarth Calculator/Solver/Plot → `compute.*`; EarthAgent "
        "`calculate_ndwi/ndti/...` → `geo_raster.calculate_index`; batch stats → "
        "`geo_statistics.batch_raster_stats`; pure-Python helpers → "
        "`compute.solver`). No `ipython.execute` remains.\n\n"
        "The system-prompt tool catalog is rebuilt from the LIVE registry and "
        "restricted to the active tool set, so the tools shown during SFT match "
        "what a restricted rollout binds (same slugs, same descriptions), and no "
        "mock/osm/bing/vlm tool appears.\n\n"
        f"- strict samples: {stats['rows']}\n"
        f"- remapped calls: {stats['remapped_calls']} ({dict(remap_by_tool)})\n"
        f"- active tools in catalog: {len(active_slugs)}\n",
        encoding="utf-8",
    )

    print(json.dumps({"manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
