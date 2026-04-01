#!/usr/bin/env python3
"""prepare_data.py — Copy image/raster files into terrabox and normalize dataset formats.

Handles two external datasets:
  - OpenEarthAgent  (/data1/yuhongjie2/OpenEarthAgent)
  - Earth-Agent     (/data1/yuhongjie2/Earth-Agent)

Subcommands:
    copy    Copy image/raster files, update paths in eval.jsonl to local absolute paths
    split   Stratified train/eval split for EarthBench (80/20 by modality)
    unify   Normalize both eval.jsonl to unified schema + generate combined_eval.jsonl
    all     Run copy → split → unify in order

Run from repo root:
    python scripts/prepare_data.py all
    python scripts/prepare_data.py copy
    python scripts/prepare_data.py split
    python scripts/prepare_data.py unify
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Destinations within terrabox
_OEA_EVAL_JSONL     = _REPO_ROOT / "data" / "openearth" / "eval.jsonl"
_OEA_IMAGES_DIR     = _REPO_ROOT / "data" / "openearth" / "images"
_EB_EVAL_JSONL      = _REPO_ROOT / "data" / "earthbench" / "eval.jsonl"
_EB_TRAIN_JSONL     = _REPO_ROOT / "data" / "earthbench" / "train.jsonl"
_EB_QDATA_DIR       = _REPO_ROOT / "data" / "earthbench" / "question_data"
_EB_QUESTION_JSON   = _REPO_ROOT / "data" / "earthbench" / "question.json"
_COMBINED_EVAL      = _REPO_ROOT / "data" / "combined_eval.jsonl"

# External source projects
_OEA_ROOT           = Path("/data1/yuhongjie2/OpenEarthAgent")
_OEA_TEST_IMG       = _OEA_ROOT / "data" / "test"
_EA_ROOT            = Path("/data1/yuhongjie2/Earth-Agent")
_EA_BENCH_DATA      = _EA_ROOT / "benchmark" / "data"
_EA_QUESTION_JSON   = _EA_ROOT / "benchmark" / "question.json"

# ─────────────────────────────────────────────────────────────────
# Earth-Agent tool name → Terrabox slug mapping
# (matches convert_datasets.py and data_loader.py)
# ─────────────────────────────────────────────────────────────────

_EA_TOOL_MAP: dict[str, str] = {
    # File utilities
    "get_filelist":              "bash.execute",
    "list_files":                "bash.execute",
    "bash":                      "bash.execute",
    "execute_bash":              "bash.execute",
    # Raster indices & computation
    "compute_tvdi":              "geo_raster.compute_tvdi",
    "calculate_tvdi":            "geo_raster.compute_tvdi",
    "compute_ndvi":              "geo_raster.calculate_index",
    "compute_ndwi":              "geo_raster.calculate_index",
    "compute_ndbi":              "geo_raster.calculate_index",
    "compute_nbr":               "geo_raster.calculate_index",
    "calculate_index":           "geo_raster.calculate_index",
    "raster_diff":               "geo_raster.raster_diff",
    "raster_average":            "geo_raster.raster_average",
    "compute_raster_stats":      "geo_raster.statistics",
    "raster_statistics":         "geo_raster.statistics",
    # Statistics & analysis
    "batch_raster_stats":        "geo_statistics.batch_raster_stats",
    "threshold_ratio":           "geo_statistics.threshold_ratio",
    "count_spikes":              "geoanalysis.count_spikes",
    "detect_spikes":             "geoanalysis.count_spikes",
    "compute_linear_trend":      "geoanalysis.compute_linear_trend",
    "trend_analysis":            "geoanalysis.compute_linear_trend",
    "mann_kendall_test":         "geoanalysis.mann_kendall",
    "seasonality_analysis":      "geoanalysis.seasonality",
    "compute_correlation":       "geoanalysis.correlation",
    "spatial_autocorrelation":   "geoanalysis.spatial_autocorrelation",
    # Perception
    "remoteclip_analysis":       "geo_perception.remoteclip_analysis",
    "strip_rcnn_detect":         "geo_perception.strip_rcnn_detect",
    "vlm_analyze":               "geo_perception.vlm_analyze",
    "mscn_enhance":              "geo_perception.mscn_enhance",
    "sam2_segment":              "geo_perception.sam2_segment",
    "remotesam":                 "geo_perception.remotesam",
    # Geocoding
    "get_area_boundary":         "osm_gis.get_area_boundary",
    "add_pois_layer":            "osm_gis.add_pois_layer",
    # Code execution
    "python_code":               "ipython_code.execute",
    "execute_python":            "ipython_code.execute",
    "calculator":                "ipython_code.execute",
}

_NOISE_TOOLS = frozenset({"Terminate", "Finish", "Done", "FinalAnswer", "Solver"})


def _ea_slug(name: str) -> str:
    """Map an Earth-Agent tool name to a Terrabox slug."""
    if name in _EA_TOOL_MAP:
        return _EA_TOOL_MAP[name]
    lo = name.lower()
    if lo in _EA_TOOL_MAP:
        return _EA_TOOL_MAP[lo]
    if "." in name:
        return name
    return lo


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────
# Step 1: copy
# ─────────────────────────────────────────────────────────────────

def do_copy() -> None:
    """Copy image/raster files into terrabox and update paths in eval.jsonl files."""
    print("\n=== COPY: Copying data files into terrabox ===")

    # 1a. OpenEarthAgent test images
    if _OEA_TEST_IMG.exists():
        _OEA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        img_files = [f for f in _OEA_TEST_IMG.iterdir() if f.is_file()]
        n_copied = 0
        for src in img_files:
            dst = _OEA_IMAGES_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                n_copied += 1
        print(f"  OEA test images: {n_copied} new / {len(img_files)} total → {_OEA_IMAGES_DIR}")

        # Update eval.jsonl paths
        if _OEA_EVAL_JSONL.exists():
            records = _load_jsonl(_OEA_EVAL_JSONL)
            updated = 0
            for rec in records:
                new_images = []
                for img in rec.get("images", []):
                    fname = Path(img).name
                    local = _OEA_IMAGES_DIR / fname
                    if local.exists():
                        new_images.append(str(local))
                        if img != str(local):
                            updated += 1
                    else:
                        new_images.append(img)  # keep original if local copy missing
                rec["images"] = new_images
            _write_jsonl(_OEA_EVAL_JSONL, records)
            print(f"  Updated {updated} image paths in {_OEA_EVAL_JSONL.name}")
    else:
        print(f"  [WARNING] OEA test image dir not found: {_OEA_TEST_IMG}")

    # Report missing training images
    _report_missing_train_images()

    # 1b. EarthBench question data directories
    if _EA_BENCH_DATA.exists():
        _EB_QDATA_DIR.mkdir(parents=True, exist_ok=True)
        q_dirs = sorted(
            d for d in _EA_BENCH_DATA.iterdir()
            if d.is_dir() and d.name.startswith("question")
        )
        n_copied = n_exist = 0
        for q_dir in q_dirs:
            dst = _EB_QDATA_DIR / q_dir.name
            if dst.exists():
                n_exist += 1
            else:
                shutil.copytree(q_dir, dst)
                n_copied += 1
        print(f"  EarthBench data: {n_copied} new / {n_exist} already present → {_EB_QDATA_DIR}")

        # Update eval.jsonl paths
        if _EB_EVAL_JSONL.exists():
            records = _load_jsonl(_EB_EVAL_JSONL)
            _update_eb_paths(records)
            _write_jsonl(_EB_EVAL_JSONL, records)
            print(f"  Updated data_dir/data_files paths in {_EB_EVAL_JSONL.name} ({len(records)} records)")
    else:
        print(f"  [WARNING] Earth-Agent benchmark data not found: {_EA_BENCH_DATA}")

    print("COPY done.")


def _update_eb_paths(records: list[dict]) -> None:
    """In-place: replace external Earth-Agent paths with local question_data paths."""
    for rec in records:
        # data_dir
        old_dir = rec.get("data_dir", "")
        if old_dir:
            q_name = Path(old_dir).name  # e.g. "question1"
            local_dir = _EB_QDATA_DIR / q_name
            if local_dir.exists():
                rec["data_dir"] = str(local_dir)

        # data_files
        new_files = []
        for f in rec.get("data_files", []):
            fname = Path(f).name
            q_name = Path(rec.get("data_dir", f)).parent.name or Path(f).parent.name
            # Try local location
            local_f = _EB_QDATA_DIR / Path(rec["data_dir"]).name / fname if rec.get("data_dir") else None
            if local_f and local_f.exists():
                new_files.append(str(local_f))
            else:
                new_files.append(f)  # keep original if not found locally
        rec["data_files"] = new_files


def _report_missing_train_images() -> None:
    """Print a summary of missing OpenEarth training images (dior/nwpu etc.)."""
    oea_train = _REPO_ROOT / "data" / "openearth" / "train.json"
    if not oea_train.exists():
        return
    with open(oea_train, encoding="utf-8") as f:
        train_data = json.load(f)
    missing_prefixes: set[str] = set()
    total_missing = 0
    for rec in train_data:
        for img in rec.get("images", []):
            p = Path(img)
            # Check both relative (from repo root) and common absolute locations
            full = _REPO_ROOT / img if not p.is_absolute() else p
            oea_abs = _OEA_ROOT / img if not p.is_absolute() else p
            if not full.exists() and not oea_abs.exists():
                prefix = p.name.split("_")[0] if "_" in p.name else p.suffix
                missing_prefixes.add(prefix)
                total_missing += 1
                break  # only count per record
    if total_missing:
        print(f"\n  [WARNING] {total_missing} training records reference images not on disk.")
        print(f"  Missing dataset prefixes: {sorted(missing_prefixes)}")
        print("  These images (DIOR, NWPU, DOTA, etc.) must be downloaded separately.")
        print("  This does NOT affect eval — eval uses test images which are now local.")


# ─────────────────────────────────────────────────────────────────
# Step 2: split
# ─────────────────────────────────────────────────────────────────

def do_split() -> None:
    """Stratified 80/20 train/eval split of EarthBench by modality."""
    print("\n=== SPLIT: Creating EarthBench train/eval split ===")

    if not _EB_EVAL_JSONL.exists():
        print(f"  [ERROR] {_EB_EVAL_JSONL} not found.")
        sys.exit(1)

    records = _load_jsonl(_EB_EVAL_JSONL)

    if len(records) <= 50:
        print(f"  Already split ({len(records)} records in eval.jsonl). Nothing to do.")
        return

    print(f"  Loaded {len(records)} records from eval.jsonl")

    # Group by modality (spectrum / products / rgb)
    by_mod: dict[str, list[dict]] = {}
    for rec in records:
        mod = rec.get("modality")
        if not mod:
            qnum = rec.get("question_number", 0)
            mod = "spectrum" if qnum <= 100 else ("products" if qnum <= 188 else "rgb")
        by_mod.setdefault(mod, []).append(rec)

    random.seed(42)
    train_records: list[dict] = []
    eval_records: list[dict] = []

    for mod in sorted(by_mod):
        items = list(by_mod[mod])
        random.shuffle(items)
        n_train = int(len(items) * 0.8)
        train_records.extend(items[:n_train])
        eval_records.extend(items[n_train:])
        print(f"    {mod}: {len(items)} total → {n_train} train + {len(items)-n_train} eval")

    # Sort by question_number for readability
    train_records.sort(key=lambda x: x.get("question_number", 0))
    eval_records.sort(key=lambda x: x.get("question_number", 0))

    # Add tools_called to train records (use expected_tools as the gold trajectory)
    for rec in train_records:
        rec.setdefault("tools_called", list(rec.get("expected_tools", [])))

    # Also enrich tools_called from question.json dialogs if available
    _enrich_tools_called(train_records)

    _write_jsonl(_EB_TRAIN_JSONL, train_records)
    _write_jsonl(_EB_EVAL_JSONL, eval_records)

    print(f"  train.jsonl → {len(train_records)} records: {_EB_TRAIN_JSONL}")
    print(f"  eval.jsonl  → {len(eval_records)} records: {_EB_EVAL_JSONL}")
    print("SPLIT done.")


def _enrich_tools_called(records: list[dict]) -> None:
    """Replace tools_called with the actual tool sequence from question.json dialogs."""
    q_json = _EA_QUESTION_JSON if _EA_QUESTION_JSON.exists() else None
    if q_json is None:
        return

    with open(q_json, encoding="utf-8") as f:
        question_data = json.load(f)

    for rec in records:
        qnum = str(rec.get("question_number", ""))
        if qnum not in question_data:
            continue
        dialogs = question_data[qnum].get("dialogs", [])
        seq: list[str] = []
        for dlg in dialogs:
            if dlg.get("role") in ("assistant", "gpt"):
                for tc in dlg.get("tool_calls", []):
                    name = tc.get("function", {}).get("name", "")
                    if name and name not in _NOISE_TOOLS:
                        seq.append(_ea_slug(name))
        if seq:
            rec["tools_called"] = seq


# ─────────────────────────────────────────────────────────────────
# Step 3: unify
# ─────────────────────────────────────────────────────────────────

# Unified schema defaults (fields that may be absent)
_UNIFIED_DEFAULTS = {
    "images":       [],
    "data_dir":     None,
    "data_files":   [],
    "ground_truth": None,
    "modality":     None,
}


def _unify_record(rec: dict) -> dict:
    """Return a copy of rec with all unified-schema fields present."""
    out = dict(rec)
    for field, default in _UNIFIED_DEFAULTS.items():
        if field not in out:
            out[field] = default if not isinstance(default, list) else list(default)
    return out


def do_unify() -> None:
    """Normalize both eval.jsonl to unified schema and create combined_eval.jsonl."""
    print("\n=== UNIFY: Normalizing schemas and generating combined_eval.jsonl ===")

    oea_records: list[dict] = []
    eb_records: list[dict] = []

    if _OEA_EVAL_JSONL.exists():
        oea_records = [_unify_record(r) for r in _load_jsonl(_OEA_EVAL_JSONL)]
        _write_jsonl(_OEA_EVAL_JSONL, oea_records)
        print(f"  openearth/eval.jsonl: {len(oea_records)} records unified")
    else:
        print(f"  [WARNING] {_OEA_EVAL_JSONL} not found — skipping")

    if _EB_EVAL_JSONL.exists():
        eb_records = [_unify_record(r) for r in _load_jsonl(_EB_EVAL_JSONL)]
        _write_jsonl(_EB_EVAL_JSONL, eb_records)
        print(f"  earthbench/eval.jsonl: {len(eb_records)} records unified")
    else:
        print(f"  [WARNING] {_EB_EVAL_JSONL} not found — skipping")

    # Also unify train.jsonl if it exists
    if _EB_TRAIN_JSONL.exists():
        eb_train = [_unify_record(r) for r in _load_jsonl(_EB_TRAIN_JSONL)]
        _write_jsonl(_EB_TRAIN_JSONL, eb_train)
        print(f"  earthbench/train.jsonl: {len(eb_train)} records unified")

    # Combined eval
    combined = oea_records + eb_records
    if combined:
        _write_jsonl(_COMBINED_EVAL, combined)
        print(f"  combined_eval.jsonl: {len(combined)} records ({len(oea_records)} OEA + {len(eb_records)} EB)")
        print(f"  → {_COMBINED_EVAL}")
    print("UNIFY done.")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare OpenEarth + EarthBench datasets for Terrabox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "command",
        choices=("copy", "split", "unify", "all"),
        help="Subcommand to run",
    )
    args = p.parse_args()

    # Validate we're running from the repo root (or nearby)
    if not (_REPO_ROOT / "src" / "terrabox").exists():
        print(f"[ERROR] Run from repo root: cd {_REPO_ROOT} && python scripts/prepare_data.py ...")
        sys.exit(1)

    if args.command in ("copy", "all"):
        do_copy()
    if args.command in ("split", "all"):
        do_split()
    if args.command in ("unify", "all"):
        do_unify()

    print("\nDone. Verification:")
    print(f"  OEA test images:    {sum(1 for _ in _OEA_IMAGES_DIR.iterdir()) if _OEA_IMAGES_DIR.exists() else 0} files")
    print(f"  EB question_data:   {sum(1 for _ in _EB_QDATA_DIR.iterdir()) if _EB_QDATA_DIR.exists() else 0} dirs")
    print(f"  OEA eval.jsonl:     {sum(1 for _ in open(_OEA_EVAL_JSONL)) if _OEA_EVAL_JSONL.exists() else 0} lines")
    print(f"  EB eval.jsonl:      {sum(1 for _ in open(_EB_EVAL_JSONL)) if _EB_EVAL_JSONL.exists() else 0} lines")
    print(f"  EB train.jsonl:     {sum(1 for _ in open(_EB_TRAIN_JSONL)) if _EB_TRAIN_JSONL.exists() else 0} lines")
    print(f"  combined_eval.jsonl:{sum(1 for _ in open(_COMBINED_EVAL)) if _COMBINED_EVAL.exists() else 0} lines")


if __name__ == "__main__":
    main()
