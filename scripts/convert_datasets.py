#!/usr/bin/env python3
"""
[DEPRECATED] 功能已拆分到：
  - convert_openearth_to_evolution.py (OpenEarth → Evolution 格式)
  - prepare_merged_dataset.py (合并 OpenEarth + EarthBench)
  - prepare_data.py (外部数据同步)

Dataset Conversion Script
=========================
Converts OpenEarthAgent and Earth-Bench datasets to Terrabox-compatible formats.

Outputs:
  data/openearth/train_sft.jsonl   - 14,538 SFT training samples (messages format)
  data/openearth/eval.jsonl        - 1,169 evaluation samples
  data/earthbench/eval.jsonl       - 248 Earth-Bench evaluation questions

Run:
  cd /data1/yuhongjie2/terrabox
  python scripts/convert_datasets.py
"""

import json
import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

OEA_ROOT     = Path("/data1/yuhongjie2/OpenEarthAgent")
OEA_TRAIN    = OEA_ROOT / "data" / "train.json"
OEA_TEST     = OEA_ROOT / "data" / "test.json"
OEA_TEST_IMG = OEA_ROOT / "data" / "test"

EA_ROOT      = Path("/data1/yuhongjie2/Earth-Agent")
EA_BENCH     = EA_ROOT / "benchmark" / "question.json"
EA_BENCH_DATA= EA_ROOT / "benchmark" / "data"

# ──────────────────────────────────────────────────────────────────────────────
# Tool name mappings
# ──────────────────────────────────────────────────────────────────────────────

# OpenEarthAgent tool name → Terrabox slug
OPENEARTH_TOOL_MAPPING: Dict[str, Optional[str]] = {
    "GetAreaBoundary":          "osm_gis.get_area_boundary",
    "AddPoisLayer":             "osm_gis.add_pois_layer",
    "ComputeDistance":          "osm_gis.compute_route_dist",
    "GetBboxFromGeotiff":       "osm_gis.get_bbox_from_raster",
    "AddIndexLayer":            "geo_raster.calculate_index",
    "ComputeIndexChange":       "geo_raster.raster_diff",
    "ChangeDetection":          "geo_perception.change_os_detect",
    "ShowIndexLayer":           "ipython_code.execute",
    "ObjectDetection":          "geo_perception.strip_rcnn_detect",
    "ImageDescription":         "geo_perception.vlm_analyze",
    "RegionAttributeDescription": "geo_perception.vlm_analyze",
    "SegmentObjectPixels":      "geo_perception.sam2_segment",
    "CountGivenObject":         "geo_perception.instructsam",
    "Calculator":               "ipython_code.execute",
    "Solver":                   "ipython_code.execute",
    "GoogleSearch":             "bing_search.search",
    "OCR":                      "geo_perception.ocr_extract",
    "TextToBbox":               "geo_perception.remotesam",
    "DrawBox":                  "geo_perception.draw_bboxes",
    "AddText":                  "geo_perception.add_text",
    "Plot":                     "ipython_code.execute",
    "DisplayOnMap":             "ipython_code.execute",
    "DisplayOnGeotiff":         "ipython_code.execute",
    "Terminate":                None,   # agent stop signal, skip
}

# Earth-Agent tool name → Terrabox slug
EARTH_AGENT_TOOL_MAPPING: Dict[str, Optional[str]] = {
    # Index Kit
    "calculate_batch_ndvi":     "geo_raster.calculate_index",
    "calculate_batch_ndwi":     "geo_raster.calculate_index",
    "calculate_batch_ndbi":     "geo_raster.calculate_index",
    "calculate_batch_evi":      "geo_raster.calculate_evi",
    "calculate_batch_nbr":      "geo_raster.calculate_index",
    "calculate_batch_fvc":      "geo_raster.calculate_fvc",
    "calculate_batch_wri":      "geo_raster.calculate_wri",
    "calculate_batch_frp":      "geo_raster.calculate_frp",
    "calculate_batch_ndti":     "geo_raster.calculate_index",
    "calculate_batch_ndsi":     "geo_raster.calculate_index",
    "calc_extreme_snow_loss_percentage_from_binary_map": "geo_raster.calc_snow_loss_stats",
    "compute_tvdi":             "geo_raster.compute_tvdi",

    # Analysis Kit
    "compute_linear_trend":     "geoanalysis.compute_linear_trend",
    "mann_kendall_test":        "geoanalysis.mann_kendall_test",
    "sens_slope":               "geoanalysis.sens_slope",
    "stl_decompose":            "geoanalysis.stl_decompose",
    "detect_change_points":     "geoanalysis.detect_change_points",
    "autocorrelation_function": "geoanalysis.autocorrelation_function",
    "detect_seasonality_acf":   "geoanalysis.detect_seasonality_acf",
    "getis_ord_gi_star":        "geoanalysis.getis_ord_gi_star",
    "analyze_hotspot_direction":"geoanalysis.analyze_hotspot_direction",
    "count_spikes_from_values": "geoanalysis.count_spikes",

    # Perception Kit
    "MSCN":                     "geo_perception.mscn_classify",
    "RemoteCLIP":               "geo_perception.remoteclip_analysis",
    "Strip_R_CNN":              "geo_perception.strip_rcnn_detect",
    "SM3Det":                   "geo_perception.sm3det_detect",
    "RemoteSAM":                "geo_perception.remotesam",
    "InstructSAM":              "geo_perception.instructsam",
    "SAM2":                     "geo_perception.sam2_segment",
    "ChangeOS":                 "geo_perception.change_os_detect",
    "threshold_segmentation":   "geo_raster.threshold_segmentation",
    "bbox_expansion":           "geo_perception.bbox_expand",
    "count_above_threshold":    "geo_raster.count_above_threshold",
    "count_skeleton_contours":  "geo_raster.count_skeleton_contours",
    "bboxes2centroids":         "geo_perception.bbox_to_centroid",
    "centroid_distance_extremes": "geo_perception.centroid_distance_extremes",
    "calculate_bbox_area":      "geo_perception.bbox_area",

    # Inversion Kit
    "lst_single_channel":       "earth_sci.calculate_lst_sc",
    "lst_multi_channel":        "earth_sci.calculate_lst_mc",
    "split_window":             "earth_sci.calculate_split_window",
    "temperature_emissivity_separation": "earth_sci.calculate_tes",
    "modis_day_night_lst":      "earth_sci.calculate_modis_day_night",
    "ttm_lst":                  "earth_sci.calculate_ttm",
    "calculate_mean_lst_by_ndvi": "earth_sci.stats_lst_ndvi",
    "calculate_max_lst_by_ndvi":  "earth_sci.stats_lst_ndvi",
    "band_ratio":               "earth_sci.calculate_pwv",
    "ATI":                      "earth_sci.calculate_ati",
    "dual_polarization_differential": "earth_sci.microwave_dpdm",
    "dual_frequency_diff":      "earth_sci.microwave_ddm",
    "dual_polarization_ratio":  "earth_sci.microwave_prm",
    "multi_freq_bt":            "earth_sci.microwave_multi_freq",
    "calculate_water_turbidity_ntu": "earth_sci.calculate_turbidity",
    "nasa_team_sea_ice_concentration": "earth_sci.calculate_sea_ice",

    # Statistics Kit
    "calc_batch_image_mean":    "geo_statistics.batch_raster_stats",
    "calc_batch_image_std":     "geo_statistics.batch_raster_stats",
    "calc_batch_image_median":  "geo_statistics.batch_raster_stats",
    "calc_batch_image_min":     "geo_statistics.batch_raster_stats",
    "calc_batch_image_max":     "geo_statistics.batch_raster_stats",
    "calc_batch_image_mean_mean": "geo_statistics.mean_of_means",
    "calc_batch_image_mean_max":  "geo_statistics.batch_raster_stats",
    "calc_batch_image_mean_threshold": "geo_statistics.threshold_ratio",
    "coefficient_of_variation": "geo_statistics.calc_cv",
    "skewness":                 "geo_statistics.calc_skewness",
    "kurtosis":                 "geo_statistics.calc_kurtosis",
    "calc_batch_image_skewness":"geo_statistics.calc_skewness",
    "calc_batch_image_kurtosis":"geo_statistics.calc_kurtosis",
    "get_percentile_value_from_image": "geo_statistics.calc_percentile",
    "calculate_threshold_ratio":"geo_statistics.threshold_ratio",
    "calculate_intersection_percentage": "geo_statistics.intersection_percentage",
    "calculate_multi_band_threshold_ratio": "geo_statistics.multi_band_threshold",
    "count_pixels_satisfying_conditions": "geo_statistics.count_pixels_condition",
    "count_images_exceeding_threshold_ratio": "geo_statistics.count_images_exceeding",
    "average_ratio_exceeding_threshold": "geo_statistics.threshold_ratio",
    "percentage_change":        "geo_statistics.percentage_change",
    "multiply":                 "geo_statistics.scalar_arithmetic",
    "difference":               "geo_statistics.scalar_arithmetic",
    "division":                 "geo_statistics.scalar_arithmetic",
    "mean":                     "geo_statistics.mean_of_means",
    "max_value_and_index":      "geo_statistics.batch_raster_stats",
    "min_value_and_index":      "geo_statistics.batch_raster_stats",
    "kelvin_to_celsius":        "geo_statistics.kelvin_to_celsius",
    "celsius_to_kelvin":        "geo_statistics.celsius_to_kelvin",
    "hotspot_percentage":       "geo_raster.hotspot_percentage",
    "calc_single_image_hotspot_percentage": "geo_raster.hotspot_percentage",
    "calc_batch_image_hotspot_percentage":  "geo_raster.hotspot_percentage",
    "calc_single_image_fire_pixels":        "geo_raster.count_fire_pixels",
    "calc_batch_fire_pixels":   "geo_raster.count_fire_pixels",
    "create_fire_increase_map": "geo_raster.create_fire_increase_map",
    "identify_fire_prone_areas":"geo_raster.identify_fire_prone_areas",

    # Utility (Earth-Agent specific)
    "get_filelist":             "bash.execute",
    "calculate_tif_average":    "geo_raster.raster_average",
    "ceil_number":              "ipython_code.execute",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def map_tool_name(name: str, mapping: Dict[str, Optional[str]]) -> Optional[str]:
    """Return terrabox slug for a tool name, or the original if unmapped."""
    if name in mapping:
        return mapping[name]
    # Try case-insensitive
    for k, v in mapping.items():
        if k.lower() == name.lower():
            return v
    return name  # unknown tool - keep as-is


def extract_tool_slugs_from_oea_conversation(conversation: List[Dict]) -> List[str]:
    """Extract tool slugs from OpenEarthAgent conversation format."""
    slugs = []
    for turn in conversation:
        if turn.get("from") != "gpt":
            continue
        val = turn.get("value", "")
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                parsed = json.loads(val)
                for action in parsed.get("actions", []):
                    name = action.get("name")
                    if name:
                        mapped = map_tool_name(name, OPENEARTH_TOOL_MAPPING)
                        if mapped:
                            slugs.append(mapped)
            except json.JSONDecodeError:
                pass
    return slugs


def extract_tool_slugs_from_ea_dialogs(dialogs: List[Dict]) -> List[str]:
    """Extract tool slugs from Earth-Agent dialog format."""
    slugs = []
    for turn in dialogs:
        if turn.get("role") != "assistant":
            continue
        for tc in turn.get("tool_calls", []):
            name = tc.get("function", {}).get("name", "")
            if name:
                mapped = map_tool_name(name, EARTH_AGENT_TOOL_MAPPING)
                if mapped:
                    slugs.append(mapped)
    return slugs


def remap_tool_names_in_gpt_value(val: str, mapping: Dict[str, Optional[str]]) -> str:
    """Replace tool names inside a GPT response JSON string."""
    if not val.strip().startswith("{"):
        return val
    try:
        parsed = json.loads(val)
    except json.JSONDecodeError:
        return val

    modified = False
    for action in parsed.get("actions", []):
        orig_name = action.get("name", "")
        if orig_name:
            new_name = mapping.get(orig_name)
            if new_name is None:
                # Remove this action (Terminate etc.)
                action["name"] = "__SKIP__"
                modified = True
            elif new_name != orig_name:
                action["name"] = new_name
                modified = True

    # Filter out skipped actions
    if modified:
        parsed["actions"] = [a for a in parsed.get("actions", []) if a.get("name") != "__SKIP__"]
        return json.dumps(parsed, ensure_ascii=False)
    return val


def resolve_image_path(img: str, base_dir: Path) -> str:
    """Resolve an image path relative to a base directory, returning absolute path."""
    p = Path(img)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative to OEA root
    candidate = base_dir / img
    if candidate.exists():
        return str(candidate)
    # Try relative to test dir
    candidate2 = OEA_TEST_IMG / p.name
    if candidate2.exists():
        return str(candidate2)
    # Return the path as-is (may not exist but preserve reference)
    return str(base_dir / img)


# ──────────────────────────────────────────────────────────────────────────────
# Converters
# ──────────────────────────────────────────────────────────────────────────────

def convert_oea_train(src: Path, dst: Path) -> int:
    """
    Convert OpenEarthAgent train.json → Terrabox SFT JSONL.
    Format: messages with role/content (OpenAI chat format).
    """
    log.info(f"Reading {src} ...")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    skipped = 0

    with open(dst, "w", encoding="utf-8") as out:
        for entry in data:
            conversation = entry.get("conversation", [])
            images = entry.get("images", [])

            if not conversation:
                skipped += 1
                continue

            messages = []
            first_human = True

            for turn in conversation:
                role = "user" if turn.get("from") == "human" else "assistant"
                raw_val = turn.get("value", "")

                if role == "user":
                    content = []
                    # Add images to the first human turn
                    if first_human and images:
                        for img in images:
                            abs_img = resolve_image_path(img, OEA_ROOT)
                            content.append({"type": "image", "image": abs_img})
                        first_human = False

                    # Clean up agent prompt prefix for training
                    text = raw_val
                    if "<AGENT_PROMPT>" in text:
                        # Extract question from "Question: ..." part
                        q_idx = text.find("\n\nQuestion:")
                        if q_idx >= 0:
                            text = text[q_idx + 2:].strip()
                    if "OBSERVATION:" in text:
                        text = raw_val  # Keep observations as-is for multi-turn

                    content.append({"type": "text", "text": text})
                    messages.append({"role": "user", "content": content})

                else:  # assistant
                    # Remap tool names in the JSON action string
                    remapped = remap_tool_names_in_gpt_value(raw_val, OPENEARTH_TOOL_MAPPING)
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": remapped}]
                    })

            if len(messages) >= 2:
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
            else:
                skipped += 1

    log.info(f"  → {written} SFT samples written, {skipped} skipped")
    return written


def convert_oea_test(src: Path, dst: Path) -> int:
    """
    Convert OpenEarthAgent test.json → Terrabox evaluation JSONL.
    Format: {id, images, question, expected_tools, conversation}
    """
    log.info(f"Reading {src} ...")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    with open(dst, "w", encoding="utf-8") as out:
        for entry in data:
            idx = entry.get("idx", 0)
            conversation = entry.get("conversation", [])
            images = entry.get("images", [])

            # Extract question from first human turn
            question = ""
            for turn in conversation:
                if turn.get("from") == "human":
                    val = turn.get("value", "")
                    q_idx = val.find("\n\nQuestion:")
                    if q_idx >= 0:
                        question = val[q_idx + 2:].strip()
                    else:
                        question = val.strip()
                    break

            expected_tools = extract_tool_slugs_from_oea_conversation(conversation)
            abs_images = [resolve_image_path(img, OEA_ROOT) for img in images]

            record = {
                "id": f"oea_test_{idx}",
                "source": "openearth",
                "images": abs_images,
                "question": question,
                "expected_tools": list(dict.fromkeys(expected_tools)),  # deduplicate, preserve order
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"  → {written} eval samples written")
    return written


def convert_earthbench(src: Path, data_root: Path, dst: Path) -> int:
    """
    Convert Earth-Bench question.json → Terrabox evaluation JSONL.
    Format: {id, question, data_dir, expected_tools, modality}
    """
    log.info(f"Reading {src} ...")
    with open(src, encoding="utf-8") as f:
        questions = json.load(f)

    # Determine modality boundaries (from Earth-Agent paper):
    # Questions 1-100: Spectrum, 101-188: Products, 189-248: RGB
    def get_modality(qnum: int) -> str:
        if qnum <= 100:
            return "spectrum"
        elif qnum <= 188:
            return "products"
        else:
            return "rgb"

    written = 0
    with open(dst, "w", encoding="utf-8") as out:
        for qid_str, qdata in questions.items():
            try:
                qnum = int(qid_str)
            except ValueError:
                qnum = 0

            dialogs = qdata.get("dialogs", [])
            if not dialogs:
                continue

            # Extract question text
            question = ""
            for d in dialogs:
                if d.get("role") == "user":
                    question = d.get("content", "")
                    break

            # Extract ground-truth tool sequence
            expected_tools = extract_tool_slugs_from_ea_dialogs(dialogs)

            # Extract final answer from last assistant turn without tool_calls
            ground_truth = ""
            for d in reversed(dialogs):
                if d.get("role") == "assistant" and not d.get("tool_calls"):
                    ground_truth = d.get("content", "")
                    if ground_truth:
                        break

            # List data files in question directory
            q_dir = data_root / f"question{qnum}"
            data_files = []
            if q_dir.exists():
                data_files = sorted([
                    str(q_dir / f)
                    for f in os.listdir(q_dir)
                    if not f.startswith(".")
                ])

            record = {
                "id": f"earthbench_{qnum}",
                "source": "earthbench",
                "question_number": qnum,
                "modality": get_modality(qnum),
                "question": question,
                "data_dir": str(q_dir),
                "data_files": data_files[:20],  # cap at 20 to avoid huge records
                "expected_tools": list(dict.fromkeys(expected_tools)),
                "ground_truth": ground_truth,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"  → {written} eval samples written")
    return written


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Terrabox Dataset Conversion")
    log.info("=" * 60)

    # Create output dirs
    oea_dir = DATA_ROOT / "openearth"
    ea_dir  = DATA_ROOT / "earthbench"
    oea_dir.mkdir(parents=True, exist_ok=True)
    ea_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Copy source JSON files ────────────────────────────────────────
    log.info("\n[1/3] Copying source JSON files...")
    for src, dst in [
        (OEA_TRAIN, oea_dir / "train.json"),
        (OEA_TEST,  oea_dir / "test.json"),
        (EA_BENCH,  ea_dir  / "question.json"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
            log.info(f"  Copied {src.name} → {dst}")
        else:
            log.warning(f"  Source not found: {src}")

    # ── Step 2: Convert OpenEarthAgent ────────────────────────────────────────
    log.info("\n[2/3] Converting OpenEarthAgent dataset...")

    if OEA_TRAIN.exists():
        n_train = convert_oea_train(OEA_TRAIN, oea_dir / "train_sft.jsonl")
    else:
        log.warning(f"  Skipping train: {OEA_TRAIN} not found")
        n_train = 0

    if OEA_TEST.exists():
        n_test = convert_oea_test(OEA_TEST, oea_dir / "eval.jsonl")
    else:
        log.warning(f"  Skipping test: {OEA_TEST} not found")
        n_test = 0

    # ── Step 3: Convert Earth-Bench ───────────────────────────────────────────
    log.info("\n[3/3] Converting Earth-Bench dataset...")

    if EA_BENCH.exists():
        n_bench = convert_earthbench(EA_BENCH, EA_BENCH_DATA, ea_dir / "eval.jsonl")
    else:
        log.warning(f"  Skipping: {EA_BENCH} not found")
        n_bench = 0

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Conversion complete!")
    log.info(f"  data/openearth/train_sft.jsonl : {n_train:,} samples")
    log.info(f"  data/openearth/eval.jsonl      : {n_test:,} samples")
    log.info(f"  data/earthbench/eval.jsonl     : {n_bench:,} samples")
    log.info("=" * 60)

    # Verify output files
    for path in [
        oea_dir / "train_sft.jsonl",
        oea_dir / "eval.jsonl",
        ea_dir  / "eval.jsonl",
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            with open(path) as f:
                lines = sum(1 for _ in f)
            log.info(f"  ✓ {path.name}: {lines:,} lines, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
