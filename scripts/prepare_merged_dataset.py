#!/usr/bin/env python3
"""
合并 OpenEarth + EarthBench 数据集，生成统一任务文件和 Evolution trajectory 文件。

功能：
  1. 加载 OpenEarth task-file 和 EarthBench train/eval
  2. EarthBench 工具名归一化（未前缀名 → Terrabox slug）
  3. 按 CLI 参数过滤 mock/bing/osm/changeos 任务
  4. 可选：EarthBench 文本扩充（改写阈值/分析角度，保持 expected_tools 不变）
  5. 输出统一格式的任务文件 + evolution trajectory JSONL

输出：
  data/merged/merged_train_tasks.json   — 合并后训练集
  data/merged/merged_eval_tasks.json    — 合并后评测集
  data/merged/merged_trajectories.jsonl — Evolution 格式（SeqGraphEvo 等可直接消费）
  data/merged/stats.json               — 工具统计

用法：
  cd /data1/yuhongjie2/terrabox
  python scripts/prepare_merged_dataset.py

  # 不扩充 EarthBench
  python scripts/prepare_merged_dataset.py --augment-factor 0

  # 允许 OSM 任务
  python scripts/prepare_merged_dataset.py --no-skip-osm

  # 使用含在线工具的 OpenEarth 数据
  python scripts/prepare_merged_dataset.py \
      --openearth-train data/openearth/openearth_train_no_mock_tasks.json \
      --openearth-test  data/openearth/openearth_test_no_mock_tasks.json \
      --no-skip-osm
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

# ──────────────────────────────────────────────────────────────────────────────
# Tool name mappings (Earth-Agent raw name → Terrabox slug)
# Sourced from convert_datasets.py EARTH_AGENT_TOOL_MAPPING
# ──────────────────────────────────────────────────────────────────────────────

EARTH_AGENT_TOOL_MAPPING: dict[str, Optional[str]] = {
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
    # Utility
    "get_filelist":             "bash.execute",
    "calculate_tif_average":    "geo_raster.raster_average",
    "ceil_number":              "ipython_code.execute",
    # Unmapped tools → fallback to ipython
    "argmax":                   "ipython_code.execute",
    "subtract":                 "geo_statistics.scalar_arithmetic",
    "calculate_band_mean_by_condition": "ipython_code.execute",
    "count_images_exceeding_mean_multiplier": "geo_statistics.count_images_exceeding",
    "calculate_tif_difference": "geo_statistics.scalar_arithmetic",
    "calc_threshold_value_mean": "geo_statistics.threshold_ratio",
    "calc_batch_image_hotspot_tif": "geo_raster.hotspot_percentage",
    "calc_batch_image_sum":     "geo_statistics.batch_raster_stats",
    "calculate_batch_image_mean_max_min": "geo_statistics.batch_raster_stats",
    "image_division_mean":      "geo_statistics.scalar_arithmetic",
    "get_list_object_via_indexes": "ipython_code.execute",
    "index_to_date_range":      "ipython_code.execute",
    "calculate_area":           "ipython_code.execute",
    "apply_cloud_mask":         "geo_raster.apply_cloud_mask",
}

# Tools that are mock/undeployed
MOCK_TOOLS = {
    "geo_perception.mscn_classify",
    "geo_perception.sm3det_detect",
    "geo_perception.change_os_detect",
}

# Tools requiring API keys
API_KEY_TOOLS = {
    "bing_search.search",
}

# Tools requiring OSM network access
OSM_TOOLS_PREFIX = "osm_gis."

# VLM service is intentionally excluded for the current merged rollout.
VLM_TOOLS = {
    "geo_perception.vlm_analyze",
}

# ChangeOS related keywords (for filtering by question text)
CHANGEOS_KEYWORDS = {"change_os", "changeos", "changedetection", "change detection"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool name normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_tool_name(name: str) -> Optional[str]:
    """Normalize an EarthBench tool name to Terrabox slug.

    If already prefixed (contains '.'), return as-is.
    Otherwise look up in EARTH_AGENT_TOOL_MAPPING.
    """
    name = name.strip()
    if not name:
        return None
    # Already a Terrabox slug
    if "." in name:
        return name
    # Look up mapping
    if name in EARTH_AGENT_TOOL_MAPPING:
        return EARTH_AGENT_TOOL_MAPPING[name]
    # Case-insensitive fallback
    for k, v in EARTH_AGENT_TOOL_MAPPING.items():
        if k.lower() == name.lower():
            return v
    log.debug(f"Unknown tool name '{name}' → ipython_code.execute")
    return "ipython_code.execute"


def normalize_tool_list(tools: list[str]) -> list[str]:
    """Normalize a list of tool names, removing None entries."""
    result = []
    for t in tools:
        slug = normalize_tool_name(t)
        if slug is not None:
            result.append(slug)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Task filtering
# ──────────────────────────────────────────────────────────────────────────────

def should_skip_task(
    task: dict,
    *,
    skip_mock: bool = True,
    skip_bing: bool = True,
    skip_osm: bool = True,
    skip_vlm: bool = True,
    skip_changeos: bool = True,
) -> Optional[str]:
    """Return skip reason if task should be filtered, else None."""
    expected = [normalize_tool_name(t) or t for t in task.get("expected_tools", [])]

    # Check mock tools
    if skip_mock and any(t in MOCK_TOOLS for t in expected):
        return "mock"

    # Check API key tools
    if skip_bing and any(t in API_KEY_TOOLS for t in expected):
        return "bing_api"

    # Check OSM tools
    if skip_osm and any(t.startswith(OSM_TOOLS_PREFIX) for t in expected):
        return "osm_online"

    # Check VLM tools
    if skip_vlm and any(t in VLM_TOOLS for t in expected):
        return "vlm"

    # Check changeos
    if skip_changeos:
        if any("change_os" in t for t in expected):
            return "changeos"
        question = task.get("question", "").lower()
        if any(kw in question for kw in CHANGEOS_KEYWORDS):
            return "changeos"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_openearth_tasks(path: Path) -> list[dict]:
    """Load OpenEarth task-file (JSON with 'tasks' list)."""
    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Expected list or {{'tasks': [...]}}: {path}")

    result = []
    for i, task in enumerate(tasks):
        result.append({
            "task_id": task.get("id", f"oea_{i}"),
            "source": "openearth",
            "question": task.get("question", ""),
            "images": task.get("images", []),
            "data_files": [],
            "expected_tools": task.get("expected_tools", []),
            "task_type": task.get("task_type", "geospatial_analysis"),
            "ground_truth": None,
        })
    return result


def load_earthbench_jsonl(path: Path, split: str = "train") -> list[dict]:
    """Load EarthBench JSONL file, normalizing tool names."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Normalize expected_tools (these are already prefixed in most cases)
            expected = normalize_tool_list(raw.get("expected_tools", []))
            # Also normalize tools_called for trajectory use
            tools_called = normalize_tool_list(raw.get("tools_called", []))

            tasks.append({
                "task_id": raw.get("id", f"earthbench_{split}_{len(tasks)}"),
                "source": "earthbench",
                "question": raw.get("question", ""),
                "images": raw.get("images", []),
                "data_files": raw.get("data_files", []),
                "data_dir": raw.get("data_dir", ""),
                "expected_tools": expected,
                "tools_called": tools_called,
                "task_type": raw.get("modality", "spectrum"),
                "ground_truth": raw.get("ground_truth"),
                "question_number": raw.get("question_number"),
            })
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# EarthBench text augmentation
# ──────────────────────────────────────────────────────────────────────────────

# Paraphrase templates: each is (regex_pattern, list_of_replacements)
# These modify question phrasing while preserving semantics and tool requirements.

_THRESHOLD_RE = re.compile(
    r"(\b(?:exceeds?|greater than|above|surpass(?:es|ed)?|>)\s*)"
    r"(\d+(?:\.\d+)?)"
    r"(\s*(?:K|%|km|m|degrees?)?\b)",
    re.IGNORECASE,
)

_PROPORTION_RE = re.compile(
    r"(\b(?:more than|over|at least|exceeding)\s*)"
    r"(\d+(?:\.\d+)?)"
    r"(%)",
    re.IGNORECASE,
)

_ANALYSIS_PHRASES = [
    (r"\bcalculate\b", ["compute", "determine", "estimate", "derive"]),
    (r"\banalyze\b", ["examine", "assess", "evaluate", "investigate"]),
    (r"\bdetermine\b", ["calculate", "identify", "find", "establish"]),
    (r"\bcompute\b", ["calculate", "derive", "estimate", "obtain"]),
    (r"\bidentify\b", ["detect", "locate", "find", "pinpoint"]),
]


def _perturb_threshold(text: str, rng: random.Random) -> str:
    """Slightly adjust numeric thresholds in text."""
    def _replace(m):
        prefix, num_str, suffix = m.group(1), m.group(2), m.group(3)
        try:
            val = float(num_str)
        except ValueError:
            return m.group(0)
        # Apply ±5-15% perturbation
        factor = rng.uniform(0.85, 1.15)
        new_val = val * factor
        # Keep same decimal precision
        if "." in num_str:
            decimals = len(num_str.split(".")[1])
            new_str = f"{new_val:.{decimals}f}"
        else:
            new_str = str(int(round(new_val)))
        return prefix + new_str + suffix

    return _THRESHOLD_RE.sub(_replace, text)


def _perturb_proportion(text: str, rng: random.Random) -> str:
    """Slightly adjust proportion thresholds."""
    def _replace(m):
        prefix, num_str, suffix = m.group(1), m.group(2), m.group(3)
        try:
            val = float(num_str)
        except ValueError:
            return m.group(0)
        delta = rng.choice([-10, -5, 5, 10])
        new_val = max(5, min(95, val + delta))
        if "." in num_str:
            decimals = len(num_str.split(".")[1])
            new_str = f"{new_val:.{decimals}f}"
        else:
            new_str = str(int(new_val))
        return prefix + new_str + suffix

    return _PROPORTION_RE.sub(_replace, text)


def _paraphrase_verbs(text: str, rng: random.Random) -> str:
    """Replace analysis verbs with synonyms."""
    for pattern, replacements in _ANALYSIS_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            chosen = rng.choice(replacements)
            # Replace first occurrence only (to keep text natural)
            text = re.sub(pattern, chosen, text, count=1, flags=re.IGNORECASE)
            break  # Only one substitution per variant
    return text


def augment_earthbench_task(task: dict, variant_idx: int, rng: random.Random) -> dict:
    """Create a text variant of an EarthBench task.

    Modifies question text (thresholds, verbs) but keeps expected_tools unchanged.
    """
    variant = copy.deepcopy(task)
    variant["task_id"] = f"{task['task_id']}_v{variant_idx}"
    variant["augmented_from"] = task["task_id"]

    question = task["question"]

    # Apply different augmentation strategies based on variant index
    if variant_idx == 1:
        question = _perturb_threshold(question, rng)
    elif variant_idx == 2:
        question = _paraphrase_verbs(question, rng)
    elif variant_idx == 3:
        question = _perturb_threshold(question, rng)
        question = _paraphrase_verbs(question, rng)
    else:
        # For higher indices, combine all
        question = _perturb_threshold(question, rng)
        question = _perturb_proportion(question, rng)
        question = _paraphrase_verbs(question, rng)

    variant["question"] = question
    return variant


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory conversion
# ──────────────────────────────────────────────────────────────────────────────

def task_to_trajectory(task: dict) -> dict:
    """Convert a unified task to evolution trajectory JSONL format.

    SeqPatternMiner expects: tool_sequence, f1, task_type
    """
    # For EarthBench, prefer tools_called (actual execution trace) over expected_tools
    tool_seq = task.get("tools_called") or task.get("expected_tools", [])
    return {
        "task_id": task["task_id"],
        "source": task.get("source", "unknown"),
        "query": task.get("question", ""),
        "tool_sequence": tool_seq,
        "f1": 1.0,  # Expert trajectories are treated as successful
        "reward": 1.0,
        "task_type": task.get("task_type", "general"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(tasks: list[dict]) -> dict:
    """Compute dataset statistics."""
    all_tools: set[str] = set()
    tools_by_source: dict[str, set[str]] = {}
    tools_by_type: dict[str, set[str]] = {}
    tool_freq: Counter = Counter()
    source_counts: Counter = Counter()
    type_counts: Counter = Counter()
    skip_reasons: Counter = Counter()

    for task in tasks:
        source = task.get("source", "unknown")
        task_type = task.get("task_type", "unknown")
        expected = task.get("expected_tools", [])

        source_counts[source] += 1
        type_counts[task_type] += 1
        all_tools.update(expected)
        tools_by_source.setdefault(source, set()).update(expected)
        tools_by_type.setdefault(task_type, set()).update(expected)
        for t in expected:
            tool_freq[t] += 1

    # Group tools by toolkit prefix
    toolkit_groups: dict[str, list[str]] = {}
    for tool in sorted(all_tools):
        prefix = tool.split(".")[0] if "." in tool else "other"
        toolkit_groups.setdefault(prefix, []).append(tool)

    return {
        "total_tasks": len(tasks),
        "unique_tools": len(all_tools),
        "tools_list": sorted(all_tools),
        "toolkit_groups": {k: sorted(v) for k, v in sorted(toolkit_groups.items())},
        "tool_frequency": dict(tool_freq.most_common()),
        "source_counts": dict(source_counts),
        "type_counts": dict(type_counts),
        "tools_by_source": {k: sorted(v) for k, v in tools_by_source.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="合并 OpenEarth + EarthBench 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input paths
    parser.add_argument(
        "--openearth-train",
        default=str(DATA_ROOT / "openearth" / "openearth_train_no_mock_no_online_tasks.json"),
        help="OpenEarth 训练集 task-file 路径",
    )
    parser.add_argument(
        "--openearth-test",
        default=str(DATA_ROOT / "openearth" / "openearth_test_no_mock_no_online_tasks.json"),
        help="OpenEarth 测试集 task-file 路径",
    )
    parser.add_argument(
        "--earthbench-train",
        default=str(DATA_ROOT / "earthbench" / "train.jsonl"),
        help="EarthBench 训练集路径",
    )
    parser.add_argument(
        "--earthbench-eval",
        default=str(DATA_ROOT / "earthbench" / "eval.jsonl"),
        help="EarthBench 评测集路径",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=str(DATA_ROOT / "merged"),
        help="输出目录",
    )

    # Filtering flags (all default to True; use --no-skip-X to disable)
    parser.add_argument("--skip-mock", action="store_true", default=True)
    parser.add_argument("--no-skip-mock", dest="skip_mock", action="store_false")
    parser.add_argument("--skip-bing", action="store_true", default=True)
    parser.add_argument("--no-skip-bing", dest="skip_bing", action="store_false")
    parser.add_argument("--skip-osm", action="store_true", default=True)
    parser.add_argument("--no-skip-osm", dest="skip_osm", action="store_false")
    parser.add_argument("--skip-vlm", action="store_true", default=True)
    parser.add_argument("--no-skip-vlm", dest="skip_vlm", action="store_false")
    parser.add_argument("--skip-changeos", action="store_true", default=True)
    parser.add_argument("--no-skip-changeos", dest="skip_changeos", action="store_false")

    # Augmentation
    parser.add_argument(
        "--augment-factor", type=int, default=3,
        help="每条 EarthBench 任务生成的变体数（0=不扩充）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load OpenEarth ──
    oe_train = load_openearth_tasks(Path(args.openearth_train))
    oe_test = load_openearth_tasks(Path(args.openearth_test))
    log.info(f"OpenEarth loaded: {len(oe_train)} train, {len(oe_test)} test")

    # ── Load EarthBench ──
    eb_train = load_earthbench_jsonl(Path(args.earthbench_train), "train")
    eb_eval = load_earthbench_jsonl(Path(args.earthbench_eval), "eval")
    log.info(f"EarthBench loaded: {len(eb_train)} train, {len(eb_eval)} eval")

    # ── Filter ──
    filter_kwargs = dict(
        skip_mock=args.skip_mock,
        skip_bing=args.skip_bing,
        skip_osm=args.skip_osm,
        skip_vlm=args.skip_vlm,
        skip_changeos=args.skip_changeos,
    )

    skip_stats: Counter = Counter()

    def filter_tasks(tasks: list[dict], label: str) -> list[dict]:
        kept = []
        for task in tasks:
            reason = should_skip_task(task, **filter_kwargs)
            if reason:
                skip_stats[f"{label}_{reason}"] += 1
            else:
                kept.append(task)
        return kept

    oe_train = filter_tasks(oe_train, "oe_train")
    oe_test = filter_tasks(oe_test, "oe_test")
    eb_train = filter_tasks(eb_train, "eb_train")
    eb_eval = filter_tasks(eb_eval, "eb_eval")

    log.info(f"After filtering: OE train={len(oe_train)}, OE test={len(oe_test)}, "
             f"EB train={len(eb_train)}, EB eval={len(eb_eval)}")
    if skip_stats:
        log.info(f"Filtered out: {dict(skip_stats)}")

    # ── Augment EarthBench ──
    eb_train_augmented = list(eb_train)  # Keep originals
    if args.augment_factor > 0:
        for task in eb_train:
            for vi in range(1, args.augment_factor + 1):
                variant = augment_earthbench_task(task, vi, rng)
                eb_train_augmented.append(variant)
        log.info(
            f"EarthBench augmented: {len(eb_train)} → {len(eb_train_augmented)} "
            f"(+{len(eb_train_augmented) - len(eb_train)} variants)"
        )

    # ── Merge ──
    merged_train = oe_train + eb_train_augmented
    merged_eval = oe_test + eb_eval

    # Shuffle train to mix sources
    rng.shuffle(merged_train)

    log.info(f"Merged: {len(merged_train)} train, {len(merged_eval)} eval")

    # ── Compute stats ──
    train_stats = compute_stats(merged_train)
    eval_stats = compute_stats(merged_eval)

    # ── Write outputs ──
    # 1. Task files
    train_out = out_dir / "merged_train_tasks.json"
    with open(train_out, "w") as f:
        json.dump({"tasks": merged_train}, f, ensure_ascii=False, indent=2)
    log.info(f"Written: {train_out} ({len(merged_train)} tasks)")

    eval_out = out_dir / "merged_eval_tasks.json"
    with open(eval_out, "w") as f:
        json.dump({"tasks": merged_eval}, f, ensure_ascii=False, indent=2)
    log.info(f"Written: {eval_out} ({len(merged_eval)} tasks)")

    # 2. Evolution trajectory JSONL
    traj_out = out_dir / "merged_trajectories.jsonl"
    with open(traj_out, "w") as f:
        for task in merged_train:
            traj = task_to_trajectory(task)
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    log.info(f"Written: {traj_out}")

    # 3. Stats
    full_stats = {
        "filter_settings": {
            "skip_mock": args.skip_mock,
            "skip_bing": args.skip_bing,
            "skip_osm": args.skip_osm,
            "skip_vlm": args.skip_vlm,
            "skip_changeos": args.skip_changeos,
            "augment_factor": args.augment_factor,
        },
        "filtered_out": dict(skip_stats),
        "train": train_stats,
        "eval": eval_stats,
    }
    stats_out = out_dir / "stats.json"
    with open(stats_out, "w") as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("  MERGED DATASET SUMMARY")
    print("=" * 70)
    print(f"\n  Train tasks: {len(merged_train)}")
    print(f"    - OpenEarth:  {sum(1 for t in merged_train if t['source'] == 'openearth')}")
    print(f"    - EarthBench: {sum(1 for t in merged_train if t['source'] == 'earthbench')}")
    print(f"  Eval tasks:  {len(merged_eval)}")
    print(f"    - OpenEarth:  {sum(1 for t in merged_eval if t['source'] == 'openearth')}")
    print(f"    - EarthBench: {sum(1 for t in merged_eval if t['source'] == 'earthbench')}")
    print(f"\n  Unique tools (train): {train_stats['unique_tools']}")
    print(f"  Unique tools (eval):  {eval_stats['unique_tools']}")

    print("\n  Tools by toolkit:")
    for toolkit, tools in train_stats["toolkit_groups"].items():
        print(f"    {toolkit}: {len(tools)} tools")
        for t in tools:
            freq = train_stats["tool_frequency"].get(t, 0)
            print(f"      - {t} ({freq}×)")

    print(f"\n  Output files:")
    print(f"    {train_out}")
    print(f"    {eval_out}")
    print(f"    {traj_out}")
    print(f"    {stats_out}")
    print()


if __name__ == "__main__":
    main()
