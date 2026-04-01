#!/usr/bin/env python3
"""
test_sft_flows.py — 验证 SFT 数据集工具调用流程

测试分两层：
  Phase 1 — disaster_response 工具单元测试（真实 mock 栅格，端到端执行）
  Phase 2 — 全量 90 条 SFT 样本静态验证
             * 参数名与源码签名比对
             * $stepN.key 引用解析检查
             * sample_output 缺失引用键检查

运行: /data1/yuhongjie2/env/earth/bin/python scripts/test_sft_flows.py
输出: data/flow_test_report.json  +  终端摘要
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# ── 路径设置 ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET_PATH = REPO_ROOT / "data" / "disaster_sft_dataset.json"
REPORT_PATH  = REPO_ROOT / "data" / "flow_test_report.json"


# ============================================================================
# 源码确认的参数签名（required 字段）
# ============================================================================

REQUIRED_PARAMS: dict[str, list[str]] = {
    # geo_basic  (geobasic.py 实测)
    "geo_basic.pixel_area":   ["pixels", "gsd_m"],              # L387
    "geo_basic.distance":     ["lon1", "lat1", "lon2", "lat2"], # L362 — dataset 用 point_a/point_b 是错误的
    "geo_basic.aoi_validate": ["geojson"],                      # L239 — dataset 用 aoi 是错误的
    "geo_basic.gridify":      ["geojson"],                      # L461 — dataset 用 aoi 是错误的

    # geo_raster  (georaster.py 实测)
    "geo_raster.calculate_index":            ["band_a_path", "band_b_path", "output_path"],
    "geo_raster.threshold_segmentation":     ["input_path", "threshold", "output_path"],
    "geo_raster.raster_stats":               ["input_path"],
    "geo_raster.raster_diff":               ["path_a", "path_b", "output_path"],
    "geo_raster.calculate_frp":             ["input_path", "output_path"],
    "geo_raster.create_fire_increase_map":  ["pre_path", "post_path", "output_path"],
    "geo_raster.calc_snow_loss_stats":      ["binary_map_path"],
    "geo_raster.calculate_fvc":             ["nir_path", "red_path", "output_path"],
    "geo_raster.compute_tvdi":              ["ndvi_path", "lst_path", "output_path"],      # L229
    "geo_raster.hotspot_percentage":        ["input_path", "threshold"],                   # L405
    "geo_raster.apply_cloud_mask":          ["sr_path", "qa_path", "output_path"],        # L436
    "geo_raster.get_percentile_value":      ["input_path", "percentile"],                 # L467
    "geo_raster.identify_fire_prone_areas": ["input_paths", "output_path"],               # L678
    "geo_raster.count_above_threshold":     ["input_path", "threshold"],

    # geo_statistics  (geo_statistics.py 实测)
    "geo_statistics.count_pixels_condition": ["input_path"],   # lower/upper 可选
    "geo_statistics.percentage_change":      ["a", "b"],
    "geo_statistics.threshold_ratio":        ["input_path", "threshold"],
    "geo_statistics.batch_raster_stats":     ["input_paths"],  # L56

    # geoanalysis  (geoanalysis.py 实测)
    "geoanalysis.getis_ord_gi_star":    ["image_path", "output_path", "weight_matrix"],
    "geoanalysis.mann_kendall_test":    ["values"],
    "geoanalysis.detect_change_points": ["values"],

    # earth_sci  (earth_sci.py 实测)
    "earth_sci.calculate_lst_sc":  ["bt_path", "red_path", "nir_path", "output_path"],  # L138
    "earth_sci.stats_lst_ndvi":    ["red_path", "nir_path", "lst_path"],                # L322 (stats_lst_by_ndvi_handler)
    "earth_sci.calculate_ati":     ["day_temp_path", "night_temp_path", "albedo_path", "output_path"],  # L348
    "earth_sci.microwave_dpdm":    ["pol1_path", "pol2_path", "output_path"],           # L375

    # osm_gis  (osm_gis.py 实测)
    # get_area_boundary: place_name 或 bbox 二选一，均非严格 required
    "osm_gis.get_area_boundary":  [],
    # add_pois_layer: place_name 或 bbox 二选一；dataset 用 boundary 是错误的
    "osm_gis.add_pois_layer":     [],
    "osm_gis.compute_route_dist": ["origin", "destination"],

    # geo_perception  (geo_perception.py 实测)
    "geo_perception.vlm_analyze":                ["image_paths", "prompt"],
    "geo_perception.remoteclip_analysis":        ["image_path", "text_queries"],
    "geo_perception.instructsam":                ["image_path", "text"],
    "geo_perception.sam2_segment":               ["image_path"],
    "geo_perception.strip_rcnn_detect":          ["image_path"],
    "geo_perception.remotesam":                  ["image_path"],
    "geo_perception.draw_bboxes":                ["image_path", "bboxes", "output_path"],
    "geo_perception.add_text":                   ["image_path", "text", "position", "output_path"],
    "geo_perception.ocr_extract":                ["image_path"],
    "geo_perception.bbox_area":                  ["bboxes"],
    "geo_perception.bbox_to_centroid":           ["bboxes"],
    "geo_perception.centroid_distance_extremes": ["centroids"],
    "geo_perception.bbox_expand":                ["bbox", "scale"],

    # disaster_response  (disaster_response.py 实测)
    "disaster_response.calc_slope":            ["dem_path", "output_path"],
    "disaster_response.calc_flow_direction":   ["dem_path", "output_path"],
    "disaster_response.zonal_stats":           ["raster_path", "zones_geojson"],
    "disaster_response.population_exposure":   ["hazard_path", "population_path", "output_path"],
    "disaster_response.flood_route_astar":     ["origin", "destination"],
    "disaster_response.accessibility_map":     ["road_mask_path", "poi_locations", "output_path"],
    "disaster_response.building_damage_stats": ["change_raster_path", "buildings_geojson"],
    "disaster_response.fire_spread_forecast":  ["fvc_path", "ignition_points", "output_path"],
}

# 已知的错误参数名 → 正确参数名
WRONG_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "geo_basic.distance": {
        "point_a": "lon1/lat1",
        "point_b": "lon2/lat2",
    },
    "geo_basic.aoi_validate": {
        "aoi": "geojson",
    },
    "geo_basic.gridify": {
        "aoi": "geojson",
        "cell_size_km": "cell_width_m (metres) / mode",
    },
    "osm_gis.add_pois_layer": {
        "boundary": "place_name (str) 或 bbox ([w,s,e,n])",
    },
}

# aoi_validate 实际返回键（与 dataset sample_output 不符）
ACTUAL_RETURNS: dict[str, list[str]] = {
    "geo_basic.aoi_validate": [
        "aoi_feature_collection", "geometry_type", "bbox",
        "total_area_m2", "total_area_km2", "total_perimeter_m", "approximate",
    ],
    # dataset 中 sample_output 用了 "aoi" / "valid" / "area_km2" 均不存在于真实返回
}


# ============================================================================
# 工具函数
# ============================================================================

def resolve_ref(val: Any, step_outputs: dict[int, dict]) -> Any:
    """将 "$stepN.key" 字符串替换为上步输出的实际值（仅做路径解析，不做深层 GeoJSON 遍历）。"""
    if not isinstance(val, str):
        return val
    m = re.fullmatch(r"\$step(\d+)\.(.+)", val)
    if not m:
        return val
    step_no = int(m.group(1))
    key_path = m.group(2).split(".")
    src = step_outputs.get(step_no, {})
    for k in key_path:
        if isinstance(src, dict):
            src = src.get(k)
        else:
            return None  # path not found
    return src


def check_step_refs(args: dict, step_outputs: dict[int, dict]) -> list[str]:
    """检查所有 $step 引用是否能在对应步骤的 sample_output 中找到。"""
    errors = []
    for key, val in args.items():
        if not isinstance(val, str):
            continue
        m = re.fullmatch(r"\$step(\d+)\.(.+)", val)
        if not m:
            continue
        step_no = int(m.group(1))
        ref_key = m.group(2).split(".")[0]
        src = step_outputs.get(step_no, {})
        if ref_key not in src:
            errors.append(
                f"参数 '{key}' 引用 {val}，但 step{step_no}.sample_output 中无键 '{ref_key}'"
                f"（可用键: {list(src.keys())}）"
            )
    return errors


def validate_required_params(tool: str, args: dict) -> list[str]:
    """检查必填参数是否存在，并检测错误别名。"""
    errors = []
    required = REQUIRED_PARAMS.get(tool)
    if required is None:
        return [f"未知工具 '{tool}'，无法验证参数"]

    for r in required:
        if r not in args:
            errors.append(f"缺少必填参数 '{r}'（工具: {tool}）")

    aliases = WRONG_PARAM_ALIASES.get(tool, {})
    for wrong, correct in aliases.items():
        if wrong in args:
            errors.append(
                f"使用了错误参数名 '{wrong}'，应改为 '{correct}'（工具: {tool}）"
            )
    return errors


def validate_return_keys(tool: str, sample_output: dict, downstream_refs: list[str]) -> list[str]:
    """检查下游步骤引用的键是否存在于本步 sample_output 中。"""
    errors = []
    actual_keys = ACTUAL_RETURNS.get(tool)
    if actual_keys is not None:
        for key in sample_output:
            if key not in actual_keys:
                errors.append(
                    f"sample_output 包含键 '{key}'，但 {tool} 实际不返回此键"
                    f"（实际返回: {actual_keys}）"
                )
    return errors


# ============================================================================
# Phase 2 — 静态流程验证
# ============================================================================

def validate_all_flows(dataset: dict) -> dict:
    """验证全部 SFT 样本的工具调用流程。"""
    results = {}
    total_errors = 0
    failed_tasks = []

    for sample in dataset["samples"]:
        sid       = sample["id"]
        task_type = sample["task_type"]
        errors_in_sample: list[dict] = []

        step_outputs: dict[int, dict] = {}

        for step_def in sample.get("tool_calls", []):
            step_no  = step_def["step"]
            tool     = step_def["tool"]
            args     = step_def.get("args", {})
            s_output = step_def.get("sample_output", {})

            step_errors = []

            # 1. 必填参数 + 错误别名
            step_errors += validate_required_params(tool, args)

            # 2. $step 引用解析
            step_errors += check_step_refs(args, step_outputs)

            # 3. sample_output 键与实际返回比对
            step_errors += validate_return_keys(tool, s_output, [])

            if step_errors:
                errors_in_sample.append({
                    "step": step_no,
                    "tool": tool,
                    "errors": step_errors,
                })

            step_outputs[step_no] = s_output

        sample_result = {
            "id": sid,
            "task_type": task_type,
            "passed": len(errors_in_sample) == 0,
            "error_count": sum(len(e["errors"]) for e in errors_in_sample),
            "errors": errors_in_sample,
        }
        results[sid] = sample_result
        total_errors += sample_result["error_count"]
        if not sample_result["passed"]:
            failed_tasks.append(sid)

    return {
        "phase": "static_validation",
        "total_samples": len(results),
        "passed": len(results) - len(failed_tasks),
        "failed": len(failed_tasks),
        "total_errors": total_errors,
        "failed_sample_ids": failed_tasks,
        "details": results,
    }


# ============================================================================
# Phase 1 — disaster_response 工具单元测试
# ============================================================================

def create_mock_rasters(tmpdir: str) -> dict[str, str]:
    """用 numpy+rasterio 创建 50×50 小型测试栅格。"""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    west, south, east, north = 113.0, 23.0, 113.01, 23.01
    transform = from_bounds(west, south, east, north, 50, 50)
    crs = CRS.from_epsg(4326)

    base_profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": 50, "height": 50, "count": 1,
        "crs": crs, "transform": transform,
    }

    np.random.seed(42)
    x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    paths = {}

    # DEM: 斜坡 100-700m
    dem = (100 + 400 * x + 200 * y + 30 * np.random.randn(50, 50)).astype(np.float32)
    paths["dem"] = os.path.join(tmpdir, "dem.tif")
    with rasterio.open(paths["dem"], "w", **base_profile) as dst:
        dst.write(dem, 1)

    # 洪水掩膜 0/255
    flood = np.zeros((50, 50), dtype=np.uint8)
    flood[10:25, 15:35] = 255
    paths["flood"] = os.path.join(tmpdir, "flood_mask.tif")
    fp = {**base_profile, "dtype": "uint8"}
    with rasterio.open(paths["flood"], "w", **fp) as dst:
        dst.write(flood, 1)

    # FVC 0-1
    fvc = np.clip(np.random.rand(50, 50), 0.1, 1).astype(np.float32)
    paths["fvc"] = os.path.join(tmpdir, "fvc.tif")
    with rasterio.open(paths["fvc"], "w", **base_profile) as dst:
        dst.write(fvc, 1)

    # 人口密度
    pop = (np.random.rand(50, 50) * 1000).astype(np.float32)
    paths["population"] = os.path.join(tmpdir, "population.tif")
    with rasterio.open(paths["population"], "w", **base_profile) as dst:
        dst.write(pop, 1)

    # 道路掩膜 (1=可通行, 0=封堵)
    road = np.ones((50, 50), dtype=np.uint8)
    road[10:25, 15:35] = 0
    paths["road"] = os.path.join(tmpdir, "road_mask.tif")
    rp = {**base_profile, "dtype": "uint8"}
    with rasterio.open(paths["road"], "w", **rp) as dst:
        dst.write(road, 1)

    # 变化掩膜 0/255
    change = np.zeros((50, 50), dtype=np.uint8)
    change[20:35, 20:35] = 255
    paths["change"] = os.path.join(tmpdir, "change_mask.tif")
    with rasterio.open(paths["change"], "w", **rp) as dst:
        dst.write(change, 1)

    return paths, transform


def create_mock_geojson(tmpdir: str, west=113.0, south=23.0, east=113.01, north=23.01) -> dict[str, str]:
    """创建简单的建筑物和行政区 GeoJSON 测试文件。"""
    mid_lon = (west + east) / 2
    mid_lat = (south + north) / 2
    q = (east - west) * 0.25

    buildings = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1, "name": "Building A"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[west + q,     south + q],
                     [mid_lon,      south + q],
                     [mid_lon,      mid_lat],
                     [west + q,     mid_lat],
                     [west + q,     south + q]]
                ]},
            },
            {
                "type": "Feature",
                "properties": {"id": 2, "name": "Building B"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[mid_lon + q,  south + q],
                     [east - q,     south + q],
                     [east - q,     mid_lat],
                     [mid_lon + q,  mid_lat],
                     [mid_lon + q,  south + q]]
                ]},
            },
        ],
    }
    buildings_path = os.path.join(tmpdir, "buildings.geojson")
    with open(buildings_path, "w") as f:
        json.dump(buildings, f)

    districts = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1, "name": "A区"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[west, south], [mid_lon, south], [mid_lon, north], [west, north], [west, south]]
                ]},
            },
            {
                "type": "Feature",
                "properties": {"id": 2, "name": "B区"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[mid_lon, south], [east, south], [east, north], [mid_lon, north], [mid_lon, south]]
                ]},
            },
        ],
    }
    districts_path = os.path.join(tmpdir, "districts.geojson")
    with open(districts_path, "w") as f:
        json.dump(districts, f)

    return {"buildings": buildings_path, "districts": districts_path}


def run_unit_tests() -> dict:
    """Phase 1: 对 disaster_response 工具做端到端单元测试。"""
    from terrabox.toolkits.disaster_response import (
        calc_slope_handler,
        calc_flow_direction_handler,
        zonal_stats_handler,
        population_exposure_handler,
        accessibility_map_handler,
        building_damage_stats_handler,
        fire_spread_forecast_handler,
    )

    results = {}

    with tempfile.TemporaryDirectory(prefix="dr_test_") as tmpdir:
        paths, transform = create_mock_rasters(tmpdir)
        geo = create_mock_geojson(tmpdir)

        # ── T1 calc_slope ─────────────────────────────────────────────────
        def _test_calc_slope():
            out = os.path.join(tmpdir, "slope.tif")
            r = calc_slope_handler(
                {"dem_path": paths["dem"], "output_path": out, "unit": "degrees"},
                {}, None
            )
            assert "output_path" in r, "缺少 output_path"
            assert "mean_slope" in r, "缺少 mean_slope"
            assert os.path.exists(r["output_path"]), "输出文件未生成"
            return r

        # ── T2 calc_flow_direction ────────────────────────────────────────
        def _test_calc_flow_direction():
            out = os.path.join(tmpdir, "flow_dir.tif")
            r = calc_flow_direction_handler(
                {"dem_path": paths["dem"], "output_path": out},
                {}, None
            )
            assert "output_path" in r, "缺少 output_path"
            assert "d8_codes" in r, "缺少 d8_codes"
            assert os.path.exists(r["output_path"]), "输出文件未生成"
            return r

        # ── T3 zonal_stats ────────────────────────────────────────────────
        def _test_zonal_stats():
            r = zonal_stats_handler(
                {"raster_path": paths["flood"], "zones_geojson": geo["districts"], "stat": "mean"},
                {}, None
            )
            assert "zones" in r, "缺少 zones"
            assert r["count"] == 2, f"期望 2 个区，得到 {r['count']}"
            return r

        # ── T4 population_exposure ────────────────────────────────────────
        def _test_population_exposure():
            out = os.path.join(tmpdir, "exposed_pop.tif")
            r = population_exposure_handler(
                {
                    "hazard_path":     paths["flood"],
                    "population_path": paths["population"],
                    "output_path":     out,
                    "threshold":       127,
                },
                {}, None
            )
            assert "exposed_population" in r, "缺少 exposed_population"
            assert "exposed_pixels" in r, "缺少 exposed_pixels"
            assert os.path.exists(r["output_path"]), "输出文件未生成"
            return r

        # ── T5 accessibility_map ─────────────────────────────────────────
        def _test_accessibility_map():
            out = os.path.join(tmpdir, "access_time.tif")
            r = accessibility_map_handler(
                {
                    "road_mask_path":  paths["road"],
                    "poi_locations":   [{"lon": 113.005, "lat": 23.002, "name": "Hospital"}],
                    "output_path":     out,
                    "travel_speed_kmh": 30,
                },
                {}, None
            )
            assert "output_path" in r, "缺少 output_path"
            assert os.path.exists(r["output_path"]), "输出文件未生成"
            return r

        # ── T6 building_damage_stats ─────────────────────────────────────
        def _test_building_damage_stats():
            r = building_damage_stats_handler(
                {
                    "change_raster_path": paths["change"],
                    "buildings_geojson":  geo["buildings"],
                    "damage_threshold":   0.3,
                },
                {}, None
            )
            assert "total_buildings" in r, "缺少 total_buildings"
            assert "damaged" in r, "缺少 damaged"
            assert r["total_buildings"] == 2, f"期望 2 栋，得到 {r['total_buildings']}"
            return r

        # ── T7 fire_spread_forecast ──────────────────────────────────────
        def _test_fire_spread_forecast():
            out = os.path.join(tmpdir, "fire_forecast.tif")
            r = fire_spread_forecast_handler(
                {
                    "fvc_path":          paths["fvc"],
                    "ignition_points":   [{"lon": 113.005, "lat": 23.005}],
                    "wind_direction_deg": 135,
                    "wind_speed_ms":     5,
                    "time_steps":        3,
                    "output_path":       out,
                },
                {}, None
            )
            assert "output_path" in r, "缺少 output_path"
            assert "affected_area_km2" in r, "缺少 affected_area_km2"
            assert os.path.exists(r["output_path"]), "输出文件未生成"
            return r

        # ── T8 flood_route_astar (需联网，跳过) ──────────────────────────
        def _test_flood_route_astar_skipped():
            return {"skipped": True, "reason": "需要 OSM 网络下载，跳过单元测试"}

        unit_tests = [
            ("calc_slope",            _test_calc_slope),
            ("calc_flow_direction",   _test_calc_flow_direction),
            ("zonal_stats",           _test_zonal_stats),
            ("population_exposure",   _test_population_exposure),
            ("accessibility_map",     _test_accessibility_map),
            ("building_damage_stats", _test_building_damage_stats),
            ("fire_spread_forecast",  _test_fire_spread_forecast),
            ("flood_route_astar",     _test_flood_route_astar_skipped),
        ]

        for name, fn in unit_tests:
            try:
                ret = fn()
                results[name] = {"passed": True, "output": ret}
            except Exception as e:
                results[name] = {
                    "passed": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

    passed = sum(1 for v in results.values() if v.get("passed"))
    skipped = sum(1 for v in results.values() if v.get("skipped"))

    return {
        "phase": "unit_tests",
        "total": len(results),
        "passed": passed,
        "skipped": skipped,
        "failed": len(results) - passed - skipped,
        "details": results,
    }


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 60)
    print("SFT 流程测试")
    print("=" * 60)

    # ── Phase 1: 单元测试 ─────────────────────────────────────────────────
    print("\n[Phase 1] disaster_response 工具单元测试 ...")
    unit_report = run_unit_tests()
    print(f"  单元测试: {unit_report['passed']}/{unit_report['total']} 通过"
          f"  ({unit_report['skipped']} 跳过, {unit_report['failed']} 失败)")
    for name, res in unit_report["details"].items():
        status = "SKIP" if res.get("skipped") else ("PASS" if res["passed"] else "FAIL")
        msg = ""
        if not res["passed"] and not res.get("skipped"):
            msg = f" — {res['error']}"
        print(f"    [{status}] disaster_response.{name}{msg}")

    # ── Phase 2: 静态流程验证 ────────────────────────────────────────────
    print("\n[Phase 2] SFT 数据集全量流程静态验证 ...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    flow_report = validate_all_flows(dataset)
    print(f"  样本总数: {flow_report['total_samples']}")
    print(f"  通过: {flow_report['passed']}  失败: {flow_report['failed']}")
    print(f"  错误总数: {flow_report['total_errors']}")

    if flow_report["failed"]:
        print("\n  失败任务详情:")
        for sid in flow_report["failed_sample_ids"]:
            r = flow_report["details"][sid]
            print(f"\n  ── {sid} (task={r['task_type']}) ──")
            for step_err in r["errors"]:
                print(f"     step {step_err['step']} [{step_err['tool']}]:")
                for e in step_err["errors"]:
                    print(f"       ✗ {e}")

    # ── 汇总报告写入 ──────────────────────────────────────────────────────
    report = {
        "summary": {
            "unit_tests":   {k: v for k, v in unit_report.items() if k != "details"},
            "flow_validation": {k: v for k, v in flow_report.items() if k not in ("details",)},
        },
        "unit_tests":    unit_report["details"],
        "flow_validation": flow_report["details"],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {REPORT_PATH}")

    # ── 退出码 ────────────────────────────────────────────────────────────
    exit_code = 0
    if unit_report["failed"] > 0:
        print("\n[!] Phase 1 有单元测试失败")
        exit_code = 1
    if flow_report["failed"] > 0:
        print(f"[!] Phase 2 有 {flow_report['failed']} 条样本存在流程错误")
        exit_code = 1
    if exit_code == 0:
        print("\n所有测试通过。")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
