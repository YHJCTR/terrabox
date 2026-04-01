"""
生成灾害救援遥感分析 SFT 数据集。
参数签名均经过源码核对，修复了原设计中的 5 处错误。

运行: python scripts/generate_disaster_sft.py
输出: data/disaster_sft_dataset.json
"""

import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 已验证的工具签名（源码核对结果）
# --------------------------------------------------------------------------- #
VERIFIED_SIGNATURES = {
    "geo_basic.pixel_area": {
        "required": ["pixels", "gsd_m"],
        "returns": ["area_m2", "area_km2", "pixel_area_m2"],
    },
    "geo_statistics.count_pixels_condition": {
        "required": ["input_path"],
        "optional": ["lower", "upper"],
        "returns": ["count", "total", "ratio"],
    },
    "geo_statistics.percentage_change": {
        "required": ["a", "b"],
        "returns": ["percentage_change"],
    },
    "geoanalysis.getis_ord_gi_star": {
        "required": ["image_path", "output_path", "weight_matrix"],
        "returns": ["status", "output_path"],
    },
    "geo_raster.calc_snow_loss_stats": {
        "required": ["binary_map_path"],          # 单张二值图，非 pre/post
        "returns": ["percentage", "total_pixels", "loss_pixels"],
    },
    "geo_raster.calculate_index": {
        "required": ["band_a_path", "band_b_path", "output_path"],
        "returns": ["status", "output_path"],     # 无统计数值
    },
    "geo_raster.calculate_frp": {
        "required": ["input_path", "output_path"],
        "returns": ["output_path", "fire_pixels", "fire_coverage_pct"],  # key 为 fire_pixels
    },
    "geo_raster.create_fire_increase_map": {
        "required": ["pre_path", "post_path", "output_path"],
        "returns": ["output_path", "new_fire_pixels"],
    },
    "geo_raster.threshold_segmentation": {
        "note": "pixels > threshold → 255, others → 0",
        "required": ["input_path", "threshold", "output_path"],
        "returns": ["output_path"],
    },
    "geo_perception.bbox_area": {
        "required": ["bboxes"],
        "optional": ["gsd_m"],
        "returns": ["total_area_px2", "per_bbox_area_px2", "count"],
    },
    "geo_perception.bbox_to_centroid": {
        "required": ["bboxes"],
        "returns": ["centroids", "count"],  # centroids: [{x, y}, ...]
    },
    "geo_perception.centroid_distance_extremes": {
        "required": ["centroids"],
        "returns": ["min_distance", "max_distance", "min_pair_indices", "max_pair_indices"],
    },
}

# --------------------------------------------------------------------------- #
# 修复记录（相比原设计）
# --------------------------------------------------------------------------- #
FIXES = [
    {
        "tool": "geo_basic.pixel_area",
        "wrong_params": ["pixel_count", "resolution_m"],
        "correct_params": ["pixels", "gsd_m"],
        "source_line": "geobasic.py:387",
    },
    {
        "tool": "geo_statistics.count_pixels_condition",
        "wrong_params": ["raster_path", "condition: '==255' (字符串)"],
        "correct_params": ["input_path", "lower: 254 (数值下界)"],
        "source_line": "geo_statistics.py:248",
    },
    {
        "tool": "geo_statistics.percentage_change",
        "wrong_params": ["before", "after"],
        "correct_params": ["a", "b"],
        "source_line": "geo_statistics.py:343",
    },
    {
        "tool": "geoanalysis.getis_ord_gi_star",
        "wrong_params": ["values (列表)", "coords (坐标列表)"],
        "correct_params": ["image_path (栅格路径)", "output_path", "weight_matrix (空间权重矩阵)"],
        "source_line": "geoanalysis.py:662",
    },
    {
        "tool": "geo_raster.calc_snow_loss_stats",
        "wrong_params": ["pre_path", "post_path"],
        "correct_params": ["binary_map_path (单张二值损失图，需先用 raster_diff 生成)"],
        "source_line": "georaster.py:858",
    },
]

# --------------------------------------------------------------------------- #
# 权重矩阵（常用）
# --------------------------------------------------------------------------- #
W3x3 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]


# --------------------------------------------------------------------------- #
# 工具调用序列定义（每个 task_type 一个）
# --------------------------------------------------------------------------- #

def make_tool(step: int, tool: str, args: dict, sample_output: dict, note: str = ""):
    entry = {"step": step, "tool": tool, "args": args, "sample_output": sample_output}
    if note:
        entry["note"] = note
    return entry


TASK_TEMPLATES = {

    # ------------------------------------------------------------------ #
    # 1. 洪水淹没范围检测
    # ------------------------------------------------------------------ #
    "flood_detection": {
        "disaster_category": "flood",
        "difficulty": "easy",
        "prompts": [
            "我有一张洪灾发生后的多光谱卫星影像，请提取淹没水体区域，计算淹没总面积（km²），并标注淹没范围。",
            "对洪灾区域遥感影像进行水体提取，识别所有被洪水覆盖的区域并统计淹没面积。",
            "利用多光谱卫星数据检测洪水淹没范围，计算受灾面积并生成带有标注的可视化图。",
            "请分析这张洪涝灾区遥感影像，找出所有积水区域，给出淹没总面积及空间分布。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"},
                      "NDWI = (Green-NIR)/(Green+NIR)，水体NDWI>0"),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi.tif", "threshold": 0.1, "output_path": "flood_mask.tif"},
                      {"output_path": "flood_mask.tif"},
                      "像素值>0.1标记为255，其余为0"),
            make_tool(3, "geo_statistics.count_pixels_condition",
                      {"input_path": "flood_mask.tif", "lower": 254},
                      {"count": 45820, "total": 1000000, "ratio": 0.04582},
                      "lower=254 等效于 ==255，容忍浮点舍入"),
            make_tool(4, "geo_basic.pixel_area",
                      {"pixels": "$step3.count", "gsd_m": 10},
                      {"area_m2": 4582000.0, "area_km2": 4.582, "pixel_area_m2": 100.0}),
            make_tool(5, "geo_perception.vlm_analyze",
                      {"image_paths": ["rgb.tif"],
                       "prompt": "请识别并标注图中被洪水淹没的区域，描述积水范围和严重程度"},
                      {"analysis": "图中大面积蓝色区域为积水...", "bboxes": [
                          {"x1": 120, "y1": 80, "x2": 340, "y2": 210}]}),
            make_tool(6, "geo_perception.draw_bboxes",
                      {"image_path": "rgb.tif", "bboxes": "$step5.bboxes",
                       "output_path": "flood_annotated.png"},
                      {"output_path": "flood_annotated.png"}),
        ],
    },

    # ------------------------------------------------------------------ #
    # 2. 洪水前后变化对比
    # ------------------------------------------------------------------ #
    "flood_change": {
        "disaster_category": "flood",
        "difficulty": "medium",
        "prompts": [
            "给定洪水前后两期遥感影像，计算新增淹没区域面积及扩张比例。",
            "请对比灾前和灾后卫星图像，量化洪水淹没面积扩张了多少，给出百分比变化。",
            "利用洪水发生前后的卫星影像，分析洪涝灾害导致的水体面积变化，计算新增淹没面积。",
            "通过前后期遥感对比，评估本次洪水事件导致的水体扩张程度。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green_pre.tif", "band_b_path": "nir_pre.tif",
                       "index_name": "NDWI", "output_path": "ndwi_pre.tif"},
                      {"status": "success", "output_path": "ndwi_pre.tif"}, "灾前NDWI"),
            make_tool(2, "geo_raster.calculate_index",
                      {"band_a_path": "green_post.tif", "band_b_path": "nir_post.tif",
                       "index_name": "NDWI", "output_path": "ndwi_post.tif"},
                      {"status": "success", "output_path": "ndwi_post.tif"}, "灾后NDWI"),
            make_tool(3, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi_pre.tif", "threshold": 0.1, "output_path": "water_pre.tif"},
                      {"output_path": "water_pre.tif"}, "灾前水体二值掩膜"),
            make_tool(4, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi_post.tif", "threshold": 0.1, "output_path": "water_post.tif"},
                      {"output_path": "water_post.tif"}, "灾后水体二值掩膜"),
            make_tool(5, "geo_statistics.count_pixels_condition",
                      {"input_path": "water_pre.tif", "lower": 254},
                      {"count": 38200, "total": 1000000, "ratio": 0.0382}, "灾前水体像素数"),
            make_tool(6, "geo_statistics.count_pixels_condition",
                      {"input_path": "water_post.tif", "lower": 254},
                      {"count": 79350, "total": 1000000, "ratio": 0.0794}, "灾后水体像素数"),
            make_tool(7, "geo_raster.raster_diff",
                      {"path_a": "water_post.tif", "path_b": "water_pre.tif",
                       "output_path": "flood_increase.tif"},
                      {"output_path": "flood_increase.tif"}, "差值>0表示新增水体"),
            make_tool(8, "geo_statistics.count_pixels_condition",
                      {"input_path": "flood_increase.tif", "lower": 127},
                      {"count": 41150, "total": 1000000, "ratio": 0.0412}, "新增淹没像素"),
            make_tool(9, "geo_basic.pixel_area",
                      {"pixels": "$step8.count", "gsd_m": 10},
                      {"area_m2": 4115000.0, "area_km2": 4.115}),
            make_tool(10, "geo_statistics.percentage_change",
                      {"a": "$step5.count", "b": "$step6.count"},
                      {"percentage_change": 107.7},
                      "a=before, b=after；水体面积增加107.7%"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 3. 洪水时序趋势分析
    # ------------------------------------------------------------------ #
    "flood_timeseries": {
        "disaster_category": "flood",
        "difficulty": "hard",
        "prompts": [
            "利用过去30天的日度NDWI影像序列，分析该区域洪水水体面积变化趋势，找出突变时间点。",
            "基于多时相卫星数据，追踪洪水蔓延的时间变化规律，判断洪峰出现时间。",
            "对一个月内的卫星影像进行时序分析，研究洪涝灾害水体面积的演变过程并检测趋势转折。",
            "请通过时间序列遥感分析，追踪洪水从发生到消退的动态过程，给出关键时间节点。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green_d1.tif", "band_b_path": "nir_d1.tif",
                       "index_name": "NDWI", "output_path": "ndwi_d1.tif"},
                      {"status": "success", "output_path": "ndwi_d1.tif"},
                      "循环执行 N 期，此处示意第1期；实际需对每期影像重复步骤1-3"),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi_d1.tif", "threshold": 0.1, "output_path": "water_d1.tif"},
                      {"output_path": "water_d1.tif"}),
            make_tool(3, "geo_statistics.count_pixels_condition",
                      {"input_path": "water_d1.tif", "lower": 254},
                      {"count": 32100, "total": 1000000, "ratio": 0.0321},
                      "收集每期水体像素数，组成时间序列列表"),
            make_tool(4, "geoanalysis.mann_kendall_test",
                      {"values": [32100, 41200, 56800, 73400, 79350, 81200, 78900, 70100, 52300, 41000]},
                      {"trend": "increasing", "p_value": 0.003, "tau": 0.71, "slope": 4823.0},
                      "values 为各期水体像素数组成的时间序列"),
            make_tool(5, "geoanalysis.detect_change_points",
                      {"values": [32100, 41200, 56800, 73400, 79350, 81200, 78900, 70100, 52300, 41000]},
                      {"change_points": [3, 7]},
                      "返回突变点的索引，3=第4天（洪峰前）、7=第8天（消退开始）"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 4. 森林火灾检测
    # ------------------------------------------------------------------ #
    "fire_detection": {
        "disaster_category": "wildfire",
        "difficulty": "easy",
        "prompts": [
            "分析一幅MODIS卫星影像，检测活跃火点，统计火灾覆盖面积并定位最高火势区域。",
            "请识别卫星热红外影像中的森林火灾活跃火点，计算过火面积和火势强度分布。",
            "利用遥感影像对林区进行火情实时监测，统计当前活跃火点数量和燃烧面积。",
            "基于FRP数据检测当前林区火灾状况，识别火势最强区域，评估过火面积。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.apply_cloud_mask",
                      {"sr_path": "fire_mask.tif", "qa_path": "cloud_qa.tif",
                       "output_path": "frp_clean.tif"},
                      {"output_path": "frp_clean.tif", "masked_pixels": 12340},
                      "sr_path为FRP影像，qa_path为云掩膜；去除云污染像素避免虚假火点"),
            make_tool(2, "geo_raster.calculate_frp",
                      {"input_path": "frp_clean.tif", "output_path": "fire_detected.tif", "threshold": 10},
                      {"output_path": "fire_detected.tif", "fire_pixels": 1243, "fire_coverage_pct": 4.12},
                      "FRP>10 MW 认定为活跃火点；返回键为 fire_pixels（非 fire_pixel_count）"),
            make_tool(3, "geo_basic.pixel_area",
                      {"pixels": "$step2.fire_pixels", "gsd_m": 500},
                      {"area_m2": 310750000.0, "area_km2": 310.75},
                      "MODIS FRP 像元分辨率 500m"),
            make_tool(4, "geo_raster.get_percentile_value",
                      {"input_path": "frp_clean.tif", "percentile": 90},
                      {"percentile": 90, "value": 85.6},
                      "找出最高10%火点的FRP阈值"),
            make_tool(5, "geo_raster.hotspot_percentage",
                      {"input_path": "fire_detected.tif", "threshold": 127},
                      {"percentage": 4.12, "count_above": 1243, "total_valid": 30195},
                      "threshold=127 对应二值掩膜的255像素"),
            make_tool(6, "geo_perception.vlm_analyze",
                      {"image_paths": ["rgb_fire.tif"],
                       "prompt": "识别图中火点位置，描述火势范围、蔓延方向和受影响植被类型"},
                      {"analysis": "图像东北方向可见大范围红橙色火点...",
                       "bboxes": [{"x1": 200, "y1": 50, "x2": 450, "y2": 280}]}),
        ],
    },

    # ------------------------------------------------------------------ #
    # 5. 火灾蔓延扩散分析
    # ------------------------------------------------------------------ #
    "fire_spread": {
        "disaster_category": "wildfire",
        "difficulty": "medium",
        "prompts": [
            "给定火灾发生前后5天的FRP影像，分析新增燃烧区域及火势蔓延方向。",
            "对比两期热红外遥感影像，量化火灾扩散范围，评估新增过火面积和方向。",
            "分析山火事件前后的卫星FRP数据，识别火线推进方向和新燃烧区域面积。",
            "通过前后期FRP对比，确定山火扩散范围，分析过火面积增量并判断蔓延趋势。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_frp",
                      {"input_path": "frp_pre.tif", "output_path": "fire_pre.tif", "threshold": 10},
                      {"output_path": "fire_pre.tif", "fire_pixels": 890, "fire_coverage_pct": 2.95}),
            make_tool(2, "geo_raster.calculate_frp",
                      {"input_path": "frp_post.tif", "output_path": "fire_post.tif", "threshold": 10},
                      {"output_path": "fire_post.tif", "fire_pixels": 1422, "fire_coverage_pct": 4.71}),
            make_tool(3, "geo_raster.create_fire_increase_map",
                      {"pre_path": "fire_pre.tif", "post_path": "fire_post.tif",
                       "output_path": "fire_spread.tif"},
                      {"output_path": "fire_spread.tif", "new_fire_pixels": 532},
                      "新增燃烧像素；返回键为 new_fire_pixels"),
            make_tool(4, "geo_basic.pixel_area",
                      {"pixels": "$step3.new_fire_pixels", "gsd_m": 500},
                      {"area_m2": 133000000.0, "area_km2": 133.0}),
            make_tool(5, "geo_perception.sam2_segment",
                      {"image_path": "rgb_post.tif", "prompt": "fire burned area and smoke"},
                      {"bboxes": [
                          {"x1": 150, "y1": 80, "x2": 420, "y2": 310},
                          {"x1": 380, "y1": 60, "x2": 600, "y2": 250}]}),
            make_tool(6, "geo_perception.bbox_to_centroid",
                      {"bboxes": "$step5.bboxes"},
                      {"centroids": [{"x": 285.0, "y": 195.0}, {"x": 490.0, "y": 155.0}],
                       "count": 2}),
            make_tool(7, "geo_perception.centroid_distance_extremes",
                      {"centroids": "$step6.centroids"},
                      {"min_distance": 212.6, "max_distance": 212.6,
                       "min_pair_indices": [0, 1], "max_pair_indices": [0, 1]},
                      "像素距离，乘以GSD得物理距离"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 6. 高火险区域识别
    # ------------------------------------------------------------------ #
    "fire_risk_zones": {
        "disaster_category": "wildfire",
        "difficulty": "hard",
        "prompts": [
            "基于过去一年的月度FRP时间序列影像，识别长期高火险区域，辅助消防预防部署。",
            "分析多年历史卫星数据，找出反复发生火灾的高风险区域，生成火险分级图。",
            "利用历史火灾遥感记录，评估各区域火灾风险等级，为防火减灾决策提供支持。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.identify_fire_prone_areas",
                      {"input_paths": ["frp_jan.tif", "frp_feb.tif", "frp_mar.tif",
                                        "frp_apr.tif", "frp_may.tif", "frp_jun.tif",
                                        "frp_jul.tif", "frp_aug.tif", "frp_sep.tif",
                                        "frp_oct.tif", "frp_nov.tif", "frp_dec.tif"],
                       "output_path": "high_risk.tif", "percentile": 90},
                      {"output_path": "high_risk.tif"},
                      "top 10%高FRP区域标记为255"),
            make_tool(2, "geo_raster.raster_stats",
                      {"input_path": "high_risk.tif"},
                      {"min": 0.0, "max": 255.0, "mean": 11.73, "std": 36.2, "valid_pixels": 900000}),
            make_tool(3, "geo_statistics.count_pixels_condition",
                      {"input_path": "high_risk.tif", "lower": 254},
                      {"count": 41570, "total": 900000, "ratio": 0.0462}),
            make_tool(4, "geo_basic.pixel_area",
                      {"pixels": "$step3.count", "gsd_m": 500},
                      {"area_m2": 10392500000.0, "area_km2": 10392.5}),
            make_tool(5, "geoanalysis.getis_ord_gi_star",
                      {"image_path": "high_risk.tif", "output_path": "fire_hotspot_gi.tif",
                       "weight_matrix": W3x3},
                      {"status": "success", "output_path": "fire_hotspot_gi.tif"},
                      "weight_matrix 为3×3空间权重矩阵，非 values/coords 列表"),
            make_tool(6, "osm_gis.get_area_boundary",
                      {"place_name": "目标林区行政区名称"},
                      {"geometry": {"type": "Polygon", "coordinates": [[[113.0, 23.0], [114.0, 23.0], [114.0, 24.0], [113.0, 24.0], [113.0, 23.0]]]},
                       "bbox": [113.0, 23.0, 114.0, 24.0]},
                      "获取行政区边界和bbox用于POI查询"),
            make_tool(7, "osm_gis.add_pois_layer",
                      {"bbox": "$step6.bbox", "tags": {"amenity": "fire_station"}},
                      {"pois": [{"name": "消防站A", "lat": 23.45, "lon": 113.22}]},
                      "bbox为[west,south,east,north]；标注消防站位置辅助部署"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 7. 地震建筑物损毁评估
    # ------------------------------------------------------------------ #
    "earthquake_damage": {
        "disaster_category": "earthquake",
        "difficulty": "hard",
        "prompts": [
            "对比地震前后卫星影像，识别受损建筑物，统计损毁面积占比，圈出重灾区域。",
            "利用震前震后高分辨率遥感影像，评估建筑物损毁程度，生成灾情分级图。",
            "分析地震灾区卫星影像，检测倒塌和严重损毁的建筑，统计受灾范围面积。",
            "请通过震前震后影像对比，评估本次地震造成的建筑物损毁情况，给出损毁面积和重灾分布。",
        ],
        "tool_calls": [
            make_tool(1, "geo_perception.vlm_analyze",
                      {"image_paths": ["pre.tif", "post.tif"],
                       "prompt": "对比两张图中的建筑物，识别地震后倒塌、严重受损的建筑，列出受损区域坐标和损毁类型"},
                      {"analysis": "震后图中可见多处建筑倒塌...",
                       "bboxes": [{"x1": 100, "y1": 120, "x2": 280, "y2": 300}]}),
            make_tool(2, "geo_perception.remoteclip_analysis",
                      {"image_path": "post.tif",
                       "text_queries": ["collapsed building", "damaged roof", "rubble", "debris"]},
                      {"matches": [
                          {"text": "collapsed building", "score": 0.82, "bboxes": [{"x1": 110, "y1": 130, "x2": 270, "y2": 290}]}]}),
            make_tool(3, "geo_perception.sam2_segment",
                      {"image_path": "post.tif", "prompt": "damaged and collapsed buildings"},
                      {"bboxes": [{"x1": 100, "y1": 120, "x2": 280, "y2": 300},
                                  {"x1": 320, "y1": 200, "x2": 480, "y2": 360}]}),
            make_tool(4, "geo_raster.raster_diff",
                      {"path_a": "post.tif", "path_b": "pre.tif", "output_path": "damage_diff.tif"},
                      {"output_path": "damage_diff.tif"},
                      "差值绝对值大的区域表示明显变化"),
            make_tool(5, "geo_raster.threshold_segmentation",
                      {"input_path": "damage_diff.tif", "threshold": 50, "output_path": "damage_mask.tif"},
                      {"output_path": "damage_mask.tif"}),
            make_tool(6, "geo_statistics.count_pixels_condition",
                      {"input_path": "damage_mask.tif", "lower": 254},
                      {"count": 87430, "total": 4000000, "ratio": 0.0219}),
            make_tool(7, "geo_basic.pixel_area",
                      {"pixels": "$step6.count", "gsd_m": 0.5},
                      {"area_m2": 21857.5, "area_km2": 0.0219},
                      "高分影像 GSD=0.5m"),
            make_tool(8, "geo_perception.draw_bboxes",
                      {"image_path": "post.tif", "bboxes": "$step3.bboxes",
                       "output_path": "damage_annotated.png"},
                      {"output_path": "damage_annotated.png"},
                      "使用 sam2_segment（step3）的 bboxes，避免嵌套 matches[0] 引用"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 8. 滑坡/泥石流范围提取
    # ------------------------------------------------------------------ #
    "landslide": {
        "disaster_category": "landslide",
        "difficulty": "medium",
        "prompts": [
            "分析山区灾后遥感影像，提取滑坡堆积体范围，计算滑坡影响面积，描述受灾地形特征。",
            "利用高分辨率遥感影像识别滑坡灾害区域，统计裸露土壤面积，评估植被破坏程度。",
            "对泥石流发生区域的卫星影像进行分析，提取受灾范围并量化堆积体面积。",
            "请识别遥感影像中的滑坡堆积区和植被损毁区，计算总受灾面积并生成标注图。",
        ],
        "tool_calls": [
            make_tool(1, "geo_perception.vlm_analyze",
                      {"image_paths": ["disaster.tif"],
                       "prompt": "识别图中滑坡堆积体、裸露土壤和植被损毁区域，给出受灾范围描述"},
                      {"analysis": "图像左侧山坡可见大面积棕黄色滑坡堆积...",
                       "bboxes": [{"x1": 80, "y1": 100, "x2": 350, "y2": 400}]}),
            make_tool(2, "geo_perception.remoteclip_analysis",
                      {"image_path": "disaster.tif",
                       "text_queries": ["landslide deposit", "bare soil", "vegetation damage", "debris flow"]},
                      {"matches": [{"text": "landslide deposit", "score": 0.88,
                                    "bboxes": [{"x1": 80, "y1": 100, "x2": 350, "y2": 400}]}]}),
            make_tool(3, "geo_raster.calculate_index",
                      {"band_a_path": "nir.tif", "band_b_path": "red.tif",
                       "index_name": "NDVI", "output_path": "ndvi.tif"},
                      {"status": "success", "output_path": "ndvi.tif"},
                      "低NDVI区域（<0.1）即裸露扰动区"),
            make_tool(4, "geo_statistics.count_pixels_condition",
                      {"input_path": "ndvi.tif", "upper": 0.1},
                      {"count": 156780, "total": 2000000, "ratio": 0.0784},
                      "upper=0.1 → 计算NDVI低于0.1的裸露区域像素"),
            make_tool(5, "geo_basic.pixel_area",
                      {"pixels": "$step4.count", "gsd_m": 2},
                      {"area_m2": 627120.0, "area_km2": 0.627},
                      "高分2m分辨率"),
            make_tool(6, "geo_perception.remotesam",
                      {"image_path": "disaster.tif", "task": "segment landslide deposits"},
                      {"bboxes": [{"x1": 82, "y1": 102, "x2": 345, "y2": 398}],
                       "masks": ["mask_0.png"]}),
            make_tool(7, "geo_perception.draw_bboxes",
                      {"image_path": "disaster.tif", "bboxes": "$step6.bboxes",
                       "output_path": "landslide_annotated.png"},
                      {"output_path": "landslide_annotated.png"}),
        ],
    },

    # ------------------------------------------------------------------ #
    # 9. 干旱严重程度评估
    # ------------------------------------------------------------------ #
    "drought_monitoring": {
        "disaster_category": "drought",
        "difficulty": "hard",
        "prompts": [
            "基于多光谱影像和热红外数据，计算TVDI旱情指数，评估区域干旱严重程度，识别极重旱区域。",
            "利用遥感数据反演植被干旱指数，对干旱区域进行严重程度分级，统计各等级面积。",
            "分析农业区遥感影像，结合地表温度和植被指数计算TVDI，评估当前旱情并划定重旱区。",
            "请通过LST和NDVI联合反演旱情指数，识别极端干旱区域，为农业减灾提供决策依据。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "nir.tif", "band_b_path": "red.tif",
                       "index_name": "NDVI", "output_path": "ndvi.tif"},
                      {"status": "success", "output_path": "ndvi.tif"}),
            make_tool(2, "earth_sci.calculate_lst_sc",
                      {"bt_path": "thermal.tif", "red_path": "red.tif", "nir_path": "nir.tif",
                       "output_path": "lst.tif"},
                      {"output_path": "lst.tif"},
                      "单通道LST算法：bt_path为热红外亮温，red/nir用于内部发射率估算"),
            make_tool(3, "geo_raster.compute_tvdi",
                      {"lst_path": "lst.tif", "ndvi_path": "ndvi.tif", "output_path": "tvdi.tif"},
                      {"output_path": "tvdi.tif"},
                      "TVDI接近1为极旱，接近0为湿润"),
            make_tool(4, "geo_raster.raster_stats",
                      {"input_path": "tvdi.tif"},
                      {"min": 0.12, "max": 0.98, "mean": 0.61, "std": 0.18, "valid_pixels": 850000}),
            make_tool(5, "geo_statistics.count_pixels_condition",
                      {"input_path": "tvdi.tif", "lower": 0.8},
                      {"count": 127500, "total": 850000, "ratio": 0.15},
                      "TVDI>0.8 为极重旱"),
            make_tool(6, "geo_statistics.count_pixels_condition",
                      {"input_path": "tvdi.tif", "lower": 0.6},
                      {"count": 425000, "total": 850000, "ratio": 0.50},
                      "TVDI>0.6 为中度以上旱"),
            make_tool(7, "geo_basic.pixel_area",
                      {"pixels": "$step5.count", "gsd_m": 30},
                      {"area_m2": 114750000.0, "area_km2": 114.75},
                      "Landsat分辨率30m"),
            make_tool(8, "geo_statistics.threshold_ratio",
                      {"input_path": "tvdi.tif", "threshold": 0.6},
                      {"ratio": 0.50, "count": 425000, "total": 850000},
                      "中度以上干旱占比50%"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 10. 洪灾救援路线规划
    # ------------------------------------------------------------------ #
    "flood_rescue_routing": {
        "disaster_category": "flood",
        "difficulty": "medium",
        "prompts": [
            "洪水淹没了某市部分道路，请提取可通行区域，找出附近救援医院和安置点，规划最近救援路线。",
            "分析受灾区域卫星影像，结合道路网络和医疗设施分布，为救援队伍提供最优路线建议。",
            "基于洪水淹没图和城市地图，识别安全通行区域，定位最近的救援物资点和医院位置。",
            "请综合分析洪水范围和道路封堵情况，推荐救援车队从安全区域抵达受灾中心的路线。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"}),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi.tif", "threshold": 0.1, "output_path": "flood_mask.tif"},
                      {"output_path": "flood_mask.tif"},
                      "淹没区标记为255，可与OSM道路叠加分析"),
            make_tool(3, "osm_gis.get_area_boundary",
                      {"place_name": "受灾城市名"},
                      {"geometry": {"type": "Polygon", "coordinates": [[[113.0, 22.5], [114.0, 22.5], [114.0, 23.5], [113.0, 23.5], [113.0, 22.5]]]},
                       "bbox": [113.0, 22.5, 114.0, 23.5]}),
            make_tool(4, "osm_gis.add_pois_layer",
                      {"bbox": "$step3.bbox",
                       "tags": {"amenity": ["hospital", "shelter", "fire_station"]}},
                      {"pois": [
                          {"name": "市人民医院", "lat": 23.12, "lon": 113.45, "type": "hospital"},
                          {"name": "体育馆安置点", "lat": 23.08, "lon": 113.38, "type": "shelter"}]},
                      "bbox=[west,south,east,north]，从step3.bbox引用"),
            make_tool(5, "osm_gis.compute_route_dist",
                      {"origin": [113.40, 22.95],
                       "destination": [113.45, 23.12]},
                      {"distance_m": 8300.0, "distance_km": 8.3, "travel_time_min": 22.0},
                      "destination为最近医院坐标[lon,lat]；单次路线规划"),
            make_tool(6, "geo_basic.distance",
                      {"lon1": 113.40, "lat1": 22.95, "lon2": 113.38, "lat2": 23.08},
                      {"distance_m": 14520.0, "distance_km": 14.52},
                      "直线距离供参考；参数为 lon1/lat1/lon2/lat2 四个独立数值"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 11. 热浪城市高温热点分析
    # ------------------------------------------------------------------ #
    "heatwave_analysis": {
        "disaster_category": "heatwave",
        "difficulty": "medium",
        "prompts": [
            "分析城市热浪期间地表温度分布，识别高温聚集热点，统计危险高温区域面积，辅助应急响应部署。",
            "利用热红外遥感数据评估城市热岛效应，找出高温脆弱区域，为热浪应急响应提供支持。",
            "对城市区域卫星热红外影像进行分析，识别极端高温区域，统计高温暴露面积和关键设施风险。",
            "请评估热浪期间城市地表温度分布，识别高温热点聚集区，定位附近冷却中心和医院。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "nir.tif", "band_b_path": "red.tif",
                       "index_name": "NDVI", "output_path": "ndvi.tif"},
                      {"status": "success", "output_path": "ndvi.tif"},
                      "先计算NDVI，供LST和后续热岛分析使用"),
            make_tool(2, "earth_sci.calculate_lst_sc",
                      {"bt_path": "thermal.tif", "red_path": "red.tif", "nir_path": "nir.tif",
                       "output_path": "lst.tif"},
                      {"output_path": "lst.tif"},
                      "bt_path=热红外亮温，red/nir用于内部发射率估算"),
            make_tool(3, "earth_sci.stats_lst_ndvi",
                      {"red_path": "red.tif", "nir_path": "nir.tif", "lst_path": "lst.tif",
                       "threshold": 0.3, "mode": "below"},
                      {"result": 48.3, "pixel_count": 234500},
                      "内部由red/nir计算NDVI，统计低植被区(NDVI<0.3)的LST均值（热岛效应）"),
            make_tool(4, "geo_statistics.count_pixels_condition",
                      {"input_path": "lst.tif", "lower": 40},
                      {"count": 234500, "total": 1500000, "ratio": 0.156},
                      "LST > 40°C 为高温危险区"),
            make_tool(5, "geo_basic.pixel_area",
                      {"pixels": "$step4.count", "gsd_m": 30},
                      {"area_m2": 211050000.0, "area_km2": 211.05}),
            make_tool(6, "geoanalysis.getis_ord_gi_star",
                      {"image_path": "lst.tif", "output_path": "lst_hotspot.tif",
                       "weight_matrix": W3x3},
                      {"status": "success", "output_path": "lst_hotspot.tif"},
                      "输出Gi*统计量栅格，高值为显著高温聚集区"),
            make_tool(7, "osm_gis.add_pois_layer",
                      {"bbox": [120.0, 30.0, 121.0, 31.0],
                       "tags": {"amenity": ["hospital", "cooling_center", "community_centre"]}},
                      {"pois": [{"name": "第一医院", "lat": 30.12, "lon": 120.21}]},
                      "bbox=[west,south,east,north]"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 12. 冰雪灾害评估
    # ------------------------------------------------------------------ #
    "snow_disaster": {
        "disaster_category": "snow",
        "difficulty": "medium",
        "prompts": [
            "对比暴雪前后卫星影像，计算积雪覆盖面积变化，评估农业和交通受灾程度。",
            "利用遥感影像监测暴雪覆盖范围，分析积雪对道路和农田的影响程度，给出受灾面积。",
            "基于暴雪前后多光谱影像，统计积雪覆盖面积增减变化，评估区域受灾情况。",
            "请分析这组暴雪前后遥感影像，计算积雪扩展面积，并统计受影响的关键道路长度。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green_pre.tif", "band_b_path": "swir_pre.tif",
                       "index_name": "NDSI", "output_path": "ndsi_pre.tif"},
                      {"status": "success", "output_path": "ndsi_pre.tif"},
                      "NDSI = (Green-SWIR)/(Green+SWIR)，雪>0.4"),
            make_tool(2, "geo_raster.calculate_index",
                      {"band_a_path": "green_post.tif", "band_b_path": "swir_post.tif",
                       "index_name": "NDSI", "output_path": "ndsi_post.tif"},
                      {"status": "success", "output_path": "ndsi_post.tif"}),
            make_tool(3, "geo_raster.threshold_segmentation",
                      {"input_path": "ndsi_pre.tif", "threshold": 0.4, "output_path": "snow_pre.tif"},
                      {"output_path": "snow_pre.tif"}, "NDSI>0.4 → 积雪区255"),
            make_tool(4, "geo_raster.threshold_segmentation",
                      {"input_path": "ndsi_post.tif", "threshold": 0.4, "output_path": "snow_post.tif"},
                      {"output_path": "snow_post.tif"}),
            make_tool(5, "geo_raster.raster_diff",
                      {"path_a": "snow_post.tif", "path_b": "snow_pre.tif",
                       "output_path": "snow_change.tif"},
                      {"output_path": "snow_change.tif"},
                      "正值=新增积雪区，负值=融雪区"),
            make_tool(6, "geo_raster.threshold_segmentation",
                      {"input_path": "snow_change.tif", "threshold": 127,
                       "output_path": "snow_increase_binary.tif"},
                      {"output_path": "snow_increase_binary.tif"},
                      "生成新增积雪区二值图，用于calc_snow_loss_stats"),
            make_tool(7, "geo_raster.calc_snow_loss_stats",
                      {"binary_map_path": "snow_increase_binary.tif"},
                      {"percentage": 18.3, "total_pixels": 900000, "loss_pixels": 164700},
                      "参数为 binary_map_path（单张二值图），非 pre_path/post_path"),
            make_tool(8, "geo_statistics.count_pixels_condition",
                      {"input_path": "snow_post.tif", "lower": 254},
                      {"count": 412300, "total": 900000, "ratio": 0.458}),
            make_tool(9, "geo_basic.pixel_area",
                      {"pixels": "$step8.count", "gsd_m": 10},
                      {"area_m2": 41230000.0, "area_km2": 41.23}),
        ],
    },

    # ------------------------------------------------------------------ #
    # 13. 台风港口损毁评估
    # ------------------------------------------------------------------ #
    "typhoon_port_damage": {
        "disaster_category": "typhoon",
        "difficulty": "medium",
        "prompts": [
            "台风过境后，对港口区域卫星图像进行分析，识别受损船只和码头设施，统计损毁目标数量。",
            "分析台风灾后港口遥感影像，检测受损设施和搁浅船只，评估港口受损程度。",
            "利用高分辨率卫星影像评估台风对沿海港口的破坏，统计受损目标数量和损毁覆盖面积。",
            "请识别台风后港口区域影像中的受损目标，给出损毁数量和面积统计，辅助灾后恢复评估。",
        ],
        "tool_calls": [
            make_tool(1, "geo_perception.vlm_analyze",
                      {"image_paths": ["port_pre.tif", "port_post.tif"],
                       "prompt": "对比台风前后港口图像，识别受损船只、倒塌码头设施和新增碎片"},
                      {"analysis": "台风后图像可见多艘船只搁浅...",
                       "bboxes": [{"x1": 200, "y1": 150, "x2": 380, "y2": 280}]}),
            make_tool(2, "geo_perception.strip_rcnn_detect",
                      {"image_path": "port_post.tif"},
                      {"detections": [
                          {"label": "ship", "confidence": 0.91,
                           "bbox": {"x1": 205, "y1": 155, "x2": 375, "y2": 275}},
                          {"label": "debris", "confidence": 0.78,
                           "bbox": {"x1": 450, "y1": 300, "x2": 600, "y2": 420}}],
                       "detections_bboxes": [
                           {"x1": 205, "y1": 155, "x2": 375, "y2": 275},
                           {"x1": 450, "y1": 300, "x2": 600, "y2": 420}]},
                      "detections_bboxes为顶层展平列表，供draw_bboxes直接引用"),
            make_tool(3, "geo_perception.instructsam",
                      {"image_path": "port_post.tif",
                       "text": "damaged ships and port structures"},
                      {"count": 7, "bboxes": [], "masks": []}),
            make_tool(4, "geo_perception.bbox_area",
                      {"bboxes": [{"x1": 205, "y1": 155, "x2": 375, "y2": 275},
                                  {"x1": 450, "y1": 300, "x2": 600, "y2": 420}],
                       "gsd_m": 0.5},
                      {"total_area_px2": 46800.0,
                       "per_bbox_area_px2": [28900.0, 17900.0],
                       "count": 2,
                       "per_bbox_area_m2": [7225.0, 4475.0]}),
            make_tool(5, "geo_perception.draw_bboxes",
                      {"image_path": "port_post.tif",
                       "bboxes": "$step2.detections_bboxes",
                       "output_path": "damage_detected.png"},
                      {"output_path": "damage_detected.png"}),
            make_tool(6, "geo_perception.add_text",
                      {"image_path": "damage_detected.png",
                       "text": "Detected 7 damaged targets",
                       "position": [10, 10],
                       "output_path": "final_annotated.png"},
                      {"output_path": "final_annotated.png"}),
        ],
    },

    # ------------------------------------------------------------------ #
    # 14. 旱区土壤湿度反演
    # ------------------------------------------------------------------ #
    "soil_moisture": {
        "disaster_category": "drought",
        "difficulty": "hard",
        "prompts": [
            "利用微波遥感数据反演旱区土壤湿度，识别极度缺水区域，辅助抗旱调度决策。",
            "基于被动微波卫星数据计算土壤含水量，评估干旱区域水分亏缺状况。",
            "对干旱灾区进行微波遥感土壤湿度反演，找出严重缺水区域，支持农业抗旱工作。",
            "请通过微波亮温数据和热惯量模型联合反演土壤湿度，识别极端干旱区并分析趋势。",
        ],
        "tool_calls": [
            make_tool(1, "earth_sci.microwave_dpdm",
                      {"pol1_path": "tb_v.tif", "pol2_path": "tb_h.tif", "output_path": "sm_dpdm.tif"},
                      {"output_path": "sm_dpdm.tif"},
                      "双极化差分法（DPDM）；pol1_path=垂直极化，pol2_path=水平极化亮温"),
            make_tool(2, "earth_sci.calculate_ati",
                      {"day_temp_path": "lst_day.tif",
                       "night_temp_path": "lst_night.tif",
                       "albedo_path": "albedo.tif",
                       "output_path": "ati.tif"},
                      {"output_path": "ati.tif"},
                      "表观热惯量法：day_temp_path/night_temp_path为日夜亮温；ATI与土壤含水量正相关"),
            make_tool(3, "geo_statistics.batch_raster_stats",
                      {"input_paths": ["sm_dpdm.tif", "ati.tif"]},
                      {"per_image": [
                          {"path": "sm_dpdm.tif", "mean": 0.18, "std": 0.09, "min": 0.02, "max": 0.48},
                          {"path": "ati.tif", "mean": 0.23, "std": 0.11, "min": 0.05, "max": 0.52}],
                       "aggregate": {"mean": 0.205}}),
            make_tool(4, "geo_statistics.count_pixels_condition",
                      {"input_path": "sm_dpdm.tif", "upper": 0.1},
                      {"count": 89600, "total": 576000, "ratio": 0.156},
                      "SM<10%为极度干旱区域"),
            make_tool(5, "geo_basic.pixel_area",
                      {"pixels": "$step4.count", "gsd_m": 25000},
                      {"area_m2": 56000000000.0, "area_km2": 56000.0},
                      "被动微波像元约25km分辨率"),
            make_tool(6, "geoanalysis.mann_kendall_test",
                      {"values": [0.22, 0.20, 0.18, 0.15, 0.13, 0.11, 0.10, 0.09]},
                      {"trend": "decreasing", "p_value": 0.001, "tau": -0.86},
                      "输入为多期SM均值时间序列，判断干旱是否持续恶化"),
        ],
    },

    # ------------------------------------------------------------------ #
    # 15. 网格化洪涝灾情分级评估
    # ------------------------------------------------------------------ #
    "grid_flood_assessment": {
        "disaster_category": "flood",
        "difficulty": "hard",
        "prompts": [
            "对受灾区域进行网格化分析，评估每个网格单元的洪水淹没程度，生成轻度/中度/重度灾情分级图。",
            "将受灾区划分为规则网格，对每个网格评估洪水严重程度，输出三级灾情分级统计报告。",
            "对洪涝灾区进行空间网格化精细评估，按轻中重三级统计淹没面积，辅助救援资源分配决策。",
            "请对洪灾区域进行1km×1km网格化分析，按淹没深度（NDWI）分级统计受灾面积和热点分布。",
        ],
        "tool_calls": [
            make_tool(1, "geo_basic.aoi_validate",
                      {"geojson": {"type": "Polygon",
                                   "coordinates": [[[113.0, 22.5], [114.0, 22.5],
                                                    [114.0, 23.5], [113.0, 23.5],
                                                    [113.0, 22.5]]]}},
                      {"aoi_feature_collection": {"type": "FeatureCollection",
                                                   "features": [{"type": "Feature",
                                                                  "geometry": {"type": "Polygon",
                                                                               "coordinates": [[[113.0, 22.5], [114.0, 22.5], [114.0, 23.5], [113.0, 23.5], [113.0, 22.5]]]},
                                                                  "properties": {}}]},
                       "geometry_type": "Polygon",
                       "bbox": [113.0, 22.5, 114.0, 23.5],
                       "total_area_km2": 12100.0,
                       "total_area_m2": 12100000000.0},
                      "参数名为 geojson（非 aoi）；返回 aoi_feature_collection 和 bbox"),
            make_tool(2, "geo_basic.gridify",
                      {"geojson": "$step1.aoi_feature_collection", "cell_width_m": 1000},
                      {"type": "FeatureCollection", "features": [{"type": "Feature"}],
                       "grid_count": 12100},
                      "参数名为 geojson + cell_width_m（米）；1000m=1km网格"),
            make_tool(3, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"}),
            make_tool(4, "geo_raster.raster_stats",
                      {"input_path": "ndwi.tif"},
                      {"min": -0.45, "max": 0.82, "mean": 0.06, "std": 0.21, "valid_pixels": 1210000}),
            make_tool(5, "geo_statistics.count_pixels_condition",
                      {"input_path": "ndwi.tif", "lower": 0.1, "upper": 0.3},
                      {"count": 181500, "total": 1210000, "ratio": 0.15},
                      "轻度淹没：0.1 ≤ NDWI < 0.3"),
            make_tool(6, "geo_statistics.count_pixels_condition",
                      {"input_path": "ndwi.tif", "lower": 0.3, "upper": 0.5},
                      {"count": 96800, "total": 1210000, "ratio": 0.08},
                      "中度淹没：0.3 ≤ NDWI < 0.5"),
            make_tool(7, "geo_statistics.count_pixels_condition",
                      {"input_path": "ndwi.tif", "lower": 0.5},
                      {"count": 48400, "total": 1210000, "ratio": 0.04},
                      "重度淹没：NDWI ≥ 0.5"),
            make_tool(8, "geo_basic.pixel_area",
                      {"pixels": "$step5.count", "gsd_m": 10},
                      {"area_m2": 18150000.0, "area_km2": 18.15}, "轻度淹没面积"),
            make_tool(9, "geo_basic.pixel_area",
                      {"pixels": "$step6.count", "gsd_m": 10},
                      {"area_m2": 9680000.0, "area_km2": 9.68}, "中度淹没面积"),
            make_tool(10, "geo_basic.pixel_area",
                      {"pixels": "$step7.count", "gsd_m": 10},
                      {"area_m2": 4840000.0, "area_km2": 4.84}, "重度淹没面积"),
            make_tool(11, "geoanalysis.getis_ord_gi_star",
                      {"image_path": "ndwi.tif", "output_path": "ndwi_hotspot.tif",
                       "weight_matrix": W3x3},
                      {"status": "success", "output_path": "ndwi_hotspot.tif"},
                      "识别淹没聚集热点区，Gi*高值区为重点救援区"),
            make_tool(12, "osm_gis.add_pois_layer",
                      {"bbox": "$step1.bbox",
                       "tags": {"amenity": ["hospital", "shelter", "school", "police"]}},
                      {"pois": [{"name": "某医院", "lat": 23.1, "lon": 113.5, "type": "hospital"}]},
                      "bbox=[west,south,east,north]，从step1.bbox引用；定位关键设施"),
        ],
    },
}

# --------------------------------------------------------------------------- #
# 新增工具集 disaster_response 的 8 类任务
# --------------------------------------------------------------------------- #

NEW_TASK_TEMPLATES = {

    # ------------------------------------------------------------------ #
    # N1. 滑坡风险坡度分析
    # ------------------------------------------------------------------ #
    "slope_landslide": {
        "disaster_category": "landslide",
        "difficulty": "easy",
        "prompts": [
            "利用DEM高程数据计算地形坡度，识别坡度大于30°的高危滑坡区域，统计高危面积。",
            "请对山区DEM数据进行坡度分析，划定滑坡高风险坡地，为灾前预警提供支撑。",
            "基于数字高程模型提取坡度信息，评估研究区内不同坡度等级的分布面积和占比。",
            "分析地形坡度与滑坡风险的关系，识别需要疏散预警的高坡度危险区。",
        ],
        "tool_calls": [
            make_tool(1, "disaster_response.calc_slope",
                      {"dem_path": "dem.tif", "output_path": "slope.tif", "unit": "degrees"},
                      {"output_path": "slope.tif", "unit": "degrees",
                       "mean_slope": 18.4, "max_slope": 62.7},
                      "Sobel梯度法计算坡度，单位度"),
            make_tool(2, "geo_raster.raster_stats",
                      {"input_path": "slope.tif"},
                      {"min": 0.1, "max": 62.7, "mean": 18.4, "std": 12.1, "valid_pixels": 900000}),
            make_tool(3, "geo_statistics.count_pixels_condition",
                      {"input_path": "slope.tif", "lower": 30},
                      {"count": 135000, "total": 900000, "ratio": 0.15},
                      "坡度>30°为高危区"),
            make_tool(4, "geo_basic.pixel_area",
                      {"pixels": "$step3.count", "gsd_m": 30},
                      {"area_m2": 121500000.0, "area_km2": 121.5}),
            make_tool(5, "geo_statistics.count_pixels_condition",
                      {"input_path": "slope.tif", "lower": 45},
                      {"count": 27000, "total": 900000, "ratio": 0.03},
                      "坡度>45°为极高危区"),
            make_tool(6, "geo_basic.pixel_area",
                      {"pixels": "$step5.count", "gsd_m": 30},
                      {"area_m2": 24300000.0, "area_km2": 24.3}),
        ],
    },

    # ------------------------------------------------------------------ #
    # N2. 洪水汇流路径追踪
    # ------------------------------------------------------------------ #
    "flood_flow_routing": {
        "disaster_category": "flood",
        "difficulty": "medium",
        "prompts": [
            "基于DEM计算D8水文流向，追踪暴雨后洪水的汇流路径，预判下游哪些村庄受灾风险最高。",
            "利用高程数据分析流域水文特征，确定洪水流动方向和汇聚点，为下游预警提供依据。",
            "通过DEM水文分析找出洪水主要流向，识别低洼汇水区，辅助防洪工程规划。",
        ],
        "tool_calls": [
            make_tool(1, "disaster_response.calc_slope",
                      {"dem_path": "dem.tif", "output_path": "slope.tif", "unit": "degrees"},
                      {"output_path": "slope.tif", "mean_slope": 12.3, "max_slope": 48.2}),
            make_tool(2, "disaster_response.calc_flow_direction",
                      {"dem_path": "dem.tif", "output_path": "flow_dir.tif"},
                      {"output_path": "flow_dir.tif",
                       "d8_codes": "1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE"},
                      "D8算法，每个像元指向最陡下坡邻居"),
            make_tool(3, "geo_raster.raster_stats",
                      {"input_path": "flow_dir.tif"},
                      {"min": 1.0, "max": 128.0, "mean": 42.5, "valid_pixels": 850000}),
            make_tool(4, "geo_statistics.count_pixels_condition",
                      {"input_path": "slope.tif", "upper": 5},
                      {"count": 102000, "total": 850000, "ratio": 0.12},
                      "坡度<5°的低洼区=潜在汇水区"),
            make_tool(5, "geo_basic.pixel_area",
                      {"pixels": "$step4.count", "gsd_m": 30},
                      {"area_m2": 91800000.0, "area_km2": 91.8}),
            make_tool(6, "osm_gis.get_area_boundary",
                      {"place_name": "下游目标区域名"},
                      {"geometry": {"type": "Polygon", "coordinates": [[[113.0, 23.0], [114.0, 23.0], [114.0, 24.0], [113.0, 24.0], [113.0, 23.0]]]}},
                      "获取行政区边界与流向叠加分析"),
        ],
    },

    # ------------------------------------------------------------------ #
    # N3. 按行政区分区统计洪水淹没
    # ------------------------------------------------------------------ #
    "zonal_flood_stats": {
        "disaster_category": "flood",
        "difficulty": "medium",
        "prompts": [
            "按行政区统计各区洪水淹没面积，排序确定救援优先级，输出各区受灾严重程度报告。",
            "将洪水NDWI栅格与行政区边界叠加，计算每个区县的平均淹没强度和受灾比例。",
            "对多个受灾区县进行分区统计，分析哪些区县淹没面积最大，优先调配救援资源。",
            "基于分区统计方法，量化每个行政单元的洪涝受灾程度，为救灾物资分配提供数据支持。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"}),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi.tif", "threshold": 0.1, "output_path": "flood_mask.tif"},
                      {"output_path": "flood_mask.tif"}),
            make_tool(3, "disaster_response.zonal_stats",
                      {"raster_path": "flood_mask.tif",
                       "zones_geojson": "districts.geojson",
                       "stat": "mean"},
                      {"zones": [
                          {"id": 1, "name": "A区", "value": 0.32},
                          {"id": 2, "name": "B区", "value": 0.18},
                          {"id": 3, "name": "C区", "value": 0.51}],
                       "count": 3, "stat": "mean"},
                      "mean>0.3表示该区超30%面积被淹"),
            make_tool(4, "disaster_response.zonal_stats",
                      {"raster_path": "flood_mask.tif",
                       "zones_geojson": "districts.geojson",
                       "stat": "count"},
                      {"zones": [
                          {"id": 1, "name": "A区", "value": 84200},
                          {"id": 2, "name": "B区", "value": 41300},
                          {"id": 3, "name": "C区", "value": 127800}],
                       "count": 3, "stat": "count"},
                      "各区淹没像素总数"),
            make_tool(5, "geo_basic.pixel_area",
                      {"pixels": 127800, "gsd_m": 10},
                      {"area_m2": 12780000.0, "area_km2": 12.78},
                      "C区淹没面积最大"),
        ],
    },

    # ------------------------------------------------------------------ #
    # N4. 洪水暴露人口估算
    # ------------------------------------------------------------------ #
    "population_flood_risk": {
        "disaster_category": "flood",
        "difficulty": "medium",
        "prompts": [
            "估算洪水淹没范围内的暴露人口数量，结合人口密度栅格计算受灾人口总数。",
            "利用洪水风险栅格和WorldPop人口数据，评估本次洪灾的受灾人口规模。",
            "将洪水淹没掩膜与人口密度数据叠加，计算暴露在洪涝风险中的居民人数。",
            "请综合灾害范围和人口分布数据，估算受此次洪水影响的人口数量，辅助救援规模决策。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"}),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi.tif", "threshold": 0.1, "output_path": "flood_mask.tif"},
                      {"output_path": "flood_mask.tif"},
                      "flood_mask: 255=淹没，0=正常，归一化为0-1供人口叠加"),
            make_tool(3, "disaster_response.population_exposure",
                      {"hazard_path": "flood_mask.tif",
                       "population_path": "population_density.tif",
                       "output_path": "exposed_pop.tif",
                       "threshold": 127},
                      {"exposed_population": 238700.0,
                       "output_path": "exposed_pop.tif",
                       "exposed_pixels": 45820,
                       "total_pixels": 1000000,
                       "exposed_ratio": 0.0458,
                       "total_area_km2": 4.582}),
            make_tool(4, "geo_raster.raster_stats",
                      {"input_path": "exposed_pop.tif"},
                      {"min": 0.0, "max": 1842.3, "mean": 5.2, "valid_pixels": 1000000},
                      "查看人口暴露的空间分布特征"),
            make_tool(5, "geoanalysis.getis_ord_gi_star",
                      {"image_path": "exposed_pop.tif", "output_path": "pop_hotspot.tif",
                       "weight_matrix": W3x3},
                      {"status": "success", "output_path": "pop_hotspot.tif"},
                      "识别高人口暴露聚集区，优先救援"),
        ],
    },

    # ------------------------------------------------------------------ #
    # N5. 洪灾A*救援路径规划
    # ------------------------------------------------------------------ #
    "disaster_astar_routing": {
        "disaster_category": "flood",
        "difficulty": "hard",
        "prompts": [
            "洪水封堵了部分道路，请用A*算法规划从救援站到受灾社区的最优路线，避开被淹路段。",
            "基于实时洪水掩膜和道路网络，为救援车队规划考虑道路封堵的最短安全路径。",
            "分析洪灾期间道路封堵情况，用A*路径规划为医疗救援队提供绕行路线建议。",
            "结合洪水淹没图和OSM路网，计算在多条道路被淹的情况下抵达受灾中心的最优路径。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"}),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi.tif", "threshold": 0.1, "output_path": "flood_mask.tif"},
                      {"output_path": "flood_mask.tif"},
                      "生成洪水二值掩膜，供路径规划工具识别封堵路段"),
            make_tool(3, "disaster_response.flood_route_astar",
                      {"origin": [113.40, 22.95],
                       "destination": [113.55, 23.10],
                       "flood_mask_path": "flood_mask.tif",
                       "flood_threshold": 127,
                       "travel_mode": "drive",
                       "penalty_factor": 1000},
                      {"passable": True,
                       "distance_m": 28450.0,
                       "distance_km": 28.45,
                       "travel_time_min": 34.1,
                       "blocked_edges": 23,
                       "blocked_ratio": 0.087,
                       "route_geojson": {"type": "Feature",
                                         "geometry": {"type": "LineString",
                                                       "coordinates": [[113.40, 22.95], [113.55, 23.10]]}}},
                      "A*+障碍权重；blocked_ratio=8.7%道路被淹；返回GeoJSON路线"),
            make_tool(4, "geo_basic.distance",
                      {"lon1": 113.40, "lat1": 22.95, "lon2": 113.55, "lat2": 23.10},
                      {"distance_m": 22380.0, "distance_km": 22.38},
                      "直线距离对比，实际路线比直线长27%（绕行被淹路段）；参数为四个独立数值"),
        ],
    },

    # ------------------------------------------------------------------ #
    # N6. 受灾区医疗可达性分析
    # ------------------------------------------------------------------ #
    "rescue_accessibility": {
        "disaster_category": "flood",
        "difficulty": "hard",
        "prompts": [
            "生成受灾区距最近医院的行程时间栅格，识别孤立区域（>60分钟不可达）。",
            "分析洪灾后的医疗可达性，找出因道路封堵而无法及时获救的孤岛区域。",
            "评估受灾区域内各地点到最近救援站的可达时间，为直升机救援优先级提供依据。",
            "结合洪水封路情况和医院位置，生成全区医疗可达性地图，识别救援盲区。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_index",
                      {"band_a_path": "green.tif", "band_b_path": "nir.tif",
                       "index_name": "NDWI", "output_path": "ndwi.tif"},
                      {"status": "success", "output_path": "ndwi.tif"}),
            make_tool(2, "geo_raster.threshold_segmentation",
                      {"input_path": "ndwi.tif", "threshold": 0.1, "output_path": "flood_mask.tif"},
                      {"output_path": "flood_mask.tif"}),
            make_tool(3, "geo_raster.raster_diff",
                      {"path_a": "ones_raster.tif", "path_b": "flood_mask_norm.tif",
                       "output_path": "road_passable.tif"},
                      {"output_path": "road_passable.tif"},
                      "1-flood_norm → 可通行掩膜（0=被淹封路，1=可通行）"),
            make_tool(4, "osm_gis.add_pois_layer",
                      {"bbox": [113.0, 22.5, 114.0, 23.5],
                       "tags": {"amenity": ["hospital", "clinic"]}},
                      {"pois": [
                          {"name": "市人民医院", "lat": 23.12, "lon": 113.45},
                          {"name": "中心卫生院", "lat": 22.98, "lon": 113.62}]},
                      "bbox=[west,south,east,north]"),
            make_tool(5, "disaster_response.accessibility_map",
                      {"road_mask_path": "road_passable.tif",
                       "poi_locations": "$step4.pois",
                       "output_path": "access_time.tif",
                       "travel_speed_kmh": 30},
                      {"output_path": "access_time.tif",
                       "mean_time_min": 23.4,
                       "max_time_min": 187.3,
                       "isolated_pixels": 12800,
                       "total_passable_pixels": 850000,
                       "isolated_ratio": 0.015}),
            make_tool(6, "geo_statistics.count_pixels_condition",
                      {"input_path": "access_time.tif", "lower": 60},
                      {"count": 42300, "total": 850000, "ratio": 0.0498},
                      "距医院>60分钟的高风险孤岛区域"),
            make_tool(7, "geo_basic.pixel_area",
                      {"pixels": "$step6.count", "gsd_m": 10},
                      {"area_m2": 4230000.0, "area_km2": 4.23}),
        ],
    },

    # ------------------------------------------------------------------ #
    # N7. 建筑物损毁逐栋评估
    # ------------------------------------------------------------------ #
    "building_damage_assess": {
        "disaster_category": "earthquake",
        "difficulty": "hard",
        "prompts": [
            "将变化检测结果与OSM建筑轮廓叠加，逐栋评估建筑损毁状态，统计损毁建筑数量和比例。",
            "利用震前震后变化检测图和建筑物矢量数据，对每栋建筑进行损毁评分，识别倒塌建筑。",
            "对地震灾区进行建筑物精细化损毁评估，结合变化掩膜和建筑轮廓，生成损毁清单。",
            "整合SAR变化检测结果与建筑足迹数据，评估建筑受损比例，为搜救优先级排序。",
        ],
        "tool_calls": [
            make_tool(1, "geo_perception.remoteclip_analysis",
                      {"image_path": "post.tif",
                       "text_queries": ["collapsed building", "damaged structure", "rubble"]},
                      {"matches": [{"text": "collapsed building", "score": 0.84,
                                    "bboxes": [{"x1": 100, "y1": 120, "x2": 280, "y2": 300}]}],
                       "first_match_bboxes": [{"x1": 100, "y1": 120, "x2": 280, "y2": 300}]},
                      "first_match_bboxes为顶层展平键，避免嵌套matches[0]引用"),
            make_tool(2, "geo_raster.raster_diff",
                      {"path_a": "post.tif", "path_b": "pre.tif", "output_path": "change_diff.tif"},
                      {"output_path": "change_diff.tif"}),
            make_tool(3, "geo_raster.threshold_segmentation",
                      {"input_path": "change_diff.tif", "threshold": 40, "output_path": "change_mask.tif"},
                      {"output_path": "change_mask.tif"},
                      "变化值>40标记为255，生成变化二值图"),
            make_tool(4, "disaster_response.building_damage_stats",
                      {"change_raster_path": "change_mask.tif",
                       "buildings_geojson": "osm_buildings.geojson",
                       "damage_threshold": 0.3},
                      {"total_buildings": 342,
                       "damaged": 87,
                       "intact": 255,
                       "damage_ratio": 0.254,
                       "buildings": [
                           {"id": 1, "status": "damaged", "change_ratio": 0.45},
                           {"id": 2, "status": "intact",  "change_ratio": 0.08}]},
                      "damage_threshold=0.3：建筑内30%像素变化→判定损毁"),
            make_tool(5, "geo_perception.draw_bboxes",
                      {"image_path": "post.tif",
                       "bboxes": "$step1.first_match_bboxes",
                       "output_path": "damage_map.png"},
                      {"output_path": "damage_map.png"},
                      "引用step1的顶层展平键first_match_bboxes"),
        ],
    },

    # ------------------------------------------------------------------ #
    # N8. 山火蔓延预测
    # ------------------------------------------------------------------ #
    "fire_spread_warning": {
        "disaster_category": "wildfire",
        "difficulty": "hard",
        "prompts": [
            "根据当前风向（东南风8m/s）和植被覆盖度，预测山火6小时内的蔓延范围，为下风向村庄提供疏散预警。",
            "利用FVC植被数据和气象条件，模拟山火扩散过程，估算6小时后的过火面积和最远蔓延距离。",
            "对正在发生的山火进行蔓延预测，结合风速风向与植被燃料，输出预测烧毁区域范围。",
            "分析山火蔓延趋势，基于细胞自动机模型预测火势6步时间内的扩展区域，辅助疏散决策。",
        ],
        "tool_calls": [
            make_tool(1, "geo_raster.calculate_frp",
                      {"input_path": "frp.tif", "output_path": "fire_current.tif", "threshold": 10},
                      {"output_path": "fire_current.tif", "fire_pixels": 892, "fire_coverage_pct": 2.97},
                      "当前活跃火点作为预测起始点"),
            make_tool(2, "geo_raster.calculate_fvc",
                      {"nir_path": "nir.tif", "red_path": "red.tif", "output_path": "fvc.tif"},
                      {"output_path": "fvc.tif"},
                      "植被覆盖度0-1，越高燃料越充足"),
            make_tool(3, "disaster_response.fire_spread_forecast",
                      {"fvc_path": "fvc.tif",
                       "ignition_points": [{"lon": 113.82, "lat": 23.45},
                                            {"lon": 113.84, "lat": 23.43}],
                       "wind_direction_deg": 135,
                       "wind_speed_ms": 8,
                       "time_steps": 6,
                       "output_path": "fire_forecast_6h.tif"},
                      {"output_path": "fire_forecast_6h.tif",
                       "affected_area_km2": 48.3,
                       "affected_pixels": 193200,
                       "max_spread_km": 12.4,
                       "time_steps": 6,
                       "wind_direction_deg": 135,
                       "wind_speed_ms": 8},
                      "135°=东南风；fire向西北方向蔓延；6小时预测过火48.3km²"),
            make_tool(4, "geo_statistics.count_pixels_condition",
                      {"input_path": "fire_forecast_6h.tif", "lower": 254},
                      {"count": 193200, "total": 6500000, "ratio": 0.0297}),
            make_tool(5, "osm_gis.add_pois_layer",
                      {"bbox": [113.6, 23.2, 114.1, 23.7],
                       "tags": {"place": ["village", "town"], "amenity": "shelter"}},
                      {"pois": [{"name": "下风向村庄A", "lat": 23.58, "lon": 113.75},
                                 {"name": "下风向村庄B", "lat": 23.61, "lon": 113.71}]},
                      "bbox=[west,south,east,north]；标注预测火场下风向居民点，发布疏散预警"),
            make_tool(6, "geo_perception.vlm_analyze",
                      {"image_paths": ["rgb_fire.tif"],
                       "prompt": "描述当前火场位置和周边村庄分布，评估6小时内哪些居民点面临威胁"},
                      {"analysis": "图中火场位于东南方向密林区，西北侧2-3km处有两处居民聚落..."}),
        ],
    },
}

# 合并两个模板字典
TASK_TEMPLATES.update(NEW_TASK_TEMPLATES)


# --------------------------------------------------------------------------- #
# 生成样本列表
# --------------------------------------------------------------------------- #
def generate_samples():
    samples = []
    for task_type, task_def in TASK_TEMPLATES.items():
        for idx, prompt in enumerate(task_def["prompts"], start=1):
            sample = {
                "id": f"{task_type}_{idx:02d}",
                "task_type": task_type,
                "disaster_category": task_def["disaster_category"],
                "difficulty": task_def["difficulty"],
                "prompt": prompt,
                "tool_calls": task_def["tool_calls"],
            }
            samples.append(sample)
    return samples


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def main():
    samples = generate_samples()

    dataset = {
        "version": "1.0",
        "description": "灾害救援遥感分析 SFT 数据集 - 基于 Terrabox 工具链",
        "created_at": "2026-03-29",
        "total_samples": len(samples),
        "disaster_categories": sorted(list({s["disaster_category"] for s in samples})),
        "task_types": sorted(list({s["task_type"] for s in samples})),
        "validation": {
            "description": "所有工具调用参数已通过源码核对，修复了5处设计错误",
            "fixes": FIXES,
            "verified_signatures": VERIFIED_SIGNATURES,
        },
        "samples": samples,
    }

    out_path = Path(__file__).parent.parent / "data" / "disaster_sft_dataset.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[OK] 写入 {out_path}")
    print(f"     样本总数  : {len(samples)}")
    print(f"     任务类型  : {len(TASK_TEMPLATES)} 类")
    print(f"     灾害类别  : {dataset['disaster_categories']}")

    # 简单参数验证
    print("\n[参数验证]")
    errors = 0
    for sample in samples:
        for call in sample["tool_calls"]:
            tool = call["tool"]
            if tool in VERIFIED_SIGNATURES:
                sig = VERIFIED_SIGNATURES[tool]
                for req in sig.get("required", []):
                    if req not in call["args"]:
                        # 允许 $ref 值作为占位符
                        val = call["args"].get(req, "MISSING")
                        if val == "MISSING":
                            print(f"  ✗ {sample['id']} step{call['step']} [{tool}] 缺少必填参数: {req}")
                            errors += 1
    if errors == 0:
        print("  ✓ 所有已知工具的必填参数均已提供")
    else:
        print(f"  共发现 {errors} 处参数缺失")


if __name__ == "__main__":
    main()
