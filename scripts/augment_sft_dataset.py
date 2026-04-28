"""
augment_sft_dataset.py — SFT 数据集多样性增强
===============================================
输入：data/disaster_sft_dataset_v2.json
输出：data/disaster_sft_dataset_v3.json

新增内容：
1. 为 8 种现有任务类型各增加 2 条新工具链模式（每种模式 4 个样本）
2. 新增 6 种全新任务类型，每类 2 种模式 × 4 个样本

新样本特点：
- 使用正确的工具参数名（与 ToolSpec 一致）
- 工具调用顺序合理，参数引用使用 $stepN.field 格式
- prompt 覆盖多种灾害场景/地点/分析侧重
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "disaster_sft_dataset_v2.json"
OUT_PATH = ROOT / "data" / "disaster_sft_dataset_v3.json"

# ─────────────────────────────────────────────────────────────────────────────
# 新样本定义（所有新增样本集中于此）
# ─────────────────────────────────────────────────────────────────────────────

NEW_SAMPLES: list[dict] = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. flood_detection  —  新模式 B：变化检测流水线
#    raster_diff → threshold_segmentation → count_pixels_condition
#    → pixel_area → draw_bboxes
# ─────────────────────────────────────────────────────────────────────────────
_FLOOD_DET_B = [
    ("flood_detection_05", "easy",
     "对比洪灾前后卫星影像，通过像素差分提取积水扩张区域，计算新增淹没面积并标注到图上。",
     "post_flood.png", "pre_flood.png", 0.5, 84320, 4194304, 0.0201, 21080.0),
    ("flood_detection_06", "easy",
     "受强降雨影响，某县城出现内涝。请利用前后两期卫星图像，提取积水范围并估算受淹面积。",
     "county_post.png", "county_pre.png", 0.5, 62140, 4194304, 0.0148, 15535.0),
    ("flood_detection_07", "medium",
     "台风导致沿海平原区域大规模洪涝。分析前后两景影像，识别洪水淹没范围变化，输出面积统计和标注图。",
     "coastal_post.png", "coastal_pre.png", 1.0, 41250, 1048576, 0.0393, 41250.0),
    ("flood_detection_08", "medium",
     "某城市排水系统超负荷导致局部积水。对比洪涝前后遥感影像，计算各积水区域面积并圈出位置。",
     "urban_post.png", "urban_pre.png", 0.3, 93600, 4194304, 0.0223, 8424.0),
]
for sid, diff, prompt, post, pre, gsd, cnt, total, ratio, area in _FLOOD_DET_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "flood_detection",
        "disaster_category": "flood", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "flood_diff.tif"},
             "sample_output": {"output_path": "flood_diff.tif", "stats": {"mean": 18.4, "std": 12.1}}},
            {"step": 2, "tool": "geo_raster.threshold_segmentation",
             "args": {"input_path": "flood_diff.tif", "threshold": 40, "output_path": "flood_mask.tif"},
             "sample_output": {"output_path": "flood_mask.tif"}},
            {"step": 3, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "flood_mask.tif", "lower": 254},
             "sample_output": {"count": cnt, "total": total, "ratio": ratio}},
            {"step": 4, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step3.count", "gsd_m": gsd},
             "sample_output": {"area_m2": area, "area_km2": round(area / 1e6, 4)}},
            {"step": 5, "tool": "geo_perception.draw_bboxes",
             "args": {"image": post, "bboxes": [{"x1": 120, "y1": 80, "x2": 420, "y2": 310}],
                      "output_path": "flood_annotated.png"},
             "sample_output": {"output_path": "flood_annotated.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 1b. flood_detection  —  新模式 C：遥感场景分类 + 目标检测
#     mscn_classify → sm3det_detect → bbox_area → draw_bboxes → add_text
# ─────────────────────────────────────────────────────────────────────────────
_FLOOD_DET_C = [
    ("flood_detection_09", "easy",
     "从单幅灾后高分卫星影像出发，先判断场景类型，再检测积水区域目标，统计并标注受淹位置。",
     "post.png", 0.5),
    ("flood_detection_10", "easy",
     "利用灾后遥感图像，通过场景分类确认洪涝类别后，对积水目标进行检测并计算总受淹面积。",
     "flood_scene.png", 0.5),
    ("flood_detection_11", "medium",
     "针对城市内涝高分图像，先分类场景，再用目标检测识别积水区，输出面积统计和标注结果。",
     "urban_flood.png", 0.3),
    ("flood_detection_12", "medium",
     "分析洪灾高分遥感影像，结合场景分类与目标检测，生成积水范围标注和面积报告。",
     "disaster_post.png", 1.0),
]
for sid, diff, prompt, img, gsd in _FLOOD_DET_C:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "flood_detection",
        "disaster_category": "flood", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.mscn_classify",
             "args": {"image": img},
             "sample_output": {"labels": {"water": 0.62, "urban": 0.28, "vegetation": 0.10},
                               "top_class": "water"}},
            {"step": 2, "tool": "geo_perception.sm3det_detect",
             "args": {"image": img, "score_threshold": 0.4},
             "sample_output": {"detections": [{"label": "water_body", "bbox": [80, 120, 380, 340], "score": 0.91},
                                              {"label": "water_body", "bbox": [420, 200, 680, 420], "score": 0.84}],
                               "count": 2}},
            {"step": 3, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step2.detections", "gsd_m": gsd},
             "sample_output": {"areas": [36900.0, 23760.0], "total_area_m2": 60660.0}},
            {"step": 4, "tool": "geo_perception.draw_bboxes",
             "args": {"image": img, "bboxes": "$step2.detections",
                      "output_path": "flood_detected.png"},
             "sample_output": {"output_path": "flood_detected.png"}},
            {"step": 5, "tool": "geo_perception.add_text",
             "args": {"image": "flood_detected.png",
                      "annotations": [{"text": "Flood area: 6.07 ha", "position": [10, 10]}],
                      "output_path": "flood_final.png"},
             "sample_output": {"output_path": "flood_final.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 2. earthquake_damage  —  新模式 B：变化检测 + 建筑受损统计
#    change_os_detect → sam2_segment → bbox_area → building_damage_stats
#    → draw_bboxes
# ─────────────────────────────────────────────────────────────────────────────
_EQ_B = [
    ("earthquake_damage_05", "medium",
     "利用地震前后双时相遥感影像，通过变化检测识别建筑倒塌区，结合语义分割统计受损面积，圈出重灾建筑群。",
     "eq_pre.png", "eq_post.png"),
    ("earthquake_damage_06", "medium",
     "对地震灾区卫星影像进行变化检测，提取建筑物损毁区域，评估破坏比率并标注受损建筑分布。",
     "before.png", "after.png"),
    ("earthquake_damage_07", "hard",
     "震后高密度城区受损评估：比对地震前后影像，识别完全倒塌和严重受损建筑，输出损毁率统计。",
     "urban_pre.png", "urban_post.png"),
    ("earthquake_damage_08", "hard",
     "利用多时相卫星图像，通过面向对象变化检测评估地震对居民区的破坏程度，生成损毁分级地图。",
     "residential_pre.png", "residential_post.png"),
]
for sid, diff, prompt, pre, post in _EQ_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "earthquake_damage",
        "disaster_category": "earthquake", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.change_os_detect",
             "args": {"pre_image": pre, "post_image": post, "mode": "building"},
             "sample_output": {"changes": [{"type": "destroyed", "bbox": [100, 120, 280, 300]},
                                           {"type": "damaged", "bbox": [320, 200, 480, 360]},
                                           {"type": "destroyed", "bbox": [520, 80, 700, 240]}],
                               "change_count": 3}},
            {"step": 2, "tool": "geo_perception.sam2_segment",
             "args": {"image": post},
             "sample_output": {"bboxes": [{"x1": 100, "y1": 120, "x2": 280, "y2": 300},
                                          {"x1": 320, "y1": 200, "x2": 480, "y2": 360}],
                               "masks_count": 5, "count": 5}},
            {"step": 3, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step2.bboxes", "gsd_m": 0.5},
             "sample_output": {"areas": [4500.0, 3600.0], "total_area_m2": 8100.0}},
            {"step": 4, "tool": "disaster_response.building_damage_stats",
             "args": {"change_raster_path": post, "buildings_geojson": "buildings.geojson",
                      "damage_threshold": 30},
             "sample_output": {"total_buildings": 186, "damaged_buildings": 42,
                               "damage_ratio": 0.226}},
            {"step": 5, "tool": "geo_perception.draw_bboxes",
             "args": {"image": post, "bboxes": "$step2.bboxes",
                      "output_path": "eq_damage_annotated.png"},
             "sample_output": {"output_path": "eq_damage_annotated.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 2b. earthquake_damage  —  新模式 C：多时相统计分析 + 热点检测
#     raster_diff → raster_stats → count_pixels_condition × 2
#     → percentage_change → getis_ord_gi_star → add_pois_layer
# ─────────────────────────────────────────────────────────────────────────────
_EQ_C = [
    ("earthquake_damage_09", "hard",
     "地震后多区域损毁统计：计算影像差值，统计中度和重度损毁像元比例，识别空间热点区域并叠加关键设施。",
     "pre.png", "post.png"),
    ("earthquake_damage_10", "hard",
     "震区损毁空间分析：通过影像差分量化损毁程度，提取损毁热点，分析损毁率与人口密集区的空间关系。",
     "pre_img.png", "post_img.png"),
    ("earthquake_damage_11", "hard",
     "对地震重灾区进行全域损毁统计，区分轻度与重度损毁，计算各级损毁面积占比，标注医院、学校等关键点。",
     "before_eq.png", "after_eq.png"),
    ("earthquake_damage_12", "hard",
     "震后城市受损空间热点分析：差分检测损毁区域，利用空间统计识别损毁聚集区，叠加避难所信息。",
     "city_pre.png", "city_post.png"),
]
for sid, diff, prompt, pre, post in _EQ_C:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "earthquake_damage",
        "disaster_category": "earthquake", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "eq_diff.tif"},
             "sample_output": {"output_path": "eq_diff.tif",
                               "stats": {"mean": 28.6, "std": 21.4, "max": 255.0}}},
            {"step": 2, "tool": "geo_raster.raster_stats",
             "args": {"input_path": "eq_diff.tif"},
             "sample_output": {"mean": 28.6, "std": 21.4, "min": 0.0, "max": 255.0,
                               "median": 22.1, "sum": 30064486.0}},
            {"step": 3, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "eq_diff.tif", "lower": 30, "upper": 80},
             "sample_output": {"count": 143200, "total": 4194304, "ratio": 0.0341}},
            {"step": 4, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "eq_diff.tif", "lower": 80},
             "sample_output": {"count": 62430, "total": 4194304, "ratio": 0.0149}},
            {"step": 5, "tool": "geo_statistics.percentage_change",
             "args": {"a": "$step3.count", "b": "$step4.count"},
             "sample_output": {"absolute_change": -80770.0, "percentage_change": -56.4}},
            {"step": 6, "tool": "geoanalysis.getis_ord_gi_star",
             "args": {"image_path": "eq_diff.tif", "output_path": "eq_hotspot.tif",
                      "weight_matrix": "queen"},
             "sample_output": {"output_path": "eq_hotspot.tif",
                               "hotspot_ratio": 0.083, "coldspot_ratio": 0.021}},
            {"step": 7, "tool": "osm_gis.add_pois_layer",
             "args": {"place_name": "灾区", "tags": {"amenity": ["hospital", "school"]},
                      "max_results": 30, "output_path": "eq_pois.geojson"},
             "sample_output": {"pois": [{"name": "市立医院", "type": "hospital",
                                         "lat": 22.31, "lon": 114.12}], "count": 12}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 3. landslide  —  新模式 B：坡度分析 + VLM 描述
#    calc_slope → raster_stats → count_pixels_condition → pixel_area
#    → vlm_analyze → draw_bboxes
# ─────────────────────────────────────────────────────────────────────────────
_LS_B = [
    ("landslide_05", "medium",
     "分析山区 DEM 数据，提取高坡度易发滑坡区域，计算危险区面积，结合影像进行 AI 视觉分析描述。",
     "dem.tif", "rgb.png"),
    ("landslide_06", "medium",
     "基于 DEM 坡度提取和遥感影像，识别滑坡易发区，评估坡度 >25° 的不稳定斜坡面积与分布。",
     "dem_data.tif", "hillside.png"),
    ("landslide_07", "hard",
     "结合 DEM 和高分影像，分析暴雨诱发滑坡的坡度条件，提取坡度超阈值危险区并进行影像描述分析。",
     "terrain.tif", "post_rain.png"),
    ("landslide_08", "hard",
     "山地灾害综合评估：从 DEM 提取坡度信息，量化高风险坡面面积，VLM 分析影像中的地表破坏特征。",
     "mountain_dem.tif", "mountain_rgb.png"),
]
for sid, diff, prompt, dem, rgb in _LS_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "landslide",
        "disaster_category": "landslide", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "disaster_response.calc_slope",
             "args": {"dem_path": dem, "output_path": "slope.tif", "unit": "degree"},
             "sample_output": {"output_path": "slope.tif",
                               "mean_slope_deg": 22.8, "max_slope_deg": 68.4}},
            {"step": 2, "tool": "geo_raster.raster_stats",
             "args": {"input_path": "slope.tif"},
             "sample_output": {"mean": 22.8, "std": 14.2, "min": 0.1,
                               "max": 68.4, "median": 19.6}},
            {"step": 3, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "slope.tif", "lower": 25},
             "sample_output": {"count": 187640, "total": 1048576, "ratio": 0.179}},
            {"step": 4, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step3.count", "gsd_m": 10},
             "sample_output": {"area_m2": 18764000.0, "area_km2": 18.764}},
            {"step": 5, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [rgb],
                      "prompt": "识别图中滑坡迹象：裸露土壤、植被破坏、堆积体和崩塌壁，描述危险区分布"},
             "sample_output": {"analysis": "图中可见大面积植被破坏和裸露土壤，坡面出现多处崩塌迹象，"
                                           "主要集中在中部斜坡区域，堆积体已延伸至沟谷底部。",
                               "bboxes": [{"x1": 180, "y1": 150, "x2": 480, "y2": 420}]}},
            {"step": 6, "tool": "geo_perception.draw_bboxes",
             "args": {"image": rgb, "bboxes": "$step5.bboxes",
                      "output_path": "landslide_risk.png"},
             "sample_output": {"output_path": "landslide_risk.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 3b. landslide  —  新模式 C：SAM2 分割 + 像素统计 + 差分验证
#     vlm_analyze → sam2_segment → bbox_area → raster_diff
#     → count_pixels_condition → draw_bboxes
# ─────────────────────────────────────────────────────────────────────────────
_LS_C = [
    ("landslide_09", "medium", "从灾后高分影像中，通过语义分割提取滑坡堆积体范围，结合前后差分图验证，计算受损面积。",
     "disaster.png", "pre.png"),
    ("landslide_10", "medium", "利用 AI 分割识别山区滑坡体，对比前后影像变化，输出滑坡面积和位置标注。",
     "post_landslide.png", "pre_landslide.png"),
    ("landslide_11", "hard", "震后山体滑坡识别：先 VLM 分析影像，再精确分割滑坡体，差分图验证，最终输出面积统计和标注图。",
     "seismic_ls.png", "before_eq.png"),
    ("landslide_12", "hard", "暴雨后多处滑坡综合分析：VLM 描述+SAM2 分割，结合差分检测，生成面积报表和圈注图。",
     "storm_post.png", "storm_pre.png"),
]
for sid, diff, prompt, post, pre in _LS_C:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "landslide",
        "disaster_category": "landslide", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [post],
                      "prompt": "识别图中的滑坡堆积体、裸露土壤和植被破坏区，给出受灾范围和坐标"},
             "sample_output": {"analysis": "图中发现 3 处明显滑坡堆积体，堆积物主要为岩石碎屑和土壤，"
                                           "植被覆盖破坏严重，影响范围约占图幅面积的 14%。",
                               "bboxes": [{"x1": 120, "y1": 200, "x2": 380, "y2": 450},
                                          {"x1": 500, "y1": 100, "x2": 720, "y2": 320}]}},
            {"step": 2, "tool": "geo_perception.sam2_segment",
             "args": {"image": post},
             "sample_output": {"bboxes": [{"x1": 120, "y1": 200, "x2": 380, "y2": 450},
                                          {"x1": 500, "y1": 100, "x2": 720, "y2": 320}],
                               "masks_count": 2, "count": 2}},
            {"step": 3, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step2.bboxes", "gsd_m": 2},
             "sample_output": {"areas": [208000.0, 88000.0], "total_area_m2": 296000.0}},
            {"step": 4, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "ls_diff.tif"},
             "sample_output": {"output_path": "ls_diff.tif",
                               "stats": {"mean": 24.1, "std": 18.6}}},
            {"step": 5, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "ls_diff.tif", "lower": 35},
             "sample_output": {"count": 74200, "total": 524288, "ratio": 0.1415}},
            {"step": 6, "tool": "geo_perception.draw_bboxes",
             "args": {"image": post, "bboxes": "$step2.bboxes",
                      "output_path": "landslide_annotated.png"},
             "sample_output": {"output_path": "landslide_annotated.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 4. fire_detection  —  新模式 B：植被覆盖变化 + FRP 复合分析
#    calculate_fvc → raster_diff → count_above_threshold → pixel_area
#    → vlm_analyze → draw_bboxes
# ─────────────────────────────────────────────────────────────────────────────
_FIRE_B = [
    ("fire_detection_05", "medium",
     "分析林火前后植被覆盖变化，计算植被损失面积，结合 VLM 描述影像中的火烧迹地分布特征。",
     "nir_pre.tif", "red_pre.tif", "nir_post.tif", "red_post.tif", "rgb_post.png"),
    ("fire_detection_06", "medium",
     "草原火灾遥感评估：通过植被覆盖差分量化过火面积，AI 分析影像中的火烧程度和植被恢复潜力。",
     "nir1.tif", "red1.tif", "nir2.tif", "red2.tif", "grassland_post.png"),
    ("fire_detection_07", "hard",
     "多源数据综合火灾评估：计算前后植被覆盖变化量，提取重度过火区，VLM 描述典型地物特征。",
     "nir_before.tif", "red_before.tif", "nir_after.tif", "red_after.tif", "fire_rgb.png"),
    ("fire_detection_08", "hard",
     "林区火灾精确评估：植被覆盖前后差分检测过火边界，统计轻中重度过火面积，影像描述分析。",
     "forest_nir_pre.tif", "forest_red_pre.tif",
     "forest_nir_post.tif", "forest_red_post.tif", "forest_post.png"),
]
for sid, diff, prompt, nir_pre, red_pre, nir_post, red_post, rgb in _FIRE_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "fire_detection",
        "disaster_category": "fire", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.calculate_fvc",
             "args": {"nir_path": nir_pre, "red_path": red_pre, "output_path": "fvc_pre.tif"},
             "sample_output": {"output_path": "fvc_pre.tif", "mean_fvc": 0.68}},
            {"step": 2, "tool": "geo_raster.calculate_fvc",
             "args": {"nir_path": nir_post, "red_path": red_post, "output_path": "fvc_post.tif"},
             "sample_output": {"output_path": "fvc_post.tif", "mean_fvc": 0.29}},
            {"step": 3, "tool": "geo_raster.raster_diff",
             "args": {"path_a": "fvc_pre.tif", "path_b": "fvc_post.tif",
                      "output_path": "fvc_diff.tif"},
             "sample_output": {"output_path": "fvc_diff.tif",
                               "stats": {"mean": 0.39, "std": 0.18}}},
            {"step": 4, "tool": "geo_raster.count_above_threshold",
             "args": {"input_path": "fvc_diff.tif", "threshold": 0.3},
             "sample_output": {"count": 182400, "total": 1048576, "ratio": 0.174}},
            {"step": 5, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step4.count", "gsd_m": 30},
             "sample_output": {"area_m2": 164160000.0, "area_km2": 164.16}},
            {"step": 6, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [rgb],
                      "prompt": "识别图中的火烧迹地、过火边界和残留植被，描述过火强度和分布"},
             "sample_output": {"analysis": "图中可见大面积黑色火烧迹地，边界清晰，中心区域为重度过火，"
                                           "边缘出现轻度过火区，部分耐火植被仍有残留。",
                               "bboxes": [{"x1": 50, "y1": 80, "x2": 650, "y2": 680}]}},
            {"step": 7, "tool": "geo_perception.draw_bboxes",
             "args": {"image": rgb, "bboxes": "$step6.bboxes",
                      "output_path": "fire_burned_annotated.png"},
             "sample_output": {"output_path": "fire_burned_annotated.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 4b. fire_detection  —  新模式 C：热点空间分布分析
#     identify_fire_prone_areas → raster_stats → count_pixels_condition
#     → pixel_area → getis_ord_gi_star → add_pois_layer
# ─────────────────────────────────────────────────────────────────────────────
_FIRE_C = [
    ("fire_detection_09", "medium",
     "基于 FRP 数据识别高风险火点区域，统计热点覆盖面积，进行空间热点分析并标注周边居民点。",
     ["frp_t1.tif", "frp_t2.tif", "frp_t3.tif"]),
    ("fire_detection_10", "medium",
     "分析 MODIS FRP 时序数据，提取高火险区，计算火险面积比例，识别火灾聚集热点并标注基础设施。",
     ["modis_frp_1.tif", "modis_frp_2.tif", "modis_frp_3.tif"]),
    ("fire_detection_11", "hard",
     "连续多日 FRP 产品综合分析：识别持续高火险区域，统计过火面积，空间热点分析，叠加防火道和水源信息。",
     ["day1_frp.tif", "day2_frp.tif", "day3_frp.tif"]),
    ("fire_detection_12", "hard",
     "森林火险综合评估：多时次 FRP 提取高风险区，面积统计，Getis-Ord 热点识别，叠加防护区边界信息。",
     ["frp_a.tif", "frp_b.tif", "frp_c.tif"]),
]
for sid, diff, prompt, frp_paths in _FIRE_C:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "fire_detection",
        "disaster_category": "fire", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.identify_fire_prone_areas",
             "args": {"input_paths": frp_paths, "output_path": "fire_prone.tif",
                      "percentile": 85},
             "sample_output": {"output_path": "fire_prone.tif",
                               "threshold_value": 68.4, "prone_pixel_count": 143200}},
            {"step": 2, "tool": "geo_raster.raster_stats",
             "args": {"input_path": "fire_prone.tif"},
             "sample_output": {"mean": 84.2, "std": 31.6, "min": 0.0,
                               "max": 255.0, "median": 91.4}},
            {"step": 3, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "fire_prone.tif", "lower": 127},
             "sample_output": {"count": 143200, "total": 4194304, "ratio": 0.0341}},
            {"step": 4, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step3.count", "gsd_m": 500},
             "sample_output": {"area_m2": 35800000000.0, "area_km2": 35800.0}},
            {"step": 5, "tool": "geoanalysis.getis_ord_gi_star",
             "args": {"image_path": "fire_prone.tif", "output_path": "fire_hotspot.tif",
                      "weight_matrix": "queen"},
             "sample_output": {"output_path": "fire_hotspot.tif",
                               "hotspot_ratio": 0.062, "coldspot_ratio": 0.018}},
            {"step": 6, "tool": "osm_gis.add_pois_layer",
             "args": {"place_name": "火险区", "tags": {"amenity": ["fire_station"],
                                                      "landuse": ["residential"]},
                      "max_results": 20, "output_path": "fire_pois.geojson"},
             "sample_output": {"pois": [{"name": "消防大队", "type": "fire_station",
                                         "lat": 22.4, "lon": 114.2}], "count": 8}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 5. building_damage_assess  —  新模式 B：VLM + 条带检测 + 面积统计
#    vlm_analyze → strip_rcnn_detect → bbox_area → draw_bboxes → add_text
# ─────────────────────────────────────────────────────────────────────────────
_BDA_B = [
    ("building_damage_assess_05", "medium",
     "利用灾后高分影像，先通过 VLM 描述受损区域，再用条带检测器精确定位受损建筑，统计损毁数量和面积。",
     "post_disaster.png"),
    ("building_damage_assess_06", "medium",
     "震后高分卫星图像建筑损毁评估：AI 视觉分析定位受损区，条带检测器精确提取倒塌建筑并统计面积。",
     "eq_post_hr.png"),
    ("building_damage_assess_07", "hard",
     "台风过后建筑损毁精细化评估：VLM 初步分析定位，条带检测精确提取受损建筑群，生成损毁面积报告。",
     "typhoon_post.png"),
    ("building_damage_assess_08", "hard",
     "城市灾后建筑损毁清查：先 AI 视觉分析，再专项检测器识别受损建筑，计算总损毁面积并圈注。",
     "urban_damage.png"),
]
for sid, diff, prompt, img in _BDA_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "building_damage_assess",
        "disaster_category": "earthquake", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [img],
                      "prompt": "识别图中受损和倒塌的建筑物，描述损毁程度、分布区域和损毁类型"},
             "sample_output": {"analysis": "图中发现多处建筑屋顶塌陷和墙体倒塌，主要集中在图像中部密集居住区，"
                                           "约有 30% 的建筑出现不同程度损毁。",
                               "bboxes": [{"x1": 200, "y1": 180, "x2": 580, "y2": 460}]}},
            {"step": 2, "tool": "geo_perception.strip_rcnn_detect",
             "args": {"image": img, "score_threshold": 0.45},
             "sample_output": {"detections_bboxes": [[120, 140, 280, 290],
                                                      [340, 200, 500, 360],
                                                      [560, 120, 720, 280]],
                               "detection_count": 8}},
            {"step": 3, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step2.detections_bboxes", "gsd_m": 0.5},
             "sample_output": {"areas": [3500.0, 4000.0, 3200.0],
                               "total_area_m2": 10700.0}},
            {"step": 4, "tool": "geo_perception.draw_bboxes",
             "args": {"image": img, "bboxes": "$step2.detections_bboxes",
                      "output_path": "building_damage_det.png"},
             "sample_output": {"output_path": "building_damage_det.png"}},
            {"step": 5, "tool": "geo_perception.add_text",
             "args": {"image": "building_damage_det.png",
                      "annotations": [{"text": "Damaged: 8 buildings | Area: 1.07 ha",
                                       "position": [10, 10]}],
                      "output_path": "building_damage_final.png"},
             "sample_output": {"output_path": "building_damage_final.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 5b. building_damage_assess  —  新模式 C：变化检测 + 统计
#     change_os_detect → raster_diff → threshold_segmentation
#     → count_pixels_condition → pixel_area
# ─────────────────────────────────────────────────────────────────────────────
_BDA_C = [
    ("building_damage_assess_09", "hard",
     "利用双时相遥感影像进行建筑物损毁检测：变化检测定位损毁区，差值分析量化损毁程度，统计受损面积比例。",
     "pre.png", "post.png"),
    ("building_damage_assess_10", "hard",
     "地震建筑损毁全域统计：变化检测与差分复合分析，阈值分割提取损毁范围，输出损毁面积数据。",
     "before.png", "after.png"),
    ("building_damage_assess_11", "hard",
     "灾后建筑损毁率评估：对比前后卫星影像，通过变化检测和阈值分割量化损毁面积，评估总体损毁比例。",
     "pre_disaster.png", "post_disaster.png"),
    ("building_damage_assess_12", "hard",
     "高密度城区灾后损毁精确量化：双时相变化检测 + 差分分析，提取不同损毁等级面积比例。",
     "city_before.png", "city_after.png"),
]
for sid, diff, prompt, pre, post in _BDA_C:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "building_damage_assess",
        "disaster_category": "earthquake", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.change_os_detect",
             "args": {"pre_image": pre, "post_image": post, "mode": "building"},
             "sample_output": {"changes": [{"type": "destroyed", "bbox": [100, 120, 280, 300]},
                                           {"type": "damaged", "bbox": [320, 200, 480, 360]}],
                               "change_count": 6}},
            {"step": 2, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "bldg_diff.tif"},
             "sample_output": {"output_path": "bldg_diff.tif",
                               "stats": {"mean": 31.2, "std": 24.8}}},
            {"step": 3, "tool": "geo_raster.threshold_segmentation",
             "args": {"input_path": "bldg_diff.tif", "threshold": 40,
                      "output_path": "bldg_damage_mask.tif"},
             "sample_output": {"output_path": "bldg_damage_mask.tif"}},
            {"step": 4, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "bldg_damage_mask.tif", "lower": 254},
             "sample_output": {"count": 94320, "total": 4194304, "ratio": 0.0225}},
            {"step": 5, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step4.count", "gsd_m": 0.5},
             "sample_output": {"area_m2": 23580.0, "area_km2": 0.0236}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 6. typhoon_port_damage  —  新模式 B：SM3Det 船舶目标检测
#    get_raster_info → vlm_analyze → sm3det_detect → bbox_area
#    → draw_bboxes → add_text
# ─────────────────────────────────────────────────────────────────────────────
_TPD_B = [
    ("typhoon_port_damage_05", "medium",
     "台风过境后对港口进行灾害评估：获取影像基本信息，AI 视觉分析，多类目标检测统计受损船只和设施。",
     "port_post.png"),
    ("typhoon_port_damage_06", "medium",
     "台风灾后港区遥感评估：多目标检测识别受损船舶和码头，统计受损面积，生成标注报告。",
     "harbor_post.png"),
    ("typhoon_port_damage_07", "hard",
     "超强台风过后港口综合损毁评估：影像元数据获取、AI 视觉分析、船舶和建筑检测、面积统计与圈注。",
     "typhoon_port.png"),
    ("typhoon_port_damage_08", "hard",
     "重大港口台风损毁精细评估：多目标检测提取受损目标，面积统计，生成含损毁数量和面积的标注图。",
     "major_port_post.png"),
]
for sid, diff, prompt, img in _TPD_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "typhoon_port_damage",
        "disaster_category": "typhoon", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.get_raster_info",
             "args": {"raster_path": img},
             "sample_output": {"width": 1024, "height": 1024, "bands": 3,
                               "gsd_m": 0.5, "crs": "EPSG:32649"}},
            {"step": 2, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [img],
                      "prompt": "对比台风前后港口影像，识别搁浅船只、倒塌仓库、受损码头和新增碎片"},
             "sample_output": {"analysis": "港区中部发现 4 艘船只搁浅或移位，仓库区屋顶大面积损毁，"
                                           "码头设施出现倒塌，碎片散布区域约占港区 12%。",
                               "bboxes": [{"x1": 150, "y1": 200, "x2": 450, "y2": 380}]}},
            {"step": 3, "tool": "geo_perception.sm3det_detect",
             "args": {"image": img, "score_threshold": 0.4},
             "sample_output": {"detections": [{"label": "ship", "bbox": [160, 210, 320, 350], "score": 0.92},
                                              {"label": "ship", "bbox": [380, 180, 540, 300], "score": 0.87},
                                              {"label": "building", "bbox": [600, 400, 780, 560], "score": 0.76}],
                               "count": 6}},
            {"step": 4, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step3.detections", "gsd_m": "$step1.gsd_m"},
             "sample_output": {"areas": [4200.0, 3840.0, 5040.0],
                               "total_area_m2": 13080.0}},
            {"step": 5, "tool": "geo_perception.draw_bboxes",
             "args": {"image": img, "bboxes": "$step3.detections",
                      "output_path": "port_damage_det.png"},
             "sample_output": {"output_path": "port_damage_det.png"}},
            {"step": 6, "tool": "geo_perception.add_text",
             "args": {"image": "port_damage_det.png",
                      "annotations": [{"text": "Detected 6 damaged targets | Area: 1.31 ha",
                                       "position": [10, 10]}],
                      "output_path": "port_damage_final.png"},
             "sample_output": {"output_path": "port_damage_final.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 7. flood_rescue_routing  —  新模式 B：可达性分析 + 路径规划
#    raster_diff → threshold_segmentation → accessibility_map
#    → get_area_boundary → compute_route_dist → add_pois_layer
# ─────────────────────────────────────────────────────────────────────────────
_FRR_B = [
    ("flood_rescue_routing_05", "hard",
     "洪灾救援路径规划：基于积水掩膜生成道路可达性图，获取受灾区边界，规划从救援中心到安置点的最优路径并标注关键设施。",
     "post.png", "pre.png", "救援中心", "安置区"),
    ("flood_rescue_routing_06", "hard",
     "洪涝区应急救援路径优化：差分图提取积水范围，可达性评估，基于 OSM 路网规划最优救援路径。",
     "flood_post.png", "flood_pre.png", "消防队", "受灾村庄"),
    ("flood_rescue_routing_07", "hard",
     "洪灾后救援可达性评估：积水范围检测，生成可达性地图，规划医疗救援队到各受灾点的最优路线。",
     "disaster_post.png", "disaster_pre.png", "医院", "灾民聚集点"),
    ("flood_rescue_routing_08", "hard",
     "大规模洪灾救援协调：基于遥感积水数据评估路网通行情况，优化多支救援队伍的行动路径。",
     "flood_scene_post.png", "flood_scene_pre.png", "救援基地", "受灾社区"),
]
for sid, diff, prompt, post, pre, origin, dest in _FRR_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "flood_rescue_routing",
        "disaster_category": "flood", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "flood_diff.tif"},
             "sample_output": {"output_path": "flood_diff.tif",
                               "stats": {"mean": 22.4, "std": 16.8}}},
            {"step": 2, "tool": "geo_raster.threshold_segmentation",
             "args": {"input_path": "flood_diff.tif", "threshold": 35,
                      "output_path": "flood_mask.tif"},
             "sample_output": {"output_path": "flood_mask.tif"}},
            {"step": 3, "tool": "disaster_response.accessibility_map",
             "args": {"road_mask_path": "flood_mask.tif",
                      "poi_locations": [[22.31, 114.12], [22.35, 114.18]],
                      "output_path": "access_map.tif", "travel_speed_kmh": 20},
             "sample_output": {"output_path": "access_map.tif",
                               "accessible_ratio": 0.58, "mean_time_min": 18.4}},
            {"step": 4, "tool": "osm_gis.get_area_boundary",
             "args": {"place_name": "受灾区", "output_path": "area_boundary.geojson"},
             "sample_output": {"output_path": "area_boundary.geojson",
                               "area_km2": 42.6}},
            {"step": 5, "tool": "osm_gis.compute_route_dist",
             "args": {"origin": origin, "destination": dest, "travel_mode": "driving"},
             "sample_output": {"distance_km": 12.4, "time_min": 38,
                               "route_geojson": {"type": "Feature"}}},
            {"step": 6, "tool": "osm_gis.add_pois_layer",
             "args": {"place_name": "受灾区", "tags": {"amenity": ["hospital", "shelter"]},
                      "max_results": 20, "output_path": "rescue_pois.geojson"},
             "sample_output": {"pois": [{"name": "区立医院", "type": "hospital",
                                         "lat": 22.33, "lon": 114.15}], "count": 9}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# 新任务类型
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 8. 新任务类型：drone_damage_survey（无人机灾情勘察）
#    模式 A：遥测 → 抽帧 → 关键帧筛选 → VLM 分析 → 标注
# ─────────────────────────────────────────────────────────────────────────────
_DRONE_A = [
    ("drone_damage_survey_01", "medium",
     "利用无人机视频对灾区进行低空勘察：解析飞行遥测数据，自动提取并筛选关键帧，AI 分析受损情况并标注。",
     "survey.mp4", "telemetry.csv"),
    ("drone_damage_survey_02", "medium",
     "无人机低空侦察灾区：从飞行视频中提取关键帧，通过 AI 视觉分析识别建筑损毁和道路阻断情况。",
     "drone_video.mp4", "flight_log.csv"),
    ("drone_damage_survey_03", "hard",
     "震后无人机快速勘察：解析 GPS 轨迹，筛选最具代表性帧序列，VLM 评估建筑损毁程度并圈注受灾位置。",
     "post_eq_drone.mp4", "eq_telemetry.csv"),
    ("drone_damage_survey_04", "hard",
     "洪灾无人机灾情调查：从长时视频提取关键帧，AI 分析积水范围和受损建筑，生成圈注报告。",
     "flood_survey.mp4", "flood_telemetry.csv"),
]
for sid, diff, prompt, video, telem in _DRONE_A:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "drone_damage_survey",
        "disaster_category": "multi_hazard", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "drone_video.parse_telemetry",
             "args": {"telemetry_path": telem, "format": "dji_csv"},
             "sample_output": {"waypoints": [{"lat": 22.31, "lon": 114.12, "alt_m": 120,
                                              "heading_deg": 45, "timestamp": "2024-08-01T08:00:00"}],
                               "total_distance_km": 8.4, "flight_duration_min": 22}},
            {"step": 2, "tool": "drone_video.extract_frames",
             "args": {"video_path": video, "output_dir": "/tmp/drone_frames",
                      "fps": 1, "max_frames": 120},
             "sample_output": {"frames": ["/tmp/drone_frames/frame_001.jpg",
                                          "/tmp/drone_frames/frame_060.jpg"],
                               "total_frames": 86}},
            {"step": 3, "tool": "drone_video.key_frame_select",
             "args": {"frame_paths": "$step2.frames", "method": "scene_change",
                      "max_frames": 12},
             "sample_output": {"selected": ["/tmp/drone_frames/frame_008.jpg",
                                            "/tmp/drone_frames/frame_034.jpg",
                                            "/tmp/drone_frames/frame_071.jpg"],
                               "count": 12}},
            {"step": 4, "tool": "geo_perception.vlm_analyze",
             "args": {"images": "$step3.selected",
                      "prompt": "逐一分析无人机帧，识别建筑受损、道路阻断、积水或滑坡迹象，给出坐标范围"},
             "sample_output": {"analysis": "第 8 帧：严重损毁建筑 3 栋，屋顶坍塌；第 34 帧：道路被滑坡堆积物阻断约 50 米；"
                                           "第 71 帧：大面积积水，居民区淹没约 30%。",
                               "bboxes": [{"x1": 120, "y1": 80, "x2": 380, "y2": 290}]}},
            {"step": 5, "tool": "geo_perception.draw_bboxes",
             "args": {"image": "/tmp/drone_frames/frame_008.jpg",
                      "bboxes": "$step4.bboxes", "output_path": "/tmp/drone_annotated.png"},
             "sample_output": {"output_path": "/tmp/drone_annotated.png"}},
        ]
    })

# 模式 B：抽帧 → 时序差分 → 目标检测 → 面积统计 → 标注
_DRONE_B = [
    ("drone_damage_survey_05", "medium",
     "无人机视频差分分析：提取飞行视频帧，计算时序差分找出变化区域，对受损目标进行检测统计。",
     "survey2.mp4"),
    ("drone_damage_survey_06", "medium",
     "基于时序帧差分检测灾区变化：无人机影像抽帧→差分→目标检测，输出受损目标数量和标注图。",
     "drone2.mp4"),
    ("drone_damage_survey_07", "hard",
     "复杂灾区无人机低空勘察：多帧差分检测异常区域，专项目标检测器识别受损建筑，面积统计与标注。",
     "complex_survey.mp4"),
    ("drone_damage_survey_08", "hard",
     "大范围灾区无人机快速普查：时序差分定位变化区，多类目标检测，输出受损目标分类统计和标注。",
     "wide_survey.mp4"),
]
for sid, diff, prompt, video in _DRONE_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "drone_damage_survey",
        "disaster_category": "multi_hazard", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "drone_video.extract_frames",
             "args": {"video_path": video, "output_dir": "/tmp/frames2",
                      "fps": 2, "max_frames": 60},
             "sample_output": {"frames": ["/tmp/frames2/frame_000.jpg",
                                          "/tmp/frames2/frame_030.jpg"],
                               "total_frames": 52}},
            {"step": 2, "tool": "drone_video.temporal_diff",
             "args": {"frame_a_path": "/tmp/frames2/frame_000.jpg",
                      "frame_b_path": "/tmp/frames2/frame_030.jpg",
                      "output_path": "/tmp/diff_map.png", "threshold": 30},
             "sample_output": {"output_path": "/tmp/diff_map.png",
                               "change_ratio": 0.28, "changed_pixels": 73728}},
            {"step": 3, "tool": "geo_perception.sm3det_detect",
             "args": {"image": "/tmp/frames2/frame_030.jpg", "score_threshold": 0.45},
             "sample_output": {"detections": [{"label": "building", "bbox": [150, 200, 320, 380],
                                               "score": 0.88},
                                              {"label": "vehicle", "bbox": [420, 100, 540, 210],
                                               "score": 0.75}],
                               "count": 4}},
            {"step": 4, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step3.detections", "gsd_m": 0.15},
             "sample_output": {"areas": [714.0, 475.0], "total_area_m2": 1189.0}},
            {"step": 5, "tool": "geo_perception.draw_bboxes",
             "args": {"image": "/tmp/frames2/frame_030.jpg",
                      "bboxes": "$step3.detections",
                      "output_path": "/tmp/drone_damage_det.png"},
             "sample_output": {"output_path": "/tmp/drone_damage_det.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 9. 新任务类型：infrastructure_damage（道路/桥梁损毁评估）
#    模式 A：变化检测 → 目标检测 → 面积统计 → 可达性 → 设施标注
# ─────────────────────────────────────────────────────────────────────────────
_INFRA_A = [
    ("infrastructure_damage_01", "medium",
     "灾后交通基础设施损毁评估：变化检测定位受损路段，目标检测识别损毁设施，评估交通可达性影响。",
     "road_pre.png", "road_post.png"),
    ("infrastructure_damage_02", "medium",
     "地震后道路损毁快速评估：双时相变化检测识别损毁路段，统计阻断面积，分析救援可达性。",
     "bridge_pre.png", "bridge_post.png"),
    ("infrastructure_damage_03", "hard",
     "洪灾公路网损毁全面评估：变化检测识别受损路段，条带检测定位损毁桥梁，可达性分析与设施标注。",
     "highway_pre.png", "highway_post.png"),
    ("infrastructure_damage_04", "hard",
     "台风后基础设施损毁精细评估：多源信息融合，变化检测+目标检测，路网可达性评估与救援设施定位。",
     "infra_pre.png", "infra_post.png"),
]
for sid, diff, prompt, pre, post in _INFRA_A:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "infrastructure_damage",
        "disaster_category": "multi_hazard", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.change_os_detect",
             "args": {"pre_image": pre, "post_image": post, "mode": "road"},
             "sample_output": {"changes": [{"type": "blocked", "bbox": [120, 80, 340, 160]},
                                           {"type": "damaged", "bbox": [500, 300, 720, 420]}],
                               "change_count": 4}},
            {"step": 2, "tool": "geo_perception.strip_rcnn_detect",
             "args": {"image": post, "score_threshold": 0.45},
             "sample_output": {"detections_bboxes": [[120, 80, 340, 160],
                                                      [500, 300, 720, 420]],
                               "detection_count": 3}},
            {"step": 3, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step1.change_count", "gsd_m": 1.0},
             "sample_output": {"area_m2": 4.0, "area_km2": 0.000004}},
            {"step": 4, "tool": "disaster_response.accessibility_map",
             "args": {"road_mask_path": post,
                      "poi_locations": [[22.31, 114.12]],
                      "output_path": "/tmp/road_access.tif",
                      "travel_speed_kmh": 30},
             "sample_output": {"output_path": "/tmp/road_access.tif",
                               "accessible_ratio": 0.72, "mean_time_min": 24.6}},
            {"step": 5, "tool": "osm_gis.add_pois_layer",
             "args": {"place_name": "受灾区域",
                      "tags": {"amenity": ["hospital"], "highway": ["primary"]},
                      "max_results": 15, "output_path": "/tmp/road_pois.geojson"},
             "sample_output": {"pois": [{"name": "交通应急指挥中心",
                                         "type": "emergency", "lat": 22.32, "lon": 114.14}],
                               "count": 7}},
            {"step": 6, "tool": "geo_perception.draw_bboxes",
             "args": {"image": post, "bboxes": "$step2.detections_bboxes",
                      "output_path": "/tmp/infra_damage_annotated.png"},
             "sample_output": {"output_path": "/tmp/infra_damage_annotated.png"}},
        ]
    })

# 模式 B：VLM + SAM2 + 可达性 + 路径规划
_INFRA_B = [
    ("infrastructure_damage_05", "medium",
     "灾后道路通行能力评估：VLM 分析受损路段，语义分割提取阻断区域，规划绕行最优路径。",
     "road_disaster.png", "道路起点", "目的地"),
    ("infrastructure_damage_06", "medium",
     "震后桥梁损毁 + 绕行路径规划：AI 分析受损桥梁，分割提取损毁范围，规划应急绕行路线。",
     "bridge_damage.png", "桥梁A侧", "桥梁B侧"),
    ("infrastructure_damage_07", "hard",
     "洪灾淹没道路综合评估：AI 识别受损路段，分割损毁范围，路网可达性分析，规划最优救援路径。",
     "flooded_road.png", "救援基地", "受灾区"),
    ("infrastructure_damage_08", "hard",
     "台风后基础设施综合评估：多点路段损毁识别，精细分割，路网可达性与绕行路径综合规划。",
     "typhoon_road.png", "应急指挥部", "受灾社区"),
]
for sid, diff, prompt, img, origin, dest in _INFRA_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "infrastructure_damage",
        "disaster_category": "multi_hazard", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [img],
                      "prompt": "识别图中损毁道路、塌方阻断段、积水路面和损坏桥梁，给出损毁位置和程度"},
             "sample_output": {"analysis": "图中发现 2 处明显道路阻断：中部约 80 米路段被滑坡物覆盖，"
                                           "右侧桥梁出现局部塌陷。",
                               "bboxes": [{"x1": 200, "y1": 150, "x2": 500, "y2": 300},
                                          {"x1": 600, "y1": 200, "x2": 800, "y2": 380}]}},
            {"step": 2, "tool": "geo_perception.sam2_segment",
             "args": {"image": img},
             "sample_output": {"bboxes": [{"x1": 200, "y1": 150, "x2": 500, "y2": 300},
                                          {"x1": 600, "y1": 200, "x2": 800, "y2": 380}],
                               "masks_count": 2, "count": 2}},
            {"step": 3, "tool": "geo_perception.bbox_area",
             "args": {"bboxes": "$step2.bboxes", "gsd_m": 0.5},
             "sample_output": {"areas": [15000.0, 9000.0], "total_area_m2": 24000.0}},
            {"step": 4, "tool": "osm_gis.compute_route_dist",
             "args": {"origin": origin, "destination": dest, "travel_mode": "driving"},
             "sample_output": {"distance_km": 16.8, "time_min": 42,
                               "route_geojson": {"type": "Feature"}}},
            {"step": 5, "tool": "geo_perception.draw_bboxes",
             "args": {"image": img, "bboxes": "$step2.bboxes",
                      "output_path": "/tmp/road_damage_annotated.png"},
             "sample_output": {"output_path": "/tmp/road_damage_annotated.png"}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 10. 新任务类型：vegetation_recovery（植被灾后恢复监测）
#     模式 A：多时次 FVC 计算 → 差分 → 统计 → 趋势分析
# ─────────────────────────────────────────────────────────────────────────────
_VR_A = [
    ("vegetation_recovery_01", "medium",
     "林火后植被恢复监测：计算灾后多时次植被覆盖度，分析恢复趋势和速率，评估植被恢复进展。",
     "nir_m1.tif", "red_m1.tif", "nir_m6.tif", "red_m6.tif",
     [0.18, 0.24, 0.31, 0.38, 0.44, 0.51]),
    ("vegetation_recovery_02", "medium",
     "草地火灾后植被恢复评估：对比火后 1 个月和 6 个月 FVC，计算恢复率，分析恢复趋势。",
     "grass_nir_t1.tif", "grass_red_t1.tif", "grass_nir_t6.tif", "grass_red_t6.tif",
     [0.12, 0.19, 0.27, 0.35, 0.42, 0.49]),
    ("vegetation_recovery_03", "hard",
     "地震后植被受损与自然恢复过程分析：多时相 FVC 序列追踪植被恢复动态，Mann-Kendall 趋势检验。",
     "eq_nir_t1.tif", "eq_red_t1.tif", "eq_nir_t12.tif", "eq_red_t12.tif",
     [0.08, 0.14, 0.22, 0.31, 0.38, 0.45]),
    ("vegetation_recovery_04", "hard",
     "洪涝淹没区植被恢复时序分析：从植被覆盖度序列评估积水消退后植被复苏速率和空间分布。",
     "flood_nir_t0.tif", "flood_red_t0.tif", "flood_nir_t6.tif", "flood_red_t6.tif",
     [0.05, 0.12, 0.21, 0.32, 0.41, 0.48]),
]
for sid, diff, prompt, n1, r1, n6, r6, values in _VR_A:
    pct_change = round((values[-1] - values[0]) / values[0] * 100, 1)
    NEW_SAMPLES.append({
        "id": sid, "task_type": "vegetation_recovery",
        "disaster_category": "fire", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.calculate_fvc",
             "args": {"nir_path": n1, "red_path": r1, "output_path": "/tmp/fvc_t1.tif"},
             "sample_output": {"output_path": "/tmp/fvc_t1.tif", "mean_fvc": values[0]}},
            {"step": 2, "tool": "geo_raster.calculate_fvc",
             "args": {"nir_path": n6, "red_path": r6, "output_path": "/tmp/fvc_t6.tif"},
             "sample_output": {"output_path": "/tmp/fvc_t6.tif", "mean_fvc": values[-1]}},
            {"step": 3, "tool": "geo_raster.raster_diff",
             "args": {"path_a": "/tmp/fvc_t6.tif", "path_b": "/tmp/fvc_t1.tif",
                      "output_path": "/tmp/fvc_recovery.tif"},
             "sample_output": {"output_path": "/tmp/fvc_recovery.tif",
                               "stats": {"mean": round(values[-1]-values[0], 3), "std": 0.12}}},
            {"step": 4, "tool": "geo_raster.raster_stats",
             "args": {"input_path": "/tmp/fvc_recovery.tif"},
             "sample_output": {"mean": round(values[-1]-values[0], 3), "std": 0.12,
                               "min": -0.05, "max": 0.48, "median": round(values[-1]-values[0]-0.02, 3)}},
            {"step": 5, "tool": "geo_statistics.percentage_change",
             "args": {"a": values[0], "b": values[-1]},
             "sample_output": {"absolute_change": round(values[-1]-values[0], 3),
                               "percentage_change": pct_change}},
            {"step": 6, "tool": "geoanalysis.mann_kendall_test",
             "args": {"values": values},
             "sample_output": {"trend": "increasing", "p_value": 0.003,
                               "tau": 0.87, "slope": round((values[-1]-values[0])/5, 4)}},
        ]
    })

# 模式 B：批量统计 → 线性趋势 → 变点检测 → 超阈值帧数统计
_VR_B = [
    ("vegetation_recovery_05", "medium",
     "灾后植被恢复多时次批量统计：批量计算各期 FVC 均值，拟合恢复线性趋势，检测关键恢复节点。",
     ["fvc_t0.tif", "fvc_t2.tif", "fvc_t4.tif", "fvc_t6.tif", "fvc_t8.tif"],
     [0.15, 0.21, 0.30, 0.39, 0.46]),
    ("vegetation_recovery_06", "medium",
     "火后植被恢复趋势分析：多期遥感数据批量统计，线性趋势拟合，识别加速恢复时间节点。",
     ["fire_fvc_m0.tif", "fire_fvc_m2.tif", "fire_fvc_m4.tif", "fire_fvc_m6.tif"],
     [0.10, 0.18, 0.28, 0.38]),
    ("vegetation_recovery_07", "hard",
     "长时序植被恢复动态监测：多期 FVC 批量统计，趋势分析，变点检测识别恢复阶段，统计达标期数。",
     ["fvc_y1.tif", "fvc_y2.tif", "fvc_y3.tif", "fvc_y4.tif", "fvc_y5.tif"],
     [0.18, 0.28, 0.37, 0.44, 0.52]),
    ("vegetation_recovery_08", "hard",
     "震后植被恢复 5 年时序分析：批量 FVC 统计、趋势分解、变点识别，综合评估植被恢复进程。",
     ["yr1_fvc.tif", "yr2_fvc.tif", "yr3_fvc.tif", "yr4_fvc.tif", "yr5_fvc.tif"],
     [0.12, 0.20, 0.31, 0.42, 0.50]),
]
for sid, diff, prompt, paths, values in _VR_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "vegetation_recovery",
        "disaster_category": "fire", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_statistics.batch_raster_stats",
             "args": {"input_paths": paths, "stat": "mean"},
             "sample_output": {"stats": [{"file": p, "mean": v}
                                          for p, v in zip(paths, values)],
                               "aggregate_mean": round(sum(values)/len(values), 3)}},
            {"step": 2, "tool": "geoanalysis.compute_linear_trend",
             "args": {"y": values, "x": list(range(len(values)))},
             "sample_output": {"slope": round((values[-1]-values[0])/(len(values)-1), 4),
                               "intercept": values[0], "r_squared": 0.98,
                               "trend": "increasing"}},
            {"step": 3, "tool": "geoanalysis.detect_change_points",
             "args": {"values": values, "model": "rbf", "penalty": 3},
             "sample_output": {"change_points": [2], "n_changes": 1,
                               "segments": [[0, 2], [2, len(values)]]}},
            {"step": 4, "tool": "geo_statistics.count_images_exceeding",
             "args": {"input_paths": paths, "pixel_threshold": 0.35, "ratio_threshold": 0.3},
             "sample_output": {"count_exceeding": len([v for v in values if v > 0.35]),
                               "total_images": len(paths)}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 11. 新任务类型：sar_flood_mapping（SAR 影像洪水制图）
#     模式 A：统计 → 阈值分割 → 像素统计 → 面积 → 热点 → 标注
# ─────────────────────────────────────────────────────────────────────────────
_SAR_A = [
    ("sar_flood_mapping_01", "medium",
     "利用合成孔径雷达（SAR）影像进行洪水制图：基于后向散射差值阈值分割，提取积水范围并计算洪水面积。",
     "sar_post.tif", "sar_pre.tif"),
    ("sar_flood_mapping_02", "medium",
     "SAR 影像洪水快速制图：差值阈值法提取积水边界，统计受淹面积，空间热点分析识别重灾区。",
     "sentinel1_post.tif", "sentinel1_pre.tif"),
    ("sar_flood_mapping_03", "hard",
     "Sentinel-1 SAR 数据洪水精确制图：差值分析提取积水范围，分级统计面积，热点分析识别持续淹没区。",
     "s1_post.tif", "s1_pre.tif"),
    ("sar_flood_mapping_04", "hard",
     "大范围 SAR 洪水制图与人口暴露评估：差值法提取积水范围，热点分析，结合人口网格评估受灾人口。",
     "palsar_post.tif", "palsar_pre.tif"),
]
for sid, diff, prompt, post, pre in _SAR_A:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "sar_flood_mapping",
        "disaster_category": "flood", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.raster_stats",
             "args": {"input_path": post},
             "sample_output": {"mean": -12.4, "std": 4.8, "min": -28.6,
                               "max": 2.1, "median": -11.8}},
            {"step": 2, "tool": "geo_raster.raster_diff",
             "args": {"path_a": pre, "path_b": post, "output_path": "/tmp/sar_diff.tif"},
             "sample_output": {"output_path": "/tmp/sar_diff.tif",
                               "stats": {"mean": 5.8, "std": 3.2}}},
            {"step": 3, "tool": "geo_raster.threshold_segmentation",
             "args": {"input_path": "/tmp/sar_diff.tif", "threshold": 3.0,
                      "output_path": "/tmp/flood_mask_sar.tif"},
             "sample_output": {"output_path": "/tmp/flood_mask_sar.tif"}},
            {"step": 4, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "/tmp/flood_mask_sar.tif", "lower": 127},
             "sample_output": {"count": 284600, "total": 4194304, "ratio": 0.0679}},
            {"step": 5, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step4.count", "gsd_m": 10},
             "sample_output": {"area_m2": 28460000.0, "area_km2": 28.46}},
            {"step": 6, "tool": "geoanalysis.getis_ord_gi_star",
             "args": {"image_path": "/tmp/flood_mask_sar.tif",
                      "output_path": "/tmp/flood_hotspot.tif",
                      "weight_matrix": "queen"},
             "sample_output": {"output_path": "/tmp/flood_hotspot.tif",
                               "hotspot_ratio": 0.092, "coldspot_ratio": 0.014}},
            {"step": 7, "tool": "geo_perception.draw_bboxes",
             "args": {"image": post,
                      "bboxes": [{"x1": 80, "y1": 120, "x2": 480, "y2": 560}],
                      "output_path": "/tmp/sar_flood_annotated.png"},
             "sample_output": {"output_path": "/tmp/sar_flood_annotated.png"}},
        ]
    })

# 模式 B：差值 → 双指标统计 → 变化量 → 人口暴露
_SAR_B = [
    ("sar_flood_mapping_05", "hard",
     "SAR 洪水监测 + 人口暴露评估：后向散射差值提取积水，分级统计面积，结合人口格网评估受灾人口数量。",
     "sar_flood_post.tif", "sar_flood_pre.tif"),
    ("sar_flood_mapping_06", "hard",
     "洪涝灾害 SAR 监测与损失评估：多阈值分级积水检测，淹没面积统计，人口和基础设施暴露量化。",
     "sar_inundation_post.tif", "sar_inundation_pre.tif"),
    ("sar_flood_mapping_07", "hard",
     "SAR 多时相洪水演变分析：差分法提取积水范围，双阈值分级，统计轻中重度淹没区比例变化。",
     "sar_t2.tif", "sar_t1.tif"),
    ("sar_flood_mapping_08", "hard",
     "重大洪涝灾害 SAR 全域制图：后向散射差分分层提取积水，人口暴露评估，生成损失评估报告数据。",
     "major_flood_sar_post.tif", "major_flood_sar_pre.tif"),
]
for sid, diff, prompt, post, pre in _SAR_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "sar_flood_mapping",
        "disaster_category": "flood", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.raster_diff",
             "args": {"path_a": pre, "path_b": post, "output_path": "/tmp/sar_diff2.tif"},
             "sample_output": {"output_path": "/tmp/sar_diff2.tif",
                               "stats": {"mean": 6.1, "std": 3.8, "max": 18.4}}},
            {"step": 2, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "/tmp/sar_diff2.tif", "lower": 2, "upper": 6},
             "sample_output": {"count": 168400, "total": 4194304, "ratio": 0.0401}},
            {"step": 3, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "/tmp/sar_diff2.tif", "lower": 6},
             "sample_output": {"count": 84200, "total": 4194304, "ratio": 0.0201}},
            {"step": 4, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step2.count", "gsd_m": 10},
             "sample_output": {"area_m2": 16840000.0, "area_km2": 16.84}},
            {"step": 5, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step3.count", "gsd_m": 10},
             "sample_output": {"area_m2": 8420000.0, "area_km2": 8.42}},
            {"step": 6, "tool": "disaster_response.population_exposure",
             "args": {"hazard_path": "/tmp/sar_diff2.tif",
                      "population_path": "population_grid.tif",
                      "output_path": "/tmp/pop_exposure.tif", "threshold": 2.0},
             "sample_output": {"affected_population": 34280, "total_population": 186000,
                               "exposure_ratio": 0.184}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 12. 新任务类型：multi_hazard_exposure（多灾种人口暴露评估）
#     模式 A：差分检测 → 人口暴露 → 热点 → 设施标注 → 圈注
# ─────────────────────────────────────────────────────────────────────────────
_MHE_A = [
    ("multi_hazard_exposure_01", "hard",
     "洪灾人口暴露与社会脆弱性评估：提取积水范围，计算受影响人口，热点分析识别高风险聚居区，叠加救援设施。",
     "post.png", "pre.png"),
    ("multi_hazard_exposure_02", "hard",
     "地震灾区人口暴露精细评估：损毁检测，受灾人口统计，高脆弱性热点识别，避难所和医院分布标注。",
     "eq_post.png", "eq_pre.png"),
    ("multi_hazard_exposure_03", "hard",
     "台风受灾区人口风险综合评估：灾害范围检测，人口暴露量化，高密度受灾热点识别，应急设施定位。",
     "typhoon_post.png", "typhoon_pre.png"),
    ("multi_hazard_exposure_04", "hard",
     "复合灾害人口暴露快速评估：多灾种叠加影响区域检测，受灾人口统计，关键设施脆弱性评估。",
     "compound_post.png", "compound_pre.png"),
]
for sid, diff, prompt, post, pre in _MHE_A:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "multi_hazard_exposure",
        "disaster_category": "multi_hazard", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "/tmp/hazard_diff.tif"},
             "sample_output": {"output_path": "/tmp/hazard_diff.tif",
                               "stats": {"mean": 24.6, "std": 19.2}}},
            {"step": 2, "tool": "geo_raster.threshold_segmentation",
             "args": {"input_path": "/tmp/hazard_diff.tif", "threshold": 30,
                      "output_path": "/tmp/hazard_mask.tif"},
             "sample_output": {"output_path": "/tmp/hazard_mask.tif"}},
            {"step": 3, "tool": "disaster_response.population_exposure",
             "args": {"hazard_path": "/tmp/hazard_mask.tif",
                      "population_path": "population_grid.tif",
                      "output_path": "/tmp/pop_exposure.tif", "threshold": 127},
             "sample_output": {"affected_population": 28640, "total_population": 142000,
                               "exposure_ratio": 0.202}},
            {"step": 4, "tool": "geoanalysis.getis_ord_gi_star",
             "args": {"image_path": "/tmp/hazard_mask.tif",
                      "output_path": "/tmp/exposure_hotspot.tif",
                      "weight_matrix": "queen"},
             "sample_output": {"output_path": "/tmp/exposure_hotspot.tif",
                               "hotspot_ratio": 0.074, "coldspot_ratio": 0.022}},
            {"step": 5, "tool": "osm_gis.add_pois_layer",
             "args": {"place_name": "受灾区",
                      "tags": {"amenity": ["hospital", "school", "shelter"]},
                      "max_results": 30, "output_path": "/tmp/hazard_pois.geojson"},
             "sample_output": {"pois": [{"name": "人民医院", "type": "hospital",
                                         "lat": 22.32, "lon": 114.15},
                                        {"name": "体育馆避难所", "type": "shelter",
                                         "lat": 22.30, "lon": 114.18}],
                               "count": 14}},
            {"step": 6, "tool": "geo_perception.add_text",
             "args": {"image": post,
                      "annotations": [{"text": f"Affected: 28640 people | Ratio: 20.2%",
                                       "position": [10, 10]}],
                      "output_path": "/tmp/exposure_report.png"},
             "sample_output": {"output_path": "/tmp/exposure_report.png"}},
        ]
    })

# 模式 B：坡度分析 → 人口暴露 → 变化量计算 → 设施标注
_MHE_B = [
    ("multi_hazard_exposure_05", "hard",
     "山地灾害人口暴露评估：基于 DEM 坡度分析高风险区，统计受影响人口，量化暴露变化，标注关键设施。",
     "dem.tif", "pop_grid.tif"),
    ("multi_hazard_exposure_06", "hard",
     "滑坡风险人口暴露精细评估：坡度危险区提取，受灾人口统计，与历史基准对比，疏散设施定位。",
     "slope_dem.tif", "pop_raster.tif"),
    ("multi_hazard_exposure_07", "hard",
     "山洪灾害人口综合风险评估：地形坡度 + 人口空间分布分析，计算不同风险等级人口暴露量。",
     "mountain_dem.tif", "population.tif"),
    ("multi_hazard_exposure_08", "hard",
     "强降雨诱发地质灾害人口暴露评估：危险坡度区统计，受灾人口量化，变化分析，疏散路线设施标注。",
     "rain_dem.tif", "pop_grid_hr.tif"),
]
for sid, diff, prompt, dem, pop in _MHE_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "multi_hazard_exposure",
        "disaster_category": "multi_hazard", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "disaster_response.calc_slope",
             "args": {"dem_path": dem, "output_path": "/tmp/slope_risk.tif",
                      "unit": "degree"},
             "sample_output": {"output_path": "/tmp/slope_risk.tif",
                               "mean_slope_deg": 24.6, "max_slope_deg": 72.1}},
            {"step": 2, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "/tmp/slope_risk.tif", "lower": 30},
             "sample_output": {"count": 213400, "total": 1048576, "ratio": 0.204}},
            {"step": 3, "tool": "disaster_response.population_exposure",
             "args": {"hazard_path": "/tmp/slope_risk.tif",
                      "population_path": pop,
                      "output_path": "/tmp/slope_exposure.tif", "threshold": 30},
             "sample_output": {"affected_population": 18620, "total_population": 94000,
                               "exposure_ratio": 0.198}},
            {"step": 4, "tool": "geo_statistics.percentage_change",
             "args": {"a": 15200, "b": "$step3.affected_population"},
             "sample_output": {"absolute_change": 3420.0, "percentage_change": 22.5}},
            {"step": 5, "tool": "osm_gis.get_area_boundary",
             "args": {"place_name": "受灾县区", "output_path": "/tmp/county_boundary.geojson"},
             "sample_output": {"output_path": "/tmp/county_boundary.geojson",
                               "area_km2": 186.4}},
            {"step": 6, "tool": "osm_gis.add_pois_layer",
             "args": {"place_name": "受灾县区",
                      "tags": {"amenity": ["shelter", "hospital"]},
                      "max_results": 20, "output_path": "/tmp/evac_pois.geojson"},
             "sample_output": {"pois": [{"name": "镇安置点", "type": "shelter",
                                         "lat": 22.28, "lon": 114.09}], "count": 6}},
        ]
    })

# ─────────────────────────────────────────────────────────────────────────────
# 13. 新任务类型：debris_flow_risk（泥石流风险评估）
#     模式 A：坡度 + 流向 → 统计 → 面积 → 热点分析
# ─────────────────────────────────────────────────────────────────────────────
_DFR_A = [
    ("debris_flow_risk_01", "medium",
     "山区泥石流风险评估：基于 DEM 计算坡度和流向，识别高危沟谷，统计易发区面积，空间热点分析。",
     "mountain_dem.tif"),
    ("debris_flow_risk_02", "medium",
     "暴雨后泥石流风险快速评估：坡度流向分析识别汇流通道，量化高风险区面积，标注威胁聚居点。",
     "rainfall_dem.tif"),
    ("debris_flow_risk_03", "hard",
     "复杂地形泥石流危险性评估：精细化坡度和流向分析，分级识别高中低风险区，热点聚集分析。",
     "complex_dem.tif"),
    ("debris_flow_risk_04", "hard",
     "震后泥石流次生灾害风险评估：地震松动坡面 + DEM 坡度流向，识别泥石流危险沟道并量化风险。",
     "post_eq_dem.tif"),
]
for sid, diff, prompt, dem in _DFR_A:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "debris_flow_risk",
        "disaster_category": "landslide", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "disaster_response.calc_slope",
             "args": {"dem_path": dem, "output_path": "/tmp/df_slope.tif",
                      "unit": "degree"},
             "sample_output": {"output_path": "/tmp/df_slope.tif",
                               "mean_slope_deg": 28.4, "max_slope_deg": 75.6}},
            {"step": 2, "tool": "disaster_response.calc_flow_direction",
             "args": {"dem_path": dem, "output_path": "/tmp/df_flow.tif"},
             "sample_output": {"output_path": "/tmp/df_flow.tif",
                               "dominant_direction": "SE", "channel_count": 18}},
            {"step": 3, "tool": "geo_raster.raster_stats",
             "args": {"input_path": "/tmp/df_slope.tif"},
             "sample_output": {"mean": 28.4, "std": 16.2, "min": 0.0,
                               "max": 75.6, "median": 25.8}},
            {"step": 4, "tool": "geo_statistics.count_pixels_condition",
             "args": {"input_path": "/tmp/df_slope.tif", "lower": 35},
             "sample_output": {"count": 164200, "total": 1048576, "ratio": 0.156}},
            {"step": 5, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step4.count", "gsd_m": 12.5},
             "sample_output": {"area_m2": 25656250.0, "area_km2": 25.66}},
            {"step": 6, "tool": "geoanalysis.getis_ord_gi_star",
             "args": {"image_path": "/tmp/df_slope.tif",
                      "output_path": "/tmp/df_hotspot.tif",
                      "weight_matrix": "queen"},
             "sample_output": {"output_path": "/tmp/df_hotspot.tif",
                               "hotspot_ratio": 0.089, "coldspot_ratio": 0.031}},
        ]
    })

# 模式 B：VLM + SAM2 → 差分验证 → 轮廓统计 → 面积 → 标注
_DFR_B = [
    ("debris_flow_risk_05", "hard",
     "灾后泥石流堆积体识别与测量：AI 视觉分析+语义分割提取堆积体，差分验证，骨架轮廓统计，计算影响范围。",
     "debris_post.png", "debris_pre.png"),
    ("debris_flow_risk_06", "hard",
     "泥石流堆积体精细制图：VLM 初步识别，SAM2 精确分割，变化差分验证，统计堆积面积和沟道数量。",
     "mudflow_post.png", "mudflow_pre.png"),
    ("debris_flow_risk_07", "hard",
     "强降雨泥石流灾后评估：AI 视觉 + 语义分割识别堆积体，差分图验证，轮廓分析量化堆积规模。",
     "storm_debris_post.png", "storm_debris_pre.png"),
    ("debris_flow_risk_08", "hard",
     "山区群发性泥石流综合评估：多处堆积体 AI 分割，差分变化确认，沟道轮廓统计，受灾面积量化。",
     "multi_debris_post.png", "multi_debris_pre.png"),
]
for sid, diff, prompt, post, pre in _DFR_B:
    NEW_SAMPLES.append({
        "id": sid, "task_type": "debris_flow_risk",
        "disaster_category": "landslide", "difficulty": diff, "prompt": prompt,
        "tool_calls": [
            {"step": 1, "tool": "geo_perception.vlm_analyze",
             "args": {"images": [post],
                      "prompt": "识别图中泥石流堆积体、沟道淤积和受冲毁植被区，描述堆积规模和影响范围"},
             "sample_output": {"analysis": "图中发现 3 处明显泥石流堆积体，主沟道被大量泥沙和砾石填充，"
                                           "两侧斜坡植被破坏严重，堆积锥延伸至沟口约 200 米。",
                               "bboxes": [{"x1": 150, "y1": 200, "x2": 550, "y2": 600}]}},
            {"step": 2, "tool": "geo_perception.sam2_segment",
             "args": {"image": post},
             "sample_output": {"bboxes": [{"x1": 150, "y1": 200, "x2": 550, "y2": 600},
                                          {"x1": 600, "y1": 100, "x2": 800, "y2": 300}],
                               "masks_count": 2, "count": 2}},
            {"step": 3, "tool": "geo_raster.raster_diff",
             "args": {"path_a": post, "path_b": pre, "output_path": "/tmp/debris_diff.tif"},
             "sample_output": {"output_path": "/tmp/debris_diff.tif",
                               "stats": {"mean": 32.6, "std": 28.4}}},
            {"step": 4, "tool": "geo_raster.count_skeleton_contours",
             "args": {"image_path": "/tmp/debris_diff.tif"},
             "sample_output": {"contour_count": 8, "total_length_px": 4280}},
            {"step": 5, "tool": "geo_basic.pixel_area",
             "args": {"pixels": "$step4.total_length_px", "gsd_m": 2},
             "sample_output": {"area_m2": 17120.0, "area_km2": 0.017}},
            {"step": 6, "tool": "geo_perception.draw_bboxes",
             "args": {"image": post, "bboxes": "$step2.bboxes",
                      "output_path": "/tmp/debris_annotated.png"},
             "sample_output": {"output_path": "/tmp/debris_annotated.png"}},
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# 生成最终数据集
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(IN_PATH, encoding="utf-8") as f:
        data = json.load(f)

    existing = data["samples"]
    existing_ids = {s["id"] for s in existing}

    # 过滤掉重复 ID（防止误重复）
    new_uniq = [s for s in NEW_SAMPLES if s["id"] not in existing_ids]

    all_samples = existing + new_uniq

    new_data = copy.deepcopy(data)
    new_data["samples"] = all_samples
    old_ver = float(data.get("version", 2))
    new_data["version"] = f"{old_ver + 0.1:.1f}"
    new_data["description"] = (
        data.get("description", "") +
        f" [v3: +{len(new_uniq)} diverse chain patterns; "
        f"new task types: drone_damage_survey, infrastructure_damage, "
        f"vegetation_recovery, sar_flood_mapping, multi_hazard_exposure, debris_flow_risk]"
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    task_counter = Counter(s["task_type"] for s in new_uniq)
    print(f"=== 完成 ===")
    print(f"原始样本: {len(existing)}")
    print(f"新增样本: {len(new_uniq)}")
    print(f"总计    : {len(all_samples)}")
    print(f"输出    : {OUT_PATH}")
    print()
    print("新增样本按任务类型分布：")
    for task, cnt in sorted(task_counter.items()):
        print(f"  {task:<35s}: {cnt:3d} 条")
    print()

    # 验证工具链多样性
    from collections import defaultdict
    by_task = defaultdict(set)
    for s in all_samples:
        chain = tuple(tc["tool"] for tc in s["tool_calls"])
        by_task[s["task_type"]].add(chain)
    print("各任务类型工具链模式数（含原有）：")
    for task, chains in sorted(by_task.items()):
        print(f"  {task:<35s}: {len(chains)} 种模式")


if __name__ == "__main__":
    main()
