#!/usr/bin/env python3
"""
将 disaster_sft_dataset.json 中的每条 SFT 数据与真实灾害影像数据进行匹配。
输出: data/sft_image_mapping.json — 与 SFT 数据一一对应的影像路径映射。
"""
import json
import os
import glob
import random
from pathlib import Path

random.seed(42)

# ── 常量 ──
DISASTER_DATA_ROOT = "/data1/yuhongjie2/disasterData"
DM3_IMAGES = f"{DISASTER_DATA_ROOT}/DisasterM3/data/DisasterM3/train_images"
DM3_BENCH  = f"{DISASTER_DATA_ROOT}/DisasterM3/data/DisasterM3/DisasterM3_Bench/test_images"
XBD_IMAGES = f"{DISASTER_DATA_ROOT}/xBD/test/images"
SN8_PRE    = f"{DISASTER_DATA_ROOT}/spaceNet8/Louisiana-West_Test_Public/PRE-event"
SN8_POST   = f"{DISASTER_DATA_ROOT}/spaceNet8/Louisiana-West_Test_Public/POST-event"
MSNET_TRAIN = f"{DISASTER_DATA_ROOT}/msnet/train"

SFT_PATH = "data/disaster_sft_dataset.json"
OUTPUT_PATH = "data/sft_image_mapping.json"


def list_images(directory, pattern="*"):
    """列出目录下匹配的文件，返回排序后的列表。"""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    return [f for f in files if os.path.isfile(f)]


def get_pre_post_pairs(directory, event_prefix, pre_key="pre", post_key="post"):
    """提取某事件的 pre/post 影像对。"""
    all_files = os.listdir(directory)
    pre_files = sorted([f for f in all_files if f.startswith(event_prefix) and pre_key in f])
    post_files = sorted([f for f in all_files if f.startswith(event_prefix) and post_key in f])

    pairs = []
    # 尝试按编号匹配
    pre_index = {}
    for f in pre_files:
        # 提取编号部分
        idx = f.replace(event_prefix, "").replace(f"_{pre_key}", "").replace("_disaster", "").replace(".png", "").replace(".tif", "").strip("_")
        pre_index[idx] = f

    for f in post_files:
        idx = f.replace(event_prefix, "").replace(f"_{post_key}", "").replace("_disaster", "").replace(".png", "").replace(".tif", "").strip("_")
        if idx in pre_index:
            pairs.append({
                "pre": os.path.join(directory, pre_index[idx]),
                "post": os.path.join(directory, f),
            })

    if not pairs and pre_files and post_files:
        # fallback: 直接配对
        for pre, post in zip(pre_files, post_files):
            pairs.append({
                "pre": os.path.join(directory, pre),
                "post": os.path.join(directory, post),
            })
    return pairs


def get_xbd_pairs(event_prefix):
    """获取 xBD 数据集的 pre/post 对。"""
    return get_pre_post_pairs(XBD_IMAGES, event_prefix, "pre_disaster", "post_disaster")


# ── 构建影像池 ──
print("构建影像池...")

# --- Flood ---
flood_pool = []

# DisasterM3: kalehe_flooding (pre/post pairs)
flood_pool.extend([
    {"source": "DisasterM3/kalehe_flooding", **p}
    for p in get_pre_post_pairs(DM3_IMAGES, "kalehe_flooding")
])

# DisasterM3: libya_flood (pre/post pairs)
flood_pool.extend([
    {"source": "DisasterM3/libya_flood", **p}
    for p in get_pre_post_pairs(DM3_IMAGES, "libya_flood")
])

# xBD: midwest-flooding
flood_pool.extend([
    {"source": "xBD/midwest-flooding", **p}
    for p in get_xbd_pairs("midwest-flooding")
])

# SpaceNet8: Louisiana hurricane/flood (GeoTIFF, pre/post)
sn8_pre_files = sorted(os.listdir(SN8_PRE))
sn8_post_files = sorted(os.listdir(SN8_POST))
sn8_common = set(sn8_pre_files) & set(sn8_post_files)
for fname in sorted(sn8_common)[:40]:  # 取前40对
    flood_pool.append({
        "source": "SpaceNet8/Louisiana-flood",
        "pre": os.path.join(SN8_PRE, fname),
        "post": os.path.join(SN8_POST, fname),
        "format": "GeoTIFF",
    })

print(f"  Flood pool: {len(flood_pool)} pairs")

# --- Wildfire ---
wildfire_pool = []

# DisasterM3: hawaii_wildfire
wildfire_pool.extend([
    {"source": "DisasterM3/hawaii_wildfire", **p}
    for p in get_pre_post_pairs(DM3_IMAGES, "hawaii_wildfire")
])

# DisasterM3: socal_fire
socal_dm3_pairs = get_pre_post_pairs(DM3_IMAGES, "socal_fire")
wildfire_pool.extend([
    {"source": "DisasterM3/socal_fire", **p}
    for p in socal_dm3_pairs[:30]
])

# DisasterM3: portugal_wildfire
portugal_pairs = get_pre_post_pairs(DM3_IMAGES, "portugal_wildfire")
wildfire_pool.extend([
    {"source": "DisasterM3/portugal_wildfire", **p}
    for p in portugal_pairs[:30]
])

# DisasterM3: woolsey_fire
woolsey_pairs = get_pre_post_pairs(DM3_IMAGES, "woolsey_fire")
wildfire_pool.extend([
    {"source": "DisasterM3/woolsey_fire", **p}
    for p in woolsey_pairs[:20]
])

# xBD: socal-fire
wildfire_pool.extend([
    {"source": "xBD/socal-fire", **p}
    for p in get_xbd_pairs("socal-fire")[:30]
])

# xBD: santa-rosa-wildfire
wildfire_pool.extend([
    {"source": "xBD/santa-rosa-wildfire", **p}
    for p in get_xbd_pairs("santa-rosa-wildfire")[:20]
])

print(f"  Wildfire pool: {len(wildfire_pool)} pairs")

# --- Earthquake ---
earthquake_pool = []

# DisasterM3: haiti_earthquake
earthquake_pool.extend([
    {"source": "DisasterM3/haiti_earthquake", **p}
    for p in get_pre_post_pairs(DM3_IMAGES, "haiti_earthquake")
])

# DisasterM3: morroco_earthquake1
morroco_pairs = get_pre_post_pairs(DM3_IMAGES, "morroco_earthquake1")
earthquake_pool.extend([
    {"source": "DisasterM3/morocco_earthquake", **p}
    for p in morroco_pairs[:30]
])

# DisasterM3: turkey_earthquake0
turkey0_pairs = get_pre_post_pairs(DM3_IMAGES, "turkey_earthquake0")
earthquake_pool.extend([
    {"source": "DisasterM3/turkey_earthquake", **p}
    for p in turkey0_pairs[:30]
])

# DisasterM3: turkey_earthquake3
turkey3_pairs = get_pre_post_pairs(DM3_IMAGES, "turkey_earthquake3")
earthquake_pool.extend([
    {"source": "DisasterM3/turkey_earthquake", **p}
    for p in turkey3_pairs[:20]
])

# xBD: mexico-earthquake
earthquake_pool.extend([
    {"source": "xBD/mexico-earthquake", **p}
    for p in get_xbd_pairs("mexico-earthquake")
])

print(f"  Earthquake pool: {len(earthquake_pool)} pairs")

# --- Landslide ---
landslide_pool = []

# DisasterM3: shovi_landslide (clear mudflow visible)
landslide_pool.extend([
    {"source": "DisasterM3/shovi_landslide", **p}
    for p in get_pre_post_pairs(DM3_IMAGES, "shovi_landslide")
])

print(f"  Landslide pool: {len(landslide_pool)} pairs")

# --- Typhoon ---
typhoon_pool = []

# xBD: hurricane-florence, harvey, matthew, michael
for event in ["hurricane-florence", "hurricane-harvey", "hurricane-matthew", "hurricane-michael"]:
    pairs = get_xbd_pairs(event)
    typhoon_pool.extend([
        {"source": f"xBD/{event}", **p}
        for p in pairs[:15]
    ])

# DisasterM3: hurricane images
for event in ["hurricane_florence", "hurricane_harvey", "hurricane_michael"]:
    dm3_pairs = get_pre_post_pairs(DM3_IMAGES, event)
    typhoon_pool.extend([
        {"source": f"DisasterM3/{event}", **p}
        for p in dm3_pairs[:10]
    ])

# MSNet: hurricane damage images (single images, no pre/post)
msnet_files = sorted(glob.glob(os.path.join(MSNET_TRAIN, "*.jpg")))[:20]
for f in msnet_files:
    typhoon_pool.append({
        "source": "MSNet/hurricane",
        "post": f,
        "pre": None,
        "format": "JPG",
        "note": "单张灾后影像，无灾前对比",
    })

print(f"  Typhoon pool: {len(typhoon_pool)} pairs")

# --- Drought (无直接匹配，用干旱区域 libya_flood 的灾前影像作为代理) ---
drought_pool = []
libya_pre_files = sorted([
    f for f in os.listdir(DM3_IMAGES)
    if f.startswith("libya_flood_pre")
])
for f in libya_pre_files:
    drought_pool.append({
        "source": "DisasterM3/libya_flood(proxy-arid-region)",
        "post": os.path.join(DM3_IMAGES, f),
        "pre": None,
        "note": "利比亚干旱/半干旱区域影像，作为干旱监测代理数据",
        "compatibility": "proxy",
    })
print(f"  Drought pool: {len(drought_pool)} images (proxy)")

# --- Heatwave (用城市区域影像作为代理：建筑密集区 → 城市热岛效应) ---
heatwave_pool = []
# 使用利比亚城市影像（有建筑密集区）
libya_post_files = sorted([
    f for f in os.listdir(DM3_IMAGES)
    if f.startswith("libya_flood_post") and not "sar" in f
])
for f in libya_post_files[:20]:
    heatwave_pool.append({
        "source": "DisasterM3/libya_flood(proxy-urban-heat)",
        "post": os.path.join(DM3_IMAGES, f),
        "pre": None,
        "note": "城市区域光学影像，作为热浪/城市热岛效应分析代理数据",
        "compatibility": "proxy",
    })
print(f"  Heatwave pool: {len(heatwave_pool)} images (proxy)")

# --- Snow (无直接匹配，标记为不可用) ---
snow_pool = []
# 使用 shovi_landslide 的高山区域影像作为代理（山区可能有雪）
for p in get_pre_post_pairs(DM3_IMAGES, "shovi_landslide"):
    snow_pool.append({
        "source": "DisasterM3/shovi_landslide(proxy-mountain-snow)",
        **p,
        "note": "高山区域影像，作为雪灾分析代理数据（山区场景）",
        "compatibility": "proxy",
    })
print(f"  Snow pool: {len(snow_pool)} pairs (proxy)")


# ── 映射逻辑 ──
# task_type → 是否需要 pre/post 对
NEEDS_PRE_POST = {
    "flood_change", "flood_timeseries", "fire_spread", "fire_spread_warning",
    "fire_risk_zones", "earthquake_damage", "building_damage_assess",
    "landslide", "typhoon_port_damage", "snow_disaster",
}

CATEGORY_POOL = {
    "flood": flood_pool,
    "wildfire": wildfire_pool,
    "earthquake": earthquake_pool,
    "landslide": landslide_pool,
    "typhoon": typhoon_pool,
    "drought": drought_pool,
    "heatwave": heatwave_pool,
    "snow": snow_pool,
}


def select_image(pool, idx, needs_pair=False):
    """从池中循环选取影像，确保多样性。"""
    if not pool:
        return None
    if needs_pair:
        # 优先选有 pre 的
        pair_pool = [p for p in pool if p.get("pre")]
        if pair_pool:
            return pair_pool[idx % len(pair_pool)]
    return pool[idx % len(pool)]


# ── 主逻辑 ──
print("\n读取 SFT 数据集...")
with open(SFT_PATH) as f:
    sft_data = json.load(f)

samples = sft_data["samples"]
print(f"共 {len(samples)} 条样本\n")

mappings = []
category_counters = {}

for sample in samples:
    sid = sample["id"]
    cat = sample["disaster_category"]
    task_type = sample["task_type"]
    needs_pair = task_type in NEEDS_PRE_POST

    pool = CATEGORY_POOL.get(cat, [])
    idx = category_counters.get(cat, 0)
    category_counters[cat] = idx + 1

    img_info = select_image(pool, idx, needs_pair)

    if img_info is None:
        mapping = {
            "sft_id": sid,
            "disaster_category": cat,
            "task_type": task_type,
            "status": "no_match",
            "note": f"当前 disasterData 中无直接匹配的 {cat} 类型影像数据",
        }
    else:
        mapping = {
            "sft_id": sid,
            "disaster_category": cat,
            "task_type": task_type,
            "status": "matched" if img_info.get("compatibility") != "proxy" else "proxy",
            "source_dataset": img_info["source"],
        }
        if img_info.get("pre"):
            mapping["image_pre"] = img_info["pre"]
        if img_info.get("post"):
            mapping["image_post"] = img_info["post"]
        if img_info.get("format"):
            mapping["format"] = img_info["format"]
        else:
            mapping["format"] = "PNG"
        mapping["image_size"] = "1024x1024"
        if img_info.get("note"):
            mapping["note"] = img_info["note"]
        if img_info.get("compatibility"):
            mapping["compatibility"] = img_info["compatibility"]

    mappings.append(mapping)

# ── 统计 ──
matched = sum(1 for m in mappings if m["status"] == "matched")
proxy = sum(1 for m in mappings if m["status"] == "proxy")
no_match = sum(1 for m in mappings if m["status"] == "no_match")

print(f"匹配结果:")
print(f"  直接匹配: {matched}")
print(f"  代理匹配: {proxy}")
print(f"  无匹配:   {no_match}")

# 按类别统计
from collections import Counter
cat_stats = Counter()
for m in mappings:
    cat_stats[f"{m['disaster_category']}/{m['status']}"] += 1
print(f"\n按类别:")
for k, v in sorted(cat_stats.items()):
    print(f"  {k}: {v}")

# ── 输出 ──
output = {
    "version": "1.0",
    "description": "SFT 数据集影像映射 — 将每条 SFT 样本与真实灾害遥感影像一一对应",
    "sft_dataset": SFT_PATH,
    "disaster_data_root": DISASTER_DATA_ROOT,
    "statistics": {
        "total": len(mappings),
        "matched": matched,
        "proxy": proxy,
        "no_match": no_match,
    },
    "compatibility_notes": {
        "matched": "直接匹配：影像类型与灾害类别完全对应",
        "proxy": "代理匹配：使用相近场景的影像作为替代（如用干旱区域影像代理旱情监测）",
        "format_note": "DisasterM3/xBD 为 PNG 格式 (1024x1024 RGB)；SpaceNet8 为 GeoTIFF；若工具要求 GeoTIFF 需先转换格式",
    },
    "source_datasets": {
        "DisasterM3": f"{DISASTER_DATA_ROOT}/DisasterM3/data/DisasterM3/train_images/",
        "xBD": f"{DISASTER_DATA_ROOT}/xBD/test/images/",
        "SpaceNet8": f"{DISASTER_DATA_ROOT}/spaceNet8/Louisiana-West_Test_Public/",
        "MSNet": f"{DISASTER_DATA_ROOT}/msnet/train/",
    },
    "mappings": mappings,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n映射文件已保存到: {OUTPUT_PATH}")
print(f"文件大小: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
