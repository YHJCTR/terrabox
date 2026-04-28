#!/usr/bin/env python3
"""
run_sft_sample.py — 用真实图像执行某条灾害 SFT 样本的工具链

用法:
  # 列出所有可用样本 ID
  python scripts/run_sft_sample.py --list

  # 自动模式：从映射文件读取影像路径，自动分配给占位符
  python scripts/run_sft_sample.py --id flood_detection_01 --auto

  # 交互模式：脚本自动检测所需输入图像，逐一提示你填写真实路径
  python scripts/run_sft_sample.py --id flood_detection_01

  # 非交互：用 --inputs 直接提供映射（占位符=真实路径）
  python scripts/run_sft_sample.py --id flood_detection_01 \\
      --inputs green.tif=/data/s2/band3.tif nir.tif=/data/s2/band8.tif

  # 指定工作目录（中间/输出文件落地于此）
  python scripts/run_sft_sample.py --id flood_detection_01 --workdir /tmp/test_run

  # 使用扩增数据集
  python scripts/run_sft_sample.py --id flood_detection_01_aug1 --auto --augmented

运行环境:
  /data1/yuhongjie2/env/unsloth/bin/python scripts/run_sft_sample.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET_PATH = REPO_ROOT / "data" / "disaster_sft_dataset.json"
AUGMENTED_PATH = REPO_ROOT / "data" / "disaster_sft_augmented.json"
MAPPING_PATH = REPO_ROOT / "data" / "sft_image_mapping.json"
AUGMENTED_MAPPING_PATH = REPO_ROOT / "data" / "sft_augmented_image_mapping.json"

# 识别文件路径的正则：不以 $step 开头，以 .tif/.tiff/.png/.geojson 结尾
_PATH_RE = re.compile(r"^(?!\$step)[\w/._-]+\.(tif|tiff|png|geojson)$", re.IGNORECASE)

# $stepN.key 引用正则
_REF_RE = re.compile(r"^\$step(\d+)\.(.+)$")

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ─────────────────────────────────────────────────────────────────────────────
# 数据集加载
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_sample(dataset: dict, sample_id: str) -> dict | None:
    for s in dataset["samples"]:
        if s["id"] == sample_id:
            return s
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 路径分析：识别叶节点输入（需要用户提供的真实文件）
# ─────────────────────────────────────────────────────────────────────────────

def _is_file_placeholder(val: Any) -> bool:
    return isinstance(val, str) and bool(_PATH_RE.match(val))


def _collect_all_paths(args: dict) -> list[str]:
    """递归收集 args 中所有看起来像文件路径的字符串。"""
    found = []
    for v in args.values():
        if _is_file_placeholder(v):
            found.append(v)
        elif isinstance(v, list):
            for item in v:
                if _is_file_placeholder(item):
                    found.append(item)
    return found


def _collect_output_paths(args: dict) -> list[str]:
    """收集 output 类参数的占位符名。"""
    outputs = []
    for k, v in args.items():
        if ("output" in k) and _is_file_placeholder(v):
            outputs.append(v)
    return outputs


def find_leaf_inputs(tool_calls: list[dict]) -> list[str]:
    """
    返回需要用户提供的输入占位符列表（按首次出现顺序，去重）。
    规则：某路径占位符在首次被用作「输入」之前，未被任何 step 的 output 参数生成。
    """
    generated: set[str] = set()
    leaf_inputs: list[str] = []
    seen: set[str] = set()

    for step in tool_calls:
        args = step.get("args", {})
        output_paths = set(_collect_output_paths(args))

        for k, v in args.items():
            candidates = []
            if _is_file_placeholder(v):
                candidates = [v]
            elif isinstance(v, list):
                candidates = [item for item in v if _is_file_placeholder(item)]

            for path in candidates:
                if path in output_paths:
                    # 这是本步的输出，不是输入
                    continue
                if path not in generated and path not in seen:
                    leaf_inputs.append(path)
                    seen.add(path)

        # 本步的输出加入 generated 集合
        generated |= output_paths

    return leaf_inputs


# ─────────────────────────────────────────────────────────────────────────────
# 自动影像映射 (--auto)
# ─────────────────────────────────────────────────────────────────────────────

# 占位符名 → (源图选择, 波段索引)
# 源图: "pre" / "post" / "post_or_pre"
# 波段: 1=R, 2=G, 3=B(proxy NIR), 0=保持多波段原样
_PLACEHOLDER_BAND_MAP = {
    # RGB 原图（多波段）
    "rgb.tif":       ("post_or_pre", 0),
    "rgb_post.tif":  ("post", 0),
    "rgb_fire.tif":  ("post", 0),
    # 光谱波段
    "green.tif":     ("post_or_pre", 2),
    "green_pre.tif": ("pre", 2),
    "green_post.tif":("post", 2),
    "green_d1.tif":  ("post_or_pre", 2),
    "red.tif":       ("post_or_pre", 1),
    "nir.tif":       ("post_or_pre", 3),
    "nir_pre.tif":   ("pre", 3),
    "nir_post.tif":  ("post", 3),
    "nir_d1.tif":    ("post_or_pre", 3),
    "swir_pre.tif":  ("pre", 1),
    "swir_post.tif": ("post", 1),
    # 变化检测 pre/post
    "pre.tif":       ("pre", 0),
    "post.tif":      ("post", 0),
    "port_pre.tif":  ("pre", 0),
    "port_post.tif": ("post", 0),
    # 热/FRP（用单波段代理）
    "thermal.tif":   ("post_or_pre", 1),
    "lst_day.tif":   ("post_or_pre", 1),
    "lst_night.tif": ("post_or_pre", 3),
    "frp.tif":       ("post_or_pre", 1),
    "frp_pre.tif":   ("pre", 1),
    "frp_post.tif":  ("post", 1),
    "albedo.tif":    ("post_or_pre", 2),
    "tb_h.tif":      ("post_or_pre", 1),
    "tb_v.tif":      ("post_or_pre", 3),
    # 地形/辅助
    "dem.tif":             ("post_or_pre", 1),
    "disaster.tif":        ("post", 1),
    "fire_mask.tif":       ("post", 1),
    "cloud_qa.tif":        ("post", 3),
    "flood_mask_norm.tif": ("post", 1),
    "ones_raster.tif":     ("post_or_pre", 0),  # 特殊：全1栅格
    "population_density.tif": ("post_or_pre", 1),
}

# 月份 FRP 文件：frp_jan.tif ~ frp_dec.tif
for _m in ["jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec"]:
    _PLACEHOLDER_BAND_MAP[f"frp_{_m}.tif"] = ("post_or_pre", 1)


def _extract_band_to_tif(src_path: str, band_idx: int, dst_path: str) -> str:
    """从多波段影像中提取单波段，保存为 GeoTIFF。"""
    import rasterio
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    with rasterio.open(src_path) as src:
        n_bands = src.count
        idx = min(band_idx, n_bands)  # 防止越界
        data = src.read(idx).astype(np.float32)
        profile = src.profile.copy()
        profile.update(count=1, driver="GTiff", dtype="float32")
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)
    return dst_path


def _copy_as_tif(src_path: str, dst_path: str) -> str:
    """将 PNG/多波段影像原样转为 GeoTIFF（保留所有波段）。"""
    import rasterio
    import warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    with rasterio.open(src_path) as src:
        data = src.read()
        profile = src.profile.copy()
        profile.update(driver="GTiff")
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
    return dst_path


def _make_ones_raster(src_path: str, dst_path: str) -> str:
    """生成与源影像同尺寸的全 1 栅格。"""
    import rasterio
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, driver="GTiff", dtype="float32")
        ones = np.ones((src.height, src.width), dtype=np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(ones, 1)
    return dst_path


def _make_dummy_geojson(src_path: str | None, dst_path: str, placeholder: str) -> str:
    """生成覆盖影像范围的简单 GeoJSON 多边形。"""
    import rasterio
    import warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # 默认用像素坐标范围
    minx, miny, maxx, maxy = 0, 0, 1024, 1024
    if src_path and os.path.exists(src_path):
        try:
            with rasterio.open(src_path) as src:
                b = src.bounds
                minx, miny, maxx, maxy = b.left, b.bottom, b.right, b.top
        except Exception:
            pass

    # 生成 4 个子区域（模拟行政区/建筑物）
    dx = (maxx - minx) / 2
    dy = (maxy - miny) / 2
    features = []
    names = ["zone_A", "zone_B", "zone_C", "zone_D"]
    for i, (ox, oy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        x0 = minx + ox * dx + dx * 0.05
        y0 = miny + oy * dy + dy * 0.05
        x1 = x0 + dx * 0.9
        y1 = y0 + dy * 0.9
        features.append({
            "type": "Feature",
            "properties": {"name": names[i], "id": i + 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
            }
        })

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(geojson, f)
    return dst_path


def auto_map_inputs(
    sample_id: str,
    leaf_inputs: list[str],
    mapping_path: Path,
    workdir: str,
) -> tuple[dict[str, str], list[str]]:
    """
    从映射文件自动生成 占位符→真实文件 的映射。

    返回 (path_map, warnings)。
    对于需要单波段的占位符，会从 RGB 影像中提取对应波段到 workdir。
    """
    if not mapping_path.exists():
        return {}, [f"映射文件不存在: {mapping_path}"]

    data = json.loads(mapping_path.read_text())
    entry = None
    for m in data["mappings"]:
        if m["sft_id"] == sample_id:
            entry = m
            break

    if entry is None:
        return {}, [f"映射文件中找不到 sft_id={sample_id}"]
    if entry.get("status") == "no_match":
        return {}, [f"该样本无可用影像匹配 (status=no_match)"]

    image_pre = entry.get("image_pre")
    image_post = entry.get("image_post")
    warnings_list = []

    if entry.get("status") == "proxy":
        warnings_list.append(f"代理匹配 (proxy): {entry.get('note', '')}")

    path_map: dict[str, str] = {}
    prep_dir = os.path.join(workdir, "_auto_inputs")
    os.makedirs(prep_dir, exist_ok=True)

    for ph in leaf_inputs:
        # GeoJSON 特殊处理
        if ph.endswith(".geojson"):
            ref_img = image_post or image_pre
            dst = os.path.join(prep_dir, ph)
            _make_dummy_geojson(ref_img, dst, ph)
            path_map[ph] = dst
            warnings_list.append(f"  {ph}: 生成模拟 GeoJSON")
            continue

        # 查找波段映射规则
        rule = _PLACEHOLDER_BAND_MAP.get(ph)
        if rule is None:
            # 未知占位符：尝试用 post 影像 band1
            warnings_list.append(f"  {ph}: 未知占位符，使用 post 影像 band1 代理")
            rule = ("post_or_pre", 1)

        src_choice, band_idx = rule

        # 选择源影像
        if src_choice == "pre":
            src_img = image_pre or image_post
        elif src_choice == "post":
            src_img = image_post or image_pre
        else:  # post_or_pre
            src_img = image_post or image_pre

        if src_img is None:
            warnings_list.append(f"  {ph}: 无可用源影像，跳过")
            continue

        if not os.path.exists(src_img):
            warnings_list.append(f"  {ph}: 源影像不存在 {src_img}")
            continue

        dst = os.path.join(prep_dir, ph.replace(".tif", ".tif"))  # 保持名字

        # ones_raster 特殊处理
        if ph == "ones_raster.tif":
            _make_ones_raster(src_img, dst)
            path_map[ph] = dst
            continue

        # 多波段原样 or 单波段提取
        if band_idx == 0:
            _copy_as_tif(src_img, dst)
        else:
            _extract_band_to_tif(src_img, band_idx, dst)

        path_map[ph] = dst

    return path_map, warnings_list


# ─────────────────────────────────────────────────────────────────────────────
# 工具链执行
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_value(val: Any, step_outputs: dict[int, dict], path_map: dict[str, str]) -> Any:
    """解析单个参数值：替换 $stepN.key 引用 和 文件路径映射。"""
    if isinstance(val, str):
        # 优先处理 $step 引用
        m = _REF_RE.match(val)
        if m:
            step_no = int(m.group(1))
            key_path = m.group(2).split(".")
            src = step_outputs.get(step_no, {})
            for k in key_path:
                src = src.get(k) if isinstance(src, dict) else None
            return src if src is not None else val
        # 文件路径替换
        if _is_file_placeholder(val):
            return path_map.get(val, val)
    elif isinstance(val, list):
        return [_resolve_value(item, step_outputs, path_map) for item in val]
    return val


def resolve_args(args: dict, step_outputs: dict[int, dict], path_map: dict[str, str]) -> dict:
    return {k: _resolve_value(v, step_outputs, path_map) for k, v in args.items()}


def run_sample(sample: dict, path_map: dict[str, str], workdir: str) -> list[dict]:
    """
    按顺序执行样本中的所有 tool_calls，返回每步的执行结果列表。
    path_map: 占位符名 → 真实路径（叶输入 + 中间/输出文件落地到 workdir）
    """
    # 初始化 registry（加载所有内置 toolkits）
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import get_handler

    load_builtin_toolkits()

    # 如果 keep-services 模式，禁用 atexit 自动清理
    if os.environ.get("_TERRABOX_KEEP_SERVICES") == "true":
        import atexit
        from terrabox.managers.base_manager import ServiceRegistry
        try:
            atexit.unregister(ServiceRegistry.cleanup_all)
        except Exception:
            pass

    # 将输出文件路径映射到 workdir
    tool_calls = sample["tool_calls"]
    generated: set[str] = set()
    for step in tool_calls:
        for output_ph in _collect_output_paths(step.get("args", {})):
            if output_ph not in path_map:
                fname = os.path.basename(output_ph)
                path_map[output_ph] = os.path.join(workdir, fname)
            generated.add(output_ph)

    step_outputs: dict[int, dict] = {}
    results = []

    for step_def in tool_calls:
        step_no = step_def["step"]
        slug = step_def["tool"]
        raw_args = step_def.get("args", {})

        resolved = resolve_args(raw_args, step_outputs, path_map)

        handler = get_handler(slug)
        if handler is None:
            result = {"error": f"找不到工具 handler: {slug}"}
            print(f"\n  [step {step_no}] {slug}  ✗ 找不到 handler")
        else:
            print(f"\n  [step {step_no}] {slug}")
            print(f"    args: {json.dumps(resolved, ensure_ascii=False, default=str)}")
            try:
                result = handler(resolved, {}, None)
                print(f"    结果: {json.dumps(result, ensure_ascii=False, default=str)}")
            except Exception as e:
                import traceback
                result = {"error": str(e), "traceback": traceback.format_exc()}
                print(f"    ✗ 异常: {e}")

        step_outputs[step_no] = result
        results.append({"step": step_no, "tool": slug, "result": result})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 结果展示
# ─────────────────────────────────────────────────────────────────────────────

def summarize_results(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("执行结果汇总")
    print("=" * 60)
    for r in results:
        step_no = r["step"]
        tool = r["tool"]
        result = r["result"]

        if "error" in result:
            print(f"\n[step {step_no}] {tool}  ✗ FAILED")
            print(f"  错误: {result['error']}")
            continue

        print(f"\n[step {step_no}] {tool}  ✓ OK")

        # 判断结果中是否有图像路径
        image_paths = []
        text_data = {}
        for k, v in result.items():
            if isinstance(v, str) and Path(v).suffix.lower() in IMAGE_EXTS and Path(v).exists():
                image_paths.append((k, v))
            else:
                text_data[k] = v

        if text_data:
            print(f"  文本结果: {json.dumps(text_data, ensure_ascii=False, indent=4, default=str)}")
        for k, img_path in image_paths:
            print(f"  图像输出 [{k}]: {img_path}")
            _try_show_image_info(img_path)


def _try_show_image_info(path: str) -> None:
    """打印图像基本信息（尺寸、波段、统计值）。"""
    try:
        import rasterio
        import numpy as np
        with rasterio.open(path) as src:
            data = src.read(1)
            print(f"    尺寸: {src.width}×{src.height}  波段: {src.count}  CRS: {src.crs}")
            finite = data[np.isfinite(data)]
            if len(finite) > 0:
                print(f"    值域: [{finite.min():.4f}, {finite.max():.4f}]  "
                      f"均值: {finite.mean():.4f}  非零像素: {(finite != 0).sum()}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="用真实图像执行某条灾害 SFT 样本工具链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="列出所有可用样本 ID")
    parser.add_argument("--id", help="要运行的样本 ID")
    parser.add_argument("--auto", action="store_true",
                        help="自动从映射文件读取影像并分配给占位符")
    parser.add_argument("--augmented", action="store_true",
                        help="使用扩增数据集和映射文件")
    parser.add_argument("--mapping", help="自定义映射文件路径（覆盖默认）")
    parser.add_argument("--docker", action="store_true",
                        help="使用 Docker 版感知服务（VLM/SAM2/CLIP 等）")
    parser.add_argument("--keep-services", action="store_true",
                        help="测试结束后保留 Docker 感知服务（避免下次重新加载模型）")
    parser.add_argument(
        "--inputs", nargs="*", metavar="placeholder=real_path",
        help="文件路径映射，格式: placeholder.tif=/real/path.tif"
    )
    parser.add_argument("--workdir", help="工作目录（存放中间/输出文件），默认使用临时目录")
    args = parser.parse_args()

    # Docker 模式：在 toolkit 导入前设置环境变量
    if args.docker:
        os.environ["TERRABOX_USE_DOCKER"] = "true"
        print("[docker] 启用 Docker 版感知服务")

    # --keep-services: 禁用 atexit 自动清理，保留 Docker 服务
    if args.keep_services:
        os.environ["_TERRABOX_KEEP_SERVICES"] = "true"

    # 选择数据集
    ds_path = AUGMENTED_PATH if args.augmented else DATASET_PATH
    if not ds_path.exists():
        print(f"错误：数据集不存在 {ds_path}")
        return 1

    dataset = json.loads(ds_path.read_text())

    if args.list:
        print(f"共 {len(dataset['samples'])} 条样本:\n")
        print(f"{'ID':40s}  {'task_type':28s}  {'disaster':12s}  difficulty")
        print("-" * 100)
        for s in dataset["samples"]:
            print(f"{s['id']:40s}  {s['task_type']:28s}  {s['disaster_category']:12s}  {s.get('difficulty','')}")
        return 0

    if not args.id:
        parser.print_help()
        return 1

    sample = find_sample(dataset, args.id)
    if sample is None:
        print(f"错误：找不到样本 ID '{args.id}'")
        print("使用 --list 查看所有可用 ID")
        return 1

    print("=" * 60)
    print(f"样本: {sample['id']}")
    print(f"类型: {sample['task_type']}  灾害: {sample['disaster_category']}")
    print(f"任务: {sample['prompt']}")
    print("=" * 60)

    # 分析叶节点输入
    leaf_inputs = find_leaf_inputs(sample["tool_calls"])
    print(f"\n该样本需要 {len(leaf_inputs)} 个输入文件: {leaf_inputs}")

    # 设置工作目录（提前，auto 模式需要用到）
    # Docker 模式下，工作目录必须在容器挂载路径内（默认 /data1），
    # 否则 SAM2/RemoteCLIP 等容器无法访问输入文件。
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        workdir = args.workdir
        _cleanup = False
    elif args.docker:
        _tmpdir = tempfile.mkdtemp(
            prefix="sft_run_",
            dir=str(REPO_ROOT / "terrabox_uploads")  # 在 /data1 下，容器可见
        )
        workdir = _tmpdir
        _cleanup = True
    else:
        _tmpdir = tempfile.mkdtemp(prefix="sft_run_")
        workdir = _tmpdir
        _cleanup = True

    # 构建路径映射
    path_map: dict[str, str] = {}

    if args.auto:
        # 自动模式：从映射文件读取影像
        if args.mapping:
            mp = Path(args.mapping)
        elif args.augmented:
            mp = AUGMENTED_MAPPING_PATH
        else:
            mp = MAPPING_PATH

        print(f"\n[auto] 使用映射文件: {mp}")
        path_map, auto_warnings = auto_map_inputs(
            sample["id"], leaf_inputs, mp, workdir
        )
        if auto_warnings:
            print("[auto] 提示:")
            for w in auto_warnings:
                print(f"  {w}")

        if not path_map:
            print("[auto] 未能生成任何映射，退出")
            return 1

    # --inputs 可叠加或覆盖 auto 映射
    if args.inputs:
        for mapping in args.inputs:
            if "=" not in mapping:
                print(f"警告：忽略无效映射格式 '{mapping}'（应为 placeholder=real_path）")
                continue
            placeholder, real_path = mapping.split("=", 1)
            path_map[placeholder.strip()] = real_path.strip()

    # 非 auto 模式：交互补全缺少的映射
    if not args.auto:
        missing = [p for p in leaf_inputs if p not in path_map]
        if missing:
            print("\n请为以下占位符提供真实文件路径（直接回车跳过该输入）:")
            for ph in missing:
                try:
                    real = input(f"  {ph} = ").strip()
                except (EOFError, KeyboardInterrupt):
                    real = ""
                if real:
                    path_map[ph] = real

    # 确认哪些输入已映射
    print("\n文件路径映射:")
    for ph in leaf_inputs:
        real = path_map.get(ph, "【未提供，将使用占位符】")
        status = "✓" if ph in path_map and os.path.exists(path_map[ph]) else ("?" if ph in path_map else "✗")
        print(f"  [{status}] {ph:30s} → {real}")

    print(f"\n工作目录: {workdir}")
    print("\n开始执行工具链 ...")

    results = run_sample(sample, path_map, workdir)
    summarize_results(results)

    # 列出工作目录中生成的所有文件
    generated_files = sorted(Path(workdir).iterdir())
    if generated_files:
        print(f"\n生成文件 ({workdir}):")
        for f in generated_files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:40s}  {size_kb:8.1f} KB")

    failed = sum(1 for r in results if "error" in r["result"])
    print(f"\n完成: {len(results) - failed}/{len(results)} 步成功")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
