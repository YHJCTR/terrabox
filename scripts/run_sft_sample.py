#!/usr/bin/env python3
"""
run_sft_sample.py — 用真实图像执行某条灾害 SFT 样本的工具链

用法:
  # 列出所有可用样本 ID
  python scripts/run_sft_sample.py --list

  # 交互模式：脚本自动检测所需输入图像，逐一提示你填写真实路径
  python scripts/run_sft_sample.py --id flood_detection_01

  # 非交互：用 --inputs 直接提供映射（占位符=真实路径）
  python scripts/run_sft_sample.py --id flood_detection_01 \\
      --inputs green.tif=/data/s2/band3.tif nir.tif=/data/s2/band8.tif

  # 指定工作目录（中间/输出文件落地于此）
  python scripts/run_sft_sample.py --id flood_detection_01 --workdir /tmp/test_run

运行环境:
  /data1/yuhongjie2/env/earth/bin/python scripts/run_sft_sample.py ...
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

# 识别文件路径的正则：不以 $step 开头，以 .tif/.tiff/.png 结尾
_PATH_RE = re.compile(r"^(?!\$step)[\w/._-]+\.(tif|tiff|png)$", re.IGNORECASE)

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
    from terrabox.extensions import load_builtin_toolkits, Registrar
    from terrabox.core.registry import get_handler

    r = Registrar()
    load_builtin_toolkits(r)

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
    parser.add_argument(
        "--inputs", nargs="*", metavar="placeholder=real_path",
        help="文件路径映射，格式: placeholder.tif=/real/path.tif"
    )
    parser.add_argument("--workdir", help="工作目录（存放中间/输出文件），默认使用临时目录")
    args = parser.parse_args()

    dataset = load_dataset()

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

    # 解析用户提供的 --inputs 映射
    path_map: dict[str, str] = {}
    if args.inputs:
        for mapping in args.inputs:
            if "=" not in mapping:
                print(f"警告：忽略无效映射格式 '{mapping}'（应为 placeholder=real_path）")
                continue
            placeholder, real_path = mapping.split("=", 1)
            path_map[placeholder.strip()] = real_path.strip()

    # 交互补全缺少的映射
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

    # 设置工作目录
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        workdir = args.workdir
        _cleanup = False
    else:
        _tmpdir = tempfile.mkdtemp(prefix="sft_run_")
        workdir = _tmpdir
        _cleanup = True

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
