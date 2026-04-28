"""
fix_sft_toolchain.py — 修正 SFT 数据集工具链
==============================================
输入：data/disaster_sft_dataset.json
输出：data/disaster_sft_dataset_v2.json

修正内容：
1. A类任务（洪水/NDWI）：移除 calculate_index(NDWI) 及其依赖链
   （threshold_segmentation → count_pixels_condition → pixel_area）
2. 冗余工具移除：在 building_damage_assess / earthquake_damage / landslide 中
   移除输出未被引用的 remoteclip_analysis 步骤
"""
import json
import re
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH  = ROOT / "data" / "disaster_sft_dataset.json"
OUT_PATH = ROOT / "data" / "disaster_sft_dataset_v2.json"

# A类任务：洪水/NDWI（移除 calculate_index 及其级联链）
# 注意：zonal_flood_stats / grid_flood_assessment / disaster_astar_routing 中
# NDWI 是算法核心输入（而非预处理），移除后工具链过于残缺，故保留 v1 版本。
A_TASKS = {
    "flood_detection", "flood_change", "flood_timeseries",
    "population_flood_risk",
    "flood_rescue_routing", "rescue_accessibility",
}

# NDWI级联链（只有当它依赖 calculate_index 输出时才移除）
CASCADE_TOOLS = {
    "geo_raster.threshold_segmentation",
    "geo_statistics.count_pixels_condition",
    "geo_basic.pixel_area",
}

# 纯数据流冗余（remoteclip 在这些任务中输出未被引用）
REDUNDANT_CLIP_TASKS = {
    "building_damage_assess",
    "earthquake_damage",
    "landslide",
}


def _collect_output_paths(tc: dict) -> set[str]:
    """收集某步骤 args 中所有 output_path / result_path 字段值。"""
    args = tc.get("args", {})
    paths = set()
    for key in ("output_path", "result_path", "out_path"):
        v = args.get(key)
        if v and isinstance(v, str) and not v.startswith("$"):
            paths.add(v)
    return paths


def _uses_file_from(tc: dict, file_paths: set[str]) -> bool:
    """判断某步骤的 input args 是否依赖给定文件路径之一。"""
    args = tc.get("args", {})
    for key, val in args.items():
        if key.startswith("output") or key.startswith("result"):
            continue
        if isinstance(val, str) and val in file_paths:
            return True
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item in file_paths:
                    return True
    return False


def find_steps_to_remove(tool_calls: list[dict], task_type: str) -> set[int]:
    """返回需要移除的 step 编号集合。"""
    remove = set()

    # --- A类：移除 NDWI 流水线（+ 全链级联）---
    if task_type in A_TASKS:
        # 找所有 calculate_index 步骤
        for tc in tool_calls:
            if tc["tool"] == "geo_raster.calculate_index":
                remove.add(tc["step"])

        # 级联：任何工具，只要其输入来自待移除步骤，就一并移除
        # 检测方式：$stepN 显式引用 OR 文件路径依赖（output_path → input_path）
        changed = True
        while changed:
            changed = False
            removed_output_files: set[str] = set()
            for tc in tool_calls:
                if tc["step"] in remove:
                    removed_output_files.update(_collect_output_paths(tc))

            for tc in tool_calls:
                if tc["step"] in remove:
                    continue
                args_str = json.dumps(tc.get("args", {}))
                ref_match = any(f"$step{s}" in args_str for s in remove)
                file_match = _uses_file_from(tc, removed_output_files)
                if ref_match or file_match:
                    remove.add(tc["step"])
                    changed = True

    # --- 冗余 remoteclip_analysis（输出未被任何后续步骤引用）---
    if task_type in REDUNDANT_CLIP_TASKS:
        all_args = " ".join(
            json.dumps(tc.get("args", {})) for tc in tool_calls
        )
        for tc in tool_calls:
            if tc["tool"] != "geo_perception.remoteclip_analysis":
                continue
            # 检查是否有后续步骤引用它的输出
            if f"$step{tc['step']}" not in all_args:
                remove.add(tc["step"])

    return remove


def remap_refs(args_str: str, step_map: dict[int, int]) -> str:
    """将 args JSON 字符串中的 $stepN 替换为新编号。"""
    # 从大到小替换，避免 step10 被 step1 误匹配
    for old, new in sorted(step_map.items(), reverse=True):
        args_str = args_str.replace(f"$step{old}", f"$step{new}")
    return args_str


def fix_sample(sample: dict) -> tuple[dict, dict]:
    """
    返回 (fixed_sample, change_log)。
    change_log: {"removed": [...], "renumbered": True/False}
    """
    s = copy.deepcopy(sample)
    task = s["task_type"]
    tool_calls = s.get("tool_calls", [])

    remove_steps = find_steps_to_remove(tool_calls, task)

    if not remove_steps:
        return s, {"removed": [], "renumbered": False}

    # 保留未移除的步骤
    kept = [tc for tc in tool_calls if tc["step"] not in remove_steps]

    # 建立旧编号 → 新编号映射
    step_map: dict[int, int] = {}
    for new_idx, tc in enumerate(kept, 1):
        step_map[tc["step"]] = new_idx

    # 重新编号 + 修正 $stepN 引用
    new_tcs = []
    for tc in kept:
        tc = copy.deepcopy(tc)
        old_step = tc["step"]
        tc["step"] = step_map[old_step]

        # 修正 args 中的引用
        args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
        args_str = remap_refs(args_str, step_map)
        tc["args"] = json.loads(args_str)

        # 修正 sample_output 中如果有引用（少见，保险起见）
        if "sample_output" in tc:
            so_str = json.dumps(tc["sample_output"], ensure_ascii=False)
            so_str = remap_refs(so_str, step_map)
            tc["sample_output"] = json.loads(so_str)

        new_tcs.append(tc)

    removed_tools = [
        tc["tool"] for tc in tool_calls if tc["step"] in remove_steps
    ]

    # 若移除后剩 0 步（整条链都依赖 NDWI），补充最小 VLM 工具链
    if not new_tcs:
        # 从影像映射中取占位路径（showcase 会注入真实路径）
        new_tcs = [
            {
                "step": 1,
                "tool": "geo_perception.vlm_analyze",
                "args": {
                    "image_paths": ["rgb.tif"],
                    "prompt": "分析图中的灾害区域，识别受影响范围并描述灾情严重程度",
                },
                "sample_output": {},
            },
            {
                "step": 2,
                "tool": "geo_perception.draw_bboxes",
                "args": {
                    "image_path": "rgb.tif",
                    "bboxes": "$step1.bboxes",
                    "output_path": "annotated.png",
                },
                "sample_output": {},
            },
        ]

    s["tool_calls"] = new_tcs

    return s, {
        "removed": removed_tools,
        "renumbered": True,
        "orig_steps": len(tool_calls),
        "new_steps": len(new_tcs),
        "fallback_added": len(removed_tools) == len(tool_calls),
    }


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    new_samples = []
    stats = {
        "total": len(samples),
        "modified": 0,
        "unchanged": 0,
        "by_task": {},
    }

    print(f"处理 {len(samples)} 条样本...")
    for s in samples:
        fixed, log = fix_sample(s)
        new_samples.append(fixed)

        task = s["task_type"]
        if log["removed"]:
            stats["modified"] += 1
            if task not in stats["by_task"]:
                stats["by_task"][task] = {"count": 0, "removed_tools": set()}
            stats["by_task"][task]["count"] += 1
            stats["by_task"][task]["removed_tools"].update(log["removed"])
            print(
                f"  ✓ {s['id']:40s}  {log['orig_steps']}步 → {log['new_steps']}步  "
                f"移除: {log['removed']}"
            )
        else:
            stats["unchanged"] += 1

    # 写入新数据集
    new_data = copy.deepcopy(data)
    new_data["samples"] = new_samples
    new_data["version"] = str(float(data.get("version", 1)) + 0.1)
    new_data["description"] = (
        data.get("description", "") +
        " [v2: NDWI pipeline removed for RGB-only tasks; redundant remoteclip_analysis removed]"
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== 完成 ===")
    print(f"输出: {OUT_PATH}")
    print(f"修改: {stats['modified']} 条  未改: {stats['unchanged']} 条")
    print()
    print("按任务类型统计:")
    for task, info in sorted(stats["by_task"].items()):
        print(f"  {task:35s}: {info['count']} 条  移除工具: {sorted(info['removed_tools'])}")


if __name__ == "__main__":
    main()
