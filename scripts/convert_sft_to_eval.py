"""
将 disaster_sft_dataset.json 转换为 evolution 框架使用的 eval.jsonl 格式。

同时导出：
  data/disaster_eval.jsonl          — evolution eval 格式（90 条）
  data/disaster_failure_refs.jsonl  — 已知失败模式参考（从 test report 中提取）

运行:
  python scripts/convert_sft_to_eval.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT_PATH   = ROOT / "data" / "disaster_sft_dataset.json"
EVAL_OUT   = ROOT / "data" / "disaster_eval.jsonl"
REPORT_PATH = ROOT / "data" / "flow_test_report.json"
FAILURE_OUT = ROOT / "data" / "disaster_failure_refs.jsonl"


def extract_input_files(tool_calls: list[dict]) -> list[str]:
    """
    从 tool_calls 的 args 中提取原始输入文件名（.tif / .geojson）。
    排除 $step 引用和明显是中间产物的输出路径。
    """
    # 中间产物关键字（这些文件由工具链内部生成）
    GENERATED = {
        "ndwi", "ndvi", "ndsi", "tvdi", "lst", "frp", "fvc",
        "flood_mask", "flood_increase", "flood_norm", "road_pass",
        "slope", "flow_dir", "access_time", "exposed_pop", "pop_hotspot",
        "fire_mask", "fire_detected", "fire_current", "fire_forecast",
        "change_diff", "change_mask", "damage_diff", "damage_mask",
        "water_pre", "water_post", "snow_pre", "snow_post", "snow_change",
        "snow_increase", "fire_pre", "fire_post", "fire_spread",
        "frp_clean", "high_risk", "gi_star", "hotspot",
        "annotated", "detected", "map", "output",
    }

    seen = set()
    result = []
    for tc in tool_calls:
        for v in tc.get("args", {}).values():
            if isinstance(v, str) and not v.startswith("$step"):
                if v.endswith(".tif") or v.endswith(".geojson"):
                    stem = Path(v).stem.lower()
                    if not any(kw in stem for kw in GENERATED):
                        if v not in seen:
                            seen.add(v)
                            result.append(v)
    return result


def convert_sft_to_eval(samples: list[dict]) -> list[dict]:
    entries = []
    for s in samples:
        # Deduplicate tools while preserving order
        seen_tools: dict[str, None] = {}
        for tc in s["tool_calls"]:
            seen_tools[tc["tool"]] = None
        expected_tools = list(seen_tools.keys())

        entry = {
            "id": f"disaster_{s['id']}",
            "source": "disaster_sft",
            "task_type": s["task_type"],
            "disaster_category": s["disaster_category"],
            "difficulty": s["difficulty"],
            "images": [],
            "question": s["prompt"],
            "expected_tools": expected_tools,
            "data_dir": None,
            "data_files": extract_input_files(s["tool_calls"]),
            "ground_truth": None,
            "modality": None,
        }
        entries.append(entry)
    return entries


def build_failure_refs(samples: list[dict], report_path: Path) -> list[dict]:
    """
    从测试报告中提取失败样本，构建 failure_refs：
    包含 question、actual_tools（bug 工具链）、expected_tools（正确工具链）。
    """
    if not report_path.exists():
        print(f"  [warn] 未找到测试报告 {report_path}，跳过 failure_refs 生成")
        return []

    report = json.loads(report_path.read_text())
    fv = report.get("flow_validation", {})

    # Index samples by id
    sample_idx = {s["id"]: s for s in samples}

    refs = []
    for entry_id, entry in fv.items():
        if isinstance(entry, dict) and not entry.get("passed", True):
            # entry_id 是 flow_validation 中的 key，格式如 "flood_detection_01"
            sample = sample_idx.get(entry_id)
            if not sample:
                continue
            # actual_tools = 当前（有 bug 的）工具序列
            actual_tools = [tc["tool"] for tc in sample["tool_calls"]]
            # errors summary
            error_msgs = []
            for step_err in entry.get("errors", []):
                error_msgs.extend(step_err.get("errors", []))

            ref = {
                "id": f"failure_{entry_id}",
                "source": "disaster_sft_failure",
                "task_type": entry["task_type"],
                "question": sample["prompt"],
                "actual_tools": actual_tools,
                "expected_tools": actual_tools,  # same sequence, errors are in params not tool choice
                "errors": error_msgs[:5],  # keep top 5
            }
            refs.append(ref)
    return refs


def main():
    print(f"读取数据集: {SFT_PATH}")
    data = json.loads(SFT_PATH.read_text())
    samples = data["samples"]
    print(f"  共 {len(samples)} 条样本")

    # --- eval.jsonl ---
    eval_entries = convert_sft_to_eval(samples)
    with EVAL_OUT.open("w", encoding="utf-8") as f:
        for e in eval_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n[OK] 写入 {EVAL_OUT}")
    print(f"     {len(eval_entries)} 行，每行含 expected_tools 列表")

    # Show a sample entry
    sample_entry = eval_entries[0]
    print(f"\n  示例（第1条）:")
    print(f"    id            : {sample_entry['id']}")
    print(f"    task_type     : {sample_entry['task_type']}")
    print(f"    expected_tools: {sample_entry['expected_tools']}")
    print(f"    data_files    : {sample_entry['data_files']}")

    # --- failure_refs.jsonl ---
    failure_refs = build_failure_refs(samples, REPORT_PATH)
    if failure_refs:
        with FAILURE_OUT.open("w", encoding="utf-8") as f:
            for r in failure_refs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[OK] 写入 {FAILURE_OUT}")
        print(f"     {len(failure_refs)} 条已知失败模式参考")
    else:
        print(f"\n[!] 未生成 failure_refs（测试报告中无失败样本，或报告不存在）")

    # --- Evolution 使用提示 ---
    print("""
[使用方法]
  # 接入 SkillRL eval（评估 evolution 效果）
  /data1/yuhongjie2/env/unsloth/bin/python -m terrabox.evolution.skillrl.runner eval \\
      --eval-data data/disaster_eval.jsonl \\
      --store-dir evolution_store/skillrl_test \\
      --top-k 3

  # 接入 MemRL populate（填充情节记忆）
  /data1/yuhongjie2/env/unsloth/bin/python -m terrabox.evolution.memrl.runner populate \\
      --train-data data/openearth/train.json \\
      --eval-data data/disaster_eval.jsonl \\
      --memory-db evolution_store/memrl/episodic_memory.db
""")


if __name__ == "__main__":
    main()
