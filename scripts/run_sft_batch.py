#!/usr/bin/env python3
"""
run_sft_batch.py — 批量执行 disaster SFT 数据集（90 条主数据集）工具链

用法:
  # Docker 模式，保留感知服务（推荐）
  python scripts/run_sft_batch.py --docker --keep-services

  # 只测试指定灾害类别
  python scripts/run_sft_batch.py --docker --keep-services --category earthquake

  # 只测试指定任务类型
  python scripts/run_sft_batch.py --docker --keep-services --task-type earthquake_damage

  # 指定输出报告路径
  python scripts/run_sft_batch.py --docker --keep-services --report /tmp/batch_report.json

  # 失败后继续（默认），或失败即停
  python scripts/run_sft_batch.py --docker --keep-services --fail-fast

运行环境:
  /data1/yuhongjie2/env/unsloth/bin/python scripts/run_sft_batch.py --docker --keep-services
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# 复用 run_sft_sample 中的所有核心逻辑
from run_sft_sample import (
    find_leaf_inputs,
    auto_map_inputs,
    run_sample,
    DATASET_PATH,
    MAPPING_PATH,
)

BATCH_WORKDIR_BASE = REPO_ROOT / "terrabox_uploads" / "batch_runs"


def run_one_sample(
    sample: dict,
    mapping_path: Path,
    base_workdir: str,
    keep_services: bool,
) -> dict:
    """执行单条样本，返回结果摘要。"""
    sid = sample["id"]
    workdir = os.path.join(base_workdir, sid)
    os.makedirs(workdir, exist_ok=True)

    leaf_inputs = find_leaf_inputs(sample["tool_calls"])

    path_map, auto_warnings = auto_map_inputs(sid, leaf_inputs, mapping_path, workdir)

    if not path_map and leaf_inputs:
        return {
            "id": sid,
            "task_type": sample["task_type"],
            "disaster_category": sample["disaster_category"],
            "status": "skip",
            "reason": "无可用影像映射",
            "warnings": auto_warnings,
            "steps_total": len(sample["tool_calls"]),
            "steps_passed": 0,
            "steps_failed": 0,
            "duration_s": 0,
            "step_results": [],
        }

    if keep_services:
        os.environ["_TERRABOX_KEEP_SERVICES"] = "true"

    t0 = time.time()
    try:
        step_results = run_sample(sample, path_map, workdir)
    except Exception as e:
        return {
            "id": sid,
            "task_type": sample["task_type"],
            "disaster_category": sample["disaster_category"],
            "status": "error",
            "reason": str(e),
            "traceback": traceback.format_exc(),
            "warnings": auto_warnings,
            "steps_total": len(sample["tool_calls"]),
            "steps_passed": 0,
            "steps_failed": len(sample["tool_calls"]),
            "duration_s": round(time.time() - t0, 1),
            "step_results": [],
        }
    duration = round(time.time() - t0, 1)

    failed_steps = [r for r in step_results if "error" in r["result"]]
    passed_steps = [r for r in step_results if "error" not in r["result"]]

    # 收集精简的步骤摘要（不保留大型图像 base64）
    steps_summary = []
    for r in step_results:
        res = r["result"]
        brief = {}
        for k, v in res.items():
            if k in ("status", "error", "message", "traceback"):
                brief[k] = v
            elif isinstance(v, str) and len(v) > 200:
                brief[k] = v[:200] + "...[truncated]"
            elif k == "bboxes" and isinstance(v, list):
                brief[k] = f"[{len(v)} bboxes]"
            elif k == "output" and isinstance(v, str) and v.startswith("SAM2"):
                brief[k] = v.split("\n")[0]
            else:
                brief[k] = v
        steps_summary.append({
            "step": r["step"],
            "tool": r["tool"],
            "passed": "error" not in res,
            "result_brief": brief,
        })

    return {
        "id": sid,
        "task_type": sample["task_type"],
        "disaster_category": sample["disaster_category"],
        "status": "pass" if not failed_steps else "fail",
        "warnings": auto_warnings,
        "steps_total": len(step_results),
        "steps_passed": len(passed_steps),
        "steps_failed": len(failed_steps),
        "duration_s": duration,
        "workdir": workdir,
        "step_results": steps_summary,
    }


def print_sample_result(result: dict, idx: int, total: int) -> None:
    sid = result["id"]
    status = result["status"]
    icon = {"pass": "✓", "fail": "✗", "skip": "○", "error": "✗"}.get(status, "?")
    print(
        f"  [{idx:3d}/{total}] [{icon}] {sid:45s}  "
        f"{result['steps_passed']}/{result['steps_total']} steps  "
        f"{result['duration_s']:.1f}s"
    )
    if status in ("fail", "error"):
        for s in result.get("step_results", []):
            if not s["passed"]:
                brief = s["result_brief"]
                msg = brief.get("message") or brief.get("error") or str(brief)
                print(f"           step {s['step']:2d} [{s['tool']}]  ✗ {msg[:100]}")
    if result.get("reason"):
        print(f"           reason: {result['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量执行 disaster SFT 主数据集（90 条）工具链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--docker", action="store_true",
                        help="使用 Docker 版感知服务（VLM/SAM2/RemoteCLIP 等）")
    parser.add_argument("--keep-services", action="store_true",
                        help="测试结束后保留 Docker 感知服务")
    parser.add_argument("--category", metavar="NAME",
                        help="只测试指定灾害类别（如 earthquake、flood）")
    parser.add_argument("--task-type", metavar="NAME",
                        help="只测试指定任务类型（如 earthquake_damage）")
    parser.add_argument("--report", metavar="PATH",
                        help="报告输出路径（默认: data/batch_test_report.json）")
    parser.add_argument("--workdir", metavar="DIR",
                        help="批量工作目录基路径（默认: terrabox_uploads/batch_runs/<timestamp>）")
    parser.add_argument("--fail-fast", action="store_true",
                        help="遇到失败立即停止")
    args = parser.parse_args()

    # Docker 模式：在 toolkit 导入前设置环境变量
    if args.docker:
        os.environ["TERRABOX_USE_DOCKER"] = "true"
        print("[docker] 启用 Docker 版感知服务")

    if args.keep_services:
        os.environ["_TERRABOX_KEEP_SERVICES"] = "true"

    # 加载数据集
    if not DATASET_PATH.exists():
        print(f"错误：数据集不存在 {DATASET_PATH}")
        return 1

    dataset = json.loads(DATASET_PATH.read_text())
    samples = dataset["samples"]

    # 过滤
    if args.category:
        samples = [s for s in samples if s["disaster_category"] == args.category]
        print(f"[filter] disaster_category={args.category}，共 {len(samples)} 条")
    if args.task_type:
        samples = [s for s in samples if s["task_type"] == args.task_type]
        print(f"[filter] task_type={args.task_type}，共 {len(samples)} 条")

    if not samples:
        print("过滤后无样本，退出")
        return 0

    # 工作目录
    if args.workdir:
        base_workdir = args.workdir
    elif args.docker:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_workdir = str(BATCH_WORKDIR_BASE / ts)
    else:
        base_workdir = tempfile.mkdtemp(prefix="sft_batch_")
    os.makedirs(base_workdir, exist_ok=True)

    report_path = Path(args.report) if args.report else REPO_ROOT / "data" / "batch_test_report.json"

    print(f"\n{'=' * 60}")
    print(f"批量测试: {len(samples)} 条样本")
    print(f"工作目录: {base_workdir}")
    print(f"报告路径: {report_path}")
    print(f"{'=' * 60}\n")

    results = []
    t_batch_start = time.time()

    for idx, sample in enumerate(samples, 1):
        sid = sample["id"]
        print(f"\n[{idx}/{len(samples)}] {sid}")
        result = run_one_sample(
            sample=sample,
            mapping_path=MAPPING_PATH,
            base_workdir=base_workdir,
            keep_services=args.keep_services,
        )
        results.append(result)
        print_sample_result(result, idx, len(samples))

        if args.fail_fast and result["status"] in ("fail", "error"):
            print("\n[fail-fast] 遇到失败，停止测试")
            break

    total_duration = round(time.time() - t_batch_start, 1)

    # 统计
    passed  = [r for r in results if r["status"] == "pass"]
    failed  = [r for r in results if r["status"] == "fail"]
    errored = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skip"]

    # 按灾害类别统计
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r["disaster_category"]
        if cat not in by_category:
            by_category[cat] = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
        by_category[cat][r["status"]] += 1

    print(f"\n{'=' * 60}")
    print("批量测试结果汇总")
    print(f"{'=' * 60}")
    print(f"总计: {len(results)} 条  耗时: {total_duration:.1f}s")
    print(f"  ✓ 通过: {len(passed)}")
    print(f"  ✗ 失败: {len(failed) + len(errored)}")
    print(f"  ○ 跳过: {len(skipped)}")

    if by_category:
        print("\n按灾害类别:")
        for cat, counts in sorted(by_category.items()):
            total_cat = sum(counts.values())
            print(f"  {cat:20s}  通过 {counts['pass']}/{total_cat}  "
                  f"失败 {counts['fail']+counts['error']}  跳过 {counts['skip']}")

    if failed or errored:
        print("\n失败任务:")
        for r in failed + errored:
            print(f"  ✗ {r['id']}  ({r['steps_passed']}/{r['steps_total']} steps)")
            for s in r.get("step_results", []):
                if not s["passed"]:
                    msg = s["result_brief"].get("message") or s["result_brief"].get("error", "")
                    print(f"      step {s['step']} [{s['tool']}]: {str(msg)[:120]}")

    # 写报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(results),
        "passed": len(passed),
        "failed": len(failed) + len(errored),
        "skipped": len(skipped),
        "total_duration_s": total_duration,
        "by_category": by_category,
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {report_path}")

    # 写失败记录到项目根目录
    failure_records = []
    for r in results:
        if r["status"] not in ("fail", "error"):
            continue
        for s in r.get("step_results", []):
            if not s["passed"]:
                brief = s["result_brief"]
                failure_records.append({
                    "sft_id": r["id"],
                    "task_type": r["task_type"],
                    "disaster_category": r["disaster_category"],
                    "step": s["step"],
                    "tool": s["tool"],
                    "error": brief.get("message") or brief.get("error") or str(brief),
                })

    failures_path = REPO_ROOT / "sft_failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_failures": len(failure_records),
            "failures": failure_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"失败记录已写入: {failures_path}  ({len(failure_records)} 条失败步骤)")

    return 0 if not (failed or errored) else 1


if __name__ == "__main__":
    sys.exit(main())
