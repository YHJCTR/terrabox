#!/usr/bin/env python3
"""
验证 90 条 SFT 数据是否能通过 agent 工具链成功执行。
测试每条样本的工具调用流程（不涉及 LLM 推理，仅验证工具可达性）。
"""
import json
import sys
import logging
from pathlib import Path
from collections import Counter

sys.path.insert(0, "src")

from terrabox.core.registry import CoreRegistry
from terrabox.evolution.shared.trajectory import Trajectory, Turn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_eval_cases(eval_jsonl: str) -> list[dict]:
    """Load evaluation cases from JSONL."""
    cases = []
    with open(eval_jsonl) as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def load_trajectories(traj_json: str) -> dict[str, Trajectory]:
    """Load training trajectories."""
    with open(traj_json) as f:
        data = json.load(f)

    traj_map = {}
    for td in data:
        turns = [Turn(**t) for t in td.get("turns", [])]
        traj = Trajectory(
            task_id=td.get("task_id"),
            question=td.get("question"),
            images=td.get("images", []),
            turns=turns,
            tools_called=td.get("tools_called", []),
            expected_tools=td.get("expected_tools", []),
            final_answer=td.get("final_answer", ""),
            success=td.get("success", True),
            source=td.get("source", ""),
            task_type=td.get("task_type", "")
        )
        traj_map[traj.task_id] = traj
    return traj_map


def verify_tools_available(registry: CoreRegistry, tools_called: list[str]) -> tuple[list[str], list[str]]:
    """
    验证所有工具是否在注册表中可用。
    返回 (available_tools, missing_tools)
    """
    available = []
    missing = []

    for tool_slug in tools_called:
        try:
            spec = registry.get_tool(tool_slug)
            if spec:
                available.append(tool_slug)
            else:
                missing.append(tool_slug)
        except Exception as e:
            missing.append(tool_slug)

    return available, missing


def verify_images_exist(images: list[str]) -> tuple[int, int]:
    """验证图像文件是否存在。返回 (exist_count, missing_count)"""
    exist = 0
    missing = 0
    for img_path in images:
        if img_path and Path(img_path).exists():
            exist += 1
        else:
            missing += 1
    return exist, missing


def main():
    eval_jsonl = "data/disaster_eval.jsonl"
    traj_json = "data/disaster_trajectories.json"

    print("\n" + "="*80)
    print("验证 90 条 SFT 数据能否通过工具链执行")
    print("="*80)

    # 加载数据
    eval_cases = load_eval_cases(eval_jsonl)
    traj_map = load_trajectories(traj_json)
    registry = CoreRegistry()

    print(f"\n✓ 已加载 {len(eval_cases)} 条评测案例")
    print(f"✓ 已加载 {len(traj_map)} 条训练轨迹")

    # 验证每条案例
    results = {
        "total": len(eval_cases),
        "passed": 0,
        "failed": 0,
        "failures": []
    }

    tool_stats = Counter()
    image_stats = {"exist": 0, "missing": 0}

    print(f"\n{'任务ID':30} {'状态':10} {'工具':15} {'图像':10}")
    print("-" * 70)

    for i, case in enumerate(eval_cases):
        task_id = case.get("id") or case.get("task_id", "")
        normalized_task_id = task_id
        if task_id.startswith("disaster_"):
            normalized_task_id = task_id.replace("disaster_", "", 1)

        # 获取训练数据中的工具列表
        if normalized_task_id in traj_map:
            traj = traj_map[normalized_task_id]
            tools_called = traj.tools_called
        else:
            tools_called = case.get("expected_tools", [])

        images = case.get("images", [])

        # 验证工具
        available, missing = verify_tools_available(registry, tools_called)
        for t in available:
            tool_stats[t] += 1

        # 验证图像
        exist_count, missing_count = verify_images_exist(images)
        image_stats["exist"] += exist_count
        image_stats["missing"] += missing_count

        # 判断通过/失败
        tools_ok = len(missing) == 0
        images_ok = missing_count == 0
        passed = tools_ok and images_ok

        if passed:
            results["passed"] += 1
            status = "✓ PASS"
        else:
            results["failed"] += 1
            status = "✗ FAIL"
            results["failures"].append({
                "task_id": task_id,
                "missing_tools": missing,
                "missing_images": missing_count
            })

        tools_str = f"{len(available)}/{len(tools_called)}"
        images_str = f"{exist_count}/{exist_count + missing_count}"

        print(f"{task_id:30} {status:10} {tools_str:15} {images_str:10}")

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_cases)}]")

    # 汇总结果
    print("\n" + "="*80)
    print("验证结果汇总")
    print("="*80)

    print(f"\n总体结果：")
    print(f"  ✓ 通过：{results['passed']}/{results['total']} ({100*results['passed']/results['total']:.1f}%)")
    print(f"  ✗ 失败：{results['failed']}/{results['total']}")

    print(f"\n图像文件验证：")
    print(f"  ✓ 存在：{image_stats['exist']} 个")
    print(f"  ✗ 缺失：{image_stats['missing']} 个")

    if tool_stats:
        print(f"\n工具使用统计（Top 10）：")
        for tool, count in tool_stats.most_common(10):
            print(f"  {tool:40} {count:3} 次")

    if results["failures"]:
        print(f"\n失败案例详情（前10）：")
        for failure in results["failures"][:10]:
            print(f"\n  {failure['task_id']}:")
            if failure['missing_tools']:
                print(f"    - 缺失工具：{failure['missing_tools']}")
            if failure['missing_images']:
                print(f"    - 缺失图像：{failure['missing_images']} 个")

    print("\n" + "="*80)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
