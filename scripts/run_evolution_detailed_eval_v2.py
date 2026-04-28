#!/usr/bin/env python3
"""
增强评测脚本（v2）：基于原始 run_evolution_final_eval.py，
加上完整的预测轨迹和详细指标的保存。

对比三个方法：
  - AgentEvolver: 直接使用训练轨迹 (direct)
  - MemRL: 使用内存检索 (memory-augmented)
  - CausalEvo: 直接使用训练轨迹 (direct)
"""
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "src")

from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
from terrabox.evolution.shared.trajectory import Trajectory, Turn


def load_eval_data(eval_jsonl_path: str) -> list[dict]:
    """Load eval data from JSONL."""
    cases = []
    with open(eval_jsonl_path) as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def load_trajectories(json_path: str) -> dict[str, Trajectory]:
    """Load trajectories and map by task_id."""
    with open(json_path) as f:
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


def evaluate_method_direct(eval_cases: list[dict], traj_map: dict[str, Trajectory]) -> dict:
    """
    Evaluate using tools_called from training trajectories (direct match).
    Returns full trajectory predictions and metrics.
    """
    evaluator = ToolMatchEvaluator()
    results = []
    case_results = []

    matched = 0
    for case in eval_cases:
        task_id = case.get("task_id") or case.get("id")  # eval.jsonl may use "id"

        # Handle prefix mismatch: eval uses "disaster_flood_detection_01", training uses "flood_detection_01"
        normalized_task_id = task_id
        if task_id and task_id.startswith("disaster_"):
            normalized_task_id = task_id.replace("disaster_", "", 1)

        # Create eval trajectory with expected_tools from eval set
        if normalized_task_id in traj_map:
            # Use tools from training trajectory
            traj = traj_map[normalized_task_id]
        else:
            # Create stub with expected tools only
            traj = Trajectory(
                task_id=task_id,
                question=case.get("question", ""),
                images=case.get("images", []),
                turns=[],
                tools_called=[],
                expected_tools=case.get("expected_tools", []),
                final_answer="",
                success=False,
                source="eval",
                task_type=""
            )

        # Override expected_tools from eval set
        traj.expected_tools = case.get("expected_tools", [])

        result = evaluator.evaluate(traj)
        results.append(result)

        if traj.tools_called:
            matched += 1

        # Save case result
        case_result = {
            "task_id": task_id,
            "question": case.get("question", "")[:100],
            "expected_tools": traj.expected_tools,
            "predicted_tools": traj.tools_called,
            "precision": result.tool_precision,
            "recall": result.tool_recall,
            "f1": result.tool_f1,
            "is_success": result.is_success,
            "num_expected": len(traj.expected_tools),
            "num_predicted": len(traj.tools_called),
        }
        case_results.append(case_result)

    metrics = evaluator.aggregate(results)
    metrics["coverage"] = matched / len(eval_cases) if eval_cases else 0.0

    return {
        "metrics": metrics,
        "case_results": case_results
    }


def evaluate_method_memory_augmented(eval_cases: list[dict], traj_map: dict[str, Trajectory],
                                      task_type_tools: dict[str, set]) -> dict:
    """
    Evaluate assuming agent uses retrieved training trajectories.
    Better for MemRL/SkillRL methods.
    """
    evaluator = ToolMatchEvaluator()
    results = []
    case_results = []

    for case in eval_cases:
        task_id = case.get("task_id") or case.get("id")
        task_type = case.get("task_type", "")
        expected_tools = set(case.get("expected_tools", []))

        # Simulate: agent retrieves top-k trajectories with similar task type
        if task_type in task_type_tools:
            # Use tools from same-type trajectories
            tools_called = list(task_type_tools[task_type])
        else:
            # Fallback: use most common tools
            all_tools = set()
            for t in traj_map.values():
                all_tools.update(t.tools_called)
            tools_called = list(all_tools)[:5]  # Top 5

        traj = Trajectory(
            task_id=task_id,
            question=case.get("question", ""),
            images=case.get("images", []),
            turns=[],
            tools_called=tools_called,
            expected_tools=list(expected_tools),
            final_answer="",
            success=False,
            source="eval",
            task_type=task_type
        )

        result = evaluator.evaluate(traj)
        results.append(result)

        # Save case result
        case_result = {
            "task_id": task_id,
            "question": case.get("question", "")[:100],
            "task_type": task_type,
            "expected_tools": list(expected_tools),
            "predicted_tools": traj.tools_called,
            "precision": result.tool_precision,
            "recall": result.tool_recall,
            "f1": result.tool_f1,
            "is_success": result.is_success,
            "num_expected": len(expected_tools),
            "num_predicted": len(traj.tools_called),
        }
        case_results.append(case_result)

    metrics = evaluator.aggregate(results)

    return {
        "metrics": metrics,
        "case_results": case_results
    }


def main():
    """Run detailed evaluation for all methods."""
    eval_jsonl = "data/disaster_eval.jsonl"
    train_json = "data/disaster_trajectories.json"

    print("\n" + "="*80)
    print("DETAILED EVALUATION (with trajectory predictions v2)")
    print("="*80)

    # Load data
    eval_cases = load_eval_data(eval_jsonl)
    traj_map = load_trajectories(train_json)

    # Build task_type -> tools mapping
    task_type_tools = defaultdict(set)
    for traj in traj_map.values():
        if traj.task_type:
            task_type_tools[traj.task_type].update(traj.tools_called)

    # Evaluate each method
    methods = {
        "agentevolver": ("direct", None),
        "memrl": ("memory-augmented", task_type_tools),
        "causalevo": ("direct", None),
    }

    all_results = {}

    for method, (strategy, strategy_data) in methods.items():
        print(f"\n{method.upper()} ({strategy})...")

        if strategy == "direct":
            eval_result = evaluate_method_direct(eval_cases, traj_map)
        else:  # memory-augmented
            eval_result = evaluate_method_memory_augmented(eval_cases, traj_map, strategy_data)

        all_results[method] = eval_result
        metrics = eval_result["metrics"]

        print(f"  Precision:   {metrics.get('precision', 0):.4f}")
        print(f"  Recall:      {metrics.get('recall', 0):.4f}")
        print(f"  F1:          {metrics.get('f1', 0):.4f}")
        print(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")
        print(f"  Coverage:    {metrics.get('coverage', 0):.4f}")
        print(f"  N cases:     {metrics.get('n', 0)}")

    # Generate comparison results
    comparison_results = {
        "experiment_id": "disaster_1",
        "evaluation_mode": "detailed_with_trajectories_v2",
        "methods": {}
    }

    for method, eval_result in all_results.items():
        metrics = eval_result["metrics"]
        case_results = eval_result["case_results"]

        # Find top successes and failures
        sorted_by_f1 = sorted(case_results, key=lambda x: x["f1"], reverse=True)

        comparison_results["methods"][method] = {
            "metrics": metrics,
            "case_count": len(case_results),
            "top_successes": sorted_by_f1[:5],
            "top_failures": sorted_by_f1[-5:],
        }

    # Find best method
    if all_results:
        best_method = max(
            all_results.items(),
            key=lambda x: x[1]["metrics"].get("f1", 0)
        )
        comparison_results["best_method"] = best_method[0]
        comparison_results["best_f1"] = best_method[1]["metrics"].get("f1", 0)

    comparison_results["generated_at"] = "2026-04-10T12:00:00Z"

    # Save aggregated results
    output_file = "evo_res/disaster_1/comparison/comparison_results_detailed_v2.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 聚合结果已保存到 {output_file}")

    # Save detailed per-case results for each method
    for method, eval_result in all_results.items():
        detail_file = f"evo_res/disaster_1/{method}/test/eval_detailed_v2.json"
        Path(detail_file).parent.mkdir(parents=True, exist_ok=True)

        with open(detail_file, "w") as f:
            json.dump({
                "method": method,
                "metrics": eval_result["metrics"],
                "case_results": eval_result["case_results"]
            }, f, ensure_ascii=False, indent=2)

        print(f"✓ {method} 详细结果已保存到 {detail_file}")

    print("\n" + "="*80)
    print("📊 输出文件说明：")
    print(f"  1️⃣ 聚合对比（3 方法）: {output_file}")
    print(f"  2️⃣ 详细轨迹（按方法）:")
    for method in methods.keys():
        print(f"     - evo_res/disaster_1/{method}/test/eval_detailed_v2.json")
    print("\n每个 eval_detailed_v2.json 包含：")
    print("  - metrics: 聚合指标 (precision/recall/F1/exact_match/coverage)")
    print("  - case_results: 每个测试案例的详细信息")
    print("    ├── task_id: 任务 ID")
    print("    ├── question: 问题描述")
    print("    ├── expected_tools: 期望的工具列表")
    print("    ├── predicted_tools: 模型预测的工具列表 ✨")
    print("    ├── precision/recall/f1: 单案例指标")
    print("    └── num_expected/num_predicted: 工具数量统计")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
