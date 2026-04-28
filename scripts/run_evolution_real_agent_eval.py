#!/usr/bin/env python3
"""
真实 Agent 评测脚本：使用 augmented_prompt 运行真实 agent，评估进化方法的实际效果。

与 run_evolution_detailed_eval_v2.py 的区别：
  - 离线评测：直接用技能库工具列表 vs expected_tools
  - 真实评测：LLM 读 augmented_prompt → 真实 agent 推理 → 实际工具调用

支持的 Agent 模式：
  - react       : 基础 ReAct agent（自由工具选择）
  - progressive : 3级工具发现 + 错误重试（推荐）
  - category    : 2阶段分类选择 + 动态扩展

流程：
  1. 加载评测数据
  2. 对每个方法（AgentEvolver, MemRL, CausalEvo, SkillRL, RewardEvo, GraphSkillEvo）
  3. 对每个测试样本：
     - 调用 method.augment(query) 获取增强提示
     - 运行选定的 agent 模式（带增强提示）
     - 记录实际 tools_called
     - 与 expected_tools 对比
  4. 保存结果到 evo_res/disaster_1/{method}/test/eval_results_real.json

注意：这个脚本会真实执行工具链，非常耗时。建议先用 --sample 10 --agent-mode progressive 测试。
"""
import json
import sys
import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, "src")

from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
from terrabox.evolution.shared.trajectory import Trajectory, Turn
from terrabox.evolution import get_prompt_augmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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


def extract_tools_from_agent_result(result: dict) -> list[str]:
    """
    从 agent 的返回结果中提取工具调用列表。

    参考自 scripts/eval/evaluate_agent.py 中的逻辑。
    支持不同的返回结构：messages, tool_calls 等。
    """
    tools_called = []

    # 方式1：从 messages 中提取 tool_calls（参考 evaluate_agent.py）
    if "messages" in result:
        messages = result.get("messages", [])
        for msg in messages:
            # ToolMessage 或 AIMessage with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    slug = tc.get("name") or tc.get("function", {}).get("name", "")
                    if slug:
                        tools_called.append(slug)

    # 方式2：直接从结果中的 tool_calls 字段
    if "tool_calls" in result and not tools_called:
        for tc in result.get("tool_calls", []):
            slug = tc.get("name") or tc.get("function", {}).get("name", "")
            if slug:
                tools_called.append(slug)

    # 方式3：从 tools_executed 字段
    if "tools_executed" in result and not tools_called:
        tools_called = result.get("tools_executed", [])

    return tools_called


def run_agent_with_mode(agent_mode: str, question: str, images: list[str],
                        augmenter=None) -> dict:
    """
    根据指定的 agent 模式运行 agent。

    Args:
        agent_mode: 'react', 'progressive', 或 'category'
        question: 用户问题
        images: 图像列表
        augmenter: evolution 方法的 augmenter（可选）

    Returns:
        agent 返回的结果字典
    """
    # 加载 agent 模块
    from terrabox.agent.config import load_config
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import registry

    # 确保工具已加载
    if not registry.list_toolkits():
        load_builtin_toolkits()

    # 构建上下文消息
    context_msg = question
    if augmenter:
        try:
            augmented = augmenter.augment(question)
            context_msg = f"{question}\n\n[增强提示]\n{augmented}"
        except Exception as e:
            log.warning(f"获取增强提示失败: {e}，使用原始问题")

    config = load_config()

    try:
        if agent_mode == "progressive":
            from terrabox.agent.progressive_graph import run_progressive_agent

            result_str = run_progressive_agent(
                session_id="eval_session",
                user_message=context_msg,
                image_paths=images,
                user=None,
                db=None,  # 评测不需要数据库持久化
                config=config,
            )
            # 返回值是字符串，转为dict格式以兼容工具提取逻辑
            return {"messages": [], "final_answer": result_str}

        elif agent_mode == "category":
            from terrabox.agent.category_graph import run_category_agent

            result_str = run_category_agent(
                session_id="eval_session",
                user_message=context_msg,
                image_paths=images,
                user=None,
                db=None,
                config=config,
            )
            return {"messages": [], "final_answer": result_str}

        else:
            raise ValueError(f"不支持的 agent 模式: {agent_mode}（仅支持: progressive, category）")

    except Exception as e:
        log.error(f"Agent 执行失败 ({agent_mode}): {e}")
        raise


def evaluate_method_with_real_agent(method_name: str, eval_cases: list[dict],
                                     agent_mode: str = "progressive",
                                     sample_size: int = None,
                                     store_root: str = "evo_res/disaster_1") -> dict:
    """
    真实 Agent 评测：使用方法的 augmented_prompt 运行真实 agent。

    Args:
        method_name: 方法名称 ('agentevolver', 'memrl', 'causalevo', 'skillrl', 'rewardevo', 'graphskillevo')
        eval_cases: 评测样本列表
        agent_mode: agent 模式 ('react', 'progressive', 'category')
        sample_size: 如果指定，只评测前 N 个样本

    Returns:
        {
            "metrics": {...},
            "case_results": [...]
        }
    """
    log.info(f"\n{method_name.upper()} - 真实 Agent 评测 ({agent_mode})")
    log.info("=" * 80)

    # 限制样本数
    if sample_size:
        eval_cases = eval_cases[:sample_size]
        log.info(f"仅评测前 {sample_size} 个样本")

    # 加载 augmenter
    try:
        kwargs = {}
        if method_name == "memrl":
            kwargs["memory_db"] = os.path.join(store_root, "memrl", "episodic_memory.db")
        elif method_name == "skillrl":
            kwargs["store_dir"] = os.path.join(store_root, "skillrl", "store")
        elif method_name == "rewardevo":
            kwargs["memory_db"] = os.path.join(store_root, "rewardevo", "store", "episodic_memory.db")
        elif method_name == "graphskillevo":
            store_v2 = os.path.join(store_root, "graphskillevo", "store_v2", "tool_graph.json")
            legacy = os.path.join(store_root, "graphskillevo", "store", "tool_graph.json")
            legacy_skill = os.path.join(store_root, "graphskillevo", "store", "skill_graph.json")
            if os.path.exists(store_v2):
                kwargs["tool_graph_path"] = store_v2
            elif os.path.exists(legacy):
                kwargs["tool_graph_path"] = legacy
            else:
                kwargs["tool_graph_path"] = legacy_skill
        elif method_name in {"agentevolver", "causalevo", "evoskill", "seqgraphevo", "causaltextevo", "causalpolicyevo"}:
            kwargs["store_dir"] = os.path.join(store_root, method_name, "store")

        augmenter = get_prompt_augmenter(method_name, **kwargs)
        log.info(f"✓ 已加载 {method_name} 的 augmenter")
    except Exception as e:
        log.error(f"无法加载 {method_name} 的 augmenter: {e}")
        return None

    evaluator = ToolMatchEvaluator()
    results = []
    case_results = []

    log.info(f"{'样本ID':30} {'状态':12} {'期望工具':10} {'实际工具':10}")
    log.info("-" * 70)

    for i, case in enumerate(eval_cases):
        task_id = case.get("task_id") or case.get("id")
        question = case.get("question", "")
        expected_tools = case.get("expected_tools", [])
        images = case.get("images", [])

        try:
            # Step 1: 运行真实 Agent（带增强提示）
            agent_result = run_agent_with_mode(agent_mode, question, images, augmenter)

            # Step 2: 从返回结果提取实际调用的工具
            tools_called = extract_tools_from_agent_result(agent_result)

            # Step 3: 评测
            traj = Trajectory(
                task_id=task_id,
                question=question,
                images=images,
                turns=[],
                tools_called=tools_called,
                expected_tools=expected_tools,
                final_answer="",
                success=len(tools_called) > 0,
                source="agent_eval",
                task_type=case.get("task_type", "")
            )

            result = evaluator.evaluate(traj)
            results.append(result)

            status = "✓" if tools_called else "⚠"

        except Exception as e:
            log.warning(f"[{task_id}] 执行失败: {str(e)[:50]}")
            status = "✗"
            tools_called = []
            result = None

        # 保存案例结果
        case_result = {
            "task_id": task_id,
            "question": question[:100],
            "expected_tools": expected_tools,
            "predicted_tools": tools_called,
            "num_expected": len(expected_tools),
            "num_predicted": len(tools_called),
            "status": status
        }

        if result:
            case_result.update({
                "precision": result.tool_precision,
                "recall": result.tool_recall,
                "f1": result.tool_f1,
                "is_success": result.is_success,
            })

        case_results.append(case_result)

        log.info(f"{task_id:30} {status:12} {len(expected_tools):10} {len(tools_called):10}")

        if (i + 1) % 10 == 0:
            log.info(f"  进度: [{i+1}/{len(eval_cases)}]")

    # 聚合结果
    valid_results = [r for r in results if r is not None]
    if valid_results:
        metrics = evaluator.aggregate(valid_results)
    else:
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "coverage": 0.0,
            "n": len(eval_cases)
        }

    return {
        "metrics": metrics,
        "case_results": case_results
    }


def main():
    """Run real agent evaluation for all methods."""
    import argparse

    parser = argparse.ArgumentParser(
        description="真实 Agent 评测：使用 evolution 方法的增强提示运行真实 agent 并评估效果",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--eval-data", default="data/disaster_eval.jsonl", help="评测数据路径")
    parser.add_argument("--train-data", default="data/disaster_trajectories.json", help="训练数据路径（用于加载 augmenter）")
    parser.add_argument("--agent-mode", choices=["progressive", "category"], default="progressive",
                       help="Agent 模式选择（默认: progressive，推荐）")
    parser.add_argument("--sample", type=int, default=None, help="仅评测前 N 个样本（用于快速测试）")
    parser.add_argument("--methods", nargs="+",
                       default=["agentevolver", "memrl", "causalevo", "skillrl", "rewardevo", "graphskillevo", "causalpolicyevo"],
                       help="要评测的方法列表（默认: 所有7个）")
    parser.add_argument("--output-suffix", default="real", help="输出文件名后缀（默认: real）")
    parser.add_argument("--store-root", default="evo_res/disaster_1", help="方法产物根目录（默认: evo_res/disaster_1）")

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"真实 AGENT 评测（Agent 模式: {args.agent_mode}）")
    print("="*80)

    # 加载数据
    eval_cases = load_eval_data(args.eval_data)
    traj_map = load_trajectories(args.train_data)

    print(f"\n✓ 已加载 {len(eval_cases)} 条评测样本")
    print(f"✓ Agent 模式: {args.agent_mode}")
    if args.sample:
        print(f"⚠ 仅评测前 {args.sample} 个样本（快速测试）")

    # 评测每个方法
    all_results = {}

    for method in args.methods:
        try:
            eval_result = evaluate_method_with_real_agent(
                method, eval_cases,
                agent_mode=args.agent_mode,
                sample_size=args.sample,
                store_root=args.store_root,
            )
            if eval_result:
                all_results[method] = eval_result
                print(f"✓ {method} 评测完成")
        except Exception as e:
            log.error(f"{method} 评测失败: {e}")
            continue

    if not all_results:
        print("\n✗ 没有成功的评测结果")
        return

    # 生成对比结果
    comparison_results = {
        "experiment_id": Path(args.store_root).name,
        "evaluation_mode": f"real_agent_{args.agent_mode}",
        "agent_mode": args.agent_mode,
        "methods": {}
    }

    for method, eval_result in all_results.items():
        metrics = eval_result["metrics"]
        case_results = eval_result["case_results"]

        sorted_by_f1 = sorted(case_results, key=lambda x: x.get("f1", 0), reverse=True)

        comparison_results["methods"][method] = {
            "metrics": metrics,
            "case_count": len(case_results),
            "top_successes": sorted_by_f1[:5],
            "top_failures": sorted_by_f1[-5:],
        }

    # 找最佳方法
    if all_results:
        best_method = max(
            all_results.items(),
            key=lambda x: x[1]["metrics"].get("f1", 0)
        )
        comparison_results["best_method"] = best_method[0]
        comparison_results["best_f1"] = best_method[1]["metrics"].get("f1", 0)

    comparison_results["generated_at"] = "2026-04-10T12:00:00Z"

    # 保存汇总结果
    output_file = os.path.join(args.store_root, "comparison", f"comparison_results_{args.output_suffix}.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 汇总结果已保存到 {output_file}")

    # 保存详细的按方法结果
    for method, eval_result in all_results.items():
        detail_file = os.path.join(args.store_root, method, "test", f"eval_results_{args.output_suffix}.json")
        Path(detail_file).parent.mkdir(parents=True, exist_ok=True)

        with open(detail_file, "w") as f:
            json.dump({
                "method": method,
                "evaluation_mode": f"real_agent_{args.agent_mode}",
                "agent_mode": args.agent_mode,
                "metrics": eval_result["metrics"],
                "case_results": eval_result["case_results"]
            }, f, ensure_ascii=False, indent=2)

        print(f"✓ {method} 详细结果已保存到 {detail_file}")

    # 打印总结
    print("\n" + "="*80)
    print(f"评测结果总结（{args.agent_mode}）")
    print("="*80)

    for method, result in all_results.items():
        metrics = result["metrics"]
        print(f"\n{method:15}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1:        {metrics.get('f1', 0):.4f}")
        print(f"  覆盖度:    {metrics.get('coverage', 0):.4f}")

    if all_results:
        best_m = comparison_results.get("best_method", "N/A")
        best_f1 = comparison_results.get("best_f1", 0)
        print(f"\n🏆 最佳方法: {best_m} (F1: {best_f1:.4f})")


if __name__ == "__main__":
    main()
