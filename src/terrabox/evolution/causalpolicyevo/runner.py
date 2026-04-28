"""CausalPolicyEvo experiment runner.

Builds and refines an external policy state for tool-calling agents.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _normalize_count(value: int, max_count: int) -> float:
    if max_count <= 0:
        return 0.0
    return value / max_count


def cmd_build(args) -> None:
    from ..causalevo.counterfactual_credit import compute_statistical_cca
    from ..causalevo.tool_function_model import CTFMBuilder
    from ..seqgraphevo.seq_pattern_miner import SeqPatternMiner
    from ..shared.data_loader import make_loader
    from .policy_state import PolicyState

    os.makedirs(args.store_dir, exist_ok=True)
    loader = make_loader(args.train_data, args.eval_data)
    trajectories = list(loader.iter_train(limit=args.limit))
    if not trajectories:
        logger.error("No trajectories loaded — check --train-data path.")
        return

    logger.info(f"Loaded {len(trajectories)} trajectories for CausalPolicyEvo build")

    global_cca = compute_statistical_cca(trajectories)
    ctfm_builder = CTFMBuilder(top_k_keywords=args.top_k_keywords, top_k_downstream=5)
    ctfm_models = ctfm_builder.build(trajectories, global_cca)

    miner = SeqPatternMiner()
    for traj in trajectories:
        miner.add_trajectory(
            traj.tools_called,
            task_type=getattr(traj, "task_type", "general"),
            f1=1.0 if traj.success else 0.0,
        )

    tool_stats: dict[str, dict] = {}
    for slug, stats in miner.get_tool_stats().items():
        tool_stats[slug] = {
            "count": stats["count"],
            "avg_f1": stats["avg_f1"],
            "task_types": stats["task_types"],
        }

    max_count = max((stats["count"] for stats in tool_stats.values()), default=1)
    tool_priors: dict[str, float] = {}
    task_tool_priors: dict[str, dict[str, float]] = {}
    tool_keywords = {slug: model.precondition_keywords for slug, model in ctfm_models.items()}

    for slug, model in ctfm_models.items():
        use_count = tool_stats.get(slug, {}).get("count", 0)
        normalized_use = _normalize_count(use_count, max_count)
        tool_priors[slug] = round(
            0.45 * model.avg_cca_score + 0.35 * model.success_rate + 0.20 * normalized_use,
            4,
        )
        for task_type, affinity in model.task_type_affinity.items():
            task_tool_priors.setdefault(task_type, {})
            task_tool_priors[task_type][slug] = round(
                0.60 * affinity + 0.25 * model.success_rate + 0.15 * model.avg_cca_score,
                4,
            )

    transition_scores = []
    for edge in miner.get_directed_edges(min_count=args.min_edge_count):
        transition_scores.append({
            "source": edge["source"],
            "target": edge["target"],
            "count": edge["count"],
            "avg_f1": edge["avg_f1"],
            "task_types": edge["task_types"],
            "score": round(0.70 * edge["avg_f1"] + 0.30 * min(edge["count"] / 10.0, 1.0), 4),
        })

    anti_patterns = miner.mine_anti_patterns(
        max_f1=args.max_anti_f1,
        min_support=args.min_anti_support,
    )
    anti_pairs = []
    for pattern in anti_patterns:
        seq = pattern.get("pattern", [])
        if len(seq) >= 2:
            anti_pairs.append(seq[:2])

    state = PolicyState(
        tool_keywords=tool_keywords,
        tool_priors=tool_priors,
        task_tool_priors=task_tool_priors,
        transition_scores=transition_scores,
        anti_pairs=anti_pairs,
        stop_rules={
            "general": "Stop when the current tool evidence covers the user question and additional tools would only repeat the same capability."
        },
        recovery_rules={
            "general": "If the first tool path is weak, switch to a task-appropriate setup tool and follow a high-confidence transition rather than repeating the same call."
        },
        policy_texts={},
        cca_scores=global_cca,
        tool_stats=tool_stats,
        trajectory_count=len(trajectories),
    )

    state_path = os.path.join(args.store_dir, "policy_state.json")
    state.save(state_path)
    logger.info(
        f"\n=== CausalPolicyEvo Build Complete ===\n"
        f"  Tools:          {len(tool_priors)}\n"
        f"  Task priors:    {len(task_tool_priors)} task types\n"
        f"  Transitions:    {len(transition_scores)}\n"
        f"  Anti-pairs:     {len(anti_pairs)}\n"
        f"  Trajectories:   {len(trajectories)}\n"
        f"  Saved to:       {state_path}"
    )


def cmd_optimize(args) -> None:
    from ..shared.data_loader import make_loader
    from ..shared.llm_client import EvolutionLLMClient
    from .optimizer import CausalPolicyOptimizer
    from .policy_state import PolicyState

    state_path = os.path.join(args.store_dir, "policy_state.json")
    if not os.path.exists(state_path):
        logger.error(f"Policy state not found at {state_path}. Run 'build' first.")
        return

    state = PolicyState.load(state_path)
    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    llm_url = getattr(args, "llm_url", None) or os.environ.get("EVOLUTION_LLM_URL")
    optimizer = CausalPolicyOptimizer(EvolutionLLMClient(llm_url=llm_url), top_k=args.top_k)
    optimized = optimizer.optimize(state, eval_cases)
    optimized.save(state_path)
    logger.info(f"Optimized policy state saved to {state_path}")


def cmd_eval(args) -> dict:
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory
    from .policy_state import PolicyState
    from .prompt_injector import CausalPolicyEvoPromptInjector, infer_task_type

    state_path = os.path.join(args.store_dir, "policy_state.json")
    if not os.path.exists(state_path):
        logger.error(f"Policy state not found at {state_path}. Run 'build' first.")
        return {}

    state = PolicyState.load(state_path)
    injector = CausalPolicyEvoPromptInjector(state, top_k=args.top_k)
    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "causalpolicyevo",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    precision_sum = recall_sum = f1_sum = exact_sum = 0.0
    for i, case in enumerate(eval_cases):
        task_id = case.get("id", case.get("task_id", f"eval_{i}"))
        question = case.get("question", "")
        expected_tools = case.get("expected_tools", [])
        task_type = infer_task_type(question, task_id)
        tools_called = injector.predict_tools(question, task_type=task_type)

        traj = Trajectory(
            task_id=task_id,
            question=question,
            images=case.get("images", []),
            turns=[],
            tools_called=tools_called,
            expected_tools=expected_tools,
            final_answer="",
            success=False,
            task_type=task_type,
        )
        eval_result = evaluator.evaluate(traj)
        results["cases"].append({
            "task_id": task_id,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "precision": eval_result.tool_precision,
            "recall": eval_result.tool_recall,
            "f1": eval_result.tool_f1,
        })
        precision_sum += eval_result.tool_precision
        recall_sum += eval_result.tool_recall
        f1_sum += eval_result.tool_f1
        if set(tools_called) == set(expected_tools):
            exact_sum += 1

    n_cases = len(eval_cases)
    results["summary"] = {
        "precision": precision_sum / n_cases if n_cases else 0.0,
        "recall": recall_sum / n_cases if n_cases else 0.0,
        "f1": f1_sum / n_cases if n_cases else 0.0,
        "exact_match": exact_sum / n_cases if n_cases else 0.0,
        "n_cases": n_cases,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(
        f"\n=== CausalPolicyEvo Evaluation Results ===\n"
        f"   Precision:   {results['summary']['precision']:.4f}\n"
        f"   Recall:      {results['summary']['recall']:.4f}\n"
        f"   F1:          {results['summary']['f1']:.4f}\n"
        f"   Exact Match: {results['summary']['exact_match']:.4f}\n"
        f"   N cases:     {n_cases}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalPolicyEvo runner")
    subparsers = parser.add_subparsers(dest="command")

    build_p = subparsers.add_parser("build", help="Build policy state from trajectories")
    build_p.add_argument("--train-data", required=True, help="Training trajectory file")
    build_p.add_argument("--eval-data", default="", help="Eval data file (used for loader compatibility)")
    build_p.add_argument("--store-dir", required=True, help="Directory to save policy_state.json")
    build_p.add_argument("--limit", type=int, default=None, help="Limit number of trajectories for build")
    build_p.add_argument("--top-k-keywords", type=int, default=10, help="Keywords kept per tool")
    build_p.add_argument("--min-edge-count", type=int, default=3, help="Minimum count for directed edges")
    build_p.add_argument("--min-anti-support", type=int, default=3, help="Minimum support for anti-patterns")
    build_p.add_argument("--max-anti-f1", type=float, default=0.3, help="Maximum F1 for anti-pattern mining")

    opt_p = subparsers.add_parser("optimize", help="Optimize policy texts with LLM")
    opt_p.add_argument("--train-data", required=True, help="Training trajectory file")
    opt_p.add_argument("--eval-data", required=True, help="Eval data JSONL")
    opt_p.add_argument("--store-dir", required=True, help="Directory containing policy_state.json")
    opt_p.add_argument("--limit", type=int, default=None, help="Limit eval cases for optimization")
    opt_p.add_argument("--top-k", type=int, default=8, help="Top-k predicted tools during optimization")
    opt_p.add_argument("--llm-url", default=None, help="Override EVOLUTION_LLM_URL")

    eval_p = subparsers.add_parser("eval", help="Offline evaluation")
    eval_p.add_argument("--train-data", default="", help="Training trajectory file (for loader compatibility)")
    eval_p.add_argument("--eval-data", required=True, help="Eval data JSONL")
    eval_p.add_argument("--store-dir", required=True, help="Directory containing policy_state.json")
    eval_p.add_argument("--output", required=True, help="Output JSON file")
    eval_p.add_argument("--limit", type=int, default=None, help="Limit eval cases")
    eval_p.add_argument("--top-k", type=int, default=8, help="Top-k predicted tools")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
