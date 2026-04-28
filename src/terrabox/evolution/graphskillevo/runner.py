"""GraphSkillEvo runner — build tool graph and offline evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_task_type(case_id: str) -> str:
    """Infer task_type from eval case id (e.g. 'flood_detection_01' → 'flood_detection')."""
    parts = case_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return case_id


def _predict_tools_offline_graphskillevo(
    tool_graph,
    question: str,
    task_type: str,
    top_k: int = 8,
) -> list[str]:
    """Predict tools using the ToolGraph.

    Strategy:
      1. Keyword-match query → seed tools
      2. Expand via co-occurrence edges
      3. Supplement with task-type top tools
    """
    from .prompt_injector import _rank_tools_by_query

    all_tools = list(tool_graph._nodes.keys())
    seed_tools = _rank_tools_by_query(question, all_tools, top_k=4)
    expanded = tool_graph.expand_toolset(seed_tools, top_k=top_k)

    # Also incorporate task-type specific tools
    task_tools = tool_graph.get_tools_for_task_type(task_type, top_k=4)
    combined: dict[str, float] = {}
    for i, t in enumerate(expanded):
        combined[t] = 1000.0 - i
    for i, t in enumerate(task_tools):
        combined.setdefault(t, 0.0)
        combined[t] += 500.0 - i * 10

    ordered = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ordered[:top_k]]


def cmd_build(args):
    """Build tool co-occurrence graph from trajectory data."""
    from .tool_graph_builder import ToolGraphBuilder
    from .skill_graph import ToolGraph

    traj_file = args.traj_file
    limit = getattr(args, 'limit', None)
    if limit:
        import tempfile
        logger.info(f"Limiting to first {limit} trajectories from {traj_file}")
        with open(traj_file) as f:
            lines = [l for i, l in enumerate(f) if l.strip() and i < limit]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        tmp.writelines(lines); tmp.close()
        traj_file = tmp.name
    logger.info(f"Loading trajectories from {traj_file} ...")
    builder = ToolGraphBuilder.load_from_trajectories(traj_file)

    # Show top co-occurrences
    logger.info("Top 10 tool co-occurrences:")
    for a, b, cnt in builder.get_top_cooccurrences(10):
        logger.info(f"  {a} + {b}: {cnt}")

    graph_data = builder.build_graph_data(min_cooccurrence=args.min_cooccurrence)
    graph = ToolGraph.from_graph_data(graph_data)

    os.makedirs(args.store_dir, exist_ok=True)
    save_path = os.path.join(args.store_dir, "tool_graph.json")
    graph.save(save_path)

    n_nodes = graph.count_nodes()
    n_edges = graph.count_edges()
    connected = len(
        set(e["source"] for e in graph_data["edges"])
        | set(e["target"] for e in graph_data["edges"])
    )
    logger.info(
        f"\n=== Tool Graph Built ===\n"
        f"  Nodes (tools):   {n_nodes}\n"
        f"  Edges:           {n_edges}\n"
        f"  Connected nodes: {connected}/{n_nodes} "
        f"({100*connected/n_nodes:.1f}%)\n"
        f"  Trajectories:    {graph_data['trajectory_count']}\n"
        f"  Saved to:        {save_path}"
    )


def cmd_eval(args):
    """Offline evaluation: predict tools from tool co-occurrence graph."""
    from ..shared.evaluator import ToolMatchEvaluator
    from .skill_graph import ToolGraph

    eval_data = []
    with open(args.eval_data) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))

    # Load tool graph
    graph_path = os.path.join(args.store_dir, "tool_graph.json")
    if not os.path.exists(graph_path):
        # Fallback: try old skill_graph.json path
        old_path = os.path.join(args.store_dir, "skill_graph.json")
        if os.path.exists(old_path):
            logger.warning(f"tool_graph.json not found; using legacy skill_graph.json")
            graph_path = old_path
        else:
            raise FileNotFoundError(
                f"No tool_graph.json found in {args.store_dir}. "
                f"Run 'build' command first."
            )

    tool_graph = ToolGraph.load(graph_path)
    logger.info(
        f"Loaded ToolGraph: {tool_graph.count_nodes()} tools, "
        f"{tool_graph.count_edges()} edges"
    )

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "graphskillevo",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    precision_sum = recall_sum = f1_sum = exact_sum = 0

    for i, case in enumerate(eval_data):
        case_id = case.get("task_id", case.get("id", f"eval_{i}"))
        task_type = _infer_task_type(case_id)
        tools_called = _predict_tools_offline_graphskillevo(
            tool_graph, case.get("question", ""), task_type
        )
        expected_tools = case.get("expected_tools", [])

        from ..shared.trajectory import Trajectory
        traj = Trajectory(
            task_id=case_id,
            question=case.get("question", ""),
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
            "task_id": case_id,
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

    n_cases = len(eval_data)
    results["summary"] = {
        "precision": precision_sum / n_cases if n_cases else 0,
        "recall": recall_sum / n_cases if n_cases else 0,
        "f1": f1_sum / n_cases if n_cases else 0,
        "exact_match": exact_sum / n_cases if n_cases else 0,
        "n_cases": n_cases,
    }

    logger.info(
        f"\n=== GraphSkillEvo Evaluation Results ===\n"
        f"   Precision:   {results['summary']['precision']:.4f}\n"
        f"   Recall:      {results['summary']['recall']:.4f}\n"
        f"   F1:          {results['summary']['f1']:.4f}\n"
        f"   Exact Match: {results['summary']['exact_match']:.4f}\n"
        f"   N cases:     {n_cases}"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="GraphSkillEvo runner")
    subparsers = parser.add_subparsers(dest="command")

    # build subcommand
    build_parser = subparsers.add_parser("build", help="Build tool co-occurrence graph")
    build_parser.add_argument("--traj-file", required=True,
                              help="Trajectory JSONL file (experience_pool.jsonl)")
    build_parser.add_argument("--store-dir", required=True,
                              help="Directory to save tool_graph.json")
    build_parser.add_argument("--min-cooccurrence", type=int, default=2,
                              help="Minimum co-occurrence count for an edge (default: 2)")
    build_parser.add_argument("--limit", type=int, default=None,
                              help="Limit number of trajectories for build")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Offline evaluation")
    eval_parser.add_argument("--store-dir", required=True,
                             help="Store directory with tool_graph.json")
    eval_parser.add_argument("--eval-data", required=True, help="Eval data file (JSONL)")
    eval_parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
