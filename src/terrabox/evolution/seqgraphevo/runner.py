"""SeqGraphEvo runner — build sequential tool graph and offline evaluation."""

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
    parts = case_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return case_id


def _keyword_score(query: str, tool_slug: str) -> float:
    slug_words = set(re.split(r"[._]", tool_slug.lower()))
    query_words = set(re.split(r"\W+", query.lower()))
    if not slug_words or not query_words:
        return 0.0
    return len(slug_words & query_words) / len(slug_words)


def _rank_tools_by_query(query: str, tool_slugs: list[str], top_k: int = 4) -> list[str]:
    scored = [(s, _keyword_score(query, s)) for s in tool_slugs]
    scored.sort(key=lambda x: x[1], reverse=True)
    positives = [s for s, sc in scored if sc > 0]
    return positives[:top_k] if positives else [s for s, _ in scored[:top_k]]


def cmd_build(args):
    """Build sequential tool graph from trajectory data."""
    from .seq_pattern_miner import SeqPatternMiner
    from .prompt_injector import _build_seq_graph

    traj_file = args.traj_file
    limit = getattr(args, 'limit', None)
    if limit:
        import tempfile, json
        logger.info(f"Limiting to first {limit} trajectories from {traj_file}")
        with open(traj_file) as f:
            lines = [l for i, l in enumerate(f) if l.strip() and i < limit]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        tmp.writelines(lines); tmp.close()
        traj_file = tmp.name
    logger.info(f"Loading trajectories from {traj_file} ...")
    miner = SeqPatternMiner.load_from_trajectories(traj_file)

    logger.info("Mining sequential patterns ...")
    graph = _build_seq_graph(
        miner,
        min_pattern_support=args.min_support,
        max_pattern_len=args.max_len,
        min_edge_count=args.min_edge_count,
        min_anti_support=args.min_anti_support,
    )

    os.makedirs(args.store_dir, exist_ok=True)
    save_path = os.path.join(args.store_dir, "seq_graph.json")
    graph.save(save_path)

    stats = graph.connectivity_stats()
    logger.info(
        f"\n=== SeqGraph Built ===\n"
        f"  Nodes:          {stats['nodes']}\n"
        f"  Forward edges:  {stats['fwd_edges']}\n"
        f"  Connected nodes:{stats['connected_nodes']}/{stats['nodes']} "
        f"({100*stats['connected_nodes']/max(stats['nodes'], 1):.1f}%)\n"
        f"  Patterns:       {stats['patterns']}\n"
        f"  Anti-patterns:  {stats['anti_patterns']}\n"
        f"  Anti-pairs:     {stats['anti_pairs']}\n"
        f"  Trajectories:   {miner._trajectory_count}\n"
        f"  Saved to:       {save_path}"
    )

    # Print top 10 patterns
    logger.info("\nTop 10 sequential patterns:")
    for p in graph.top_patterns(10):
        seq = " → ".join(p["pattern"])
        dominant = max(p["task_types"].items(), key=lambda x: x[1], default=("?", 0))
        logger.info(
            f"  [{p['support']}x, F1={p['avg_f1']:.2f}, "
            f"task={dominant[0]}] {seq}"
        )


def cmd_eval(args):
    """Offline evaluation: predict tools using SeqGraph."""
    from ..shared.evaluator import ToolMatchEvaluator
    from .seq_graph import SeqGraph

    eval_data = []
    with open(args.eval_data) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))

    graph_path = os.path.join(args.store_dir, "seq_graph.json")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"seq_graph.json not found in {args.store_dir}. Run 'build' first."
        )

    seq_graph = SeqGraph.load(graph_path)
    all_tools = list(seq_graph._nodes.keys())

    logger.info(
        f"Loaded SeqGraph: {seq_graph.count_nodes()} nodes, "
        f"{seq_graph.count_fwd_edges()} edges, "
        f"{seq_graph.count_patterns()} patterns"
    )

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "seqgraphevo",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    precision_sum = recall_sum = f1_sum = exact_sum = 0

    for i, case in enumerate(eval_data):
        case_id = case.get("task_id", case.get("id", f"eval_{i}"))
        task_type = _infer_task_type(case_id)
        question = case.get("question", "")

        seed_tools = _rank_tools_by_query(question, all_tools, top_k=4)
        tools_called = seq_graph.expand_with_composition(
            seed_tools, task_type=task_type, top_k=8
        )
        expected_tools = case.get("expected_tools", [])

        from ..shared.trajectory import Trajectory
        traj = Trajectory(
            task_id=case_id,
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
        f"\n=== SeqGraphEvo Evaluation Results ===\n"
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
    parser = argparse.ArgumentParser(description="SeqGraphEvo runner")
    subparsers = parser.add_subparsers(dest="command")

    # build subcommand
    build_p = subparsers.add_parser("build", help="Build sequential tool graph")
    build_p.add_argument("--traj-file", required=True,
                         help="Trajectory JSONL (experience_pool.jsonl)")
    build_p.add_argument("--store-dir", required=True,
                         help="Directory to save seq_graph.json")
    build_p.add_argument("--min-support", type=int, default=5,
                         help="Minimum support for sequential patterns (default: 5)")
    build_p.add_argument("--max-len", type=int, default=4,
                         help="Maximum pattern length (default: 4)")
    build_p.add_argument("--min-edge-count", type=int, default=3,
                         help="Minimum count for directed edges (default: 3)")
    build_p.add_argument("--min-anti-support", type=int, default=3,
                         help="Minimum support for anti-patterns (default: 3)")
    build_p.add_argument("--limit", type=int, default=None,
                         help="Limit number of trajectories for build")

    # eval subcommand
    eval_p = subparsers.add_parser("eval", help="Offline evaluation")
    eval_p.add_argument("--store-dir", required=True,
                        help="Store directory with seq_graph.json")
    eval_p.add_argument("--eval-data", required=True, help="Eval data JSONL")
    eval_p.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
