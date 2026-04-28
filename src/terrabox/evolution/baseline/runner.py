"""Baseline (No Evolution) experiment runner.

Predicts tools using only task-type frequency from training data,
without any self-evolution processing. Serves as the lower bound
for comparing all evolution methods.

Usage:
    python -m terrabox.evolution.baseline.runner eval \
        --train-data data/disaster_trajectories.json \
        --eval-data data/disaster_eval.jsonl \
        --output evo_res/disaster3/baseline/test/eval_offline.json
"""

import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_task_type(case_id: str) -> str:
    """Infer task_type from eval case id (e.g. 'flood_detection_01' → 'flood_detection')."""
    parts = case_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return case_id


def _build_task_type_tool_map(train_data: list[dict]) -> dict[str, list[str]]:
    """Build task_type → most common tool sequence from training data.

    For each task_type, pick the tool sequence that appears most often
    (majority vote). This is the simplest possible prediction without
    any evolution or learning.
    """
    type_sequences = {}
    for record in train_data:
        tt = record.get("task_type", "unknown")
        tools = record.get("tools_called", record.get("tool_sequence", []))
        # Use tuple as hashable key for counting
        type_sequences.setdefault(tt, []).append(tuple(tools))

    result = {}
    for tt, sequences in type_sequences.items():
        most_common = Counter(sequences).most_common(1)[0][0]
        result[tt] = list(most_common)

    return result


def _predict_tools_baseline(tool_map: dict, task_type: str) -> list[str]:
    """Predict tools by returning the majority tool sequence for this task_type."""
    return tool_map.get(task_type, [])


def cmd_eval(args):
    """Offline evaluation: predict tools using task-type frequency (no evolution)."""
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory

    # Load training data for frequency statistics
    with open(args.train_data) as f:
        train_data = json.load(f)
    tool_map = _build_task_type_tool_map(train_data)
    logger.info(f"Built tool map from {len(train_data)} training trajectories, "
                f"{len(tool_map)} task types")

    # Load eval data
    eval_data = []
    with open(args.eval_data) as f:
        for line in f:
            eval_data.append(json.loads(line))

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "baseline",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": []
    }

    precision_sum, recall_sum, f1_sum, exact_sum = 0, 0, 0, 0

    for i, case in enumerate(eval_data):
        case_id = case.get("task_id", case.get("id", f"eval_{i}"))
        task_type = _infer_task_type(case_id)
        tools_called = _predict_tools_baseline(tool_map, task_type)
        expected_tools = case.get("expected_tools", [])

        traj = Trajectory(
            task_id=case_id,
            question=case.get("question", ""),
            images=case.get("images", []),
            turns=[],
            tools_called=tools_called,
            expected_tools=expected_tools,
            final_answer="",
            success=False,
            task_type=task_type
        )

        eval_result = evaluator.evaluate(traj)

        results["cases"].append({
            "task_id": case_id,
            "task_type": task_type,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "precision": eval_result.tool_precision,
            "recall": eval_result.tool_recall,
            "f1": eval_result.tool_f1
        })

        precision_sum += eval_result.tool_precision
        recall_sum += eval_result.tool_recall
        f1_sum += eval_result.tool_f1
        if set(tools_called) == set(expected_tools):
            exact_sum += 1

    n_cases = len(eval_data)
    results["summary"] = {
        "precision": precision_sum / n_cases if n_cases > 0 else 0,
        "recall": recall_sum / n_cases if n_cases > 0 else 0,
        "f1": f1_sum / n_cases if n_cases > 0 else 0,
        "exact_match": exact_sum / n_cases if n_cases > 0 else 0,
        "n_cases": n_cases
    }

    logger.info(f"\n=== Baseline (No Evolution) Evaluation Results ===")
    logger.info(f"   Precision:   {results['summary']['precision']:.4f}")
    logger.info(f"   Recall:      {results['summary']['recall']:.4f}")
    logger.info(f"   F1:          {results['summary']['f1']:.4f}")
    logger.info(f"   Exact Match: {results['summary']['exact_match']:.4f}")
    logger.info(f"   N cases:     {n_cases}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Baseline (No Evolution) runner")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="Offline evaluation (task-type frequency)")
    eval_parser.add_argument("--train-data", required=True, help="Training data (JSON)")
    eval_parser.add_argument("--eval-data", required=True, help="Eval data file (JSONL)")
    eval_parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    if args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
