#!/usr/bin/env python3
"""
Terrabox Agent Evaluation Script
=================================
Evaluates the Terrabox agent against Earth-Bench and OpenEarthAgent evaluation sets.

Usage:
  cd /data1/yuhongjie2/terrabox
  python scripts/eval/evaluate_agent.py --dataset earthbench [--limit 20]
  python scripts/eval/evaluate_agent.py --dataset openearth  [--limit 100]
  python scripts/eval/evaluate_agent.py --dataset earthbench --output eval_results.json

Metrics computed:
  - tool_any_order   : fraction of questions where all expected tools were called (any order)
  - tool_in_order    : fraction of questions where expected tools appear in order
  - tool_exact_match : fraction of questions where tool sequence exactly matches
  - coverage         : average fraction of expected tools that were called

DO NOT RUN during development — run only after deploying the terrabox service.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_ROOT = REPO_ROOT / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(expected: List[str], actual: List[str]) -> Dict[str, float]:
    """Compute all tool-use metrics for one question."""
    if not expected:
        return {"tool_any_order": 1.0, "tool_in_order": 1.0, "tool_exact_match": 1.0, "coverage": 1.0}

    expected_set = set(expected)
    actual_set = set(actual)

    # Tool-Any-Order: did we call all expected tools (regardless of order)?
    intersection = expected_set & actual_set
    tool_any_order = 1.0 if intersection == expected_set else len(intersection) / len(expected_set)
    coverage = len(intersection) / len(expected_set)

    # Tool-In-Order: expected tools appear in actual in the same relative order
    # (subsequence check)
    def is_subsequence(subseq: List[str], seq: List[str]) -> bool:
        it = iter(seq)
        return all(item in it for item in subseq)

    tool_in_order = 1.0 if is_subsequence(expected, actual) else 0.0

    # Tool-Exact-Match: sequences are identical
    tool_exact_match = 1.0 if expected == actual else 0.0

    return {
        "tool_any_order": round(tool_any_order, 4),
        "tool_in_order": tool_in_order,
        "tool_exact_match": tool_exact_match,
        "coverage": round(coverage, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Agent runner (calls local Terrabox agent via Python API)
# ──────────────────────────────────────────────────────────────────────────────

def run_agent_and_extract_tools(question: str, data_dir: Optional[str] = None) -> Tuple[List[str], str]:
    """
    Run the Terrabox progressive agent on a question.
    Returns (tool_slugs_called, final_answer).
    """
    # Dynamically import to avoid loading at module level
    from terrabox.agent.progressive_graph import run_progressive_agent
    from terrabox.agent.config import load_config
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import registry

    # Ensure toolkits are loaded
    if not registry.list_toolkits():
        load_builtin_toolkits()

    config = load_config()

    # Build context with data directory hint if provided
    context_msg = question
    if data_dir:
        context_msg = f"{question}\n\n[Data directory: {data_dir}]"

    tool_calls_made: List[str] = []
    final_answer = ""

    try:
        result = run_progressive_agent(
            message=context_msg,
            config=config,
            user=None,
        )
        # Extract tool calls from result
        messages = result.get("messages", [])
        for msg in messages:
            # ToolMessage or AIMessage with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    slug = tc.get("name") or tc.get("function", {}).get("name", "")
                    if slug:
                        tool_calls_made.append(slug)
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                final_answer = msg.content

    except Exception as e:
        log.error(f"Agent failed: {e}")

    return tool_calls_made, final_answer


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def load_eval_data(dataset: str, limit: Optional[int] = None) -> List[Dict]:
    """Load evaluation records from JSONL file."""
    if dataset == "earthbench":
        path = DATA_ROOT / "earthbench" / "eval.jsonl"
    elif dataset == "openearth":
        path = DATA_ROOT / "openearth" / "eval.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'earthbench' or 'openearth'")

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    log.info(f"Loaded {len(records)} evaluation records from {path}")
    return records


def evaluate(dataset: str, limit: Optional[int], output: Optional[str]) -> Dict[str, Any]:
    """Run full evaluation loop and compute aggregate metrics."""
    records = load_eval_data(dataset, limit)

    results = []
    metric_sums = {
        "tool_any_order": 0.0,
        "tool_in_order": 0.0,
        "tool_exact_match": 0.0,
        "coverage": 0.0,
    }

    for i, record in enumerate(records):
        qid = record.get("id", f"q{i}")
        question = record.get("question", "")
        expected_tools = record.get("expected_tools", [])
        data_dir = record.get("data_dir")

        log.info(f"[{i+1}/{len(records)}] {qid}: {question[:60]}...")

        # Run agent
        actual_tools, answer = run_agent_and_extract_tools(question, data_dir)

        # Compute metrics
        metrics = compute_metrics(expected_tools, actual_tools)

        result = {
            "id": qid,
            "question": question,
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "answer": answer[:200] if answer else "",
            "metrics": metrics,
        }
        results.append(result)

        for k in metric_sums:
            metric_sums[k] += metrics[k]

        log.info(
            f"  → tools expected={len(expected_tools)} actual={len(actual_tools)} "
            f"any_order={metrics['tool_any_order']:.2f} "
            f"exact={metrics['tool_exact_match']:.2f}"
        )

    n = len(results)
    aggregate = {k: round(v / n, 4) if n > 0 else 0.0 for k, v in metric_sums.items()}

    report = {
        "dataset": dataset,
        "total": n,
        "aggregate_metrics": aggregate,
        "results": results,
    }

    # Print summary
    log.info("\n" + "=" * 50)
    log.info(f"EVALUATION SUMMARY — {dataset} ({n} questions)")
    log.info("=" * 50)
    for k, v in aggregate.items():
        log.info(f"  {k:<22}: {v:.4f} ({v*100:.1f}%)")
    log.info("=" * 50)

    # Save report
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"Report saved to {out_path}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Terrabox agent on benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["earthbench", "openearth"],
        default="earthbench",
        help="Which dataset to evaluate on (default: earthbench)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON report (e.g., eval_results.json)",
    )
    args = parser.parse_args()

    evaluate(args.dataset, args.limit, args.output)


if __name__ == "__main__":
    main()
