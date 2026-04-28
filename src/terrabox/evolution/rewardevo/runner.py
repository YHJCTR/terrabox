"""RewardEvo experiment runner.

Usage:
    # Phase 1: Pseudo-label trajectories via LLM-as-Judge (requires Docker vLLM)
    python -m terrabox.evolution.rewardevo.runner label \
        --train-data data/disaster_trajectories.json \
        --store-dir evo_res/disaster3/rewardevo/store

    # Phase 2: Offline evaluation
    python -m terrabox.evolution.rewardevo.runner eval \
        --eval-data data/disaster_eval.jsonl \
        --output evo_res/disaster3/rewardevo/test/eval_offline.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_label(args):
    """Pseudo-label trajectories using LLM-as-Judge (like SkillRL distill)."""
    from ..shared.data_loader import make_loader
    from ..shared.llm_client import EvolutionLLMClient
    from ..shared.trajectory import Trajectory, Turn
    from .self_consistent_labeler import SelfConsistentLabeler

    logger.info(f"Loading trajectories from {args.train_data}")

    # Load trajectories (same pattern as SkillRL distill)
    loader = make_loader(args.train_data, args.train_data)  # eval not needed for labeling
    trajectories = loader.load_train_trajectories(limit=args.limit)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Create LLM client (same as SkillRL: EvolutionLLMClient() with default env detection)
    llm = EvolutionLLMClient()
    logger.info(f"LLM client: docker={llm._use_docker}, url={llm._llm_url}")

    if not llm._use_docker:
        logger.error("Docker vLLM not available — RewardEvo requires LLM for judging")
        sys.exit(1)

    os.makedirs(args.store_dir, exist_ok=True)

    # Label trajectories
    labeler = SelfConsistentLabeler(llm_client=llm)
    labeled = labeler.label_trajectories(trajectories)

    # Write to MemRL episodic memory
    memory_db = os.path.join(args.store_dir, "episodic_memory.db")
    result = labeler.write_to_memrl(labeled, memory_db, llm_client=llm)
    logger.info(f"LLM judge: pos={result['positive_written']}, "
                f"neg={result['negative_written']}, unc={result['uncertain_written']}")

    # Save pseudo labels JSONL
    records = labeler.generate_pseudo_labels_jsonl(labeled)
    labels_path = os.path.join(args.store_dir, "pseudo_labels.jsonl")
    with open(labels_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(records)} pseudo labels to {labels_path}")


def _predict_tools_offline_rewardevo(memory, question: str, images: list) -> list[str]:
    """Predict tools using episodic memory retrieval (same DB as MemRL)."""
    try:
        from ..memrl.intent_parser import IntentParser
        intent_parser = IntentParser()
        intent = intent_parser.parse(question, images)
        candidates = memory.retrieve_candidates(intent, top_k=5)
        # Prefer high-utility (positive-labeled) memories
        candidates.sort(key=lambda m: m.get("utility", 0.0), reverse=True)
        tools: list[str] = []
        for mem in candidates[:3]:
            for t in mem.get("experience", {}).get("tool_sequence", []):
                if t not in tools:
                    tools.append(t)
        return tools
    except Exception as e:
        logger.warning(f"RewardEvo predict failed: {e}")
        return []


def cmd_eval(args):
    """Offline evaluation: predict tools from pseudo-labeled episodic memory."""
    from ..memrl.episodic_memory import EpisodicMemory
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory

    eval_data = []
    with open(args.eval_data) as f:
        for line in f:
            eval_data.append(json.loads(line))

    # Load episodic memory built by the label phase
    memory_db = os.path.join(args.store_dir, "episodic_memory.db")
    memory = EpisodicMemory(memory_db)
    logger.info(f"Loaded episodic memory: {memory.count()} entries from {memory_db}")

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "rewardevo",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": []
    }

    precision_sum, recall_sum, f1_sum, exact_sum = 0, 0, 0, 0

    for i, case in enumerate(eval_data):
        tools_called = _predict_tools_offline_rewardevo(
            memory, case.get("question", ""), case.get("images", [])
        )
        expected_tools = case.get("expected_tools", [])

        traj = Trajectory(
            task_id=case.get("task_id", f"eval_{i}"),
            question=case.get("question", ""),
            images=case.get("images", []),
            turns=[],
            tools_called=tools_called,
            expected_tools=expected_tools,
            final_answer="",
            success=False,
            task_type=""
        )

        eval_result = evaluator.evaluate(traj)

        results["cases"].append({
            "task_id": traj.task_id,
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

    logger.info(f"\n=== RewardEvo Evaluation Results ===")
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
    parser = argparse.ArgumentParser(description="RewardEvo runner")
    subparsers = parser.add_subparsers(dest="command")

    label_parser = subparsers.add_parser("label", help="Pseudo-label trajectories via LLM-as-Judge")
    label_parser.add_argument("--train-data", required=True, help="Training data (JSON)")
    label_parser.add_argument("--store-dir", required=True, help="Store directory for outputs")
    label_parser.add_argument("--limit", type=int, default=None, help="Limit trajectories")

    eval_parser = subparsers.add_parser("eval", help="Offline evaluation")
    eval_parser.add_argument("--eval-data", required=True, help="Eval data file (JSONL)")
    eval_parser.add_argument("--store-dir", default="evo_res/rewardevo", help="Store directory")
    eval_parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    if args.command == "label":
        cmd_label(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
