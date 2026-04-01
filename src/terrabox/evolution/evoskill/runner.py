"""EvoSkill experiment runner.

Implements the multi-agent failure-driven skill discovery from:
EvoSkill — Automated Skill Discovery for Multi-Agent Systems (arXiv 2603.02766)

Three-agent loop:
  1. BaseAgent executes task → Trajectory
  2. ProposerAgent analyzes failure → FailureAnalysis
  3. SkillBuilderAgent builds SkillModule
  4. ParetoManager evaluates and retains Pareto-optimal skills

Usage:
    # Run multi-agent skill discovery loop
    python -m terrabox.evolution.evoskill.runner discover \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/evoskill \\
        --n-episodes 50 \\
        --pareto-size 30

    # Evaluate agent with Pareto-optimal skills
    python -m terrabox.evolution.evoskill.runner eval \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/evoskill \\
        --top-n 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_pipeline(store_dir: str):
    from ..shared.llm_client import EvolutionLLMClient
    from .agents import BaseAgent, ProposerAgent, SkillBuilderAgent
    from .pareto_manager import ParetoManager
    from .transfer_engine import CrossDomainTransfer

    os.makedirs(store_dir, exist_ok=True)
    llm = EvolutionLLMClient()
    base_agent = BaseAgent()
    proposer = ProposerAgent(llm)
    builder = SkillBuilderAgent(llm)
    pareto = ParetoManager(os.path.join(store_dir, "pareto_frontier.json"))
    transfer = CrossDomainTransfer()
    return base_agent, proposer, builder, pareto, transfer


def _compute_baseline_f1(eval_cases: list[dict], base_agent, system_prompt: str) -> float:
    """Compute baseline tool-match F1 without skill injection."""
    from ..shared.evaluator import ToolMatchEvaluator

    evaluator = ToolMatchEvaluator()
    sample = eval_cases[:min(10, len(eval_cases))]
    f1_sum = 0.0
    for case in sample:
        try:
            traj = base_agent.execute(
                case["question"], case.get("images", []), system_prompt,
                expected_tools=case.get("expected_tools", []),
            )
            result = evaluator.evaluate(traj)
            f1_sum += result.tool_f1
        except Exception as e:
            logger.warning(f"Baseline eval failed: {e}")
    return f1_sum / len(sample) if sample else 0.0


def cmd_discover(args):
    """Run multi-agent skill discovery loop.

    IMPORTANT — data separation:
      Discovery uses train.json trajectories (as synthetic execution seeds),
      NOT eval.jsonl. Pareto skill evaluation uses a held-out validation split
      of train data. eval.jsonl is reserved exclusively for final evaluation.
    """
    from ..shared.data_loader import OpenEarthLoader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.prompt_builder import PromptAugmenter

    loader = OpenEarthLoader(args.train_data, args.eval_data)

    # --- Use train.json for discovery (NOT eval.jsonl) ---
    # Convert train trajectories into discovery "cases" (same format as eval cases)
    train_trajs = loader.load_train_trajectories(limit=args.n_episodes * 3)
    discovery_cases = [
        {
            "id": t.task_id,
            "question": t.question,
            "images": t.images,
            # Use the tool sequence from train as soft expected_tools (ground truth proxy)
            "expected_tools": t.tools_called,
        }
        for t in train_trajs
        if t.tools_called   # only cases where tool usage is known
    ][:args.n_episodes]

    # Held-out validation split from train (last 20% of discovery_cases)
    # Used for Pareto skill evaluation — never the same as discovery batch
    n_val = max(5, len(discovery_cases) // 5)
    discovery_batch = discovery_cases[:-n_val]
    validation_cases = discovery_cases[-n_val:]

    logger.info(
        f"Discovery: {len(discovery_batch)} train cases  |  "
        f"Validation: {len(validation_cases)} held-out train cases  |  "
        f"eval.jsonl reserved for final evaluation only"
    )

    if args.reset:
        _, _, _, pareto, _ = _build_pipeline(args.store_dir)
        pareto._store.clear()
        logger.info("Cleared existing Pareto frontier")

    base_agent, proposer, builder, pareto, transfer = _build_pipeline(args.store_dir)
    evaluator = ToolMatchEvaluator()
    base_prompt = PromptAugmenter.BASE_SYSTEM

    logger.info(f"Computing baseline F1 on {min(10, len(validation_cases))} validation samples...")
    baseline_f1 = _compute_baseline_f1(validation_cases[:10], base_agent, base_prompt)
    logger.info(f"Baseline F1: {baseline_f1:.4f}")

    skills_discovered = 0
    skills_added = 0

    for i, case in enumerate(discovery_batch):
        logger.info(f"\n[Episode {i+1}/{len(discovery_batch)}]")

        # Step 1: BaseAgent executes on train case
        try:
            traj = base_agent.execute(
                case["question"], case.get("images", []), base_prompt,
                expected_tools=case.get("expected_tools", []),
            )
        except Exception as e:
            logger.warning(f"BaseAgent failed: {e}")
            continue

        result = evaluator.evaluate(traj)
        logger.info(f"  F1={result.tool_f1:.3f}")

        # Only analyze failures (F1 < 0.5)
        if result.tool_f1 >= 0.5:
            logger.info("  Success — skipping failure analysis")
            continue

        # Step 2: ProposerAgent analyzes failure
        analysis = proposer.analyze(traj)
        if not analysis:
            continue
        logger.info(f"  Failure mode: {analysis.get('failure_mode')}")

        # Step 3: SkillBuilderAgent generates skill
        skill = builder.build(analysis, traj)
        if not skill:
            continue
        skills_discovered += 1
        logger.info(f"  Proposed skill: {skill.name}")

        # Step 4: Evaluate on HELD-OUT VALIDATION set (not discovery batch, not eval.jsonl)
        f1_delta, generality = pareto.evaluate_skill(
            skill, validation_cases, base_agent, baseline_f1
        )
        added = pareto.add_candidate(skill, f1_delta, generality)
        if added:
            skills_added += 1

    logger.info(f"\n=== EvoSkill Discovery Complete ===")
    logger.info(f"  Episodes run:    {len(discovery_batch)}")
    logger.info(f"  Skills proposed: {skills_discovered}")
    logger.info(f"  Skills added:    {skills_added}")
    logger.info(f"  Frontier size:   {pareto.count()}  (eval.jsonl not used here)")


def cmd_eval(args):
    """Evaluate agent with Pareto-optimal skill injection."""
    from ..shared.data_loader import OpenEarthLoader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory
    from .prompt_injector import EvoSkillPromptInjector

    loader = OpenEarthLoader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    base_agent, _, _, pareto, transfer = _build_pipeline(args.store_dir)
    injector = EvoSkillPromptInjector(pareto, transfer, top_n=args.top_n)
    evaluator = ToolMatchEvaluator()

    logger.info(f"Evaluating {len(eval_cases)} cases with {pareto.count()} Pareto skills")
    results = []

    for i, case in enumerate(eval_cases):
        system_prompt = injector.augment(
            case["question"], images=case.get("images", [])
        )
        try:
            traj = base_agent.execute(
                case["question"], case.get("images", []), system_prompt,
                expected_tools=case.get("expected_tools", []),
            )
            result = evaluator.evaluate(traj)
            results.append(result)
        except Exception as e:
            logger.warning(f"Eval case {i} failed: {e}")

        if (i + 1) % 20 == 0:
            avg_f1 = sum(r.tool_f1 for r in results) / len(results)
            logger.info(f"  [{i+1}/{len(eval_cases)}] avg_f1={avg_f1:.3f}")

    metrics = evaluator.aggregate(results)
    logger.info("\n=== EvoSkill Evaluation Results ===")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  N cases:     {metrics['n']}")

    out_path = args.output or os.path.join(args.store_dir, "eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EvoSkill: multi-agent failure-driven skill discovery")
    parser.add_argument("mode", choices=["discover", "eval"])
    parser.add_argument("--train-data", default="data/openearth/train.json")
    parser.add_argument("--eval-data", default="data/openearth/eval.jsonl")
    parser.add_argument("--store-dir", default="evolution_store/evoskill")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--pareto-size", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing skill store before running (prevents accumulation on re-run)")
    args = parser.parse_args()

    if args.mode == "discover":
        cmd_discover(args)
    elif args.mode == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
