"""SelfCritic experiment runner.

Build: reads trajectories → LLM critiques successes → stores optimal chains.
Eval:  retrieves critic skills → injects into agent → measures tool F1.

Works with any trajectory source (SFT JSON or OpenEarth format).
Does NOT modify any existing evolution module.

Usage:
    # Build from SFT data (all samples treated as success, min-f1=0.0)
    python -m terrabox.evolution.selfcritic.runner build \\
        --train-data data/disaster_sft_dataset_v2.json \\
        --store-dir evolution_store/selfcritic \\
        --min-f1 0.0

    # Build from OpenEarth trajectories (only high-quality successes)
    python -m terrabox.evolution.selfcritic.runner build \\
        --train-data data/openearth/train.json \\
        --store-dir evolution_store/selfcritic \\
        --min-f1 0.8

    # Offline eval (no agent, extract tool slugs from critic text)
    python -m terrabox.evolution.selfcritic.runner eval \\
        --eval-data data/disaster_sft_dataset_v2.json \\
        --store-dir evolution_store/selfcritic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_task_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["flood", "water", "inundation"]):
        return "flood_detection"
    if any(k in q for k in ["fire", "burn", "frp", "wildfire"]):
        return "fire_detection"
    if any(k in q for k in ["earthquake", "damage", "collapse"]):
        return "earthquake_damage"
    if any(k in q for k in ["drought", "ndvi", "vegetation"]):
        return "drought_monitoring"
    if any(k in q for k in ["typhoon", "cyclone", "port"]):
        return "typhoon_port_damage"
    if any(k in q for k in ["heat", "lst", "temperature"]):
        return "heatwave_analysis"
    return "general"


def cmd_build(args):
    """Build critic skill bank from training trajectories."""
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.llm_client import EvolutionLLMClient
    from .bank import CriticSkillBank
    from .distiller import SelfCriticDistiller

    logger.info(f"Building critic bank from {args.train_data} (min_f1={args.min_f1})")

    bank = CriticSkillBank(args.store_dir)
    if getattr(args, "reset", False):
        bank.clear()
        logger.info("Critic bank cleared (--reset)")

    llm = EvolutionLLMClient(llm_url=getattr(args, "llm_url", None))
    if not llm._use_docker:
        raise RuntimeError(
            "SelfCritic build requires Docker vLLM. "
            "Ensure EVOLUTION_LLM_URL is reachable."
        )

    distiller = SelfCriticDistiller(bank, llm, min_f1=args.min_f1)
    evaluator = ToolMatchEvaluator()

    loader = make_loader(args.train_data, args.train_data)

    try:
        import tqdm as _tqdm
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        _iter = _tqdm.tqdm(
            loader.iter_train(limit=args.limit),
            desc="[selfcritic build]", unit="traj", ncols=90,
            position=tqdm_pos, leave=True, total=args.limit,
        )
    except ImportError:
        _iter = loader.iter_train(limit=args.limit)

    episodes = [evaluator.evaluate(t) for t in _iter]
    logger.info(f"Loaded {len(episodes)} trajectories")

    created = distiller.distill_batch(episodes)
    logger.info(f"Done — critic skills created: {created}, bank total: {bank.count()}")


def cmd_eval(args):
    """Offline evaluation: retrieve critic skills and measure tool prediction quality."""
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory
    from .bank import CriticSkillBank
    from .retriever import CriticRetriever

    bank = CriticSkillBank(args.store_dir)
    if bank.count() == 0:
        raise RuntimeError(f"No critic skills in {args.store_dir}. Run 'build' first.")

    retriever = CriticRetriever(bank)
    evaluator = ToolMatchEvaluator()

    eval_cases = _load_eval_cases(args.eval_data)
    if args.limit:
        eval_cases = eval_cases[:args.limit]
    logger.info(f"Evaluating {len(eval_cases)} cases")

    results = []
    t0 = time.time()
    for i, case in enumerate(eval_cases):
        question = case.get("question", "")
        task_type = case.get("task_type") or _infer_task_type(question)
        expected_tools = case.get("expected_tools", [])

        skills = retriever.retrieve(question, task_type=task_type, top_k=args.top_k)
        predicted = _extract_tools_from_skills(skills)

        traj = Trajectory(
            task_id=case.get("id", str(i)),
            question=question,
            images=case.get("images", []),
            turns=[],
            tools_called=predicted,
            expected_tools=expected_tools,
            final_answer="",
            success=False,
            task_type=task_type,
        )
        results.append(evaluator.evaluate(traj))

        if (i + 1) % 20 == 0:
            avg_f1 = sum(r.tool_f1 for r in results) / len(results)
            logger.info(f"  [{i+1}/{len(eval_cases)}] avg_f1={avg_f1:.3f}  "
                        f"elapsed={time.time()-t0:.1f}s")

    metrics = evaluator.aggregate(results)
    logger.info("\n=== SelfCritic Eval Results ===")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  N cases:     {metrics['n']}")

    out = args.output or os.path.join(args.store_dir, "eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"metrics": metrics, "method": "selfcritic"}, f, indent=2)
    logger.info(f"Results → {out}")


def _load_eval_cases(path: str) -> list[dict]:
    """Load eval cases from JSON or JSONL."""
    cases = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return cases


def _extract_tools_from_skills(skills: list[str]) -> list[str]:
    """Extract tool slugs from optimal_chain lines in critic skill texts."""
    seen: list[str] = []
    for text in skills:
        for line in text.splitlines():
            if line.startswith("Optimized chain:"):
                chain_str = line.split(":", 1)[1].strip()
                for slug in re.split(r"\s*→\s*|\s*->\s*", chain_str):
                    slug = slug.strip()
                    if "." in slug and slug not in seen:
                        seen.append(slug)
    return seen[:10]


def main():
    parser = argparse.ArgumentParser(description="SelfCritic: optimal chain distillation")
    parser.add_argument("mode", choices=["build", "eval"])
    parser.add_argument("--train-data", default="data/disaster_sft_dataset_v2.json")
    parser.add_argument("--eval-data", default="data/disaster_sft_dataset_v2.json")
    parser.add_argument("--store-dir", default="evolution_store/selfcritic")
    parser.add_argument("--min-f1", type=float, default=0.0,
                        help="Min F1 to critique (0.0=all for SFT data, 0.8=high-quality only)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reset", action="store_true", help="Clear bank before build")
    parser.add_argument("--llm-url", default=None, help="Override EVOLUTION_LLM_URL")
    args = parser.parse_args()

    if args.mode == "build":
        cmd_build(args)
    elif args.mode == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
