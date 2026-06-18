"""ExpeL runner: build principles and evaluate offline tool prediction."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

from .principle_bank import PrincipleBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_task_type(case: dict) -> str:
    t = (case.get("task_type") or "").strip()
    if t:
        return t
    cid = case.get("task_id", case.get("id", "unknown"))
    parts = cid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return "unknown"


def cmd_build(args):
    from ..shared.data_loader import make_loader
    from ..shared.llm_client import EvolutionLLMClient
    from .distiller import ExpeLDistiller

    loader = make_loader(args.train_data, args.train_data)
    trajectories = loader.load_train_trajectories(limit=args.limit)
    logger.info("Loaded %d trajectories", len(trajectories))

    bank = PrincipleBank(args.store_dir)
    llm = EvolutionLLMClient(llm_url=args.llm_url)
    if not llm._use_docker:
        raise RuntimeError(
            "ExpeL build requires Docker vLLM. "
            "Please ensure EVOLUTION_LLM_URL (or --llm-url) is reachable."
        )
    distiller = ExpeLDistiller(bank, llm)
    created = distiller.distill_batch(trajectories, limit=args.limit)

    logger.info("Saved ExpeL principles to %s", bank.path)
    logger.info(
        "Distilled principles: general=%d task_specific=%d mistakes=%d",
        created["general"],
        created["task"],
        created["mistakes"],
    )


def _predict_tools_from_principles(bank: PrincipleBank, question: str, task_type: str, top_k: int) -> list[str]:
    # Lightweight parser: extract quoted tool slugs from principle texts and rank by support/score.
    import re

    candidates: dict[str, float] = {}

    def extract_tool_slugs(text: str) -> list[str]:
        pattern = r"(?<![A-Za-z0-9_.-])([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)(?![A-Za-z0-9_.-])"
        seen = set()
        slugs = []
        for slug in re.findall(pattern, text or ""):
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
        return slugs

    def add_from_entries(entries: list[dict], weight: float) -> None:
        for e in entries:
            text = e.get("text", "")
            score = float(e.get("score", 1.0)) * weight
            for slug in extract_tool_slugs(text):
                candidates[slug] = max(candidates.get(slug, 0.0), score)

    add_from_entries(bank.data.get("general", []), 1.0)
    add_from_entries(bank.data.get("task_specific", {}).get(task_type, []), 1.5)

    # If nothing extracted, fallback to any tools in expected style from question tokens (none -> empty)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [slug for slug, _ in ranked[:top_k]]


def cmd_eval(args):
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory

    bank = PrincipleBank(args.store_dir)
    if not bank.data.get("general") and not bank.data.get("task_specific"):
        raise RuntimeError(
            f"No principles found in {bank.path}. Run 'build' first."
        )

    eval_data = []
    with open(args.eval_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "expel",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    precision_sum = recall_sum = f1_sum = exact_sum = 0.0

    for i, case in enumerate(eval_data):
        task_id = case.get("task_id", case.get("id", f"eval_{i}"))
        task_type = _infer_task_type(case)
        question = case.get("question", "")
        expected_tools = case.get("expected_tools", [])

        predicted = _predict_tools_from_principles(bank, question, task_type, top_k=args.top_k)

        traj = Trajectory(
            task_id=task_id,
            question=question,
            images=case.get("images", []),
            turns=[],
            tools_called=predicted,
            expected_tools=expected_tools,
            final_answer="",
            success=False,
            task_type=task_type,
        )
        ev = evaluator.evaluate(traj)

        results["cases"].append(
            {
                "task_id": task_id,
                "tools_called": predicted,
                "expected_tools": expected_tools,
                "precision": ev.tool_precision,
                "recall": ev.tool_recall,
                "f1": ev.tool_f1,
            }
        )

        precision_sum += ev.tool_precision
        recall_sum += ev.tool_recall
        f1_sum += ev.tool_f1
        if set(predicted) == set(expected_tools):
            exact_sum += 1.0

    n = len(eval_data)
    results["summary"] = {
        "precision": precision_sum / n if n else 0.0,
        "recall": recall_sum / n if n else 0.0,
        "f1": f1_sum / n if n else 0.0,
        "exact_match": exact_sum / n if n else 0.0,
        "n_cases": n,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved eval results to %s", args.output)


def main():
    parser = argparse.ArgumentParser(description="ExpeL runner")
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="Build principle bank from train trajectories")
    p_build.add_argument("--train-data", required=True, help="Training trajectory file")
    p_build.add_argument("--store-dir", required=True, help="Store dir for principles.json")
    p_build.add_argument("--limit", type=int, default=None, help="Optional train sample limit")
    p_build.add_argument("--llm-url", default=None, help="Optional override for EVOLUTION_LLM_URL")

    p_eval = sub.add_parser("eval", help="Offline evaluation")
    p_eval.add_argument("--store-dir", required=True, help="Store dir with principles.json")
    p_eval.add_argument("--eval-data", required=True, help="Eval JSONL file")
    p_eval.add_argument("--output", required=True, help="Output JSON")
    p_eval.add_argument("--top-k", type=int, default=8, help="Max predicted tools")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
