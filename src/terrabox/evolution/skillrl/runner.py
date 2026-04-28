"""SkillRL experiment runner.

Implements the Recursive Skill-Augmented Reinforcement Learning approach from:
SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning
(arXiv 2602.08234)

Usage:
    # Phase 1: Distill skills from training data (offline)
    python -m terrabox.evolution.skillrl.runner distill \\
        --train-data data/openearth/train.json \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/skillrl \\
        --limit 500

    # Phase 2: Evaluate agent with skill injection
    python -m terrabox.evolution.skillrl.runner eval \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/skillrl \\
        --top-k 3

    # Phase 3: Online evolution (eval + recursive re-distillation on drift)
    python -m terrabox.evolution.skillrl.runner online \\
        --train-data data/openearth/train.json \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/skillrl \\
        --top-k 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_pipeline(store_dir: str, top_k: int):
    """Build the full SkillRL pipeline."""
    from ..shared.llm_client import EvolutionLLMClient
    from .distiller import ExperienceDistiller
    from .evolver import SkillEvolver
    from .prompt_injector import SkillRLPromptInjector
    from .retriever import SkillRetriever
    from .skill_bank import HierarchicalSkillBank

    os.makedirs(store_dir, exist_ok=True)
    llm = EvolutionLLMClient()
    bank = HierarchicalSkillBank(store_dir)
    distiller = ExperienceDistiller(bank, llm)
    retriever = SkillRetriever(bank)
    evolver = SkillEvolver(bank, distiller)
    injector = SkillRLPromptInjector(bank, retriever, top_k=top_k)
    return bank, distiller, retriever, evolver, injector


def cmd_distill(args):
    """Offline distillation: extract skills from training trajectories."""
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator

    logger.info(f"Distilling skills from {args.train_data} (limit={args.limit})")
    bank, distiller, _, _, _ = _build_pipeline(args.store_dir, top_k=3)

    if getattr(args, "reset", False):
        bank.clear()
        logger.info("Skill bank cleared (--reset)")

    evaluator = ToolMatchEvaluator()

    loader = make_loader(args.train_data, args.eval_data)
    try:
        import tqdm as _tqdm_mod
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        _iter = _tqdm_mod.tqdm(
            loader.iter_train(limit=args.limit),
            desc="[skillrl build]", unit="traj", ncols=90,
            position=tqdm_pos, leave=True, total=args.limit,
        )
    except ImportError:
        _iter = loader.iter_train(limit=args.limit)
    episodes = [evaluator.evaluate(t) for t in _iter]

    logger.info(f"Processing {len(episodes)} trajectories...")
    created = distiller.distill_batch(episodes)
    counts = bank.counts()
    logger.info(f"Skills created: {created}")
    logger.info(f"Bank: general={counts['general']}, specific={counts['specific']}, "
                f"mistakes={counts['mistakes']}")


def _run_agent_on_task(task: dict, system_prompt: str) -> dict:
    """Run agent on a task; return {tools_called, final_answer, expected_tools}."""
    question = task.get("question", "")
    images = task.get("images", [])
    expected_tools = task.get("expected_tools", [])

    try:
        from langchain_core.messages import HumanMessage
        from langgraph.prebuilt import create_react_agent
        from ..shared.mock_user import MockUser
        from ...agent.config import load_config
        from ...agent.llm import get_llm
        from ...agent.tools import build_langchain_tools

        from ...extensions import load_builtin_toolkits
        load_builtin_toolkits()

        config = load_config()
        llm = get_llm(config)
        user = MockUser()
        tools = build_langchain_tools(user)
        content = question
        if images:
            content += f"\n\n[Images: {', '.join(images)}]"

        agent = create_react_agent(llm, tools, prompt=system_prompt)
        result = agent.invoke({"messages": [HumanMessage(content=content)]})

        tools_called = []
        for msg in result.get("messages", []):
            for tc in getattr(msg, "tool_calls", []):
                name = tc.get("name", "").replace("__", ".")
                if name:
                    tools_called.append(name)
        final = ""
        msgs = result.get("messages", [])
        if msgs and hasattr(msgs[-1], "content"):
            final = msgs[-1].content

        return {"tools_called": tools_called, "final_answer": final, "expected_tools": expected_tools}

    except Exception as e:
        logger.warning(f"Agent execution failed: {e}")
        return {"tools_called": [], "final_answer": "", "expected_tools": expected_tools}


def _infer_task_type(question: str) -> str:
    """Simple heuristic task type inference from question text."""
    q = question.lower()
    if any(kw in q for kw in ["change", "difference", "compare", "between", "before", "after"]):
        return "change_detection"
    if any(kw in q for kw in ["nearest", "closest", "route", "station"]):
        return "poi_routing"
    if any(kw in q for kw in ["detect", "segment", "count", "measure"]):
        return "segmentation"
    if any(kw in q for kw in ["ndvi", "ndbi", "nbr", "index", "calculate"]):
        return "index_calculation"
    return "general_qa"


def _predict_tools_offline_skillrl(retriever, question: str, task_type: str) -> list[str]:
    """Predict tools by extracting slugs from retrieved skill texts (no agent needed)."""
    import re
    try:
        skill_texts = retriever.retrieve(question, task_type=task_type, top_k=3)
        all_text = " ".join(
            text for tier in skill_texts.values() for text in tier
        )
        # Extract tool slugs (format: word.word, e.g. geo_raster.calculate_index)
        slugs = re.findall(r'\b[\w]+\.[\w]+\b', all_text)
        seen: list[str] = []
        for s in slugs:
            if s not in seen and "." in s:
                seen.append(s)
        return seen[:8]
    except Exception:
        return []


def cmd_eval(args, online: bool = False):
    """Evaluate agent with skill injection; optionally run online evolution."""
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory

    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    _, distiller, retriever, evolver, injector = _build_pipeline(args.store_dir, args.top_k)
    evaluator = ToolMatchEvaluator()

    logger.info(f"Evaluating on {len(eval_cases)} cases, online={online}")
    results = []
    t0 = time.time()

    for i, case in enumerate(eval_cases):
        task_type = _infer_task_type(case.get("question", ""))
        if online:
            system_prompt = injector.augment(case["question"], task_type=task_type)
            run = _run_agent_on_task(case, system_prompt)
            tools_called = run["tools_called"]
        else:
            # Offline: predict tools via skill bank retrieval (no agent)
            tools_called = _predict_tools_offline_skillrl(retriever, case["question"], task_type)

        traj = Trajectory(
            task_id=case.get("id", str(i)),
            question=case["question"],
            images=case.get("images", []),
            turns=[],
            tools_called=tools_called,
            expected_tools=case.get("expected_tools", []),
            final_answer="",
            success=False,
            task_type=task_type,
        )
        episode = evaluator.evaluate(traj)
        results.append(episode)

        if online:
            evolved = evolver.record_episode(episode)
            if evolved:
                logger.info(f"  [Evolution triggered at step {i+1}]")

        if (i + 1) % 20 == 0:
            avg_f1 = sum(r.tool_f1 for r in results) / len(results)
            logger.info(f"  [{i+1}/{len(eval_cases)}] avg_f1={avg_f1:.3f}  "
                        f"elapsed={time.time()-t0:.1f}s")

    metrics = evaluator.aggregate(results)
    logger.info("\n=== SkillRL Evaluation Results ===")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  N cases:     {metrics['n']}")
    if online:
        logger.info(f"  Evolution cycles: {evolver.evolution_count}")

    out_path = args.output or f"evolution_store/skillrl/eval_results.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "mode": "online" if online else "offline"}, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SkillRL: hierarchical skill evolution")
    parser.add_argument("mode", choices=["distill", "eval", "online"])
    parser.add_argument("--train-data", default="data/openearth/train.json")
    parser.add_argument("--eval-data", default="data/openearth/eval.jsonl")
    parser.add_argument("--store-dir", default="evolution_store/skillrl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing skill bank before distillation")
    args = parser.parse_args()

    if args.mode == "distill":
        cmd_distill(args)
    elif args.mode == "eval":
        cmd_eval(args, online=False)
    elif args.mode == "online":
        cmd_distill(args)
        cmd_eval(args, online=True)


if __name__ == "__main__":
    main()
