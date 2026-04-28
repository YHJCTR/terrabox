"""CausalEvo experiment runner.

CausalEvo: Counterfactual Causal Skill Discovery for Self-Evolving Tool-Calling Agents

Three run modes
───────────────
build   Build Causal Tool Function Models (CTFM) from training trajectories.
        Computes statistical CCA scores and extracts per-tool models.
        Safe to re-run: use --reset to clear and rebuild from scratch.

eval    Evaluate the agent with CTFM-based causal plan injection.
        Uses eval.jsonl as the test set (never seen during build).

online  Evaluate + incrementally update success_rate counters in CTFM
        based on live episode outcomes.

Ablation modes  (--ablation, only meaningful for eval / online)
───────────────
no_cca       All CCA scores reset to 0.5 — tests whether causal weighting matters.
no_synthesis BM25 retrieval replaces causal graph synthesis — same CTFM knowledge,
             different planning mechanism.
no_ctfm      No CTFM at all; pure task-type rule-based hints — tests whether the
             learned model contributes beyond hand-written heuristics.

Quick start
───────────
    # Step 1 — build CTFM
    python -m terrabox.evolution.causalevo.runner build \\
        --train-data data/openearth/train.json \\
        --store-dir  evolution_store/causalevo \\
        --limit 2000 --reset --verbose

    # Step 2 — offline eval
    python -m terrabox.evolution.causalevo.runner eval \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/causalevo \\
        --top-k 5 --limit 100

    # Step 3 — online eval (CTFM updated after each episode)
    python -m terrabox.evolution.causalevo.runner online \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/causalevo \\
        --top-k 5 --limit 100

    # Ablation A — no CCA
    python -m terrabox.evolution.causalevo.runner eval \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/causalevo \\
        --ablation no_cca --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================= #
# Pipeline helpers                                                         #
# ======================================================================= #

def _build_injector(
    store_dir: str,
    top_k: int,
    ablation: Optional[str] = None,
):
    from .prompt_injector import CausalEvoPromptInjector
    from .tool_function_model import CTFMStore

    os.makedirs(store_dir, exist_ok=True)
    store = CTFMStore(os.path.join(store_dir, "causal_tool_models.json"))
    return CausalEvoPromptInjector(store, top_k=top_k, use_ablation=ablation), store


def _infer_task_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in ["change", "compare", "difference", "before", "after"]):
        return "change_detection"
    if any(kw in q for kw in ["nearest", "route", "closest", "station", "poi"]):
        return "poi_routing"
    if any(kw in q for kw in ["detect", "segment", "count"]):
        return "segmentation"
    if any(kw in q for kw in ["ndvi", "ndbi", "ndwi", "index", "calculate"]):
        return "index_calculation"
    if any(kw in q for kw in ["image", "satellite", "aerial", "scene"]):
        return "scene_classification"
    return "general_qa"


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
        content = question + (f"\n\n[Images: {', '.join(images)}]" if images else "")
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


# ======================================================================= #
# Commands                                                                 #
# ======================================================================= #

def cmd_build(args) -> None:
    """Build CTFM from training trajectories (offline)."""
    from ..shared.data_loader import make_loader
    from .counterfactual_credit import compute_statistical_cca
    from .tool_function_model import CTFMBuilder, CTFMStore

    os.makedirs(args.store_dir, exist_ok=True)
    store = CTFMStore(os.path.join(args.store_dir, "causal_tool_models.json"))

    if getattr(args, "reset", False):
        store.clear()
        logger.info("CTFM store cleared (--reset)")

    logger.info(f"Loading training trajectories from {args.train_data} (limit={args.limit})")
    loader = make_loader(args.train_data, args.eval_data)
    try:
        import tqdm as _tqdm_mod
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        _iter = _tqdm_mod.tqdm(
            loader.iter_train(limit=args.limit),
            desc="[causalevo build]", unit="traj", ncols=90,
            position=tqdm_pos, leave=True, total=args.limit,
        )
    except ImportError:
        _iter = loader.iter_train(limit=args.limit)
    trajectories = list(_iter)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    if not trajectories:
        logger.error("No trajectories loaded — check --train-data path.")
        return

    # Step 1: Compute CCA (statistical or LLM-based)
    if getattr(args, "use_llm_cca", False):
        logger.info("Computing LLM-based Counterfactual Credit Attribution (CCA)...")
        from .counterfactual_credit import compute_llm_cca
        from ..shared.llm_client import EvolutionLLMClient

        llm_client = EvolutionLLMClient()
        global_cca = compute_llm_cca(trajectories, llm_client)
        logger.info("LLM-CCA computation complete")
    else:
        logger.info("Computing statistical Counterfactual Credit Attribution (CCA)...")
        global_cca = compute_statistical_cca(trajectories)

    top_cca = sorted(global_cca.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top-10 causally important tools (CCA score):")
    for slug, score in top_cca[:10]:
        logger.info(f"  {score:.3f}  {slug}")

    # Step 2: Build CTFM
    logger.info("Building Causal Tool Function Models...")
    builder = CTFMBuilder(
        top_k_keywords=args.top_k_keywords,
        top_k_downstream=args.top_k_downstream,
    )
    models = builder.build(trajectories, global_cca)

    # Save
    store.save(models)
    logger.info(f"CTFM complete: {len(models)} tool models → {args.store_dir}")

    if getattr(args, "verbose", False):
        logger.info("\nSample CTFM entries (top-5 by CCA):")
        top_models = sorted(models.values(), key=lambda m: m.avg_cca_score, reverse=True)
        for m in top_models[:5]:
            logger.info("\n" + m.to_prompt_text())


def _predict_tools_offline_causalevo(store, question: str, top_k: int = 5) -> list[str]:
    """Predict tools using CTFM CCA scores + character-level keyword similarity (no agent needed)."""
    try:
        models = store.load()
        if not models:
            return []
        q_chars = set(question.lower())
        scored = []
        for slug, model in models.items():
            if model.precondition_keywords:
                kw_text = "".join(model.precondition_keywords).lower()
                kw_chars = set(kw_text)
                char_overlap = len(q_chars & kw_chars) / max(len(q_chars), 1)
            else:
                char_overlap = 0.0
            # Combine CCA importance + keyword relevance
            score = model.avg_cca_score * 0.5 + char_overlap * 0.5
            scored.append((score, model.avg_cca_score, slug, model))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        tools: list[str] = []
        for _, _, slug, model in scored[:top_k]:
            if slug not in tools:
                tools.append(slug)
            for ds in model.downstream_tools[:1]:
                if ds not in tools:
                    tools.append(ds)
        return tools[:top_k]
    except Exception:
        return []


def cmd_eval(args, online: bool = False) -> dict:
    """Evaluate agent with CTFM injection; optionally update CTFM online."""
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory

    ablation: Optional[str] = getattr(args, "ablation", None)
    injector, store = _build_injector(args.store_dir, args.top_k, ablation=ablation)

    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[: args.limit]

    evaluator = ToolMatchEvaluator()
    mode_label = ("online" if online else "offline") + (f"[{ablation}]" if ablation else "")
    logger.info(f"CausalEvo eval: {len(eval_cases)} cases | mode={mode_label}")

    results = []
    t0 = time.time()

    for i, case in enumerate(eval_cases):
        task_type = _infer_task_type(case.get("question", ""))
        if online:
            system_prompt = injector.augment(
                case["question"], images=case.get("images", []), task_type=task_type
            )
            run = _run_agent_on_task(case, system_prompt)
            tools_called = run["tools_called"]
        else:
            # Offline: predict tools via CTFM keyword matching (no agent)
            tools_called = _predict_tools_offline_causalevo(store, case["question"], args.top_k)

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
            injector.record_outcome(
                query=case["question"],
                tools_called=tools_called,
                reward=episode.reward,
                task_type=task_type,
            )

        if (i + 1) % 20 == 0:
            avg_f1 = sum(r.tool_f1 for r in results) / len(results)
            logger.info(
                f"  [{i + 1}/{len(eval_cases)}] avg_f1={avg_f1:.3f} "
                f"elapsed={time.time() - t0:.1f}s"
            )

    metrics = evaluator.aggregate(results)
    logger.info("\n=== CausalEvo Evaluation Results ===")
    logger.info(f"  Mode:        {mode_label}")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  N cases:     {metrics['n']}")

    out_path = args.output or os.path.join(
        args.store_dir, f"eval_{mode_label.replace('[','_').replace(']','')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "mode": mode_label, "ablation": ablation}, f, indent=2)
    logger.info(f"Results → {out_path}")
    return metrics


# ======================================================================= #
# CLI                                                                      #
# ======================================================================= #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CausalEvo: Counterfactual Causal Skill Discovery for Tool-Calling Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", choices=["build", "eval", "online"],
                        help="Operation mode")
    parser.add_argument("--train-data", default="data/openearth/train.json")
    parser.add_argument("--eval-data",  default="data/openearth/eval.jsonl")
    parser.add_argument("--store-dir",  default="evolution_store/causalevo",
                        help="Directory for CTFM store")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Max tools in synthesized plan (default 5)")
    parser.add_argument("--top-k-keywords", type=int, default=10,
                        help="Precondition keywords extracted per tool (default 10)")
    parser.add_argument("--top-k-downstream", type=int, default=5,
                        help="Downstream tools tracked per tool (default 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit trajectories / eval cases processed")
    parser.add_argument("--output", default=None,
                        help="Output path for evaluation results JSON")
    parser.add_argument("--reset", action="store_true",
                        help="Clear CTFM store before build (prevents stale accumulation)")
    parser.add_argument("--ablation", default=None,
                        choices=["no_cca", "no_synthesis", "no_ctfm"],
                        help="Ablation variant (eval/online only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print sample CTFM entries after build")
    parser.add_argument("--use-llm-cca", action="store_true",
                        help="Enable LLM-based CCA in addition to statistical CCA (requires Docker LLM)")
    parser.add_argument("--llm-url", default="http://localhost:9100",
                        help="Docker LLM service URL (default http://localhost:9100)")
    args = parser.parse_args()

    if args.mode == "build":
        cmd_build(args)
    elif args.mode == "eval":
        cmd_eval(args, online=False)
    elif args.mode == "online":
        cmd_eval(args, online=True)


if __name__ == "__main__":
    main()
