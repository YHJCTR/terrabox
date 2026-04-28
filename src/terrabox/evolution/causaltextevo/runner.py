"""CausalTextEvo experiment runner.

CausalTextEvo: Causal-Aware Textual Gradient Optimization for Agent Self-Evolution

Run modes
─────────
build      Initialize knowledge state θ from training trajectories.
           Computes CCA scores, mines SeqGraph patterns, extracts keywords.

optimize   Run textual gradient optimization loop on θ.
           Iteratively refines keywords, patterns, anti-patterns, and strategy
           text using LLM-generated gradients guided by CCA attribution.

eval       Evaluate agent with knowledge-state-based prompt injection.
           Offline (tool prediction) or LLM (real agent execution).

Ablation modes (--ablation, for eval / optimize)
──────────────
no_textgrad       Use initial statistical knowledge only (no gradient loop)
no_cca_guide      Don't weight gradient by CCA; treat all tools equally
no_pattern        Don't use SeqGraph patterns; keyword matching only
no_anti_pattern   Don't filter anti-patterns during composition
no_skill_text     Don't inject strategy text
static_keywords   Don't update keywords via gradient (patterns + skills only)

Quick start
───────────
    # Step 1 — build initial θ
    python -m terrabox.evolution.causaltextevo.runner build \\
        --train-data data/disaster_sft_dataset.json \\
        --store-dir  evo_res/disaster3/causaltextevo/store

    # Step 2 — optimize θ with textual gradients
    python -m terrabox.evolution.causaltextevo.runner optimize \\
        --train-data data/disaster_sft_dataset.json \\
        --eval-data  data/disaster_sft_dataset.json \\
        --store-dir  evo_res/disaster3/causaltextevo/store \\
        --max-epochs 10 --patience 3

    # Step 3 — offline eval
    python -m terrabox.evolution.causaltextevo.runner eval \\
        --eval-data  data/disaster_sft_dataset.json \\
        --store-dir  evo_res/disaster3/causaltextevo/store \\
        --output     evo_res/disaster3/causaltextevo/test/eval_offline.json

    # Ablation — no textual gradients
    python -m terrabox.evolution.causaltextevo.runner eval \\
        --eval-data  data/disaster_sft_dataset.json \\
        --store-dir  evo_res/disaster3/causaltextevo/store \\
        --ablation no_textgrad
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================= #
# Build                                                                     #
# ======================================================================= #

def cmd_build(args) -> None:
    """Build initial knowledge state θ from training data."""
    from ..shared.data_loader import make_loader
    from ..causalevo.counterfactual_credit import compute_statistical_cca
    from ..causalevo.tool_function_model import CTFMBuilder
    from ..seqgraphevo.seq_pattern_miner import SeqPatternMiner
    from ..seqgraphevo.prompt_injector import _build_seq_graph
    from .knowledge_state import KnowledgeState

    os.makedirs(args.store_dir, exist_ok=True)

    # Load trajectories
    logger.info(f"Loading training data from {args.train_data} ...")
    loader = make_loader(args.train_data, args.eval_data)
    try:
        import tqdm as _tqdm_mod
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        _iter = _tqdm_mod.tqdm(
            loader.iter_train(limit=args.limit),
            desc="[causaltextevo build]", unit="traj", ncols=90,
            position=tqdm_pos, leave=True, total=args.limit,
        )
    except ImportError:
        _iter = loader.iter_train(limit=args.limit)
    trajectories = list(_iter)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    if not trajectories:
        logger.error("No trajectories loaded — check --train-data path.")
        return

    # Step 1: Compute CCA scores
    logger.info("Computing statistical CCA scores ...")
    global_cca = compute_statistical_cca(trajectories)
    top_cca = sorted(global_cca.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top-10 CCA tools:")
    for slug, score in top_cca[:10]:
        logger.info(f"  {score:.3f}  {slug}")

    # Step 2: Extract CTFM keywords
    logger.info("Extracting per-tool keywords ...")
    builder = CTFMBuilder(
        top_k_keywords=args.top_k_keywords,
        top_k_downstream=5,
    )
    ctfm_models = builder.build(trajectories, global_cca)
    tool_keywords = {
        slug: model.precondition_keywords
        for slug, model in ctfm_models.items()
    }

    # Step 3: Mine SeqGraph patterns
    logger.info("Mining sequential patterns ...")
    miner = SeqPatternMiner()
    for traj in trajectories:
        f1 = 1.0 if traj.success else 0.0
        miner.add_trajectory(
            traj.tools_called,
            task_type=getattr(traj, "task_type", "general"),
            f1=f1,
        )
    seq_graph = _build_seq_graph(
        miner,
        min_pattern_support=args.min_pattern_support,
        max_pattern_len=4,
        min_edge_count=args.min_edge_count,
        min_anti_support=args.min_anti_support,
    )

    # Step 4: Assemble knowledge state
    tool_stats = {}
    for slug, stats in miner.get_tool_stats().items():
        tool_stats[slug] = {
            "count": stats["count"],
            "avg_f1": stats["avg_f1"],
            "task_types": stats["task_types"],
        }

    forward_edges = []
    for edge in miner.get_directed_edges(min_count=args.min_edge_count):
        forward_edges.append(edge)

    state = KnowledgeState.initialize_from_data(
        cca_scores=global_cca,
        seq_patterns=seq_graph._patterns,
        anti_patterns=seq_graph._anti_patterns,
        anti_pairs=[list(p) for p in seq_graph._anti_pairs],
        forward_edges=forward_edges,
        tool_stats=tool_stats,
        tool_keywords=tool_keywords,
        trajectory_count=miner._trajectory_count,
    )

    state_path = os.path.join(args.store_dir, "knowledge_state.json")
    state.save(state_path)

    logger.info(
        f"\n=== CausalTextEvo Build Complete ===\n"
        f"  Tools:          {len(tool_keywords)}\n"
        f"  CCA scores:     {len(global_cca)}\n"
        f"  Patterns:        {len(state.seq_patterns)}\n"
        f"  Anti-patterns:  {len(state.anti_patterns)}\n"
        f"  Anti-pairs:     {len(state.anti_pairs)}\n"
        f"  Forward edges:  {len(forward_edges)}\n"
        f"  Trajectories:   {miner._trajectory_count}\n"
        f"  Saved to:       {state_path}"
    )


# ======================================================================= #
# Optimize                                                                  #
# ======================================================================= #

def cmd_optimize(args) -> None:
    """Run textual gradient optimization loop."""
    from ..shared.data_loader import make_loader
    from ..shared.llm_client import EvolutionLLMClient
    from .knowledge_state import KnowledgeState
    from .optimizer import CausalTextOptimizer

    state_path = os.path.join(args.store_dir, "knowledge_state.json")
    if not os.path.exists(state_path):
        logger.error(f"Knowledge state not found at {state_path}. Run 'build' first.")
        return

    state = KnowledgeState.load(state_path)

    # Load eval cases
    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]
    logger.info(f"Loaded {len(eval_cases)} eval cases for optimization")

    # Initialize LLM client
    llm_url = getattr(args, "llm_url", None) or os.environ.get("EVOLUTION_LLM_URL")
    llm = EvolutionLLMClient(llm_url=llm_url)

    # Run optimization
    optimizer = CausalTextOptimizer(
        llm_client=llm,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_agreement=args.min_agreement,
        top_k=args.top_k,
    )

    ablation = getattr(args, "ablation", None)
    optimized = optimizer.optimize(state, eval_cases, ablation=ablation)

    # Save optimized state
    optimized.save(state_path)
    logger.info(f"Optimized knowledge state saved to {state_path}")


# ======================================================================= #
# Eval                                                                      #
# ======================================================================= #

def cmd_eval(args, use_llm: bool = False) -> dict:
    """Evaluate with knowledge-state-based prompt injection."""
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory
    from .knowledge_state import KnowledgeState
    from .prompt_injector import CausalTextEvoPromptInjector

    state_path = os.path.join(args.store_dir, "knowledge_state.json")
    if not os.path.exists(state_path):
        logger.error(f"Knowledge state not found at {state_path}. Run 'build' first.")
        return {}

    state = KnowledgeState.load(state_path)
    ablation = getattr(args, "ablation", None)
    injector = CausalTextEvoPromptInjector(state, top_k=args.top_k, ablation=ablation)

    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    evaluator = ToolMatchEvaluator()
    mode_label = "llm_offline" if use_llm else "offline"
    if ablation:
        mode_label += f"_{ablation}"

    logger.info(
        f"CausalTextEvo eval: {len(eval_cases)} cases | "
        f"mode={mode_label} | epoch={state.epoch}"
    )

    results = []
    all_cases_output = []
    t0 = time.time()

    for i, case in enumerate(eval_cases):
        question = case.get("question", "")
        expected = case.get("expected_tools", [])
        task_id = case.get("id", case.get("task_id", str(i)))
        task_type = _infer_task_type_from_case(task_id, question)

        if use_llm:
            system_prompt = injector.augment(question, task_type=task_type)
            run = _run_agent_on_task(case, system_prompt)
            tools_called = run["tools_called"]
        else:
            tools_called = injector.predict_tools(question, task_type)

        traj = Trajectory(
            task_id=task_id,
            question=question,
            images=case.get("images", []),
            turns=[],
            tools_called=tools_called,
            expected_tools=expected,
            final_answer="",
            success=False,
            task_type=task_type,
        )
        episode = evaluator.evaluate(traj)
        results.append(episode)

        all_cases_output.append({
            "task_id": task_id,
            "task_type": task_type,
            "tools_called": tools_called,
            "expected_tools": expected,
            "precision": episode.tool_precision,
            "recall": episode.tool_recall,
            "f1": episode.tool_f1,
        })

        if (i + 1) % 20 == 0:
            avg_f1 = sum(r.tool_f1 for r in results) / len(results)
            logger.info(
                f"  [{i + 1}/{len(eval_cases)}] avg_f1={avg_f1:.3f} "
                f"elapsed={time.time() - t0:.1f}s"
            )

    metrics = evaluator.aggregate(results)
    logger.info("\n=== CausalTextEvo Evaluation Results ===")
    logger.info(f"  Mode:        {mode_label}")
    logger.info(f"  Epoch:       {state.epoch}")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  N cases:     {metrics['n']}")

    # Save results
    out_path = args.output or os.path.join(
        args.store_dir, "test", f"eval_{mode_label}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    output = {
        "method": "causaltextevo",
        "eval_mode": mode_label,
        "timestamp": datetime.now().isoformat(),
        "epoch": state.epoch,
        "ablation": ablation,
        "metrics": metrics,
        "cases": all_cases_output,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results → {out_path}")
    return metrics


# ======================================================================= #
# Helpers                                                                   #
# ======================================================================= #

def _infer_task_type_from_case(task_id: str, question: str) -> str:
    parts = task_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    q = question.lower()
    if any(kw in q for kw in ["change", "compare", "difference", "before", "after"]):
        return "change_detection"
    if any(kw in q for kw in ["flood", "inundation"]):
        return "flood_detection"
    if any(kw in q for kw in ["fire", "burn", "wildfire"]):
        return "fire_detection"
    if any(kw in q for kw in ["earthquake", "damage"]):
        return "earthquake_assessment"
    if any(kw in q for kw in ["drought", "vegetation", "ndvi"]):
        return "drought_monitoring"
    if any(kw in q for kw in ["landslide", "slope"]):
        return "landslide_detection"
    return "general_qa"


def _run_agent_on_task(task: dict, system_prompt: str) -> dict:
    """Run agent on a task; return {tools_called, final_answer}."""
    question = task.get("question", "")
    images = task.get("images", [])
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
        return {"tools_called": tools_called, "final_answer": final}
    except Exception as e:
        logger.warning(f"Agent execution failed: {e}")
        return {"tools_called": [], "final_answer": ""}


# ======================================================================= #
# CLI                                                                       #
# ======================================================================= #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CausalTextEvo: Causal-Aware Textual Gradient Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", choices=["build", "eval", "optimize"],
                        help="Operation mode")

    # Data paths
    parser.add_argument("--train-data", default="data/disaster_sft_dataset.json")
    parser.add_argument("--eval-data", default="data/disaster_sft_dataset.json")
    parser.add_argument("--store-dir", default="evo_res/causaltextevo/store",
                        help="Directory for knowledge state store")
    parser.add_argument("--output", default=None,
                        help="Output path for eval results JSON")

    # Build params
    parser.add_argument("--top-k-keywords", type=int, default=10)
    parser.add_argument("--min-pattern-support", type=int, default=5)
    parser.add_argument("--min-edge-count", type=int, default=3)
    parser.add_argument("--min-anti-support", type=int, default=3)

    # Eval params
    parser.add_argument("--top-k", type=int, default=8,
                        help="Max tools to predict (default 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit trajectories / eval cases")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use real agent for evaluation (slow)")

    # Optimize params
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Max optimization epochs (default 10)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after N epochs without improvement (default 3)")
    parser.add_argument("--min-agreement", type=int, default=1,
                        help="Min cases agreeing for keyword update (default 1)")
    parser.add_argument("--llm-url", default=None,
                        help="Docker LLM service URL (default from env)")

    # Ablation
    parser.add_argument("--ablation", default=None,
                        choices=[
                            "no_textgrad", "no_cca_guide", "no_pattern",
                            "no_anti_pattern", "no_skill_text", "static_keywords",
                        ],
                        help="Ablation variant")

    # Misc
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing knowledge state before build")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.mode == "build":
        if getattr(args, "reset", False):
            state_path = os.path.join(args.store_dir, "knowledge_state.json")
            if os.path.exists(state_path):
                os.remove(state_path)
                logger.info(f"Cleared existing state: {state_path}")
        cmd_build(args)
    elif args.mode == "eval":
        cmd_eval(args, use_llm=getattr(args, "use_llm", False))
    elif args.mode == "optimize":
        cmd_optimize(args)


if __name__ == "__main__":
    main()
