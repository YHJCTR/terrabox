"""MemRL experiment runner.

Implements the non-parametric runtime RL on episodic memory approach
from: MemRL: Self-Evolving Agents via Runtime Reinforcement Learning
      on Episodic Memory (arXiv 2601.03192)

Usage:
    # Populate memory from training data (offline)
    python -m terrabox.evolution.memrl.runner populate \\
        --train-data data/openearth/train.json \\
        --eval-data data/openearth/eval.jsonl \\
        --memory-db evolution_store/memrl/episodic_memory.db \\
        --limit 1000

    # Evaluate agent with memory injection (offline)
    python -m terrabox.evolution.memrl.runner eval \\
        --eval-data data/openearth/eval.jsonl \\
        --memory-db evolution_store/memrl/episodic_memory.db \\
        --top-k 5

    # Online mode: evaluate + update Q-values after each episode
    python -m terrabox.evolution.memrl.runner online \\
        --eval-data data/openearth/eval.jsonl \\
        --memory-db evolution_store/memrl/episodic_memory.db \\
        --top-k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_injector(memory_db: str, top_k: int, llm_client=None):
    """Construct the full MemRL injection pipeline."""
    from .bellman_updater import BellmanUpdater
    from .episodic_memory import EpisodicMemory
    from .intent_parser import IntentParser
    from .prompt_injector import MemRLPromptInjector
    from .retriever import TwoPhaseRetriever

    os.makedirs(os.path.dirname(os.path.abspath(memory_db)), exist_ok=True)
    intent_parser = IntentParser()
    memory = EpisodicMemory(memory_db, llm_client=llm_client)
    retriever = TwoPhaseRetriever(intent_parser)
    bellman = BellmanUpdater()
    injector = MemRLPromptInjector(memory, retriever, intent_parser, bellman, top_k=top_k)
    return injector, memory


def _run_agent_on_task(task: dict, system_prompt: str) -> dict:
    """Run the agent on a single eval task and return result dict.

    Attempts to use the real agent infrastructure. Falls back to a stub
    that returns an empty tools_called list if imports fail (for offline testing).
    """
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

        # Add image context to question if images present
        content = question
        if images:
            content += f"\n\n[Images: {', '.join(images)}]"

        agent = create_react_agent(llm, tools, prompt=system_prompt)
        result = agent.invoke({"messages": [HumanMessage(content=content)]})

        # Extract tool calls from message history
        tools_called = []
        for msg in result.get("messages", []):
            for tc in getattr(msg, "tool_calls", []):
                tool_name = tc.get("name", "").replace("__", ".")
                if tool_name:
                    tools_called.append(tool_name)

        final_answer = ""
        messages = result.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            final_answer = messages[-1].content

        return {
            "tools_called": tools_called,
            "final_answer": final_answer,
            "expected_tools": expected_tools,
        }

    except Exception as e:
        logger.warning(f"Agent execution failed: {e}")
        return {"tools_called": [], "final_answer": "", "expected_tools": expected_tools}


def cmd_populate(args):
    """Load training trajectories into episodic memory.

    Safe to re-run: each trajectory is keyed by task_id and skipped if already stored.
    Use --reset to clear and start fresh.
    """
    from ..shared.data_loader import OpenEarthLoader
    from ..shared.evaluator import ToolMatchEvaluator
    from .episodic_memory import EpisodicMemory
    from .intent_parser import IntentParser

    logger.info(f"Populating episodic memory from {args.train_data}")
    os.makedirs(os.path.dirname(os.path.abspath(args.memory_db)), exist_ok=True)

    memory = EpisodicMemory(args.memory_db)

    if getattr(args, "reset", False):
        memory.clear()
        logger.info("Memory cleared (--reset)")

    loader = OpenEarthLoader(args.train_data, args.eval_data)
    intent_parser = IntentParser()
    evaluator = ToolMatchEvaluator()

    added = skipped = 0
    try:
        import tqdm as _tqdm_mod
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        _iter = _tqdm_mod.tqdm(
            loader.iter_train(limit=args.limit),
            desc="[memrl build]", unit="traj", ncols=90,
            position=tqdm_pos, leave=True, total=args.limit,
        )
    except ImportError:
        _iter = loader.iter_train(limit=args.limit)
    for traj in _iter:
        result = evaluator.evaluate(traj)
        intent = intent_parser.parse(traj.question, traj.images)
        memory_id = memory.add_memory(
            traj, intent=intent, initial_utility=float(result.reward), skip_if_exists=True
        )
        if memory_id is None:
            skipped += 1
        else:
            added += 1

    logger.info(f"Done. added={added} skipped={skipped} total_in_db={memory.count()}")


def cmd_eval(args, online: bool = False):
    """Evaluate agent with memory injection; optionally update Q-values online."""
    from ..shared.data_loader import OpenEarthLoader
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory, Turn

    loader = OpenEarthLoader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    injector, memory = _build_injector(args.memory_db, args.top_k)
    evaluator = ToolMatchEvaluator()

    logger.info(f"Evaluating on {len(eval_cases)} cases (memory size: {memory.count()})")
    logger.info(f"Mode: {'online (Q-value updates)' if online else 'offline (no updates)'}")

    results = []
    t0 = time.time()

    for i, case in enumerate(eval_cases):
        system_prompt = injector.augment(
            case["question"], images=case.get("images", [])
        )
        run = _run_agent_on_task(case, system_prompt)

        traj = Trajectory(
            task_id=case.get("id", str(i)),
            question=case["question"],
            images=case.get("images", []),
            turns=[],
            tools_called=run["tools_called"],
            expected_tools=run["expected_tools"],
            final_answer=run["final_answer"],
            success=False,
            source=case.get("source", "openearth"),
        )
        episode = evaluator.evaluate(traj)
        results.append(episode)

        if online:
            # Update Q-values based on observed reward
            injector.record_outcome(episode.reward)
            # Add this episode to memory
            memory.add_memory(
                traj,
                initial_utility=episode.reward,
            )

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg_f1 = sum(r.tool_f1 for r in results) / len(results)
            logger.info(
                f"  [{i+1}/{len(eval_cases)}] avg_f1={avg_f1:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

    metrics = evaluator.aggregate(results)
    logger.info("\n=== MemRL Evaluation Results ===")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  N cases:     {metrics['n']}")

    # Save results
    out_path = args.output or "evolution_store/memrl/eval_results.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "mode": "online" if online else "offline"}, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MemRL: episodic memory evolution for geospatial agents"
    )
    parser.add_argument("mode", choices=["populate", "eval", "online"],
                        help="Operation mode")
    parser.add_argument("--train-data", default="data/openearth/train.json",
                        help="Path to OpenEarth train.json")
    parser.add_argument("--eval-data", default="data/openearth/eval.jsonl",
                        help="Path to eval.jsonl")
    parser.add_argument("--memory-db", default="evolution_store/memrl/episodic_memory.db",
                        help="Path to episodic memory SQLite database")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of memories to inject per query")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to process")
    parser.add_argument("--output", default=None,
                        help="Path for evaluation results JSON")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing memory before populate (prevents accumulation on re-run)")
    args = parser.parse_args()

    if args.mode == "populate":
        cmd_populate(args)
    elif args.mode == "eval":
        cmd_eval(args, online=False)
    elif args.mode == "online":
        cmd_eval(args, online=True)


if __name__ == "__main__":
    main()
