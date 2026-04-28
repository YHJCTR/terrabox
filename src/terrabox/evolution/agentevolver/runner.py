"""AgentEvolver experiment runner.

Implements the three-mechanism self-evolving agent system from:
AgentEvolver: Towards Efficient Self-Evolving Agent System (arXiv 2511.10395)

Mechanisms:
  1. Self-Questioning: mine task templates, generate synthetic tasks
  2. Self-Navigating: experience pool + hybrid policy guidance
  3. Self-Attributing: ADCA-GRPO per-step credit assignment

Usage:
    # Mine templates from training data
    python -m terrabox.evolution.agentevolver.runner mine \\
        --train-data data/openearth/train.json \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/agentevolver \\
        --limit 500

    # Generate synthetic tasks and run agent on them
    python -m terrabox.evolution.agentevolver.runner generate \\
        --store-dir evolution_store/agentevolver \\
        --n-per-template 5

    # Evaluate with navigation guidance + credit attribution
    python -m terrabox.evolution.agentevolver.runner eval \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/agentevolver \\
        --limit 100
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
    from .self_attributing import ADCAGRPOAttributor
    from .self_navigating import ExperiencePool, HybridPolicy
    from .self_questioning import TaskGenerator
    from .trainer import AgentEvolverTrainer

    os.makedirs(store_dir, exist_ok=True)
    llm = EvolutionLLMClient()
    pool = ExperiencePool(os.path.join(store_dir, "experience_pool.jsonl"))
    task_gen = TaskGenerator(llm_client=llm)
    policy = HybridPolicy()
    attributor = ADCAGRPOAttributor()
    trainer = AgentEvolverTrainer(pool, task_gen, policy, attributor, llm_client=llm)
    return pool, task_gen, policy, attributor, trainer


def _build_injector(pool, policy, attributor, credit_experiences=None):
    from .prompt_injector import AgentEvolverPromptInjector
    return AgentEvolverPromptInjector(pool, policy, attributor, credit_experiences)


def _run_agent_on_task(task: dict, system_prompt: str) -> dict:
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
        logger.warning(f"Agent failed: {e}")
        return {"tools_called": [], "final_answer": "", "expected_tools": expected_tools}


def _infer_task_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in ["change", "compare", "between", "before", "after"]):
        return "change_detection"
    if any(kw in q for kw in ["nearest", "route", "closest", "station"]):
        return "poi_routing"
    if any(kw in q for kw in ["detect", "segment", "count"]):
        return "segmentation"
    if any(kw in q for kw in ["ndvi", "ndbi", "index", "calculate"]):
        return "index_calculation"
    return "general_qa"


def cmd_mine(args):
    from ..shared.data_loader import make_loader

    loader = make_loader(args.train_data, args.eval_data)
    try:
        import tqdm as _tqdm_mod
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        _iter = _tqdm_mod.tqdm(
            loader.iter_train(limit=args.limit),
            desc="[agentevolver build]", unit="traj", ncols=90,
            position=tqdm_pos, leave=True, total=args.limit,
        )
    except ImportError:
        _iter = loader.iter_train(limit=args.limit)
    trajectories = list(_iter)
    pool, task_gen, _, _, trainer = _build_pipeline(args.store_dir)

    if getattr(args, "reset", False):
        pool.clear()
        logger.info("Experience pool cleared (--reset)")

    n_templates = trainer.mine_round(trajectories)
    logger.info(f"Mining complete: {n_templates} templates, pool size={pool.size()}")


def cmd_generate(args):
    pool, task_gen, policy, attributor, trainer = _build_pipeline(args.store_dir)
    generated = trainer.generate_round(n_tasks_per_template=args.n_per_template)
    logger.info(f"Generated {len(generated)} tasks")

    # Save generated tasks for inspection
    out_path = os.path.join(args.store_dir, "generated_tasks.json")
    with open(out_path, "w") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


def _predict_tools_offline_agentevolver(pool, question: str, task_type: str) -> list[str]:
    """Predict tools for a query using experience pool retrieval (no agent needed).

    Falls back to high-reward episodes across all task types if task_type-specific
    lookup returns nothing (common when pool uses disaster-specific task types).
    """
    try:
        similar = pool.get_similar(question, task_type, top_k=3)
        if not similar:
            # Fallback: use highest-reward episodes regardless of task type
            similar = pool.get_high_reward(task_type, top_k=5)
        tools: list[str] = []
        for ep in similar:
            # pool stores tools under "tool_sequence" key
            for t in ep.get("tool_sequence", ep.get("tools_called", [])):
                if t not in tools:
                    tools.append(t)
        return tools[:8]
    except Exception:
        return []


def cmd_eval(args):
    from ..shared.data_loader import make_loader
    from ..shared.evaluator import ToolMatchEvaluator

    loader = make_loader(args.train_data, args.eval_data)
    eval_cases = loader.load_eval_cases()
    if args.limit:
        eval_cases = eval_cases[:args.limit]

    # Add task_type to cases
    for case in eval_cases:
        case["task_type"] = _infer_task_type(case.get("question", ""))

    pool, task_gen, policy, attributor, trainer = _build_pipeline(args.store_dir)
    injector = _build_injector(pool, policy, attributor)

    evaluator = ToolMatchEvaluator()
    logger.info(f"Evaluating {len(eval_cases)} cases (pool size={pool.size()})")

    # Offline eval: predict tools using experience pool retrieval (no agent)
    from ..shared.trajectory import Trajectory
    results = []
    for i, case in enumerate(eval_cases):
        task_type = case.get("task_type", "unknown")
        tools_called = _predict_tools_offline_agentevolver(pool, case.get("question", ""), task_type)
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
        results.append(evaluator.evaluate(traj))

    metrics = evaluator.aggregate(results)
    logger.info("\n=== AgentEvolver Evaluation Results ===")
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
    parser = argparse.ArgumentParser(description="AgentEvolver: three-mechanism self-evolving agent")
    parser.add_argument("mode", choices=["mine", "generate", "eval"])
    parser.add_argument("--train-data", default="data/openearth/train.json")
    parser.add_argument("--eval-data", default="data/openearth/eval.jsonl")
    parser.add_argument("--store-dir", default="evolution_store/agentevolver")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-per-template", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reset", action="store_true",
                        help="Clear experience pool before mining (prevents accumulation on re-run)")
    args = parser.parse_args()

    if args.mode == "mine":
        cmd_mine(args)
    elif args.mode == "generate":
        cmd_generate(args)
    elif args.mode == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
