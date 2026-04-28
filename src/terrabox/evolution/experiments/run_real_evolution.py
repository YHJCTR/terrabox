#!/usr/bin/env python3
"""run_real_evolution.py — Real-data + real-vLLM self-evolution experiment runner.

Runs build (offline, no vLLM) and/or eval (real vLLM) phases for supported
evolution methods, including causalpolicyevo.
Use --method all to run all methods and print a comparison table.

Usage (from repo root):
    PYTHONPATH=src python src/terrabox/evolution/experiments/run_real_evolution.py \\
        --method causalevo --phase both --build-limit 5000 --eval-limit 50

    PYTHONPATH=src python src/terrabox/evolution/experiments/run_real_evolution.py \\
        --method all --phase both --build-limit 5000 --eval-limit 100 --reset
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

# Ensure src/ is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Early environment check ───────────────────────────────────────────────────
# This script requires the same Python env as the agent (langchain, langgraph, etc.)
# Run with: /home/yuhongjie/miniconda3/envs/unsloth/bin/python  OR  conda activate unsloth
try:
    import langchain_core  # noqa: F401
except ModuleNotFoundError:
    print(
        "\n[ERROR] langchain_core not found in current Python environment.\n"
        "  This script requires the agent's conda environment.\n"
        "  Please run with:\n\n"
        "    conda activate unsloth\n"
        "    PYTHONPATH=src python src/terrabox/evolution/experiments/run_real_evolution.py ...\n\n"
        "  Or use the full Python path:\n\n"
        "    PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python "
        "src/terrabox/evolution/experiments/run_real_evolution.py ...\n",
        file=sys.stderr,
    )
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_real_evolution")

VALID_METHODS = ("baseline", "causalevo", "memrl", "skillrl", "agentevolver", "evoskill", "causalpolicyevo")
METHOD_LABELS = {
    "baseline":     "Baseline (no evo)   ",
    "causalevo":    "CausalEvo (ours) ★  ",
    "memrl":        "MemRL               ",
    "skillrl":      "SkillRL             ",
    "agentevolver": "AgentEvolver        ",
    "evoskill":     "EvoSkill            ",
    "causalpolicyevo": "CausalPolicyEvo    ",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────

def check_vllm_health(timeout: int = 60) -> bool:
    """Verify LLM connectivity.

    For local LLM (Docker/subprocess): calls get_llm() which triggers start_service().
    start_service() already blocks until the service is healthy (up to 600 s), so no
    additional polling is needed — we trust it and return True immediately on success.

    For remote API: uses requests.get (proxy-aware bypass) to poll the health endpoint
    every 5 s until it responds or timeout expires.

    NOTE: urllib.request.urlopen is intentionally avoided because it routes through
    system HTTP_PROXY env vars, causing spurious failures for localhost connections
    on machines with a proxy configured.
    """
    import time
    import requests as _req

    try:
        from terrabox.agent.config import load_config
        from terrabox.agent.llm import get_llm

        cfg = load_config()
        logger.info("Starting/connecting to LLM service (may take a moment if Docker is loading model)...")
        get_llm(cfg)   # triggers start_service() which blocks until healthy (local LLM)

        if cfg.use_local_llm:
            # start_service() already verified the service is ready — no need to re-poll.
            base_url = f"http://{cfg.local_llm_host}:{cfg.local_llm_port}"
            logger.info(f"LLM service ready at {base_url}")
            return True

        # Remote API: start_service() was not called; verify connectivity manually.
        base_url = cfg.remote_llm_api_base.rstrip("/v1").rstrip("/")
        deadline = time.time() + timeout
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            for path in ("/v1/models", "/health"):
                url = base_url.rstrip("/") + path
                try:
                    resp = _req.get(url, timeout=10, proxies={"http": None, "https": None})
                    if resp.status_code < 500:
                        logger.info(f"LLM service ready at {base_url} (attempt {attempt})")
                        return True
                except Exception:
                    continue
            elapsed = time.time() - (deadline - timeout)
            logger.info(f"  LLM not ready yet (attempt {attempt}, {elapsed:.0f}s elapsed), retrying in 5s...")
            time.sleep(5)

        print(f"\n[ERROR] Remote LLM service not ready after {timeout}s at {base_url}")
        print("  You can still run --phase build without vLLM.\n")
        return False

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize LLM: {e}")
        print("  Check agent_config.yaml (use_local_llm, local_llm_port, etc.).")
        print("  You can still run --phase build without vLLM.\n")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Build helpers (call each runner's cmd_* directly)
# ─────────────────────────────────────────────────────────────────────────────

def _make_build_args(args: argparse.Namespace, store_subdir: str) -> types.SimpleNamespace:
    """Create a Namespace that satisfies each runner's cmd_* argument contract."""
    ns = types.SimpleNamespace(
        train_data=args.train_data,
        eval_data=args.eval_data,
        store_dir=store_subdir,
        limit=args.build_limit,
        top_k=args.top_k,
        # causalevo-specific
        top_k_keywords=10,
        top_k_downstream=5,
        min_edge_count=3,
        min_anti_support=3,
        max_anti_f1=0.3,
        verbose=False,
        ablation=None,
        # memrl-specific
        memory_db=os.path.join(store_subdir, "episodic_memory.db"),
        # evoskill-specific
        n_episodes=min(args.build_limit, 200),
        reset=args.reset,
    )
    return ns


def _merge_skillrl_chunks_if_needed(store_dir: str, skillrl_store: str) -> None:
    """Merge skillrl_chunk_* dirs into skillrl/ if the main bank is empty.

    Called before eval so parallel-build chunk results are picked up automatically.
    """
    import glob as _glob

    general_path = os.path.join(skillrl_store, "general_skills.json")
    try:
        with open(general_path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("skills"):
            return  # already populated
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    chunk_dirs = sorted(_glob.glob(os.path.join(store_dir, "skillrl_chunk_*")))
    if not chunk_dirs:
        return

    logger.info(f"[skillrl] Merging {len(chunk_dirs)} chunk dir(s) into {skillrl_store} ...")
    os.makedirs(skillrl_store, exist_ok=True)

    for filename in ("general_skills.json", "task_skills.json", "mistakes.json"):
        merged: list[dict] = []
        seen: set[str] = set()
        for chunk_dir in chunk_dirs:
            path = os.path.join(chunk_dir, filename)
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            for skill in data.get("skills", []):
                key = skill.get("_content_hash") or skill.get("id", "")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                merged.append(skill)

        final_path = os.path.join(skillrl_store, filename)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(
                {"skills": merged, "metadata": {"merged_chunks": len(chunk_dirs)}},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"[skillrl] {filename}: {len(merged)} skills merged")


def run_build(method: str, args: argparse.Namespace) -> None:
    store_subdir = os.path.join(args.store_dir, method)
    os.makedirs(store_subdir, exist_ok=True)
    bargs = _make_build_args(args, store_subdir)

    try:
        if method == "baseline":
            logger.info("Baseline: no build phase needed.")
            return

        elif method == "causalevo":
            from terrabox.evolution.causalevo.runner import cmd_build
            logger.info(f"[causalevo] Building CTFM from {args.build_limit} train trajectories...")
            cmd_build(bargs)

        elif method == "memrl":
            from terrabox.evolution.memrl.runner import cmd_populate
            logger.info(f"[memrl] Populating episodic memory from {args.build_limit} train trajectories...")
            cmd_populate(bargs)

        elif method == "skillrl":
            from terrabox.evolution.skillrl.runner import cmd_distill
            logger.info(f"[skillrl] Distilling skills from {args.build_limit} train trajectories...")
            cmd_distill(bargs)

        elif method == "agentevolver":
            from terrabox.evolution.agentevolver.runner import cmd_mine
            logger.info(f"[agentevolver] Mining experience pool from {args.build_limit} train trajectories...")
            cmd_mine(bargs)

        elif method == "evoskill":
            from terrabox.evolution.evoskill.runner import cmd_discover
            logger.info(f"[evoskill] Running skill discovery on {bargs.n_episodes} train episodes...")
            cmd_discover(bargs)

        elif method == "causalpolicyevo":
            from terrabox.evolution.causalpolicyevo.runner import cmd_build
            logger.info(f"[causalpolicyevo] Building policy state from {args.build_limit} train trajectories...")
            cmd_build(bargs)

    except Exception as e:
        logger.error(f"Build failed for method={method}: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Eval core
# ─────────────────────────────────────────────────────────────────────────────

def _run_agent_one_task(case: dict, system_prompt: str) -> dict:
    """Execute agent on a single eval case. Raises on error (caller handles timeout)."""
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from terrabox.agent.config import load_config
    from terrabox.agent.llm import get_llm
    from terrabox.agent.tools import build_langchain_tools
    from terrabox.evolution.shared.mock_user import MockUser

    question = case.get("question", "")
    images = case.get("images", [])
    expected_tools = case.get("expected_tools", [])

    from terrabox.extensions import load_builtin_toolkits
    load_builtin_toolkits()

    config = load_config()
    llm = get_llm(config)
    user = MockUser()
    tools = build_langchain_tools(user)

    content = question
    if images:
        content += f"\n\n[Images: {', '.join(images)}]"
    if case.get("data_dir"):
        content += f"\n\n[Data directory: {case['data_dir']}]"
    if case.get("data_files"):
        # Show first 5 file paths so agent knows what's available
        shown = case["data_files"][:5]
        more = len(case["data_files"]) - len(shown)
        content += f"\n[Data files: {', '.join(shown)}"
        if more > 0:
            content += f" ... and {more} more"
        content += "]"

    agent = create_react_agent(llm, tools, prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    tools_called: list[str] = []
    for msg in result.get("messages", []):
        for tc in getattr(msg, "tool_calls", []):
            name = tc.get("name", "").replace("__", ".")
            if name:
                tools_called.append(name)

    final_answer = ""
    msgs = result.get("messages", [])
    if msgs and hasattr(msgs[-1], "content"):
        final_answer = str(msgs[-1].content)

    return {
        "tools_called": tools_called,
        "final_answer": final_answer,
        "expected_tools": expected_tools,
    }


_MOCK_EVAL_SYSTEM = """\
You are a geospatial AI assistant planning tool usage.

Given a user question, output ONLY a JSON array of tool slugs you would call \
to solve the task, in the order you would call them.

Available tool slugs (use exact names):
{tool_list}

Rules:
- Output ONLY valid JSON, e.g. ["tool_a", "tool_b"]
- Use the exact slug names from the list above
- Order matters: list the tools in the sequence you would call them
- Do NOT include explanations, just the JSON array
"""


def _build_mock_tool_list() -> str:
    """Build sorted tool slug list from registry (called once, cached in module)."""
    from terrabox.core.registry import registry
    from terrabox.extensions import load_builtin_toolkits
    load_builtin_toolkits()
    tool_slugs: list[str] = []
    for tk in registry.list_toolkits():
        for tool in registry.list_tools(toolkit=tk.name):
            tool_slugs.append(tool.slug)
    return "\n".join(f"- {s}" for s in sorted(tool_slugs))


_MOCK_TOOL_LIST: str = ""          # populated on first call
_MOCK_LLM = None                   # shared LLM instance across tasks


def _run_agent_mock(case: dict, system_prompt: str) -> dict:
    """Single-shot eval: LLM outputs ordered tool list, no tool execution.

    One LLM call per task (no ReAct loop), so eval is ~10x faster than
    the full agent.  The augmented system_prompt is prepended before the
    planning instruction so evolution knowledge is still injected.
    """
    import re
    from langchain_core.messages import HumanMessage, SystemMessage

    global _MOCK_TOOL_LIST, _MOCK_LLM

    if not _MOCK_TOOL_LIST:
        _MOCK_TOOL_LIST = _build_mock_tool_list()

    if _MOCK_LLM is None:
        from terrabox.agent.config import load_config
        from terrabox.agent.llm import get_llm
        _MOCK_LLM = get_llm(load_config())

    llm = _MOCK_LLM
    tool_list_str = _MOCK_TOOL_LIST

    planning_system = system_prompt.rstrip() + "\n\n" + _MOCK_EVAL_SYSTEM.format(tool_list=tool_list_str)

    question = case.get("question", "")
    images = case.get("images", [])
    content = question
    if images:
        content += f"\n[Images: {', '.join(images)}]"

    messages = [SystemMessage(content=planning_system), HumanMessage(content=content)]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # Parse JSON array from response (robust: allow ```json ... ``` fences)
    tools_called: list[str] = []
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`").strip()
        # Find first [...] in the response
        m = re.search(r"\[.*?\]", clean, re.DOTALL)
        if m:
            tools_called = json.loads(m.group())
        else:
            tools_called = json.loads(clean)
    except Exception:
        # Fallback: find quoted strings that look like slugs
        tools_called = re.findall(r'"([a-z_]+\.[a-z_]+)"', raw)

    return {
        "tools_called": [t.replace("__", ".") for t in tools_called if isinstance(t, str)],
        "final_answer": raw,
        "expected_tools": case.get("expected_tools", []),
    }


def run_eval_trajectory(method: str, args: argparse.Namespace) -> dict:
    """Fast eval: parse test.json trajectories directly, no agent execution.

    tools_called is read from the pre-existing test trajectories and compared
    against expected_tools from eval.jsonl.  This is useful as a fast baseline
    to verify data loading and metric computation without waiting for LLM calls.
    """
    from terrabox.evolution.shared.data_loader import OpenEarthLoader
    from terrabox.evolution.shared.evaluator import ToolMatchEvaluator

    loader = OpenEarthLoader(args.train_data, args.eval_data)
    trajectories = loader.load_test_trajectories(limit=args.eval_limit or None)

    logger.info(f"[{method}] Trajectory-eval on {len(trajectories)} test.json records (no agent)...")

    evaluator = ToolMatchEvaluator()
    results = evaluator.batch_evaluate(trajectories)
    metrics = evaluator.aggregate(results)
    metrics["errors"] = 0

    os.makedirs(os.path.join(args.store_dir, "results"), exist_ok=True)
    output_path = args.output or os.path.join(args.store_dir, "results", f"{method}.trajectory_eval.json")
    with open(output_path, "w") as f:
        json.dump({
            "method": method,
            "eval_mode": "trajectory",
            "metrics": metrics,
            "eval_limit": len(trajectories),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    logger.info(
        f"[{method}]  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  EM={metrics['exact_match']:.4f}  n={metrics['n']}"
    )
    return metrics


def run_eval(method: str, args: argparse.Namespace) -> dict:
    """Evaluate a single method; returns aggregated metrics dict."""
    from terrabox.evolution.shared.data_loader import OpenEarthLoader
    from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
    from terrabox.evolution.shared.trajectory import Trajectory

    # Fast path: read pre-existing trajectories from test.json, skip agent
    if getattr(args, "trajectory_eval", False):
        return run_eval_trajectory(method, args)

    # Mock eval: single LLM call per task, no tool execution
    task_runner = _run_agent_mock if getattr(args, "mock_eval", False) else _run_agent_one_task

    # Load eval cases (supports --combined to include earthbench)
    loader = OpenEarthLoader(args.train_data, args.eval_data)
    if getattr(args, "combined", False):
        eval_cases = loader.load_combined_eval_cases()
    else:
        eval_cases = loader.load_eval_cases()
    if args.eval_limit:
        eval_cases = eval_cases[: args.eval_limit]

    logger.info(f"[{method}] Evaluating {len(eval_cases)} cases (timeout={args.timeout}s each)...")

    # Build augmenter
    store_subdir = os.path.join(args.store_dir, method)
    if method == "skillrl":
        _merge_skillrl_chunks_if_needed(args.store_dir, store_subdir)
    if method == "baseline":
        from terrabox.evolution.shared.prompt_builder import PromptAugmenter
        base_prompt = PromptAugmenter.BASE_SYSTEM
        def get_prompt(_q, **_kw):
            return base_prompt
    else:
        from terrabox.evolution import get_prompt_augmenter
        augmenter_kwargs: dict = {"store_dir": store_subdir, "top_k": args.top_k}
        if method == "memrl":
            augmenter_kwargs["memory_db"] = os.path.join(store_subdir, "episodic_memory.db")
        if method == "causalevo" and args.ablation:
            augmenter_kwargs["ablation"] = args.ablation
        try:
            augmenter = get_prompt_augmenter(method, **augmenter_kwargs)
        except Exception as e:
            logger.warning(f"[{method}] Failed to load augmenter: {e} — will use base prompt")
            from terrabox.evolution.shared.prompt_builder import PromptAugmenter
            base_prompt = PromptAugmenter.BASE_SYSTEM
            def get_prompt(q, **kw):
                return base_prompt
        else:
            def get_prompt(q, images=None, **kw):
                return augmenter.augment(q, images=images or [])

    evaluator = ToolMatchEvaluator()
    results = []
    errors = 0

    # Partial results path
    os.makedirs(os.path.join(args.store_dir, "results"), exist_ok=True)
    partial_path = os.path.join(args.store_dir, "results", f"{method}.partial.json")
    output_path = args.output or os.path.join(args.store_dir, "results", f"{method}.json")

    try:
        tqdm_mod = __import__("tqdm")
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        progress = tqdm_mod.tqdm(
            eval_cases, desc=f"[{method}]", unit="task", ncols=90,
            position=tqdm_pos, leave=True,
        )
    except ImportError:
        progress = eval_cases

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, case in enumerate(progress):
            question = case.get("question", "")
            images = case.get("images", [])
            system_prompt = get_prompt(question, images=images)

            run: dict = {}
            try:
                future = executor.submit(task_runner, case, system_prompt)
                run = future.result(timeout=args.timeout)
            except FuturesTimeoutError:
                logger.warning(f"[{method}] Task {i} timed out after {args.timeout}s")
                run = {"tools_called": [], "final_answer": "", "expected_tools": case.get("expected_tools", []),
                       "error": "timeout"}
                errors += 1
            except Exception as e:
                logger.warning(f"[{method}] Task {i} failed: {e}")
                run = {"tools_called": [], "final_answer": "", "expected_tools": case.get("expected_tools", []),
                       "error": str(e)}
                errors += 1

            traj = Trajectory(
                task_id=case.get("id", str(i)),
                question=question,
                images=images,
                turns=[],
                tools_called=run.get("tools_called", []),
                expected_tools=run.get("expected_tools", []),
                final_answer=run.get("final_answer", ""),
                success=False,
                source=case.get("source", "openearth"),
            )
            episode = evaluator.evaluate(traj)
            results.append(episode)

            # Online update if supported
            if method != "baseline" and args.online:
                try:
                    augmenter.record_outcome(
                        query=question,
                        tools_called=run.get("tools_called", []),
                        reward=episode.reward,
                    )
                except Exception:
                    pass

            # Save partial results every 10 tasks
            if (i + 1) % 10 == 0:
                partial_metrics = evaluator.aggregate(results)
                partial_data = {
                    "method": method,
                    "completed": i + 1,
                    "total": len(eval_cases),
                    "errors": errors,
                    "metrics": partial_metrics,
                }
                with open(partial_path, "w") as f:
                    json.dump(partial_data, f, indent=2)

    metrics = evaluator.aggregate(results)
    metrics["errors"] = errors

    output_data = {
        "method": method,
        "metrics": metrics,
        "build_limit": args.build_limit,
        "eval_limit": len(eval_cases),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        f"[{method}]  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  EM={metrics['exact_match']:.4f}  "
        f"errors={errors}/{len(eval_cases)}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics: dict[str, dict]) -> None:
    header = f"{'Method':<24}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'EM':>6}  {'Errors':>6}  {'Delta F1':>9}"
    print()
    print("=" * 80)
    print("SELF-EVOLUTION COMPARISON RESULTS")
    print("=" * 80)
    print(header)
    print("-" * 80)
    baseline_f1 = all_metrics.get("baseline", {}).get("f1")
    for method in ("baseline",) + tuple(m for m in VALID_METHODS if m != "baseline"):
        m_data = all_metrics.get(method)
        if m_data is None:
            continue
        label = METHOD_LABELS.get(method, method)
        p = m_data.get("precision", 0.0)
        r = m_data.get("recall", 0.0)
        f1 = m_data.get("f1", 0.0)
        em = m_data.get("exact_match", 0.0)
        err = m_data.get("errors", 0)
        delta = ""
        if baseline_f1 is not None and method != "baseline":
            diff = f1 - baseline_f1
            delta = f"{diff:+.4f}"
        print(f"{label:<24}  {p:>6.4f}  {r:>6.4f}  {f1:>6.4f}  {em:>6.4f}  {err:>6}  {delta:>9}")
    print("=" * 80)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-data + real-vLLM self-evolution experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method",
        default="causalevo",
        help=(
            "Which method(s) to run. "
            "Choices: " + ", ".join(VALID_METHODS) + ", all"
        ),
    )
    p.add_argument(
        "--phase",
        choices=("build", "eval", "both"),
        default="both",
        help="build=offline knowledge building (no vLLM); eval=online eval (vLLM required); both=both",
    )
    p.add_argument("--train-data", default="data/openearth/train.json")
    p.add_argument("--eval-data", default="data/openearth/eval.jsonl")
    p.add_argument("--combined", action="store_true",
                   help="Combine openearth + earthbench eval sets (uses data/earthbench/eval.jsonl alongside --eval-data)")
    p.add_argument("--store-dir", default="evolution_store", help="Root directory for all stores")
    p.add_argument("--build-limit", type=int, default=5000, help="Max train trajectories for build phase")
    p.add_argument("--eval-limit", type=int, default=50, help="Max eval cases")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ablation", default=None,
                   choices=(None, "no_cca", "no_synthesis", "no_ctfm"),
                   help="CausalEvo ablation mode (only used when method=causalevo)")
    p.add_argument("--online", action="store_true",
                   help="Update knowledge base after each eval task (online learning)")
    p.add_argument("--timeout", type=int, default=120, help="Per-task agent timeout in seconds")
    p.add_argument("--output", default=None, help="Output JSON path (default: store-dir/results/{method}.json)")
    p.add_argument("--reset", action="store_true", help="Clear existing knowledge base before build")
    p.add_argument(
        "--mock-eval", action="store_true",
        help=(
            "Fast eval: single LLM call per task asking it to list tools in order (no tool execution). "
            "~10x faster than full agent eval. LLM outputs a JSON array of slugs it would call."
        ),
    )
    p.add_argument(
        "--trajectory-eval", action="store_true",
        help=(
            "Fast eval: parse test.json trajectories directly instead of running the agent. "
            "Reads tools_called from pre-existing test trajectories and compares against "
            "expected_tools from eval.jsonl.  No LLM calls — completes in seconds."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve methods list
    if args.method.lower() == "all":
        methods = list(VALID_METHODS)
    elif args.method.lower() in VALID_METHODS:
        methods = [args.method.lower()]
    else:
        print(f"[ERROR] Unknown method {args.method!r}. Choose from: {', '.join(VALID_METHODS)}, all")
        sys.exit(1)

    # vLLM health check for eval phase
    if args.phase in ("eval", "both"):
        vllm_ok = check_vllm_health()
        if not vllm_ok:
            if args.phase == "eval":
                sys.exit(1)
            else:
                print("[WARNING] vLLM not available — running build phase only.\n")
                args.phase = "build"

    print()
    print("=" * 80)
    print("Self-Evolution Real-Data Experiment")
    print(f"  Methods      : {', '.join(methods)}")
    print(f"  Phase        : {args.phase}")
    print(f"  Train data   : {args.train_data}")
    print(f"  Eval data    : {args.eval_data}")
    print(f"  Build limit  : {args.build_limit}")
    print(f"  Eval limit   : {args.eval_limit}")
    print(f"  Store dir    : {args.store_dir}")
    print(f"  Online update: {args.online}")
    print("=" * 80)

    all_metrics: dict[str, dict] = {}

    for method in methods:
        print(f"\n{'─'*40}")
        print(f"Method: {method.upper()}")
        print(f"{'─'*40}")

        # ── Build ──
        if args.phase in ("build", "both"):
            t0 = time.time()
            try:
                run_build(method, args)
                logger.info(f"[{method}] Build done in {time.time()-t0:.1f}s")
            except Exception as e:
                logger.error(f"[{method}] Build FAILED: {e}")
                if args.phase == "build":
                    continue

        # ── Eval ──
        if args.phase in ("eval", "both"):
            t0 = time.time()
            try:
                metrics = run_eval(method, args)
                all_metrics[method] = metrics
                logger.info(f"[{method}] Eval done in {time.time()-t0:.1f}s")
            except Exception as e:
                logger.error(f"[{method}] Eval FAILED: {e}", exc_info=args.verbose)

    # ── Comparison table ──
    if all_metrics:
        print_comparison_table(all_metrics)

        # Save combined results
        combined_path = os.path.join(args.store_dir, "results", "comparison.json")
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
