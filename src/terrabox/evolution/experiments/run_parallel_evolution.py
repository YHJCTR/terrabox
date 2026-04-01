#!/usr/bin/env python3
"""run_parallel_evolution.py — Multi-GPU parallel self-evolution experiment runner.

Runs up to --n-gpus methods simultaneously, each on its own GPU and vLLM instance.
Inherits all arguments from run_real_evolution.py and adds:
  --n-gpus    Number of GPUs to use in parallel (default 4)
  --base-port Starting port for vLLM instances (default 9100)
  --gpu-ids   Comma-separated GPU IDs to use (default "0,1,2,3")

GPU assignment:
  GPU <base-port+0>: baseline
  GPU <base-port+1>: causalevo
  GPU <base-port+2>: memrl
  GPU <base-port+3>: skillrl
  (second batch) agentevolver, evoskill reuse GPU 0,1

Build phase parallelism:
  CPU-only methods (causalevo, memrl, agentevolver, skillrl) run in parallel subprocesses.
  LLM-required methods (evoskill only) run sequentially on GPU 0.

Usage:
    # Step 1 — build all knowledge bases (no vLLM for most methods)
    PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \\
        src/terrabox/evolution/experiments/run_parallel_evolution.py \\
        --phase build --method all --build-limit 14538 --reset --n-gpus 4

    # Step 2 — evaluate all methods in parallel (requires vLLM)
    PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \\
        src/terrabox/evolution/experiments/run_parallel_evolution.py \\
        --phase eval --method all --eval-limit 1169 --online --timeout 180 --n-gpus 4
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import types
from typing import Optional

# Ensure src/ is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import langchain_core  # noqa: F401
except ModuleNotFoundError:
    print(
        "\n[ERROR] langchain_core not found.\n"
        "  Run with: /home/yuhongjie/miniconda3/envs/unsloth/bin/python\n",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_parallel_evolution")

VALID_METHODS = ("baseline", "causalevo", "memrl", "skillrl", "agentevolver", "evoskill")
# Methods that need LLM during build phase (only evoskill uses multi-agent LLM discovery)
# skillrl build is CPU-only (BM25 distillation from trajectories, no LLM calls)
_BUILD_NEEDS_LLM = {"evoskill"}
# Methods that need LLM during eval phase (all of them do)
_EVAL_ORDER = ["baseline", "causalevo", "memrl", "skillrl", "agentevolver", "evoskill"]


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup registry — tracks child processes & Docker containers for this run
# Triggered on Ctrl+C (SIGINT) or SIGTERM.
# ─────────────────────────────────────────────────────────────────────────────

_cleanup_lock = threading.Lock()
_cleanup_started = False           # prevents re-entrant cleanup on repeated Ctrl+C
_active_procs: list = []          # multiprocessing.Process objects still running
_managed_containers: set = set()  # container names started/used by this run


def _register_proc(p) -> None:
    with _cleanup_lock:
        _active_procs.append(p)


def _unregister_proc(p) -> None:
    with _cleanup_lock:
        try:
            _active_procs.remove(p)
        except ValueError:
            pass


def _register_container(name: str) -> None:
    with _cleanup_lock:
        _managed_containers.add(name)


def _do_cleanup() -> None:
    """Terminate child processes and stop all Docker containers from this run."""
    with _cleanup_lock:
        procs = list(_active_procs)
        containers = set(_managed_containers)

    if procs:
        logger.info(f"[cleanup] Terminating {len(procs)} child process(es)...")
        for p in procs:
            try:
                if p.is_alive():
                    # Kill entire process group so grandchild docker procs also die
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        p.kill()
            except Exception:
                pass
        deadline = time.time() + 3
        for p in procs:
            remaining = max(0.1, deadline - time.time())
            p.join(timeout=remaining)

    if containers:
        logger.info(f"[cleanup] Stopping Docker containers: {sorted(containers)}")
        # Parallel docker stop with short timeout (-t 2: send SIGKILL after 2s)
        stop_procs = [
            subprocess.Popen(
                ["docker", "stop", "-t", "2", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for name in sorted(containers)
        ]
        deadline = time.time() + 6
        for p in stop_procs:
            remaining = max(0.1, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()
        # Parallel docker rm -f
        rm_procs = [
            subprocess.Popen(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for name in sorted(containers)
        ]
        deadline = time.time() + 3
        for p in rm_procs:
            remaining = max(0.1, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()

    # Scan for orphan containers (pattern match — catches anything missed by registration)
    try:
        scan = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=terrabox-agent-llm",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        orphans = {n.strip() for n in scan.stdout.splitlines() if n.strip()} - containers
        if orphans:
            logger.info(f"[cleanup] Removing orphan containers: {sorted(orphans)}")
            orphan_procs = [
                subprocess.Popen(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                for name in sorted(orphans)
            ]
            deadline = time.time() + 5
            for p in orphan_procs:
                remaining = max(0.1, deadline - time.time())
                try:
                    p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    p.kill()
    except Exception:
        pass


def _sigint_handler(sig, frame) -> None:
    global _cleanup_started
    if _cleanup_started:
        # Second Ctrl+C — force-exit immediately without waiting
        os._exit(130)
    _cleanup_started = True
    logger.warning("\n[cleanup] Interrupted — terminating child processes and containers...")
    _do_cleanup()
    sys.exit(130)


def _setup_cleanup_handlers() -> None:
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions (run in child processes)
# ─────────────────────────────────────────────────────────────────────────────

def _build_worker(method: str, slot: int, args_dict: dict, result_queue: multiprocessing.Queue) -> None:
    """Build worker. Configures logging with method prefix so output is identifiable."""
    # Must set TQDM_POSITION before any tqdm import so each worker's bar has its own line
    os.environ["TQDM_POSITION"] = str(slot)
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s  INFO     [build:{method}]  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    try:
        import sys as _sys
        if _ROOT not in _sys.path:
            _sys.path.insert(0, _ROOT)

        _logging.getLogger("run_parallel_evolution").info(f"Starting build for {method}...")
        from terrabox.evolution.experiments.run_real_evolution import run_build
        args = argparse.Namespace(**args_dict)
        run_build(method, args)
        _logging.getLogger("run_parallel_evolution").info(f"Build DONE for {method}")
        result_queue.put(("build_ok", method, None))
    except Exception as e:
        import traceback
        _logging.getLogger("run_parallel_evolution").error(f"Build FAILED for {method}: {e}\n{traceback.format_exc()}")
        result_queue.put(("build_err", method, str(e)))


def _eval_worker(
    method: str,
    gpu_id: int,
    port: int,
    slot: int,
    args_dict: dict,
    result_queue: multiprocessing.Queue,
) -> None:
    """Eval worker: set GPU/port/tqdm env vars BEFORE importing terrabox, then run eval."""
    # Must set ALL env vars before any terrabox imports
    os.environ["AGENT_LLM_PORT"] = str(port)
    os.environ["AGENT_LLM_GPU_DEVICES"] = str(gpu_id)
    os.environ["TQDM_POSITION"] = str(slot)  # each method's tqdm bar at its own line

    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s  INFO     [eval:{method}:GPU{gpu_id}]  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    _log = _logging.getLogger("run_parallel_evolution")

    try:
        import sys as _sys
        if _ROOT not in _sys.path:
            _sys.path.insert(0, _ROOT)

        from terrabox.evolution.experiments.run_real_evolution import (
            check_vllm_health,
            run_eval,
        )
        args = argparse.Namespace(**args_dict)

        _log.info(f"Waiting for vLLM on GPU {gpu_id}, port {port}...")
        ok = check_vllm_health(timeout=300)
        if not ok:
            result_queue.put(("eval_err", method, f"vLLM not ready on port {port}"))
            return

        _log.info(f"vLLM ready — starting eval ({args_dict.get('eval_limit')} cases)")
        metrics = run_eval(method, args)
        result_queue.put(("eval_ok", method, metrics))
    except Exception as e:
        import traceback
        _log.error(f"Eval FAILED: {e}\n{traceback.format_exc()}")
        result_queue.put(("eval_err", method, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# SkillRL multi-GPU chunk worker + merge
# ─────────────────────────────────────────────────────────────────────────────

def _skillrl_chunk_worker(
    chunk_idx: int,
    slot: int,
    port: int,
    gpu_id: int,
    trajectories: list,  # list[Trajectory] — picklable dataclasses
    store_dir_chunk: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """SkillRL build worker: distill a chunk of trajectories on its own GPU."""
    os.environ["AGENT_LLM_PORT"] = str(port)
    os.environ["AGENT_LLM_GPU_DEVICES"] = str(gpu_id)
    os.environ["TQDM_POSITION"] = str(slot)
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s  INFO     [build:skillrl:GPU{gpu_id}]  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    _log = _logging.getLogger("run_parallel_evolution")
    try:
        import sys as _sys
        if _ROOT not in _sys.path:
            _sys.path.insert(0, _ROOT)

        from terrabox.evolution.experiments.run_real_evolution import check_vllm_health
        _log.info(f"Waiting for vLLM on GPU {gpu_id}, port {port} ...")
        if not check_vllm_health(timeout=300):
            result_queue.put(("build_err", f"skillrl_chunk{chunk_idx}", "vLLM not ready"))
            return

        from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
        from terrabox.evolution.shared.llm_client import EvolutionLLMClient
        from terrabox.evolution.skillrl.distiller import ExperienceDistiller
        from terrabox.evolution.skillrl.skill_bank import HierarchicalSkillBank

        os.makedirs(store_dir_chunk, exist_ok=True)
        bank = HierarchicalSkillBank(store_dir_chunk)
        llm = EvolutionLLMClient()
        distiller = ExperienceDistiller(bank, llm)
        evaluator = ToolMatchEvaluator()

        episodes = [evaluator.evaluate(t) for t in trajectories]
        _log.info(f"Chunk {chunk_idx}: {len(trajectories)} trajs → {len(episodes)} episodes, distilling...")
        created = distiller.distill_batch(episodes)
        counts = bank.counts()
        _log.info(f"Chunk {chunk_idx} done: skills_created={created}, bank={counts}")
        result_queue.put(("build_ok", f"skillrl_chunk{chunk_idx}", {"created": created}))
    except Exception as e:
        import traceback
        _log.error(f"Chunk {chunk_idx} FAILED: {e}\n{traceback.format_exc()}")
        result_queue.put(("build_err", f"skillrl_chunk{chunk_idx}", str(e)))


def _merge_skillrl_banks(chunk_dirs: list[str], final_store_dir: str) -> None:
    """Merge JSON skill banks from parallel chunk workers into final_store_dir."""
    import json
    import shutil
    os.makedirs(final_store_dir, exist_ok=True)
    for filename in ("general_skills.json", "task_skills.json", "mistakes.json"):
        merged_skills: list[dict] = []
        for chunk_dir in chunk_dirs:
            path = os.path.join(chunk_dir, filename)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                merged_skills.extend(data.get("skills", []))
        # Deduplicate by content hash (prefer keeping first occurrence)
        seen: set[str] = set()
        deduped: list[dict] = []
        for skill in merged_skills:
            key = skill.get("_content_hash") or skill.get("id", "")
            if not key or key not in seen:
                if key:
                    seen.add(key)
                deduped.append(skill)
        final_path = os.path.join(final_store_dir, filename)
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(
                {"skills": deduped, "metadata": {"merged_chunks": len(chunk_dirs)}},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"  Merged {filename}: {len(deduped)} unique skills from {len(chunk_dirs)} chunks")
    for chunk_dir in chunk_dirs:
        shutil.rmtree(chunk_dir, ignore_errors=True)
    logger.info("SkillRL bank merge complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Build phase
# ─────────────────────────────────────────────────────────────────────────────

def run_build_parallel(methods: list[str], args: argparse.Namespace, gpu_ids: list[int], base_port: int) -> None:
    """Run build phase — CPU workers and skillrl workers start simultaneously:

    1. CPU-only  (causalevo, memrl, agentevolver): parallel subprocess workers, no GPU.
    2. skillrl   : parallel multi-GPU — episodes split across N GPUs, banks merged after.
       CPU-only and skillrl workers all start at the same time; none waits for the other.
    3. evoskill  : sequential single GPU, runs after 1+2 finish.
    baseline has no build phase.

    NOTE: skillrl's distill_batch calls LLM once per high-signal episode.
    memrl build is CPU-only (llm_client not passed to EpisodicMemory in cmd_populate).
    """
    _LLM_SEQUENTIAL = _BUILD_NEEDS_LLM  # {"evoskill"}
    cpu_methods = [m for m in methods if m not in _LLM_SEQUENTIAL and m not in ("baseline", "skillrl")]
    skillrl_requested = "skillrl" in methods
    llm_seq_methods = [m for m in methods if m in _LLM_SEQUENTIAL]

    args_dict = vars(args).copy()
    ctx = multiprocessing.get_context("spawn")

    # ── Pre-load skillrl trajectories in main process (fast, ~1s) ─────────────
    # Must happen before spawning workers so chunks can be passed as pickle args.
    skillrl_chunks: list[list] = []
    skillrl_chunk_dirs: list[str] = []
    store_dir_skillrl = os.path.join(args.store_dir, "skillrl")
    if skillrl_requested:
        logger.info("Pre-loading trajectories for skillrl chunk split...")
        from terrabox.evolution.shared.data_loader import OpenEarthLoader
        loader = OpenEarthLoader(args.train_data, args.eval_data)
        try:
            import tqdm as _tqdm_mod
            all_trajs = list(_tqdm_mod.tqdm(
                loader.iter_train(limit=args.build_limit),
                desc="[skillrl] loading", unit="traj", ncols=90, total=args.build_limit,
            ))
        except ImportError:
            all_trajs = list(loader.iter_train(limit=args.build_limit))
        logger.info(f"  Loaded {len(all_trajs)} trajectories → splitting across {len(gpu_ids)} GPU(s)")
        n = len(gpu_ids)
        chunk_size = (len(all_trajs) + n - 1) // n
        skillrl_chunks = [all_trajs[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
        os.makedirs(store_dir_skillrl, exist_ok=True)
        if getattr(args, "reset", False):
            for fname in ("general_skills.json", "task_skills.json", "mistakes.json"):
                fpath = os.path.join(store_dir_skillrl, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

    # ── Spawn ALL workers simultaneously (CPU + skillrl) ──────────────────────
    cpu_q: multiprocessing.Queue = ctx.Queue()
    skillrl_q: multiprocessing.Queue = ctx.Queue()
    all_procs: list = []

    if cpu_methods:
        logger.info(f"Spawning {len(cpu_methods)} CPU build workers: {cpu_methods}")
        for slot, method in enumerate(cpu_methods):
            p = ctx.Process(target=_build_worker, args=(method, slot, args_dict, cpu_q))
            p.start()
            _register_proc(p)
            logger.info(f"  [build:{method}] PID={p.pid} (tqdm line {slot})")
            all_procs.append(p)

    if skillrl_requested:
        logger.info(f"Spawning {len([c for c in skillrl_chunks if c])} skillrl chunk workers across GPUs {gpu_ids}")
        for chunk_idx, (chunk, gpu_id) in enumerate(zip(skillrl_chunks, gpu_ids)):
            if not chunk:
                continue
            port = base_port + chunk_idx
            chunk_dir = os.path.join(args.store_dir, f"skillrl_chunk_{chunk_idx}")
            skillrl_chunk_dirs.append(chunk_dir)
            _register_container(f"terrabox-agent-llm-{port}")
            p = ctx.Process(
                target=_skillrl_chunk_worker,
                args=(chunk_idx, chunk_idx, port, gpu_id, chunk, chunk_dir, skillrl_q),
            )
            p.start()
            _register_proc(p)
            logger.info(f"  [skillrl chunk{chunk_idx}] GPU={gpu_id} port={port} trajs={len(chunk)} PID={p.pid}")
            all_procs.append(p)

    # ── Wait for ALL workers ───────────────────────────────────────────────────
    for p in all_procs:
        p.join()
        _unregister_proc(p)

    # ── Collect CPU results ───────────────────────────────────────────────────
    if cpu_methods:
        errors: list[str] = []
        while not cpu_q.empty():
            status, m, detail = cpu_q.get()
            if status == "build_err":
                errors.append(f"{m}: {detail}")
                logger.error(f"[{m}] Build FAILED: {detail}")
            else:
                logger.info(f"[{m}] Build OK")
        if errors:
            logger.warning(f"Some CPU builds failed: {errors}")

    # ── Collect skillrl results + merge banks ─────────────────────────────────
    if skillrl_requested:
        skillrl_errors: list[str] = []
        while not skillrl_q.empty():
            status, m, detail = skillrl_q.get()
            if status == "build_err":
                skillrl_errors.append(f"{m}: {detail}")
                logger.error(f"  [{m}] Build FAILED: {detail}")
            else:
                logger.info(f"  [{m}] Build OK: {detail}")
        if not skillrl_errors:
            logger.info("Merging skillrl chunk banks...")
            _merge_skillrl_banks(skillrl_chunk_dirs, store_dir_skillrl)
        else:
            logger.warning(f"SkillRL chunks failed, skipping merge: {skillrl_errors}")

    # ── evoskill — sequential after all parallel work finishes ────────────────
    if llm_seq_methods:
        logger.info(f"Building (sequential, GPU {gpu_ids[0]}, needs vLLM): {llm_seq_methods}")
        env = os.environ.copy()
        env["AGENT_LLM_PORT"] = str(base_port)
        env["AGENT_LLM_GPU_DEVICES"] = str(gpu_ids[0])
        _register_container(f"terrabox-agent-llm-{base_port}")
        for method in llm_seq_methods:
            logger.info(f"[{method}] Running LLM build...")
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_real_evolution.py"),
                "--method", method,
                "--phase", "build",
                "--train-data", args.train_data,
                "--eval-data", args.eval_data,
                "--store-dir", args.store_dir,
                "--build-limit", str(args.build_limit),
            ]
            if args.reset:
                cmd.append("--reset")
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                logger.error(f"[{method}] LLM build failed (exit {result.returncode})")
            else:
                logger.info(f"[{method}] Build OK")


# ─────────────────────────────────────────────────────────────────────────────
# Eval phase
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_parallel(
    methods: list[str],
    args: argparse.Namespace,
    gpu_ids: list[int],
    base_port: int,
    n_gpus: int,
) -> dict[str, dict]:
    """Run eval phase: n_gpus methods at a time in parallel."""
    args_dict = vars(args).copy()
    # Remove fields not in run_eval's Namespace contract
    for key in ("n_gpus", "base_port", "gpu_ids"):
        args_dict.pop(key, None)
    args_dict["phase"] = "eval"  # workers only do eval

    all_metrics: dict[str, dict] = {}
    ctx = multiprocessing.get_context("spawn")

    # Split methods into batches of n_gpus
    batches = [methods[i:i + n_gpus] for i in range(0, len(methods), n_gpus)]
    for batch_idx, batch in enumerate(batches):
        logger.info(f"\nEval batch {batch_idx + 1}/{len(batches)}: {batch}")
        q: multiprocessing.Queue = ctx.Queue()
        procs = []

        for slot, method in enumerate(batch):
            gpu_id = gpu_ids[slot % len(gpu_ids)]
            port = base_port + slot
            _register_container(f"terrabox-agent-llm-{port}")
            p = ctx.Process(
                target=_eval_worker,
                args=(method, gpu_id, port, slot, args_dict, q),
            )
            p.start()
            _register_proc(p)
            procs.append((method, p))
            logger.info(f"  [{method}] → GPU {gpu_id}, port {port}, tqdm line {slot}")

        for method, p in procs:
            p.join()
            _unregister_proc(p)

        while not q.empty():
            status, m, data = q.get()
            if status == "eval_ok":
                all_metrics[m] = data
                logger.info(f"  [{m}] Eval OK: F1={data.get('f1', 0):.4f}")
            else:
                logger.error(f"  [{m}] Eval FAILED: {data}")

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table (reuse from run_real_evolution)
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(all_metrics: dict[str, dict]) -> None:
    from terrabox.evolution.experiments.run_real_evolution import print_comparison_table
    print_comparison_table(all_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU parallel self-evolution experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", default="all",
                   help="Method(s) to run: " + ", ".join(VALID_METHODS) + ", all")
    p.add_argument("--phase", choices=("build", "eval", "both"), default="both")
    p.add_argument("--train-data", default="data/openearth/train.json")
    p.add_argument("--eval-data", default="data/openearth/eval.jsonl")
    p.add_argument("--store-dir", default="evolution_store")
    p.add_argument("--build-limit", type=int, default=14538)
    p.add_argument("--eval-limit", type=int, default=1169)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ablation", default=None,
                   choices=(None, "no_cca", "no_synthesis", "no_ctfm"))
    p.add_argument("--online", action="store_true",
                   help="Online learning during eval")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-task agent timeout (seconds)")
    p.add_argument("--output", default=None)
    p.add_argument("--reset", action="store_true",
                   help="Clear knowledge bases before build")
    p.add_argument("--mock-eval", action="store_true",
                   help="Fast eval: single LLM call per task, no tool execution (~10x faster)")
    p.add_argument("--trajectory-eval", action="store_true",
                   help="Fast eval: read pre-existing test.json trajectories, no LLM calls")
    p.add_argument("--verbose", action="store_true")
    # Parallel-specific args
    p.add_argument("--n-gpus", type=int, default=4,
                   help="Number of GPUs to use in parallel")
    p.add_argument("--base-port", type=int, default=9100,
                   help="Base port for vLLM instances (uses base_port + slot)")
    p.add_argument("--gpu-ids", default=None,
                   help="Comma-separated GPU IDs (default: 0,1,...,n-gpus-1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_cleanup_handlers()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve methods (supports "all", single name, or comma-separated list)
    if args.method.lower() == "all":
        methods = list(VALID_METHODS)
    elif "," in args.method:
        methods = [m.strip().lower() for m in args.method.split(",")]
        invalid = [m for m in methods if m not in VALID_METHODS]
        if invalid:
            print(f"[ERROR] Unknown method(s): {invalid}. Valid: {VALID_METHODS}")
            sys.exit(1)
    elif args.method.lower() in VALID_METHODS:
        methods = [args.method.lower()]
    else:
        print(f"[ERROR] Unknown method {args.method!r}.")
        sys.exit(1)

    # Resolve GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.n_gpus))
    n_gpus = min(args.n_gpus, len(gpu_ids))

    print()
    print("=" * 80)
    print("Self-Evolution Parallel Experiment")
    print(f"  Methods    : {', '.join(methods)}")
    print(f"  Phase      : {args.phase}")
    print(f"  GPUs       : {gpu_ids[:n_gpus]}")
    print(f"  Ports      : {[args.base_port + i for i in range(n_gpus)]}")
    print(f"  Build limit: {args.build_limit}")
    print(f"  Eval limit : {args.eval_limit}")
    print(f"  Eval mode  : {'mock (no tool exec)' if args.mock_eval else 'trajectory (no LLM)' if args.trajectory_eval else 'full agent'}")
    print(f"  Online     : {args.online}")
    print("=" * 80)

    t_start = time.time()

    # ── Build ──
    if args.phase in ("build", "both"):
        t0 = time.time()
        logger.info("Starting build phase...")
        run_build_parallel(methods, args, gpu_ids, args.base_port)
        logger.info(f"Build phase done in {time.time() - t0:.1f}s")

    # ── Eval ──
    all_metrics: dict[str, dict] = {}
    if args.phase in ("eval", "both"):
        t0 = time.time()
        logger.info("Starting eval phase...")
        all_metrics = run_eval_parallel(methods, args, gpu_ids, args.base_port, n_gpus)
        logger.info(f"Eval phase done in {time.time() - t0:.1f}s")

    # ── Summary ──
    if all_metrics:
        _print_comparison(all_metrics)
        combined_path = os.path.join(args.store_dir, "results", "comparison.json")
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Combined results → {combined_path}")

    logger.info(f"Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
