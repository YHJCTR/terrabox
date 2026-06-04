"""CLI runner for fixed-shuffle reflection experiments."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ..ReAct.runner import build_rollout_env
from .data_split import DEFAULT_SHUFFLED_DATA, DEFAULT_STRICT_DATA, write_shuffled_jsonl, write_task_slice
from .memory import ReflectionMemoryBank
from .metrics import write_reflection_metrics
from .reflector import build_reflections_from_rollout


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent


def experiment_dir(experiment: str) -> Path:
    return MODULE_ROOT / "exp" / experiment


def build_rollout_command(
    *,
    task_file: str | Path,
    experiment: str,
    output_dir: str | Path,
    port: int,
    evolution_store: str | Path | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    limit: int | None = None,
    max_iterations: int = 15,
    resume: bool = True,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_trajectory_experiment.py",
        "rollout",
        "--task-file",
        str(task_file),
        "--experiment",
        experiment,
        "--mode",
        "standard",
        "--output-dir",
        str(output_dir),
        "--port",
        str(port),
        "--max-iterations",
        str(max_iterations),
        "--use-docker",
    ]
    if resume:
        cmd.append("--resume")
    if start_index is not None:
        cmd.extend(["--start-index", str(start_index)])
    if end_index is not None:
        cmd.extend(["--end-index", str(end_index)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if evolution_store is not None:
        cmd.extend(["--evolution-method", "reflection", "--evolution-store", str(evolution_store)])
    return cmd


def cmd_shuffle_data(args: argparse.Namespace) -> None:
    count = write_shuffled_jsonl(args.strict_data, args.output, seed=args.seed)
    print(json.dumps({"output": args.output, "seed": args.seed, "num_samples": count}, ensure_ascii=False, indent=2))


def cmd_prepare_tasks(args: argparse.Namespace) -> None:
    count = write_task_slice(
        args.strict_data,
        args.output,
        seed=args.seed,
        start=args.start,
        limit=args.limit,
        split_name=args.split_name,
    )
    print(json.dumps({"output": args.output, "seed": args.seed, "start": args.start, "limit": args.limit, "num_tasks": count}, ensure_ascii=False, indent=2))


def cmd_build_memory(args: argparse.Namespace) -> None:
    store_dir = Path(args.store_dir or experiment_dir(args.experiment))
    memory_path = store_dir / "memory" / "reflection_memory.jsonl"
    entries = build_reflections_from_rollout(
        args.trajectory_dir,
        low_f1_threshold=args.low_f1_threshold,
        include_success=args.include_success,
    )
    bank = ReflectionMemoryBank(memory_path)
    bank.entries = entries
    bank.save()
    stats = bank.stats()
    stats["trajectory_dir"] = args.trajectory_dir
    stats["low_f1_threshold"] = args.low_f1_threshold
    (store_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (store_dir / "metrics" / "memory_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_rollout(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / args.phase
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_rollout_command(
        task_file=args.task_file,
        experiment=f"{args.experiment}_{args.phase}",
        output_dir=out_dir,
        port=args.port,
        evolution_store=args.evolution_store,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
        max_iterations=args.max_iterations,
        resume=args.resume,
    )
    log_path = exp_dir / "logs" / f"{args.phase}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_script = exp_dir / f"run_{args.phase}.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(log_path))}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    print("Running:", " ".join(cmd))
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("Running: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        log_file.flush()
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=build_rollout_env(args.agent_gpu, args.tool_gpu),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def cmd_stats(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    rollout_dir = Path(args.rollout_dir or exp_dir / "eval")
    memory_path = Path(args.memory_path or exp_dir / "memory" / "reflection_memory.jsonl")
    metrics = write_reflection_metrics(
        rollout_dir=rollout_dir,
        memory_path=memory_path,
        task_file=args.task_file,
        output_path=exp_dir / "metrics" / "metrics.json",
        top_k=args.top_k,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrabox reflection baseline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_shuffle = sub.add_parser("shuffle-data", help="Write deterministic shuffled strict SFT JSONL")
    p_shuffle.add_argument("--strict-data", default=DEFAULT_STRICT_DATA)
    p_shuffle.add_argument("--output", default=DEFAULT_SHUFFLED_DATA)
    p_shuffle.add_argument("--seed", type=int, default=42)
    p_shuffle.set_defaults(func=cmd_shuffle_data)

    p_tasks = sub.add_parser("prepare-tasks", help="Write a rollout task JSON from a shuffled slice")
    p_tasks.add_argument("--strict-data", default=DEFAULT_SHUFFLED_DATA)
    p_tasks.add_argument("--output", required=True)
    p_tasks.add_argument("--seed", type=int, default=42)
    p_tasks.add_argument("--start", type=int, default=0)
    p_tasks.add_argument("--limit", type=int)
    p_tasks.add_argument("--split-name", default="split")
    p_tasks.set_defaults(func=cmd_prepare_tasks)

    p_memory = sub.add_parser("build-memory", help="Build reflection memory from train rollout trajectories")
    p_memory.add_argument("--experiment", required=True)
    p_memory.add_argument("--trajectory-dir", required=True)
    p_memory.add_argument("--store-dir")
    p_memory.add_argument("--low-f1-threshold", type=float, default=0.8)
    p_memory.add_argument("--include-success", action="store_true", default=True)
    p_memory.set_defaults(func=cmd_build_memory)

    p_rollout = sub.add_parser("rollout", help="Run standard rollout, optionally with reflection memory")
    p_rollout.add_argument("--experiment", required=True)
    p_rollout.add_argument("--phase", default="eval", choices=["train", "eval"])
    p_rollout.add_argument("--task-file", required=True)
    p_rollout.add_argument("--output-dir")
    p_rollout.add_argument("--evolution-store")
    p_rollout.add_argument("--port", type=int, default=9102)
    p_rollout.add_argument("--agent-gpu", default=2)
    p_rollout.add_argument("--tool-gpu", default=3)
    p_rollout.add_argument("--start-index", type=int)
    p_rollout.add_argument("--end-index", type=int)
    p_rollout.add_argument("--limit", type=int)
    p_rollout.add_argument("--max-iterations", type=int, default=15)
    p_rollout.add_argument("--resume", action="store_true", default=True)
    p_rollout.add_argument("--no-resume", dest="resume", action="store_false")
    p_rollout.set_defaults(func=cmd_rollout)

    p_stats = sub.add_parser("stats", help="Write reflection rollout, memory, and retrieval metrics")
    p_stats.add_argument("--experiment", required=True)
    p_stats.add_argument("--rollout-dir")
    p_stats.add_argument("--memory-path")
    p_stats.add_argument("--task-file")
    p_stats.add_argument("--top-k", type=int, default=5)
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
