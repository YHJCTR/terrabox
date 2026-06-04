"""CLI runner for the plain ReAct baseline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..full_shared.sft_schema import load_sft_samples
from .data_adapter import write_task_file
from .metrics import write_metrics


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = "data/newdata/sft_train_strict.jsonl"


def experiment_dir(experiment: str) -> Path:
    return MODULE_ROOT / "exp" / experiment


def build_rollout_env(agent_gpu: int | str, tool_gpu: int | str) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    env.setdefault("no_proxy", "localhost,127.0.0.1")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1")
    env["AGENT_LLM_GPU_DEVICES"] = str(agent_gpu)
    env["TERRABOX_TOOL_GPU_DEVICES"] = str(tool_gpu)
    env["CUDA_VISIBLE_DEVICES"] = f"{agent_gpu},{tool_gpu}"
    env.setdefault("TERRABOX_USE_DOCKER", "true")
    env.setdefault("TERRABOX_TOOL_MAX_GPUS", "1")
    env.setdefault("TERRABOX_TOOL_SERVICE_SCOPE", "call")
    for key in (
        "VLM_GPU_DEVICES",
        "SAM2_GPU_DEVICES",
        "REMOTESAM_GPU_DEVICES",
        "STRIP_RCNN_GPU_DEVICES",
        "INSTRUCTSAM_GPU_DEVICES",
        "REMOTECLIP_GPU_DEVICES",
    ):
        env[key] = str(tool_gpu)
    return env


def build_rollout_command(
    *,
    task_file: str | Path,
    experiment: str,
    output_dir: str | Path,
    mode: str = "standard",
    port: int = 9100,
    limit: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    max_iterations: int = 15,
    resume: bool = True,
    no_restrict_tools: bool = False,
    include_osm: bool = False,
    include_bing: bool = False,
    include_mock: bool = False,
    include_vlm: bool = False,
    include_changeos: bool = False,
    python_executable: str | None = None,
) -> list[str]:
    cmd = [
        python_executable or sys.executable,
        "scripts/run_trajectory_experiment.py",
        "rollout",
        "--task-file",
        str(task_file),
        "--experiment",
        experiment,
        "--mode",
        mode,
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
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if start_index is not None:
        cmd.extend(["--start-index", str(start_index)])
    if end_index is not None:
        cmd.extend(["--end-index", str(end_index)])
    if no_restrict_tools:
        cmd.append("--no-restrict-tools")
    if include_osm:
        cmd.append("--no-skip-osm")
    if include_bing:
        cmd.append("--no-skip-bing")
    if include_mock:
        cmd.append("--no-skip-mock")
    if include_vlm:
        cmd.append("--no-skip-vlm")
    if include_changeos:
        cmd.append("--no-skip-changeos")
    return cmd


def prepare_tasks(args: argparse.Namespace) -> Path:
    samples = load_sft_samples(args.strict_data, limit=args.prepare_limit)
    exp_dir = experiment_dir(args.experiment)
    task_file = exp_dir / "tasks.json"
    count = write_task_file(samples, task_file, source_path=args.strict_data)
    config = {
        "strict_data": args.strict_data,
        "task_file": str(task_file),
        "num_tasks": count,
        "gold_leakage_policy": "task file omits SFT messages, gold tool calls, and ground truth",
    }
    (exp_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {task_file} ({count} tasks)")
    return task_file


def cmd_rollout(args: argparse.Namespace) -> None:
    task_file = Path(args.task_file) if args.task_file else prepare_tasks(args)
    out_dir = Path(args.output_dir) if args.output_dir else experiment_dir(args.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_rollout_command(
        task_file=task_file,
        experiment=args.experiment,
        output_dir=out_dir,
        mode=args.mode,
        port=args.port,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        max_iterations=args.max_iterations,
        resume=args.resume,
        no_restrict_tools=args.no_restrict_tools,
        include_osm=args.include_osm,
        include_bing=args.include_bing,
        include_mock=args.include_mock,
        include_vlm=args.include_vlm,
        include_changeos=args.include_changeos,
    )
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=build_rollout_env(args.agent_gpu, args.tool_gpu), check=True)
    metrics = write_metrics(out_dir, out_dir / "metrics" / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    target = Path(args.output_dir) if args.output_dir else experiment_dir(args.experiment)
    metrics = write_metrics(target, target / "metrics" / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrabox ReAct baseline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    common: dict[str, Any] = {}
    p_prepare = sub.add_parser("prepare-tasks", help="Convert strict SFT JSONL to rollout task JSON")
    p_prepare.add_argument("--strict-data", default=DEFAULT_DATA)
    p_prepare.add_argument("--experiment", required=True)
    p_prepare.add_argument("--prepare-limit", type=int)
    p_prepare.set_defaults(func=prepare_tasks)

    p_rollout = sub.add_parser("rollout", help="Run real ReAct rollout through existing trajectory script")
    p_rollout.add_argument("--strict-data", default=DEFAULT_DATA)
    p_rollout.add_argument("--task-file")
    p_rollout.add_argument("--experiment", required=True)
    p_rollout.add_argument("--output-dir")
    p_rollout.add_argument("--prepare-limit", type=int)
    p_rollout.add_argument("--mode", default="standard", choices=["standard", "progressive", "category_scoped"])
    p_rollout.add_argument("--port", type=int, default=9100)
    p_rollout.add_argument("--agent-gpu", default=0)
    p_rollout.add_argument("--tool-gpu", default=1)
    p_rollout.add_argument("--limit", type=int)
    p_rollout.add_argument("--start-index", type=int)
    p_rollout.add_argument("--end-index", type=int)
    p_rollout.add_argument("--max-iterations", type=int, default=15)
    p_rollout.add_argument("--resume", action="store_true", default=True)
    p_rollout.add_argument("--no-resume", dest="resume", action="store_false")
    p_rollout.add_argument("--no-restrict-tools", action="store_true")
    p_rollout.add_argument("--include-osm", action="store_true")
    p_rollout.add_argument("--include-bing", action="store_true")
    p_rollout.add_argument("--include-mock", action="store_true")
    p_rollout.add_argument("--include-vlm", action="store_true")
    p_rollout.add_argument("--include-changeos", action="store_true")
    p_rollout.set_defaults(func=cmd_rollout)

    p_smoke = sub.add_parser("smoke", help="Run a small ReAct rollout")
    p_smoke.add_argument("--strict-data", default=DEFAULT_DATA)
    p_smoke.add_argument("--experiment", default="react_smoke")
    p_smoke.add_argument("--output-dir")
    p_smoke.add_argument("--prepare-limit", type=int)
    p_smoke.add_argument("--mode", default="standard", choices=["standard", "progressive", "category_scoped"])
    p_smoke.add_argument("--port", type=int, default=9100)
    p_smoke.add_argument("--agent-gpu", default=0)
    p_smoke.add_argument("--tool-gpu", default=1)
    p_smoke.add_argument("--limit", type=int, default=5)
    p_smoke.add_argument("--start-index", type=int)
    p_smoke.add_argument("--end-index", type=int)
    p_smoke.add_argument("--max-iterations", type=int, default=15)
    p_smoke.add_argument("--resume", action="store_true", default=True)
    p_smoke.add_argument("--no-restrict-tools", action="store_true")
    p_smoke.add_argument("--include-osm", action="store_true")
    p_smoke.add_argument("--include-bing", action="store_true")
    p_smoke.add_argument("--include-mock", action="store_true")
    p_smoke.add_argument("--include-vlm", action="store_true")
    p_smoke.add_argument("--include-changeos", action="store_true")
    p_smoke.set_defaults(func=cmd_rollout)

    p_stats = sub.add_parser("stats", help="Aggregate metrics from an experiment directory")
    p_stats.add_argument("--experiment", required=True)
    p_stats.add_argument("--output-dir")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
