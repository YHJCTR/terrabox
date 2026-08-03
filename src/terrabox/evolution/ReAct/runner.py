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


def build_rollout_env(
    agent_gpu: int | str,
    tool_gpu: int | str,
    vlm_gpus: str | None = None,
    vlm_max_model_len: int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    env.setdefault("no_proxy", "localhost,127.0.0.1")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1")
    # Redirect relative tool output paths (e.g. output_path="ndti.tif",
    # "question141/...") into a gitignored tmp dir instead of the repo root.
    # tool_executor._redirect_relative_output_path only activates when set.
    env.setdefault("TERRABOX_ARTIFACT_OUTPUT_DIR", str(REPO_ROOT / "tmp" / "artifacts"))
    env["AGENT_LLM_GPU_DEVICES"] = str(agent_gpu)
    env.setdefault("TERRABOX_USE_DOCKER", "true")
    env.setdefault("TERRABOX_TOOL_SERVICE_SCOPE", "call")
    env.setdefault("TERRABOX_SERVICE_CALL_LOCKS", "1")
    env.setdefault("TERRABOX_SERVICE_LOCK_TIMEOUT_SECONDS", "1800")
    env.setdefault("TERRABOX_SERVICE_LOCK_DIR", str(REPO_ROOT / "tmp" / "service_locks"))
    # OCR is cheap enough on CPU and otherwise competes with VLM/InstructSAM for
    # GPU memory. Keep it off GPU in rollout environments unless explicitly
    # overridden by the caller.
    env.setdefault("TERRABOX_OCR_USE_GPU", "0")
    # Text-targeted sam2_segment routes through RemoteSAM and often cold-starts a
    # model service; the default 120s outer tool timeout can kill it while the
    # inner service call is still allowed to run for 300s.
    env.setdefault("TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_SAM2_SEGMENT", "420")
    env.setdefault("TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_REMOTESAM", "420")
    env.setdefault("REMOTESAM_TOOL_TIMEOUT", "360")
    # OSM geocoding/Overpass calls are network-bound and can legitimately exceed
    # 120s on migrated hosts. Keep the longer timeout scoped to the two external
    # query tools instead of raising the global tool timeout.
    env.setdefault("TERRABOX_TOOL_TIMEOUT_OSM_GIS_GET_AREA_BOUNDARY", "240")
    env.setdefault("TERRABOX_TOOL_TIMEOUT_OSM_GIS_ADD_POIS_LAYER", "240")
    # fixdata + VLM-backed instructsam require aliases OFF (otherwise the old
    # Calculator/Solver/Plot→ipython aliases re-enter and shadow compute.*).
    # Default to off; callers can still override by exporting the env var.
    env.setdefault("TERRABOX_ENABLE_SOURCE_SCHEMA_TOOL_ALIASES", "false")
    # Perception tool containers share the tool GPU.
    for key in (
        "SAM2_GPU_DEVICES",
        "REMOTESAM_GPU_DEVICES",
        "STRIP_RCNN_GPU_DEVICES",
        "INSTRUCTSAM_GPU_DEVICES",
        "REMOTECLIP_GPU_DEVICES",
        "CHANGEOS_GPU_DEVICES",
    ):
        env[key] = str(tool_gpu)

    if vlm_gpus:
        # Dedicate the VLM (Qwen3-VL, instructsam backend) to its own GPU set,
        # tensor-parallel across them. The bf16 model (~17GB) needs >1×24GB GPU
        # at the default 16384 context; pass a SINGLE GPU here to switch to the
        # light profile below (short context fits one card → fewer concurrent
        # GPUs → lower peak power, avoids breaker trips).
        # Per-service GPU env vars hard-pin each service (allocate_gpu(s) returns
        # the env value as-is), so we do NOT set TERRABOX_TOOL_GPU_DEVICES (its
        # static pool-prefix would mis-assign the VLM and bypass per-service pins).
        vlm_list = [g.strip() for g in str(vlm_gpus).split(",") if g.strip()]
        env["VLM_GPU_DEVICES"] = ",".join(vlm_list)
        env["VLM_TENSOR_PARALLEL_SIZE"] = str(len(vlm_list))
        env.pop("TERRABOX_TOOL_GPU_DEVICES", None)
        env.pop("TERRABOX_TOOL_MAX_GPUS", None)
        # Keep the slow VLM container warm across calls; lighter perception tools
        # on the tool GPU still cycle per call (scope=call) so they don't pile up.
        env["TERRABOX_KEEP_VLM_WARM"] = "1"
        # First instructsam call cold-starts the ~17GB Qwen3-VL container
        # (~2 min); give it headroom so the tool call doesn't time out.
        env.setdefault("TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_INSTRUCTSAM", "600")
        if len(vlm_list) == 1:
            # Single-GPU profile. 8192 covers most OEA visual prompts without
            # falling back to context retries; keep one sequence per batch to
            # avoid spiky KV-cache allocation on 24GB cards. MUST also lower
            # VLM_MIN_IMAGE_MODEL_LEN, else the manager's image-context guard
            # bumps max-model-len back to 16384.
            # These env vars only affect THIS rollout subprocess; global/normal
            # VLM use (e.g. vlm_analyze at 2-GPU 16384) is untouched.
            mlen = str(int(vlm_max_model_len) if vlm_max_model_len else 8192)
            env["VLM_MAX_MODEL_LEN"] = mlen
            env["VLM_MIN_IMAGE_MODEL_LEN"] = mlen
            env.setdefault("VLM_GPU_MEMORY_UTILIZATION", "0.95")
            env.setdefault("VLM_MAX_NUM_SEQS", "1")
            env.setdefault("TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS", "4096")
        all_gpus = []
        for g in [str(agent_gpu), str(tool_gpu), *vlm_list]:
            if g not in all_gpus:
                all_gpus.append(g)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(all_gpus)
        service_gpus = []
        for g in [str(tool_gpu), *vlm_list]:
            if g not in service_gpus:
                service_gpus.append(g)
        env["TERRABOX_GPU_ALLOWED_DEVICES"] = ",".join(service_gpus)
        env["TERRABOX_GPU_FALLBACK_DEVICE"] = str(tool_gpu)
        env["TERRABOX_GPU_FALLBACK_DEVICES"] = ",".join(service_gpus)
    else:
        env["VLM_GPU_DEVICES"] = str(tool_gpu)
        env.pop("TERRABOX_TOOL_GPU_DEVICES", None)
        env.pop("TERRABOX_TOOL_MAX_GPUS", None)
        env["CUDA_VISIBLE_DEVICES"] = f"{agent_gpu},{tool_gpu}"
        env["TERRABOX_GPU_ALLOWED_DEVICES"] = str(tool_gpu)
        env["TERRABOX_GPU_FALLBACK_DEVICE"] = str(tool_gpu)
        env["TERRABOX_GPU_FALLBACK_DEVICES"] = str(tool_gpu)
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
    exclude_tools: str | None = None,
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
    if exclude_tools:
        cmd.extend(["--exclude-tools", exclude_tools])
    return cmd


def stop_managed_services() -> None:
    """Stop all terrabox-managed docker service containers (VLM, agent-llm,
    perception tools) started for an experiment. Filters by the
    ``terrabox.managed=true`` label so unrelated containers are never touched.
    Called after a rollout so experiments don't leave GPUs occupied."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=terrabox.managed=true"],
            capture_output=True, text=True,
        )
        ids = [x for x in out.stdout.split() if x]
        if not ids:
            print("[teardown] no terrabox-managed containers running.")
            return
        names = subprocess.run(
            ["docker", "ps", "--filter", "label=terrabox.managed=true",
             "--format", "{{.Names}}"], capture_output=True, text=True,
        ).stdout.split()
        subprocess.run(["docker", "stop", *ids], capture_output=True, text=True)
        print(f"[teardown] stopped {len(ids)} terrabox-managed container(s): {', '.join(names)}")
    except Exception as exc:  # noqa: BLE001 - teardown must never crash the run
        print(f"[teardown] warning: failed to stop managed containers: {exc}")


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
        exclude_tools=getattr(args, "exclude_tools", None),
    )
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=build_rollout_env(args.agent_gpu, args.tool_gpu, getattr(args, "vlm_gpus", None),
                                  getattr(args, "vlm_max_model_len", None)),
            check=True,
        )
    finally:
        # Stop experiment-started docker services unless explicitly kept warm
        # (e.g. chaining train→eval). Default: tear down so GPUs are freed.
        if not getattr(args, "keep_services", False):
            stop_managed_services()
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
    p_rollout.add_argument("--exclude-tools", default=None,
                           help="逗号分隔的工具 slug，从 allowed 中额外剔除（如 ipython.execute）")
    p_rollout.add_argument("--vlm-gpus", default=None,
                           help="给 VLM(instructsam后端)独立 GPU,逗号分隔(如 '2,3');tensor-parallel=卡数。"
                                "传单卡(如 '2')自动启用单卡轻量档(短上下文),只用3张卡降功率避免跳闸")
    p_rollout.add_argument("--vlm-max-model-len", type=int, default=None,
                           help="单卡轻量档的 VLM 上下文长度(默认 4096,仅 --vlm-gpus 为单卡时生效)")
    p_rollout.add_argument("--keep-services", action="store_true",
                           help="跑完不停 docker 服务(默认跑完自动停掉所有 terrabox 托管容器)")
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
    p_smoke.add_argument("--exclude-tools", default=None,
                         help="逗号分隔的工具 slug，从 allowed 中额外剔除（如 ipython.execute）")
    p_smoke.add_argument("--vlm-gpus", default=None,
                         help="给 VLM(instructsam后端)独立 GPU,逗号分隔(如 '2,3');传单卡启用单卡轻量档")
    p_smoke.add_argument("--vlm-max-model-len", type=int, default=None,
                         help="单卡轻量档的 VLM 上下文长度(默认 4096)")
    p_smoke.add_argument("--keep-services", action="store_true",
                         help="跑完不停 docker 服务(默认跑完自动停掉所有 terrabox 托管容器)")
    p_smoke.set_defaults(func=cmd_rollout)

    p_stats = sub.add_parser("stats", help="Aggregate metrics from an experiment directory")
    p_stats.add_argument("--experiment", required=True)
    p_stats.add_argument("--output-dir")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
