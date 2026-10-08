"""公共 Agentic RL 数据与启动命令入口。

该 runner 只放框架无关的公共工作：构建 public-view 数据、导出 Swift
Gym-env 数据、写出 Swift/veRL 启动脚本。具体方法逻辑仍由 method 目录
决定，例如 ``rl_grpo``、``experience_evo_rl`` 或后续 ``qnr_rl``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from terrabox.evolution.experience_evo_rl.data_builder import (
    DEFAULT_STORE,
    DEFAULT_TOOL_CATALOG,
    DEFAULT_TRAIN_DATA,
    prepare_dataset,
)

from .data import swift_rows_from_verl_rows, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_ROOT = REPO_ROOT / "tmp/agent_rl_runs"
DEFAULT_SWIFT_DIR = Path("/data1/yuhongjie2/ms-swift")
DEFAULT_VERL_DIR = Path("/data1/yuhongjie2/verl")
DEFAULT_MODEL_PATH = Path("/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct")
DEFAULT_LOCAL_PY_DEPS = REPO_ROOT / "tmp/python_deps/unsloth"
DEFAULT_OEA_TEST_TASKS = REPO_ROOT / "data/oea_full_sft/openearth_test_tasks.json"


def _method_dir(method: str, experiment: str, run_root: str | Path = DEFAULT_RUN_ROOT) -> Path:
    return Path(run_root) / method / experiment


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _default_top_k(method: str) -> int:
    return 5 if method == "experience_evo_rl" else 0


def _write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    """Return a stable digest for a small configuration file in a manifest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shell_join_multiline(cmd: list[str]) -> str:
    return shlex.join(cmd)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    exp_dir = _method_dir(args.method, args.experiment, args.run_root)
    data_dir = exp_dir / "data"
    verl_public_dir = data_dir / "verl_public"
    swift_dir = data_dir / "swift"
    metrics_dir = exp_dir / "metrics"
    artifact_root = Path(args.artifact_root or exp_dir / "artifacts")
    prompt_top_k = args.prompt_top_k if args.prompt_top_k is not None else _default_top_k(args.method)
    reward_top_k = args.reward_top_k if args.reward_top_k is not None else _default_top_k(args.method)
    top_k = args.top_k if args.top_k is not None else max(prompt_top_k, reward_top_k)

    stats = prepare_dataset(
        task_file=args.task_file,
        tool_catalog_file=args.tool_catalog,
        store_dir=args.store_dir,
        output_dir=verl_public_dir,
        limit=args.limit,
        train_ratio=args.train_ratio,
        top_k=top_k,
        prompt_top_k=prompt_top_k,
        reward_top_k=reward_top_k,
        parquet=True,
        online=True,
        artifact_root=artifact_root,
    )

    train_rows = _read_jsonl(verl_public_dir / "train.jsonl")
    val_rows = _read_jsonl(verl_public_dir / "val.jsonl")
    swift_train = swift_rows_from_verl_rows(
        train_rows,
        method=args.method,
        tool_catalog_path=args.tool_catalog,
        artifact_root=artifact_root,
        trace_path=metrics_dir / "online_episode_traces.jsonl",
        raw_observation_trace_path=metrics_dir / "raw_tool_observations.jsonl",
        max_turns=args.max_turns,
    )
    swift_val = swift_rows_from_verl_rows(
        val_rows,
        method=args.method,
        tool_catalog_path=args.tool_catalog,
        artifact_root=artifact_root,
        trace_path=metrics_dir / "online_episode_traces.jsonl",
        raw_observation_trace_path=metrics_dir / "raw_tool_observations.jsonl",
        max_turns=args.max_turns,
    )
    write_jsonl(swift_train, swift_dir / "train.jsonl")
    write_jsonl(swift_val, swift_dir / "val.jsonl")

    manifest = {
        "method": args.method,
        "experiment": args.experiment,
        "run_dir": str(exp_dir),
        "framework_data": {
            "verl_train_jsonl": str(verl_public_dir / "train.jsonl"),
            "verl_train_parquet": str(verl_public_dir / "train.parquet"),
            "verl_val_jsonl": str(verl_public_dir / "val.jsonl"),
            "verl_val_parquet": str(verl_public_dir / "val.parquet"),
            "swift_train_jsonl": str(swift_dir / "train.jsonl"),
            "swift_val_jsonl": str(swift_dir / "val.jsonl"),
        },
        "artifact_root": str(artifact_root),
        "metrics_dir": str(metrics_dir),
        "top_k": top_k,
        "prompt_top_k": prompt_top_k,
        "reward_top_k": reward_top_k,
        "max_turns": args.max_turns,
        "dataset_stats": stats,
    }
    _write_manifest(exp_dir / "configs" / "canonical_run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _swift_train_command(args: argparse.Namespace, *, exp_dir: Path) -> list[str]:
    data_dir = exp_dir / "data" / "swift"
    output_dir = Path(args.output_dir or exp_dir / "checkpoints" / "swift")
    dataset_arg = f"{data_dir / 'train.jsonl'}"
    cmd = [
        str(args.python_executable),
        "-m",
        "swift.cli.main",
        "rlhf",
        "--rlhf_type",
        "grpo",
        "--advantage_estimator",
        str(args.advantage_estimator),
        "--model",
        str(args.model_path),
        "--model_type",
        args.model_type,
        "--template",
        args.template,
        "--tuner_type",
        "lora",
        "--external_plugins",
        str(REPO_ROOT / "src/terrabox/evolution/agent_rl/adapters/swift.py"),
        "--use_vllm",
        "true",
        "--vllm_mode",
        args.vllm_mode,
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_max_model_len",
        str(args.vllm_max_model_len),
        "--vllm_max_num_seqs",
        str(args.vllm_max_num_seqs),
        "--vllm_enforce_eager",
        str(args.vllm_enforce_eager).lower(),
        "--no_vllm_enable_prefix_caching",
        "--vllm_server_pass_dataset",
        "true",
        "--multi_turn_scheduler",
        "gym_scheduler",
        "--gym_env",
        "terrabox_oea",
        "--use_gym_env",
        "true",
        "--dataset",
        dataset_arg,
        "--split_dataset_ratio",
        "0",
        "--max_length",
        str(args.max_length),
        "--max_completion_length",
        str(args.max_completion_length),
        "--max_turns",
        str(args.max_turns),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        args.learning_rate,
        "--optim",
        args.optim,
        "--num_generations",
        str(args.num_generations),
        "--steps_per_generation",
        str(args.steps_per_generation),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--no_dataloader_persistent_workers",
        "--dataset_num_proc",
        str(args.dataset_num_proc),
        "--local_rollout_forward_batch_size",
        str(args.local_rollout_forward_batch_size),
        "--gradient_checkpointing_kwargs",
        '{"use_reentrant": false}',
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--logging_steps",
        str(args.logging_steps),
        "--output_dir",
        str(output_dir),
        "--report_to",
        "tensorboard",
        "--log_completions",
        str(args.log_completions).lower(),
    ]
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    if args.vllm_mode == "server":
        cmd.extend(["--vllm_server_host", args.vllm_server_host, "--vllm_server_port", str(args.vllm_server_port)])
    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])
    if args.resume_only_model:
        cmd.extend(["--resume_only_model", "true"])
    if args.deepspeed and args.deepspeed.lower() not in {"none", "null", "false", "0"}:
        cmd.extend(["--deepspeed", args.deepspeed])
    return cmd


def cmd_write_swift_command(args: argparse.Namespace) -> None:
    exp_dir = _method_dir(args.method, args.experiment, args.run_root)
    if not (exp_dir / "data" / "swift" / "train.jsonl").exists():
        raise FileNotFoundError(f"缺少 Swift 训练数据，请先运行 prepare-data: {exp_dir / 'data' / 'swift' / 'train.jsonl'}")
    script = exp_dir / "configs" / "run_swift_grpo_command.sh"
    status_script = exp_dir / "configs" / "run_swift_grpo_with_status.sh"
    cmd = _swift_train_command(args, exp_dir=exp_dir)
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"OUT={shlex.quote(str(output_dir))}",
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}",
        f"export NPROC_PER_NODE=${{NPROC_PER_NODE:-{args.nproc_per_node}}}",
        f"export PYTHONPATH={shlex.quote(str(args.local_py_deps))}:{shlex.quote(str(REPO_ROOT / 'src'))}:{shlex.quote(str(args.swift_dir))}:${{PYTHONPATH:-}}",
        f"export HF_HOME=${{HF_HOME:-{shlex.quote(str(REPO_ROOT / 'tmp/hf_home'))}}}",
        f"export HF_DATASETS_CACHE=${{HF_DATASETS_CACHE:-{shlex.quote(str(REPO_ROOT / 'tmp/hf_datasets_cache'))}}}",
        "export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}",
        "export WANDB_DISABLED=${WANDB_DISABLED:-true}",
        "export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}",
        "export TERRABOX_SWIFT_DATASETS_COMPAT=${TERRABOX_SWIFT_DATASETS_COMPAT:-1}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}",
        "export TERRABOX_USE_DOCKER=true",
        "export no_proxy=localhost,127.0.0.1",
        "export TERRABOX_SERVICE_CALL_LOCKS=1",
        f"export TERRABOX_SERVICE_LOCK_DIR={shlex.quote(str(REPO_ROOT / 'tmp/service_locks'))}",
        # Online GRPO backpropagates through the full multi-turn prompt.  The
        # compact OEA tool catalog is safe, but raw tool observations can push
        # 24GB cards into OOM after one or two turns.  Keep a reproducible
        # default that preserves key artifact/error lines while leaving room
        # for GRPO old-policy/actor backward passes.
        "export TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS=${TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS:-1024}",
        "export TERRABOX_ONLINE_TOOL_COMPACT_THRESHOLD_CHARS=${TERRABOX_ONLINE_TOOL_COMPACT_THRESHOLD_CHARS:-900}",
        f"export TERRABOX_RESOURCE_MONITOR_PATH={shlex.quote(str(exp_dir / 'metrics' / 'resource_monitor.jsonl'))}",
        "export VLLM_USE_V1=${VLLM_USE_V1:-0}",
        "export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}",
        "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
        "",
        shlex.join(cmd),
        "",
    ]
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(env_lines), encoding="utf-8")
    script.chmod(0o755)
    status_lines = [
        "#!/usr/bin/env bash",
        "set -o pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"LOG={shlex.quote(str(exp_dir / 'train_swift_tmux.log'))}",
        f"RESOURCE_LOG={shlex.quote(str(exp_dir / 'metrics' / 'resource_monitor.jsonl'))}",
        "mkdir -p \"$(dirname \"$LOG\")\" \"$(dirname \"$RESOURCE_LOG\")\"",
        "resource_monitor() {",
        "  while true; do",
        "    ts=$(date -Is)",
        "    mem_line=$(free -m | awk '/^Mem:/ {print \"\\\"mem_total_mb\\\":\"$2\",\\\"mem_used_mb\\\":\"$3\",\\\"mem_available_mb\\\":\"$7}')",
        "    gpu_line=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{printf \"[\"} {if(NR>1)printf \",\"; printf \"{\\\"index\\\":%s,\\\"memory_used_mb\\\":%s,\\\"memory_total_mb\\\":%s,\\\"utilization_gpu\\\":%s}\", $1,$2,$3,$4} END{printf \"]\"}')",
        "    proc_line=$(ps -eo rss,cmd 2>/dev/null | awk '/swift|rlhf.py|vllm|terrabox.evolution/ && !/awk/ {rss+=$1; n+=1} END{printf \"\\\"matched_processes\\\":%d,\\\"matched_rss_mb\\\":%.1f\", n, rss/1024}')",
        "    printf '{\"time\":\"%s\",%s,%s,\"gpus\":%s}\n' \"$ts\" \"${mem_line:-\\\"mem_total_mb\\\":null,\\\"mem_used_mb\\\":null,\\\"mem_available_mb\\\":null}\" \"${proc_line:-\\\"matched_processes\\\":0,\\\"matched_rss_mb\\\":0}\" \"${gpu_line:-[]}\" >> \"$RESOURCE_LOG\"",
        "    sleep \"${TERRABOX_RESOURCE_MONITOR_INTERVAL_SECONDS:-30}\"",
        "  done",
        "}",
        "resource_monitor &",
        "MONITOR_PID=$!",
        "trap 'kill \"$MONITOR_PID\" 2>/dev/null || true' EXIT",
        "{",
        "  echo \"===== SWIFT GRPO START $(date -Is) =====\"",
        f"  bash {shlex.quote(str(script))}",
        "  status=$?",
        "  echo \"===== SWIFT GRPO EXIT $status $(date -Is) =====\"",
        "  exit \"$status\"",
        "} 2>&1 | tee -a \"$LOG\"",
        "",
    ]
    status_script.write_text("\n".join(status_lines), encoding="utf-8")
    status_script.chmod(0o755)
    payload = {
        "script": str(script),
        "status_script": str(status_script),
        "command": cmd,
        "algorithm": {
            "trainer": "grpo",
            "advantage_estimator": args.advantage_estimator,
            "note": "RLOO/REINFORCE++ are Swift GRPO trainer advantage-estimator variants, not separate trainer classes.",
        },
        "launch": bool(args.launch),
    }
    _write_manifest(exp_dir / "configs" / "swift_command_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.launch:
        subprocess.run(["bash", str(script)], check=True)


def cmd_write_verl_command(args: argparse.Namespace) -> None:
    from terrabox.evolution.experience_evo_rl.runner import build_grpo_command, write_online_configs

    exp_dir = _method_dir(args.method, args.experiment, args.run_root)
    train_file = exp_dir / "data" / "verl_public" / "train.parquet"
    val_file = exp_dir / "data" / "verl_public" / "val.parquet"
    if not train_file.exists():
        raise FileNotFoundError(f"缺少 veRL 训练数据，请先运行 prepare-data: {train_file}")
    config_dir = exp_dir / "configs" / "verl"
    online_configs = write_online_configs(tool_catalog=args.tool_catalog, output_dir=config_dir)
    metrics_dir = exp_dir / "metrics"
    cmd = build_grpo_command(
        train_file=train_file.resolve(),
        val_file=val_file.resolve(),
        model_path=args.model_path,
        output_dir=exp_dir / "checkpoints" / "verl",
        reward_path=REPO_ROOT / "src/terrabox/evolution/experience_evo_rl/reward_fn.py",
        n_gpus=args.n_gpus,
        max_steps=args.max_steps,
        rollout_tensor_parallel_size=args.rollout_tensor_parallel_size,
        train_batch_size=args.train_batch_size,
        val_max_samples=args.val_max_samples,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        ray_num_cpus=args.ray_num_cpus,
        ray_object_store_memory_gib=args.ray_object_store_memory_gib,
        actor_param_offload=args.actor_param_offload,
        min_available_memory_gib=args.min_available_memory_gib,
        val_before_train=args.val_before_train,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        rollout_n=args.rollout_n,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        rollout_max_model_len=args.rollout_max_model_len,
        rollout_max_num_batched_tokens=args.rollout_max_num_batched_tokens,
        actor_ppo_max_token_len_per_gpu=args.actor_ppo_max_token_len_per_gpu,
        rollout_log_prob_max_token_len_per_gpu=args.rollout_log_prob_max_token_len_per_gpu,
        ref_log_prob_max_token_len_per_gpu=args.ref_log_prob_max_token_len_per_gpu,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        resume_mode=args.resume_mode,
        resume_from_path=args.resume_from_path,
        reward_trace_path=metrics_dir / "reward_traces.jsonl",
        online=True,
        tool_config_path=online_configs["tool_config"],
        agent_loop_config_path=online_configs["agent_loop_config"],
        online_max_turns=args.online_max_turns,
        online_max_tool_response_length=args.online_max_tool_response_length,
        python_executable=sys.executable,
    )
    script = exp_dir / "configs" / "run_verl_grpo_command.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(args.verl_dir))}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        f"export PYTHONPATH={shlex.quote(str(REPO_ROOT / 'src'))}:{shlex.quote(str(args.verl_dir))}:${{PYTHONPATH:-}}\n"
        "export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}\n"
        "export WANDB_DISABLED=${WANDB_DISABLED:-true}\n"
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}\n"
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}\n"
        "export VLLM_USE_V1=0\n"
        "unset PYTORCH_CUDA_ALLOC_CONF\n"
        "export TERRABOX_USE_DOCKER=true\n"
        "export no_proxy=localhost,127.0.0.1\n"
        "export TERRABOX_SERVICE_CALL_LOCKS=1\n"
        f"export TERRABOX_SERVICE_LOCK_DIR={shlex.quote(str(REPO_ROOT / 'tmp/service_locks'))}\n"
        f"export TERRABOX_ONLINE_RL_TRACE_PATH={shlex.quote(str(metrics_dir / 'online_episode_traces.jsonl'))}\n"
        f"export TERRABOX_ONLINE_RAW_OBSERVATION_TRACE_PATH={shlex.quote(str(metrics_dir / 'raw_tool_observations.jsonl'))}\n"
        f"export TERRABOX_EXPEVO_RL_REWARD_TRACE_PATH={shlex.quote(str(metrics_dir / 'reward_traces.jsonl'))}\n"
        "export TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS=${TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS:-8192}\n\n"
        + shlex.join(cmd)
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    payload = {"script": str(script), "command": cmd, "online_configs": online_configs, "launch": bool(args.launch)}
    _write_manifest(exp_dir / "configs" / "verl_command_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.launch:
        subprocess.run(["bash", str(script)], cwd=args.verl_dir, check=True)


def _module_status(module: str, python_executable: str, pythonpath: str) -> dict[str, Any]:
    code = (
        "import importlib, json\n"
        f"name={module!r}\n"
        "try:\n"
        "    mod=importlib.import_module(name)\n"
        "    print(json.dumps({'ok': True, 'version': getattr(mod, '__version__', '')}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': repr(e)}))\n"
    )
    env = {"PYTHONPATH": pythonpath}
    result = subprocess.run([python_executable, "-c", code], env=env, text=True, capture_output=True)
    last = (result.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        payload = json.loads(last[0])
    except json.JSONDecodeError:
        payload = {"ok": False, "error": (result.stderr or result.stdout or "unknown import error")[-1000:]}
    payload["module"] = module
    return payload


def cmd_check_env(args: argparse.Namespace) -> None:
    pythonpath = ":".join([str(REPO_ROOT / "src"), str(args.local_py_deps), str(args.swift_dir)])
    modules = ["json_repair", "dacite", "swift", "datasets", "transformers", "trl", "accelerate", "peft", "vllm", "ray"]
    if args.check_deepspeed:
        modules.append("deepspeed")
    module_results = [_module_status(module, args.python_executable, pythonpath) for module in modules]
    swift_help = subprocess.run(
        [args.python_executable, "-m", "swift.cli.main", "rlhf", "--help"],
        env={"PYTHONPATH": pythonpath},
        text=True,
        capture_output=True,
    )
    key_params = [
        "--rlhf_type",
        "--external_plugins",
        "--use_vllm",
        "--vllm_mode",
        "--vllm_server_pass_dataset",
        "--multi_turn_scheduler",
        "--gym_env",
        "--use_gym_env",
        "--max_turns",
        "--num_generations",
    ]
    help_text = (swift_help.stdout or "") + (swift_help.stderr or "")
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    payload = {
        "python_executable": args.python_executable,
        "pythonpath": pythonpath,
        "modules": module_results,
        "swift_rlhf_help_ok": swift_help.returncode == 0,
        "swift_key_params_ok": {param: param in help_text for param in key_params},
        "nvidia_smi_ok": nvidia.returncode == 0,
        "nvidia_smi_output": (nvidia.stdout if nvidia.returncode == 0 else nvidia.stderr).strip()[:4000],
        "notes": [
            "默认 Swift 脚本不传 --deepspeed，避免当前环境缺 deepspeed 时无法启动。",
            "如需 ZeRO2，先安装 deepspeed 且显式传 --deepspeed zero2。",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_export_swift_adapter(args: argparse.Namespace) -> None:
    """把 Swift/PEFT LoRA checkpoint 合并成 ReAct/vLLM 可直接加载的 HF 模型目录。"""
    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "adapter_config.json").exists():
        raise FileNotFoundError(f"不是有效 Swift/PEFT adapter checkpoint: {checkpoint}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"输出目录已存在且非空: {output_dir}; 如需覆盖写入请传 --exist-ok")

    pythonpath = ":".join([
        str(args.local_py_deps),
        str(REPO_ROOT / "src"),
        str(args.swift_dir),
        os.environ.get("PYTHONPATH", ""),
    ])
    cmd = [
        str(args.python_executable),
        str(Path(args.swift_dir) / "swift/cli/export.py"),
        "--model",
        str(args.model_path),
        "--model_type",
        args.model_type,
        "--template",
        args.template,
        "--adapters",
        str(checkpoint),
        "--merge_lora",
        "true",
        "--safe_serialization",
        "true",
        "--max_shard_size",
        args.max_shard_size,
        "--output_dir",
        str(output_dir),
    ]
    if args.exist_ok:
        cmd.extend(["--exist_ok", "true"])
    script = output_dir.parent / f"export_{checkpoint.name}.sh"
    log_path = output_dir.parent / f"export_{checkpoint.name}.log"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}",
        "export TERRABOX_SWIFT_DATASETS_COMPAT=${TERRABOX_SWIFT_DATASETS_COMPAT:-1}",
        f"export PYTHONPATH={shlex.quote(pythonpath)}",
        shlex.join(cmd) + f" 2>&1 | tee {shlex.quote(str(log_path))}",
        "",
    ]
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o755)
    manifest = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "script": str(script),
        "log": str(log_path),
        "command": cmd,
        "model_path": str(args.model_path),
        "note": "合并后目录可作为 AGENT_LLM_MODEL_PATH 供 OEA real-tool rollout 使用。",
    }
    _write_manifest(output_dir.parent / "export_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.launch:
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, check=True)


def _oea_rollout_command(args: argparse.Namespace, *, gpu_class: str, workers: int) -> list[str]:
    cmd = [
        str(args.python_executable),
        "scripts/run_trajectory_experiment.py",
        "rollout",
        "--task-file",
        str(args.task_file),
        "--experiment",
        args.eval_experiment,
        "--mode",
        args.mode,
        "--output-dir",
        str(args.output_dir),
        "--port",
        str(args.port),
        "--max-iterations",
        str(args.max_iterations),
        "--workers",
        str(workers),
        "--tool-protocol",
        str(args.tool_protocol),
        "--agent-context-length",
        str(args.agent_context_length),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--gpu-class",
        gpu_class,
        "--use-docker",
        "--resume",
        "--no-skip-osm",
        "--no-skip-bing",
        "--no-skip-vlm",
        "--no-skip-changeos",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.tool_protocol == "sft-json":
        cmd.extend(["--sft-system-prompt-file", str(args.sft_system_prompt_file)])
    if args.no_restrict_tools:
        cmd.append("--no-restrict-tools")
    return cmd


def cmd_write_oea_eval_command(args: argparse.Namespace) -> None:
    """写出并可启动固定 OEA test 的真实工具评测脚本。"""
    model_path = Path(args.model_path).resolve()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"model_path 需为可服务的 HF 模型目录: {model_path}")
    if args.tool_protocol == "sft-json":
        if not args.sft_system_prompt_file:
            raise ValueError(
                "--tool-protocol sft-json requires --sft-system-prompt-file "
                "(the prompt must match the SFT training catalog)"
            )
        sft_system_prompt_file = Path(args.sft_system_prompt_file).expanduser().resolve()
        if not sft_system_prompt_file.is_file():
            raise FileNotFoundError(
                f"SFT system prompt file does not exist: {sft_system_prompt_file}"
            )
        # Use the canonical path in both generated commands and the manifest;
        # a watcher may be launched from a different working directory later.
        args.sft_system_prompt_file = str(sft_system_prompt_file)
    else:
        if args.sft_system_prompt_file:
            raise ValueError(
                "--sft-system-prompt-file is only valid with --tool-protocol sft-json"
            )
        sft_system_prompt_file = None
    if args.agent_context_length > args.agent_model_len:
        raise ValueError(
            "--agent-context-length cannot exceed --agent-model-len; otherwise "
            "the runner can request a context the vLLM service cannot serve"
        )
    if args.agent_model_len <= 0 or args.agent_context_length <= 0:
        raise ValueError("agent model/context lengths must be positive")
    if args.max_completion_tokens <= 0 or args.max_completion_tokens >= args.agent_context_length:
        raise ValueError(
            "max completion tokens must be positive and smaller than the agent context"
        )
    output_dir = Path(args.output_dir).resolve()
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    gpu_cmd = _oea_rollout_command(args, gpu_class="gpu", workers=args.gpu_workers)
    nogpu_cmd = _oea_rollout_command(args, gpu_class="nogpu", workers=args.nogpu_workers)
    stats_cmd = [
        str(args.python_executable),
        "scripts/run_trajectory_experiment.py",
        "stats",
        "--trajectory-dir",
        str(output_dir),
    ]
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"OUT={shlex.quote(str(output_dir))}",
        f"export PYTHONPATH={shlex.quote(str(REPO_ROOT / 'src'))}:${{PYTHONPATH:-}}",
        f"export AGENT_LLM_MODEL_PATH={shlex.quote(str(model_path))}",
        f"export AGENT_LLM_GPU_DEVICES=${{AGENT_LLM_GPU_DEVICES:-{args.agent_gpu}}}",
        f"export AGENT_LLM_MAX_MODEL_LEN=${{AGENT_LLM_MAX_MODEL_LEN:-{args.agent_model_len}}}",
        f"export AGENT_LLM_MODEL_LEN=${{AGENT_LLM_MODEL_LEN:-{args.agent_model_len}}}",
        f"export AGENT_LLM_GPU_MEMORY_UTILIZATION=${{AGENT_LLM_GPU_MEMORY_UTILIZATION:-{args.agent_gpu_memory_utilization}}}",
        f"export VLM_GPU_DEVICES=${{VLM_GPU_DEVICES:-{args.vlm_gpu}}}",
        f"export SAM2_GPU_DEVICES=${{SAM2_GPU_DEVICES:-{args.tool_gpu}}}",
        f"export REMOTESAM_GPU_DEVICES=${{REMOTESAM_GPU_DEVICES:-{args.tool_gpu}}}",
        f"export STRIP_RCNN_GPU_DEVICES=${{STRIP_RCNN_GPU_DEVICES:-{args.tool_gpu}}}",
        f"export INSTRUCTSAM_GPU_DEVICES=${{INSTRUCTSAM_GPU_DEVICES:-{args.tool_gpu}}}",
        f"export REMOTECLIP_GPU_DEVICES=${{REMOTECLIP_GPU_DEVICES:-{args.tool_gpu}}}",
        f"export CHANGEOS_GPU_DEVICES=${{CHANGEOS_GPU_DEVICES:-{args.tool_gpu}}}",
        "export TERRABOX_USE_DOCKER=true",
        "export TERRABOX_TOOL_SERVICE_SCOPE=call",
        "export TERRABOX_KEEP_VLM_WARM=1",
        "export TERRABOX_SERVICE_CALL_LOCKS=1",
        f"export TERRABOX_SERVICE_LOCK_DIR={shlex.quote(str(REPO_ROOT / 'tmp/service_locks'))}",
        "export TERRABOX_OCR_USE_GPU=0",
        "export no_proxy=localhost,127.0.0.1",
        "export NO_PROXY=localhost,127.0.0.1",
        "if [ \"${TERRABOX_AGENT_LLM_CLEAN_START:-0}\" = \"1\" ]; then",
        f"  docker rm -f terrabox-agent-llm-{args.port} >/dev/null 2>&1 || true",
        "fi",
        "check_rollout_status() {",
        "  lane=$1",
        "  OUT_DIR=\"$OUT\" LANE=\"$lane\" "
        + shlex.join(
            [
                str(args.python_executable),
                "-c",
                "import json,os,sys; d=json.load(open(os.path.join(os.environ['OUT_DIR'],'run_status.json'))); s=d.get('status'); n=d.get('completed',0); total=d.get('manifest_total','?'); print(f'{s}: {n}/{total} after {os.environ[\"LANE\"]}'); sys.exit(0 if s == 'complete' and d.get('invocation_complete') is True else 1)",
            ]
        ),
        "}",
        "",
    ]
    script = config_dir / "run_oea_eval.sh"
    script.write_text(
        "\n".join(env_lines)
        + "echo \"===== OEA eval GPU lane start $(date -Is) =====\"\n"
        + shlex.join(gpu_cmd)
        + f" 2>&1 | tee -a {shlex.quote(str(output_dir / 'eval_gpu_lane.log'))}\n"
        + "LANE=gpu check_rollout_status gpu\n"
        + "echo \"===== OEA eval NOGPU lane start $(date -Is) =====\"\n"
        + shlex.join(nogpu_cmd)
        + f" 2>&1 | tee -a {shlex.quote(str(output_dir / 'eval_nogpu_lane.log'))}\n"
        + "LANE=nogpu check_rollout_status nogpu\n"
        + "echo \"===== OEA eval stats $(date -Is) =====\"\n"
        + shlex.join(stats_cmd)
        + f" 2>&1 | tee -a {shlex.quote(str(output_dir / 'eval_stats.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    progress_script = config_dir / "check_oea_eval_progress.sh"
    progress_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(REPO_ROOT))}",
                f"OUT={shlex.quote(str(output_dir))}",
                "echo \"===== process =====\"",
                "tmux ls 2>/dev/null | grep -E 'qwen.*eval|oea.*eval|swift.*eval' || true",
                "ps -eo pid,ppid,user,stat,etime,cmd | grep -E 'run_trajectory_experiment|AGENT_LLM_MODEL_PATH|9103' | grep -v grep || true",
                "echo \"===== files =====\"",
                "find \"$OUT/results\" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l",
                "echo \"===== rollout_report =====\"",
                "PYTHONPATH=src "
                + shlex.quote(str(args.python_executable))
                + " -m terrabox.evolution.shared.rollout_report status --results-dir \"$OUT/results\" --scope all --total 1162 2>/dev/null || true",
                "echo \"===== recent errors =====\"",
        "grep -RInE --exclude-dir=configs --exclude='check_oea_eval_progress.sh' 'Traceback|OutOfMemory|CUDA out of memory|Exception|Agent LLM container exited|HTTP/1\\.1 400|maximum context length' \"$OUT\" 2>/dev/null | tail -n 80 || true",
                "echo \"===== gpu =====\"",
                "systemd-run --user --wait --collect --pipe /bin/bash -lc '/usr/bin/nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits' 2>/dev/null || nvidia-smi || true",
                "echo \"===== log tail =====\"",
                "tail -n 80 \"$OUT/eval_gpu_lane.log\" 2>/dev/null || true",
                "tail -n 80 \"$OUT/eval_nogpu_lane.log\" 2>/dev/null || true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    progress_script.chmod(0o755)
    unit_name = "terrabox-" + "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in args.eval_experiment)[:180]
    start_script = config_dir / "start_oea_eval_systemd.sh"
    start_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(REPO_ROOT))}",
                f"UNIT={shlex.quote(unit_name)}",
                f"OUT={shlex.quote(str(output_dir))}",
                "systemctl --user stop \"$UNIT.service\" >/dev/null 2>&1 || true",
                "systemd-run --user --unit \"$UNIT\" --collect "
                + f"--working-directory={shlex.quote(str(REPO_ROOT))} "
                + "/bin/bash -lc "
                + shlex.quote(f"bash {script} >> {output_dir / 'run_oea_eval_systemd.log'} 2>&1"),
                "echo \"started $UNIT.service\"",
                "echo \"log: $OUT/run_oea_eval_systemd.log\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    start_script.chmod(0o755)
    manifest = {
        "eval_experiment": args.eval_experiment,
        "output_dir": str(output_dir),
        "task_file": str(args.task_file),
        "model_path": str(model_path),
        "tool_protocol": args.tool_protocol,
        "sft_system_prompt_file": str(sft_system_prompt_file or ""),
        "sft_system_prompt_sha256": (
            _sha256_file(sft_system_prompt_file) if sft_system_prompt_file else ""
        ),
        "agent_model_len": args.agent_model_len,
        "agent_context_length": args.agent_context_length,
        "max_completion_tokens": args.max_completion_tokens,
        "script": str(script),
        "progress_script": str(progress_script),
        "start_script": str(start_script),
        "systemd_unit": unit_name + ".service",
        "lanes": {
            "gpu": {"gpu_class": "gpu", "workers": args.gpu_workers, "command": gpu_cmd},
            "nogpu": {"gpu_class": "nogpu", "workers": args.nogpu_workers, "command": nogpu_cmd},
        },
        "metric_note": "这是固定 OEA test 真实工具 rollout；完成后再跑 stats / answer judge 生成正式主表。",
    }
    _write_manifest(config_dir / "oea_eval_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.launch:
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrabox Agentic RL public adapter runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("prepare-data", help="构建 veRL/Swift 共用 public-view 在线 RL 数据")
    p_data.add_argument("--method", required=True, choices=["rl_grpo", "experience_evo_rl", "qnr_rl", "rewardevo_rl"])
    p_data.add_argument("--experiment", required=True)
    p_data.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p_data.add_argument("--task-file", default=str(DEFAULT_TRAIN_DATA))
    p_data.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    p_data.add_argument("--store-dir", default=str(DEFAULT_STORE))
    p_data.add_argument("--limit", type=int)
    p_data.add_argument("--train-ratio", type=float, default=0.95)
    p_data.add_argument("--top-k", type=int)
    p_data.add_argument("--prompt-top-k", type=int)
    p_data.add_argument("--reward-top-k", type=int)
    p_data.add_argument("--max-turns", type=int, default=12)
    p_data.add_argument("--artifact-root")
    p_data.set_defaults(func=cmd_prepare_data)

    p_swift = sub.add_parser("write-swift-command", help="写出低资源 MS-Swift GRPO/RLOO 启动脚本")
    p_swift.add_argument("--method", required=True, choices=["rl_grpo", "experience_evo_rl", "qnr_rl", "rewardevo_rl"])
    p_swift.add_argument("--experiment", required=True)
    p_swift.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p_swift.add_argument("--swift-dir", default=str(DEFAULT_SWIFT_DIR))
    p_swift.add_argument("--local-py-deps", default=str(DEFAULT_LOCAL_PY_DEPS))
    p_swift.add_argument("--python-executable", default=sys.executable)
    p_swift.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_swift.add_argument("--model-type", default="qwen2")
    p_swift.add_argument("--template", default="qwen2_5")
    p_swift.add_argument("--output-dir")
    p_swift.add_argument("--cuda-visible-devices", default="2")
    p_swift.add_argument("--nproc-per-node", type=int, default=1)
    p_swift.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    p_swift.add_argument("--vllm-server-host", default="127.0.0.1")
    p_swift.add_argument("--vllm-server-port", type=int, default=8000)
    p_swift.add_argument("--max-length", type=int, default=8192)
    p_swift.add_argument("--max-completion-length", type=int, default=2048)
    p_swift.add_argument("--max-turns", type=int, default=12)
    p_swift.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    p_swift.add_argument("--vllm-max-model-len", type=int, default=8192)
    p_swift.add_argument("--vllm-max-num-seqs", type=int, default=1)
    p_swift.add_argument("--vllm-enforce-eager", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p_swift.add_argument("--num-train-epochs", type=float, default=1.0)
    p_swift.add_argument("--max-steps", type=int)
    p_swift.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_swift.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p_swift.add_argument("--learning-rate", default="1e-6")
    p_swift.add_argument(
        "--optim",
        default="adamw_torch",
        help="Swift/HF optimizer；默认 adamw_torch，避免 fused Adam 在 checkpoint resume 时出现 dtype/device/layout mismatch。",
    )
    p_swift.add_argument("--num-generations", type=int, default=2)
    p_swift.add_argument(
        "--advantage-estimator",
        choices=["grpo", "rloo", "reinforce_plus_plus"],
        default="grpo",
        help="Swift GRPO trainer 的优势估计器；rloo/reinforce_plus_plus 仍使用同一真实工具 gym rollout。",
    )
    p_swift.add_argument("--steps-per-generation", type=int, default=4)
    p_swift.add_argument("--temperature", type=float, default=1.0)
    p_swift.add_argument("--top-p", type=float, default=1.0)
    p_swift.add_argument(
        "--deepspeed",
        default="none",
        help="Swift 训练的 deepspeed 配置；默认 none，避免当前环境缺 deepspeed 时不能启动。资源充足且依赖完整时可显式传 zero2。",
    )
    p_swift.add_argument("--dataloader-num-workers", type=int, default=0)
    p_swift.add_argument("--dataset-num-proc", type=int, default=1)
    p_swift.add_argument(
        "--local-rollout-forward-batch-size",
        type=int,
        default=1,
        help="Swift/TRL 计算 rollout logprob 的本地前向 micro-batch；单卡 24GB 多轮 GRPO 默认 1，避免默认 64 带来的显存峰值。",
    )
    p_swift.add_argument("--save-steps", type=int, default=50)
    p_swift.add_argument("--save-total-limit", type=int, default=2)
    p_swift.add_argument("--logging-steps", type=int, default=1)
    p_swift.add_argument(
        "--log-completions",
        action="store_true",
        help="额外启用 Swift completions.jsonl；默认关闭，因为 Terrabox 已保存 episode/raw observation JSONL，关闭可降低 CPU 内存与 I/O 压力。",
    )
    p_swift.add_argument("--resume-from-checkpoint")
    p_swift.add_argument(
        "--resume-only-model",
        action="store_true",
        help="仅从 checkpoint 加载模型/adapter 权重，不恢复 optimizer/scheduler；用于 optimizer state 不兼容时的受控续跑。",
    )
    p_swift.add_argument("--launch", action="store_true")
    p_swift.set_defaults(func=cmd_write_swift_command)

    p_verl = sub.add_parser("write-verl-command", help="写出 veRL online GRPO 启动脚本")
    p_verl.add_argument("--method", required=True, choices=["rl_grpo", "experience_evo_rl", "qnr_rl", "rewardevo_rl"])
    p_verl.add_argument("--experiment", required=True)
    p_verl.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p_verl.add_argument("--verl-dir", default=str(DEFAULT_VERL_DIR))
    p_verl.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_verl.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    p_verl.add_argument("--cuda-visible-devices", default="2,3")
    p_verl.add_argument("--n-gpus", type=int, default=2)
    p_verl.add_argument("--max-steps", type=int)
    p_verl.add_argument("--rollout-tensor-parallel-size", type=int, default=2)
    p_verl.add_argument("--train-batch-size", type=int, default=1)
    p_verl.add_argument("--val-max-samples", type=int, default=8)
    p_verl.add_argument("--val-batch-size", type=int, default=1)
    p_verl.add_argument("--dataloader-num-workers", type=int, default=0)
    p_verl.add_argument("--ray-num-cpus", type=int, default=8)
    p_verl.add_argument("--ray-object-store-memory-gib", type=int, default=4)
    p_verl.add_argument("--actor-param-offload", action="store_true")
    p_verl.add_argument("--min-available-memory-gib", type=int, default=64)
    p_verl.add_argument("--val-before-train", action="store_true")
    p_verl.add_argument("--max-prompt-length", type=int, default=6144)
    p_verl.add_argument("--max-response-length", type=int, default=2048)
    p_verl.add_argument("--rollout-n", type=int, default=2)
    p_verl.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.15)
    p_verl.add_argument("--rollout-max-model-len", type=int, default=8192)
    p_verl.add_argument("--rollout-max-num-batched-tokens", type=int, default=8192)
    p_verl.add_argument("--actor-ppo-max-token-len-per-gpu", type=int, default=8192)
    p_verl.add_argument("--rollout-log-prob-max-token-len-per-gpu", type=int, default=8192)
    p_verl.add_argument("--ref-log-prob-max-token-len-per-gpu", type=int, default=8192)
    p_verl.add_argument("--save-freq", type=int, default=50)
    p_verl.add_argument("--test-freq", type=int, default=100)
    p_verl.add_argument("--resume-mode", choices=["disable", "auto", "resume_path"], default="disable")
    p_verl.add_argument("--resume-from-path")
    p_verl.add_argument("--online-max-turns", type=int, default=12)
    p_verl.add_argument("--online-max-tool-response-length", type=int, default=6144)
    p_verl.add_argument("--launch", action="store_true")
    p_verl.set_defaults(func=cmd_write_verl_command)

    p_check = sub.add_parser("check-env", help="检查 Swift/veRL 训练前依赖、参数和 GPU 可见性")
    p_check.add_argument("--swift-dir", default=str(DEFAULT_SWIFT_DIR))
    p_check.add_argument("--local-py-deps", default=str(DEFAULT_LOCAL_PY_DEPS))
    p_check.add_argument("--python-executable", default=sys.executable)
    p_check.add_argument("--check-deepspeed", action="store_true")
    p_check.set_defaults(func=cmd_check_env)

    p_export = sub.add_parser("export-swift-adapter", help="把 Swift/PEFT LoRA checkpoint 合并成 OEA eval 可服务的 HF 模型")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output-dir", required=True)
    p_export.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_export.add_argument("--model-type", default="qwen2")
    p_export.add_argument("--template", default="qwen2_5")
    p_export.add_argument("--swift-dir", default=str(DEFAULT_SWIFT_DIR))
    p_export.add_argument("--local-py-deps", default=str(DEFAULT_LOCAL_PY_DEPS))
    p_export.add_argument("--python-executable", default=sys.executable)
    p_export.add_argument("--max-shard-size", default="2GB")
    p_export.add_argument("--cuda-visible-devices", default="3")
    p_export.add_argument("--exist-ok", action="store_true")
    p_export.add_argument("--launch", action="store_true")
    p_export.set_defaults(func=cmd_export_swift_adapter)

    p_eval = sub.add_parser("write-oea-eval-command", help="写出固定 OEA test 的真实工具 rollout 评测脚本")
    p_eval.add_argument("--eval-experiment", required=True)
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--output-dir", required=True)
    p_eval.add_argument("--task-file", default=str(DEFAULT_OEA_TEST_TASKS))
    p_eval.add_argument("--python-executable", default=sys.executable)
    p_eval.add_argument("--mode", default="standard")
    p_eval.add_argument("--port", type=int, default=9103)
    p_eval.add_argument("--max-iterations", type=int, default=15)
    p_eval.add_argument("--limit", type=int)
    p_eval.add_argument("--agent-gpu", default="3")
    p_eval.add_argument("--tool-gpu", default="1")
    p_eval.add_argument("--vlm-gpu", default="0")
    p_eval.add_argument(
        "--agent-model-len",
        type=int,
        default=24576,
        help="vLLM 服务 context 上限；默认与 Qwen2.5-3B rollout 合同保持 24576",
    )
    p_eval.add_argument(
        "--agent-context-length",
        type=int,
        default=24576,
        help="runner 使用的 agent context；必须不大于 --agent-model-len",
    )
    p_eval.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="每次 agent 请求的最大 completion；默认 4096",
    )
    p_eval.add_argument("--agent-gpu-memory-utilization", type=float, default=0.80)
    p_eval.add_argument("--gpu-workers", type=int, default=1)
    p_eval.add_argument("--nogpu-workers", type=int, default=1)
    p_eval.add_argument(
        "--tool-protocol",
        choices=["native", "sft-json"],
        default="native",
        help="模型工具协议；SFT JSON-actions checkpoint 必须显式选择 sft-json",
    )
    p_eval.add_argument(
        "--sft-system-prompt-file",
        help="sft-json 训练 system prompt.txt；必须与 checkpoint 的 SFT 数据版本一致",
    )
    p_eval.add_argument("--no-restrict-tools", action="store_true")
    p_eval.add_argument("--launch", action="store_true")
    p_eval.set_defaults(func=cmd_write_oea_eval_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
