"""CLI runner for the SFT baseline.

This module intentionally mirrors ReAct/Reflection experiment slicing. It
prepares chat SFT files for training and prompt-only task files for rollout.
The long training job is written as a shell script by default; use ``--launch``
only when the GPU plan is ready.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ..ReAct.runner import build_rollout_env
from ..reflection.data_split import load_shuffled_samples, sample_slice
from .data_adapter import write_chat_jsonl, write_eval_task_file
from .verl_backend import build_verl_sft_command, write_verl_sft_parquet


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
# Aligned, de-collapsed strict data (44 tools) — the SAME source ReAct/Reflection
# now evaluate on. (Pre-alignment data lived in data/newdata; use fixdata_decollapse
# so the SFT model is trained on the same canonical, executable tool schema.)
DEFAULT_DATA = "data/fixdata_decollapse/sft_train_strict.jsonl"
DEFAULT_MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
DEFAULT_UNSLOTH_PYTHON = "/home/yuhongjie/miniconda3/envs/unsloth/bin/python"
DEFAULT_UNSLOTH_BIN = "/home/yuhongjie/miniconda3/envs/unsloth/bin"
DEFAULT_VERL_DIR = "/data1/yuhongjie2/verl"


def apply_verl_preset(args: argparse.Namespace) -> None:
    """Apply named veRL SFT presets after argparse defaults are loaded."""
    if args.preset == "default":
        return
    if args.preset != "3090-safe":
        raise ValueError(f"Unsupported veRL SFT preset: {args.preset}")

    args.train_batch_size = 1
    args.micro_batch_size_per_gpu = 1
    args.lora_rank = 8
    args.lora_alpha = 16
    args.sequence_parallel_size = args.nproc_per_node
    args.param_offload = True
    args.optimizer_offload = True
    args.activation_offload = True
    args.use_torch_compile = False
    if args.save_freq is None:
        args.save_freq = "1000"
    if args.test_freq is None:
        args.test_freq = "-1"


def experiment_dir(experiment: str) -> Path:
    return MODULE_ROOT / "exp" / experiment


def prepare_sft_files(
    *,
    strict_data: str | Path = DEFAULT_DATA,
    output_dir: str | Path,
    seed: int = 42,
    train_start: int = 200,
    train_limit: int | None = 400,
    val_start: int = 0,
    val_limit: int | None = 200,
    compact_long_context: bool = False,
    compact_system_catalog: bool = False,
    max_observation_chars: int = 1200,
    max_string_chars: int = 800,
    max_list_items: int = 8,
) -> dict[str, object]:
    """Prepare train/val SFT JSONL plus prompt-only eval tasks."""
    out_dir = Path(output_dir)
    data_dir = out_dir / "sft_data"
    samples = load_shuffled_samples(strict_data, seed=seed)
    train_samples = sample_slice(samples, start=train_start, limit=train_limit)
    val_samples = sample_slice(samples, start=val_start, limit=val_limit)

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    eval_tasks = out_dir / "eval_tasks.json"
    train_rows = write_chat_jsonl(
        train_samples,
        train_file,
        source_path=strict_data,
        split_name="train",
        compact_long_context=compact_long_context,
        compact_system_catalog=compact_system_catalog,
        max_observation_chars=max_observation_chars,
        max_string_chars=max_string_chars,
        max_list_items=max_list_items,
    )
    val_rows = write_chat_jsonl(
        val_samples,
        val_file,
        source_path=strict_data,
        split_name="val",
        compact_long_context=compact_long_context,
        compact_system_catalog=compact_system_catalog,
        max_observation_chars=max_observation_chars,
        max_string_chars=max_string_chars,
        max_list_items=max_list_items,
    )
    eval_task_rows = write_eval_task_file(val_samples, eval_tasks, source_path=strict_data, split_name="eval")
    stats: dict[str, object] = {
        "strict_data": str(strict_data),
        "shuffle_seed": seed,
        "train_start": train_start,
        "train_limit": train_limit,
        "val_start": val_start,
        "val_limit": val_limit,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "eval_task_rows": eval_task_rows,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "eval_task_file": str(eval_tasks),
        "gold_usage_policy": "SFT train/val use messages; rollout eval uses prompt-only task file",
        "compact_long_context": compact_long_context,
        "compact_system_catalog": compact_system_catalog,
        "max_observation_chars": max_observation_chars,
        "max_string_chars": max_string_chars,
        "max_list_items": max_list_items,
    }
    (data_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def build_sft_train_command(
    *,
    train_file: str | Path,
    val_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 8192,
    save_steps: int = 50,
    save_total_limit: int = 2,
    eval_steps: int = 50,
    save_merged_model: bool = True,
    drop_overlength: bool = False,
    python_executable: str | None = None,
) -> list[str]:
    """Build the local QLoRA SFT training command."""
    cmd = [
        python_executable or DEFAULT_UNSLOTH_PYTHON,
        "-m",
        "terrabox.evolution.sft.train_lora",
        "--train-file",
        str(train_file),
        "--val-file",
        str(val_file),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--num-train-epochs",
        str(num_train_epochs),
        "--learning-rate",
        str(learning_rate),
        "--per-device-train-batch-size",
        str(per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--lora-rank",
        str(lora_rank),
        "--lora-alpha",
        str(lora_alpha),
        "--max-seq-length",
        str(max_seq_length),
        "--save-steps",
        str(save_steps),
        "--save-total-limit",
        str(save_total_limit),
        "--eval-steps",
        str(eval_steps),
    ]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if save_merged_model:
        cmd.append("--save-merged-model")
    if drop_overlength:
        cmd.append("--drop-overlength")
    return cmd


def build_sft_rollout_command(
    *,
    task_file: str | Path,
    experiment: str,
    output_dir: str | Path,
    model_path: str | Path,
    port: int = 9100,
    start_index: int | None = None,
    limit: int | None = None,
    max_iterations: int = 15,
    python_executable: str | None = None,
) -> list[str]:
    """Build a prompt-only rollout command for a trained SFT checkpoint."""
    cmd = [
        python_executable or DEFAULT_UNSLOTH_PYTHON,
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
        "--resume",
    ]
    if start_index is not None:
        cmd.extend(["--start-index", str(start_index)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def cmd_prepare_data(args: argparse.Namespace) -> None:
    stats = prepare_sft_files(
        strict_data=args.strict_data,
        output_dir=experiment_dir(args.experiment),
        seed=args.seed,
        train_start=args.train_start,
        train_limit=args.train_limit,
        val_start=args.val_start,
        val_limit=args.val_limit,
        compact_long_context=args.compact_long_context,
        compact_system_catalog=args.compact_system_catalog,
        max_observation_chars=args.max_observation_chars,
        max_string_chars=args.max_string_chars,
        max_list_items=args.max_list_items,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    data_dir = exp_dir / "sft_data"
    model_dir = Path(args.output_model_dir or MODULE_ROOT / "model" / args.experiment)
    train_file = Path(args.train_file or data_dir / "train.jsonl")
    val_file = Path(args.val_file or data_dir / "val.jsonl")
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("Missing SFT data files. Run prepare-data first.")
    cmd = build_sft_train_command(
        train_file=train_file,
        val_file=val_file,
        model_path=args.model_path,
        output_dir=model_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        save_merged_model=args.save_merged_model,
        drop_overlength=getattr(args, "drop_overlength", False),
    )
    script = exp_dir / "run_sft_train.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        "export UNSLOTH_DISABLE_STATISTICS=1\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f"export PYTHONPATH={REPO_ROOT / 'src'}:$PYTHONPATH\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(exp_dir / 'sft_train.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(f"Wrote {script}")
    if args.launch:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated script before launching SFT.")


def cmd_rollout(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    out_dir = Path(args.output_dir or exp_dir / "eval")
    task_file = Path(args.task_file or exp_dir / "eval_tasks.json")
    model_path = Path(args.model_path or MODULE_ROOT / "model" / args.experiment / "merged")
    cmd = build_sft_rollout_command(
        task_file=task_file,
        experiment=f"{args.experiment}_sft_eval",
        output_dir=out_dir,
        model_path=model_path,
        port=args.port,
        start_index=args.start_index,
        limit=args.limit,
        max_iterations=args.max_iterations,
    )
    run_script = exp_dir / "run_sft_eval.sh"
    run_script.parent.mkdir(parents=True, exist_ok=True)
    run_script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"export AGENT_LLM_MODEL_PATH={shlex.quote(str(model_path))}\n"
        "export no_proxy=localhost,127.0.0.1\n"
        "export NO_PROXY=localhost,127.0.0.1\n"
        f"export PYTHONPATH={REPO_ROOT / 'src'}:$PYTHONPATH\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(exp_dir / 'sft_eval.log'))}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    print(f"Wrote {run_script}")
    if args.launch:
        env = build_rollout_env(args.agent_gpu, args.tool_gpu)
        env["AGENT_LLM_MODEL_PATH"] = str(model_path)
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated script before launching SFT eval rollout.")


def _find_latest_global_step(model_dir: Path) -> Path | None:
    """Return the newest global_step_* checkpoint dir under a veRL model dir."""
    steps = []
    for child in model_dir.glob("global_step_*"):
        if child.is_dir():
            try:
                steps.append((int(child.name.rsplit("_", 1)[-1]), child))
            except ValueError:
                continue
    if not steps:
        return None
    return max(steps, key=lambda item: item[0])[1]


def build_convert_hf_command(
    *,
    local_dir: str | Path,
    target_dir: str | Path,
    verl_dir: str | Path = DEFAULT_VERL_DIR,
    python_executable: str | None = None,
    use_cpu_initialization: bool = True,
) -> list[str]:
    """veRL FSDP shard checkpoint → HuggingFace model (ReAct/vLLM-servable).

    Runs offline on CPU so it never triggers the inline full-gather that OOMs the
    server during training. Output is a standard HF dir (config.json + sharded
    safetensors + tokenizer) — the exact format the agent LLM Docker mounts as
    /model.
    """
    cmd = [
        python_executable or DEFAULT_UNSLOTH_PYTHON,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(target_dir),
        "--trust-remote-code",
    ]
    if use_cpu_initialization:
        cmd.append("--use_cpu_initialization")
    return cmd


def cmd_convert_hf(args: argparse.Namespace) -> None:
    """Convert a veRL FSDP checkpoint into a HuggingFace model for ReAct rollout."""
    exp_dir = experiment_dir(args.experiment)
    model_dir = Path(args.model_dir or MODULE_ROOT / "model" / f"{args.experiment}_verl")
    # Resolve the checkpoint: explicit --global-step-dir, else newest global_step_*.
    if args.global_step_dir:
        step_dir = Path(args.global_step_dir)
    else:
        latest = _find_latest_global_step(model_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No global_step_* checkpoint under {model_dir}. Pass --global-step-dir explicitly."
            )
        step_dir = latest
    # veRL stores actor weights under <global_step_x>/actor.
    local_dir = step_dir / "actor" if (step_dir / "actor").exists() else step_dir
    target_dir = Path(args.target_dir or model_dir / f"{step_dir.name}_hf")
    cmd = build_convert_hf_command(
        local_dir=local_dir,
        target_dir=target_dir,
        verl_dir=args.verl_dir,
        use_cpu_initialization=not args.no_cpu_init,
    )
    script = exp_dir / "run_convert_hf.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"export PYTHONPATH={args.verl_dir}:{REPO_ROOT / 'src'}:$PYTHONPATH\n"
        # CPU-only conversion: keep GPUs out of it so a running rollout is unaffected.
        "export CUDA_VISIBLE_DEVICES=\n"
        + " ".join(shlex.quote(str(part)) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(exp_dir / 'convert_hf.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(json.dumps({
        "checkpoint": str(step_dir),
        "local_dir": str(local_dir),
        "target_dir": str(target_dir),
        "script": str(script),
        "serve_hint": f"rollout --experiment {args.experiment} --model-path {target_dir}",
    }, ensure_ascii=False, indent=2))
    if args.launch:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{args.verl_dir}:{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated script before launching the HF conversion.")


def cmd_prepare_verl_data(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    data_dir = exp_dir / "verl_data"
    train_jsonl = Path(args.train_file or exp_dir / "sft_data" / "train.jsonl")
    val_jsonl = Path(args.val_file or exp_dir / "sft_data" / "val.jsonl")
    if not train_jsonl.exists() or not val_jsonl.exists():
        raise FileNotFoundError("Missing chat SFT JSONL files. Run prepare-data first.")
    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"
    train_rows = write_verl_sft_parquet(train_jsonl, train_parquet)
    val_rows = write_verl_sft_parquet(val_jsonl, val_parquet)
    stats = {
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "format": "verl_multiturn_sft_parquet",
        "messages_key": "messages",
    }
    (data_dir / "verl_dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_train_verl(args: argparse.Namespace) -> None:
    apply_verl_preset(args)
    if args.save_freq is None:
        args.save_freq = "after_each_epoch"
    if args.test_freq is None:
        args.test_freq = "after_each_epoch"
    exp_dir = experiment_dir(args.experiment)
    data_dir = exp_dir / "verl_data"
    train_file = Path(args.train_file or data_dir / "train.parquet")
    val_file = Path(args.val_file or data_dir / "val.parquet")
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("Missing veRL parquet files. Run prepare-verl-data first.")
    model_dir = Path(args.output_model_dir or MODULE_ROOT / "model" / f"{args.experiment}_verl")
    cmd = build_verl_sft_command(
        verl_dir=args.verl_dir,
        train_file=train_file,
        val_file=val_file,
        model_path=args.model_path,
        output_dir=model_dir,
        nproc_per_node=args.nproc_per_node,
        max_length=args.max_length,
        max_token_len_per_gpu=args.max_token_len_per_gpu,
        micro_batch_size_per_gpu=args.micro_batch_size_per_gpu,
        train_batch_size=args.train_batch_size,
        total_epochs=args.total_epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        sequence_parallel_size=args.sequence_parallel_size,
        param_offload=args.param_offload,
        optimizer_offload=args.optimizer_offload,
        activation_offload=args.activation_offload,
        use_torch_compile=args.use_torch_compile,
        save_hf_model=args.save_hf_model,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        max_ckpt_to_keep=args.max_ckpt_to_keep,
    )
    script = exp_dir / "run_verl_sft_train.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        f"export PATH={DEFAULT_UNSLOTH_BIN}:$PATH\n"
        f"export PYTHONPATH={args.verl_dir}:{REPO_ROOT / 'src'}:$PYTHONPATH\n"
        "export TOKENIZERS_PARALLELISM=true\n"
        "export HYDRA_FULL_ERROR=1\n"
        "export NCCL_DEBUG=WARN\n"
        "export NCCL_IB_DISABLE=1\n"
        "export NCCL_SOCKET_IFNAME=lo\n"
        "export NCCL_CROSS_NIC=0\n"
        "export GLOO_SOCKET_IFNAME=lo\n"
        + " ".join(shlex.quote(str(part)) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(exp_dir / 'verl_sft_train.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(f"Wrote {script}")
    if args.launch:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["PATH"] = f"{DEFAULT_UNSLOTH_BIN}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{args.verl_dir}:{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        env["TOKENIZERS_PARALLELISM"] = "true"
        env["HYDRA_FULL_ERROR"] = "1"
        env["NCCL_DEBUG"] = "WARN"
        env["NCCL_IB_DISABLE"] = "1"
        env["NCCL_SOCKET_IFNAME"] = "lo"
        env["NCCL_CROSS_NIC"] = "0"
        env["GLOO_SOCKET_IFNAME"] = "lo"
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated veRL SFT script before launching.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrabox SFT baseline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("prepare-data", help="Prepare fixed-shuffle SFT train/val and eval task files")
    p_data.add_argument("--strict-data", default=DEFAULT_DATA)
    p_data.add_argument("--experiment", required=True)
    p_data.add_argument("--seed", type=int, default=42)
    p_data.add_argument("--train-start", type=int, default=200)
    p_data.add_argument("--train-limit", type=int, default=400)
    p_data.add_argument("--val-start", type=int, default=0)
    p_data.add_argument("--val-limit", type=int, default=200)
    p_data.add_argument("--compact-long-context", action="store_true")
    p_data.add_argument("--compact-system-catalog", action="store_true", default=False)
    p_data.add_argument("--no-compact-system-catalog", dest="compact_system_catalog", action="store_false")
    p_data.add_argument("--max-observation-chars", type=int, default=1200)
    p_data.add_argument("--max-string-chars", type=int, default=800)
    p_data.add_argument("--max-list-items", type=int, default=8)
    p_data.set_defaults(func=cmd_prepare_data)

    p_train = sub.add_parser("train", help="Write or launch QLoRA SFT training")
    p_train.add_argument("--experiment", required=True)
    p_train.add_argument("--train-file")
    p_train.add_argument("--val-file")
    p_train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--output-model-dir")
    p_train.add_argument("--cuda-visible-devices", default="0,1")
    p_train.add_argument("--max-steps", type=int)
    p_train.add_argument("--num-train-epochs", type=float, default=1.0)
    p_train.add_argument("--learning-rate", type=float, default=2e-5)
    p_train.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--max-seq-length", type=int, default=8192)
    p_train.add_argument("--save-steps", type=int, default=50,
                         help="每 N 步保存一次 checkpoint(便于少量数据跑通+中途存档验证)")
    p_train.add_argument("--save-total-limit", type=int, default=2)
    p_train.add_argument("--eval-steps", type=int, default=50)
    p_train.add_argument("--save-merged-model", action="store_true", default=True)
    p_train.add_argument("--no-save-merged-model", dest="save_merged_model", action="store_false")
    p_train.add_argument("--drop-overlength", action="store_true",
                         help="超长行丢弃而非报错(verbatim 保留其余,与 rollout 对齐,不做有损截断)")
    p_train.add_argument("--launch", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("rollout", help="Evaluate an SFT checkpoint through real Terrabox rollout")
    p_eval.add_argument("--experiment", required=True)
    p_eval.add_argument("--task-file")
    p_eval.add_argument("--output-dir")
    p_eval.add_argument("--model-path")
    p_eval.add_argument("--port", type=int, default=9100)
    p_eval.add_argument("--agent-gpu", default=0)
    p_eval.add_argument("--tool-gpu", default=1)
    p_eval.add_argument("--start-index", type=int)
    p_eval.add_argument("--limit", type=int)
    p_eval.add_argument("--max-iterations", type=int, default=15)
    p_eval.add_argument("--launch", action="store_true")
    p_eval.set_defaults(func=cmd_rollout)

    p_verl_data = sub.add_parser("prepare-verl-data", help="Convert chat SFT JSONL to veRL Parquet")
    p_verl_data.add_argument("--experiment", required=True)
    p_verl_data.add_argument("--train-file")
    p_verl_data.add_argument("--val-file")
    p_verl_data.set_defaults(func=cmd_prepare_verl_data)

    p_verl_train = sub.add_parser("train-verl", help="Write or launch veRL FSDP SFT training")
    p_verl_train.add_argument("--experiment", required=True)
    p_verl_train.add_argument("--preset", choices=["default", "3090-safe"], default="default")
    p_verl_train.add_argument("--train-file")
    p_verl_train.add_argument("--val-file")
    p_verl_train.add_argument("--verl-dir", default=DEFAULT_VERL_DIR)
    p_verl_train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p_verl_train.add_argument("--output-model-dir")
    # 3 GPUs by default: 4-GPU FSDP tripped the breaker; 2-GPU FSDP OOMs at init
    # for an 8B model. 3 cards is the stable middle ground. (3090-safe preset sets
    # sequence_parallel_size = nproc_per_node, so SP becomes 3 too.)
    p_verl_train.add_argument("--cuda-visible-devices", default="0,1,2")
    p_verl_train.add_argument("--nproc-per-node", type=int, default=3)
    p_verl_train.add_argument("--max-length", type=int, default=13312)
    p_verl_train.add_argument("--max-token-len-per-gpu", type=int, default=13312)
    p_verl_train.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    p_verl_train.add_argument("--train-batch-size", type=int, default=4)
    p_verl_train.add_argument("--total-epochs", type=int, default=1)
    p_verl_train.add_argument("--learning-rate", type=float, default=2e-5)
    p_verl_train.add_argument("--lora-rank", type=int, default=16)
    p_verl_train.add_argument("--lora-alpha", type=int, default=32)
    p_verl_train.add_argument("--lora-targets", default="all-linear")
    p_verl_train.add_argument("--sequence-parallel-size", type=int, default=1)
    p_verl_train.add_argument("--param-offload", action="store_true", default=False)
    p_verl_train.add_argument("--no-param-offload", dest="param_offload", action="store_false")
    p_verl_train.add_argument("--optimizer-offload", action="store_true", default=True)
    p_verl_train.add_argument("--no-optimizer-offload", dest="optimizer_offload", action="store_false")
    p_verl_train.add_argument("--activation-offload", action="store_true", default=False)
    p_verl_train.add_argument("--no-activation-offload", dest="activation_offload", action="store_false")
    p_verl_train.add_argument("--use-torch-compile", action="store_true", default=True)
    p_verl_train.add_argument("--no-use-torch-compile", dest="use_torch_compile", action="store_false")
    p_verl_train.add_argument("--save-freq")
    p_verl_train.add_argument("--test-freq")
    p_verl_train.add_argument("--max-ckpt-to-keep", type=int, default=2)
    p_verl_train.add_argument("--save-hf-model", action="store_true", default=False)
    p_verl_train.add_argument("--no-save-hf-model", dest="save_hf_model", action="store_false")
    p_verl_train.add_argument("--launch", action="store_true")
    p_verl_train.set_defaults(func=cmd_train_verl)

    p_convert = sub.add_parser(
        "convert-hf",
        help="Convert a veRL FSDP shard checkpoint into a ReAct-servable HuggingFace model (offline, CPU)",
    )
    p_convert.add_argument("--experiment", required=True)
    p_convert.add_argument("--model-dir", help="veRL model dir holding global_step_* (default: model/<exp>_verl)")
    p_convert.add_argument("--global-step-dir", help="Specific global_step_* dir (default: newest)")
    p_convert.add_argument("--target-dir", help="Output HF model dir (default: <model-dir>/<step>_hf)")
    p_convert.add_argument("--verl-dir", default=DEFAULT_VERL_DIR)
    p_convert.add_argument("--no-cpu-init", action="store_true",
                           help="Disable --use_cpu_initialization (faster but may OOM for 8B)")
    p_convert.add_argument("--launch", action="store_true")
    p_convert.set_defaults(func=cmd_convert_hf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
