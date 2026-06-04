"""CLI runner for the veRL GRPO baseline."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ..full_shared.sft_schema import load_sft_samples
from .data_adapter import load_tool_catalog, samples_to_verl_rows, write_verl_jsonl, write_verl_parquet
from .metrics import write_reward_metrics


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = "data/newdata/sft_train_strict.jsonl"
DEFAULT_VERL_DIR = "/data1/yuhongjie2/verl"
DEFAULT_MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B"


def experiment_dir(experiment: str) -> Path:
    return MODULE_ROOT / "exp" / experiment


def build_grpo_command(
    *,
    train_file: str | Path,
    val_file: str | Path,
    model_dir: str | Path,
    reward_path: str | Path,
    model_path: str = DEFAULT_MODEL_PATH,
    n_gpus: int = 2,
    max_steps: int | None = None,
    execute_tools: bool = True,
    reward_trace_path: str | Path | None = None,
) -> list[str]:
    """Build a conservative 2-GPU LoRA GRPO command for veRL."""
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.train_batch_size=4",
        "data.max_prompt_length=8192",
        "data.max_response_length=1536",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.lora_rank=16",
        "actor_rollout_ref.model.lora_alpha=32",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.layered_summon=True",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        f"+reward.custom_reward_function.reward_kwargs.execute_tools={str(execute_tools)}",
        "reward.num_workers=1",
        "trainer.balance_batch=True",
        "trainer.logger=['console']",
        "trainer.project_name=terrabox_grpo",
        f"trainer.experiment_name={Path(model_dir).name}",
        f"trainer.default_local_dir={model_dir}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        "trainer.save_freq=20",
        "trainer.test_freq=10",
        "trainer.total_epochs=1",
    ]
    if reward_trace_path is not None:
        cmd.append(f"+reward.custom_reward_function.reward_kwargs.trace_path={reward_trace_path}")
    if max_steps is not None:
        cmd.append(f"trainer.total_training_steps={max_steps}")
    return cmd


def cmd_prepare_data(args: argparse.Namespace) -> None:
    samples = load_sft_samples(args.strict_data, limit=args.limit)
    split = int(len(samples) * args.train_ratio)
    catalog = load_tool_catalog(args.tool_catalog)
    out_dir = experiment_dir(args.experiment) / "verl_data"
    train_rows = samples_to_verl_rows(samples[:split], tool_catalog=catalog)
    val_rows = samples_to_verl_rows(samples[split:] or samples[: min(len(samples), 1)], tool_catalog=catalog)
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"
    write_verl_jsonl(train_rows, train_jsonl)
    write_verl_jsonl(val_rows, val_jsonl)
    if args.parquet:
        write_verl_parquet(train_rows, out_dir / "train.parquet")
        write_verl_parquet(val_rows, out_dir / "val.parquet")
    stats = {
        "strict_data": args.strict_data,
        "tool_catalog": args.tool_catalog,
        "num_samples": len(samples),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "parquet": bool(args.parquet),
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_train_grpo(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    data_dir = exp_dir / "verl_data"
    model_dir = MODULE_ROOT / "model" / args.experiment
    metrics_dir = exp_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train_file = Path(args.train_file or data_dir / "train.parquet")
    val_file = Path(args.val_file or data_dir / "val.parquet")
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}. Run prepare-data --parquet first.")
    cmd = build_grpo_command(
        train_file=train_file,
        val_file=val_file,
        model_dir=model_dir,
        reward_path=REPO_ROOT / "src/terrabox/evolution/ppo/reward_fn.py",
        model_path=args.model_path,
        n_gpus=args.n_gpus,
        max_steps=args.max_steps,
        execute_tools=not args.static_reward,
        reward_trace_path=metrics_dir / "reward_traces.jsonl",
    )
    if args.config:
        cmd.insert(3, f"--config-path={Path(args.config).parent}")
        cmd.insert(4, f"--config-name={Path(args.config).stem}")
    script = exp_dir / "run_grpo_command.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {args.verl_dir}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        f"export PYTHONPATH={REPO_ROOT / 'src'}:$PYTHONPATH\n"
        "export TERRABOX_USE_DOCKER=true\n"
        f"export TERRABOX_TOOL_GPU_DEVICES={args.tool_gpu}\n"
        "export TERRABOX_TOOL_MAX_GPUS=1\n"
        "export TERRABOX_TOOL_SERVICE_SCOPE=call\n"
        f"export TERRABOX_GRPO_ARTIFACT_DIR={metrics_dir / 'artifacts'}\n"
        f"export TERRABOX_GRPO_REWARD_TRACE_PATH={metrics_dir / 'reward_traces.jsonl'}\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(f"Wrote {script}")
    if args.launch:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        subprocess.run(["bash", str(script)], cwd=args.verl_dir, env=env, check=True)
    else:
        print("Review the generated script before launching a long GRPO run.")


def cmd_stats(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    metrics_dir = exp_dir / "metrics"
    trace_path = Path(args.trace_path or metrics_dir / "reward_traces.jsonl")
    metrics = write_reward_metrics(trace_path, metrics_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrabox GRPO/PPO baseline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("prepare-data", help="Convert strict SFT JSONL to veRL JSONL/Parquet")
    p_data.add_argument("--strict-data", default=DEFAULT_DATA)
    p_data.add_argument("--tool-catalog", default="data/newdata/tools_catalog.json")
    p_data.add_argument("--experiment", required=True)
    p_data.add_argument("--limit", type=int)
    p_data.add_argument("--train-ratio", type=float, default=0.95)
    p_data.add_argument("--parquet", action="store_true")
    p_data.set_defaults(func=cmd_prepare_data)

    p_train = sub.add_parser("train-grpo", help="Write or launch veRL GRPO command")
    p_train.add_argument("--experiment", required=True)
    p_train.add_argument("--verl-dir", default=DEFAULT_VERL_DIR)
    p_train.add_argument("--config")
    p_train.add_argument("--train-file")
    p_train.add_argument("--val-file")
    p_train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--n-gpus", type=int, default=2)
    p_train.add_argument("--max-steps", type=int)
    p_train.add_argument("--cuda-visible-devices", default="2,3")
    p_train.add_argument("--tool-gpu", default="3")
    p_train.add_argument("--static-reward", action="store_true", help="Disable real Terrabox tool execution in reward")
    p_train.add_argument("--launch", action="store_true")
    p_train.set_defaults(func=cmd_train_grpo)

    p_stats = sub.add_parser("stats", help="Summarize dynamic reward traces")
    p_stats.add_argument("--experiment", required=True)
    p_stats.add_argument("--trace-path")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
