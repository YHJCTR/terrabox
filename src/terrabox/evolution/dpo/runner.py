"""CLI runner for the standalone DPO baseline.

Independent comparison experiment. Reuses only shared infra (strict SFT schema,
the deterministic shuffle split, the prompt-only task writer, and the ReAct
rollout environment) — exactly what ReAct/Reflection/SFT also reuse. No coupling
to any evolution *method*.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ..ReAct.runner import build_rollout_env
from ..ReAct.data_adapter import samples_to_tasks
from ..reflection.data_split import load_shuffled_samples, sample_slice
from .pair_builder import build_pairs, load_rollout_index, write_pairs_jsonl


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
# Same aligned, de-collapsed strict data ReAct/Reflection/SFT use (44 tools).
DEFAULT_DATA = "data/fixdata_decollapse/sft_train_strict.jsonl"
DEFAULT_MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
DEFAULT_PYTHON = "/home/yuhongjie/miniconda3/envs/unsloth/bin/python"


def experiment_dir(experiment: str) -> Path:
    return MODULE_ROOT / "exp" / experiment


# prepare-pairs -------------------------------------------------------------

def cmd_prepare_pairs(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    data_dir = exp_dir / "dpo_data"
    samples = load_shuffled_samples(args.strict_data, seed=args.seed)
    train_samples = sample_slice(samples, start=args.train_start, limit=args.train_limit)
    val_samples = sample_slice(samples, start=args.val_start, limit=args.val_limit)

    if not args.rollout_dir:
        raise SystemExit("--rollout-dir is required (one or more rollout result dirs / jsonl files).")
    rollout_index = load_rollout_index(args.rollout_dir)

    train_pairs, train_stats = build_pairs(
        train_samples, rollout_index,
        rejected_policy=args.rejected_policy,
        max_rollout_f1=args.max_rollout_f1,
        max_obs_chars=args.max_obs_chars,
    )
    val_pairs, val_stats = build_pairs(
        val_samples, rollout_index,
        rejected_policy=args.rejected_policy,
        max_rollout_f1=args.max_rollout_f1,
        max_obs_chars=args.max_obs_chars,
    )
    train_file = data_dir / "train_pairs.jsonl"
    val_file = data_dir / "val_pairs.jsonl"
    write_pairs_jsonl(train_pairs, train_file)
    write_pairs_jsonl(val_pairs, val_file)

    # Prompt-only eval tasks for rolling out the DPO'd model (no gold leakage),
    # using the same writer/format as ReAct/Reflection/SFT for comparability.
    eval_tasks = exp_dir / "eval_tasks.json"
    tasks = samples_to_tasks(val_samples)
    eval_tasks.parent.mkdir(parents=True, exist_ok=True)
    eval_tasks.write_text(
        json.dumps({
            "metadata": {
                "split_name": "eval",
                "num_tasks": len(tasks),
                "gold_leakage_policy": "messages/gold_tool_calls/ground_truth omitted from model-facing task file",
            },
            "tasks": tasks,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "strict_data": args.strict_data,
        "rollout_dir": args.rollout_dir,
        "rejected_policy": args.rejected_policy,
        "train": train_stats,
        "val": val_stats,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "eval_task_file": str(eval_tasks),
        "eval_task_rows": len(tasks),
    }
    (data_dir / "dpo_dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


# train ---------------------------------------------------------------------

def build_dpo_train_command(
    *,
    train_file: str | Path,
    val_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    beta: float = 0.1,
    max_length: int = 8192,
    max_prompt_length: int = 4096,
    num_train_epochs: float = 1.0,
    max_steps: int | None = None,
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    lora_rank: int = 16,
    save_steps: int = 50,
    save_total_limit: int = 2,
    eval_steps: int = 50,
    save_merged_model: bool = True,
    python_executable: str | None = None,
) -> list[str]:
    cmd = [
        python_executable or DEFAULT_PYTHON,
        "-m", "terrabox.evolution.dpo.train_dpo",
        "--train-file", str(train_file),
        "--val-file", str(val_file),
        "--model-path", str(model_path),
        "--output-dir", str(output_dir),
        "--beta", str(beta),
        "--max-length", str(max_length),
        "--max-prompt-length", str(max_prompt_length),
        "--num-train-epochs", str(num_train_epochs),
        "--learning-rate", str(learning_rate),
        "--per-device-train-batch-size", str(per_device_train_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--lora-rank", str(lora_rank),
        "--save-steps", str(save_steps),
        "--save-total-limit", str(save_total_limit),
        "--eval-steps", str(eval_steps),
    ]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if save_merged_model:
        cmd.append("--save-merged-model")
    return cmd


def cmd_train(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    data_dir = exp_dir / "dpo_data"
    model_dir = Path(args.output_model_dir or MODULE_ROOT / "model" / args.experiment)
    train_file = Path(args.train_file or data_dir / "train_pairs.jsonl")
    val_file = Path(args.val_file or data_dir / "val_pairs.jsonl")
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("Missing DPO pair files. Run prepare-pairs first.")
    cmd = build_dpo_train_command(
        train_file=train_file, val_file=val_file,
        model_path=args.model_path, output_dir=model_dir,
        beta=args.beta, max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank, save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, eval_steps=args.eval_steps,
        save_merged_model=args.save_merged_model,
    )
    script = exp_dir / "run_dpo_train.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        "export UNSLOTH_DISABLE_STATISTICS=1\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f"export PYTHONPATH={REPO_ROOT / 'src'}:$PYTHONPATH\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(exp_dir / 'dpo_train.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(f"Wrote {script}")
    if args.launch:
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated script before launching DPO training.")


# rollout (eval) ------------------------------------------------------------

def cmd_rollout(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    out_dir = Path(args.output_dir or exp_dir / "eval")
    task_file = Path(args.task_file or exp_dir / "eval_tasks.json")
    model_path = Path(args.model_path or MODULE_ROOT / "model" / args.experiment / "merged")
    cmd = [
        DEFAULT_PYTHON, "scripts/run_trajectory_experiment.py", "rollout",
        "--task-file", str(task_file),
        "--experiment", f"{args.experiment}_dpo_eval",
        "--mode", "standard",
        "--output-dir", str(out_dir),
        "--port", str(args.port),
        "--max-iterations", str(args.max_iterations),
        "--use-docker", "--resume",
    ]
    if args.start_index is not None:
        cmd.extend(["--start-index", str(args.start_index)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    run_script = exp_dir / "run_dpo_eval.sh"
    run_script.parent.mkdir(parents=True, exist_ok=True)
    run_script.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"export AGENT_LLM_MODEL_PATH={shlex.quote(str(model_path))}\n"
        f"export PYTHONPATH={REPO_ROOT / 'src'}:$PYTHONPATH\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(exp_dir / 'dpo_eval.log'))}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    print(f"Wrote {run_script}")
    if args.launch:
        env = build_rollout_env(args.agent_gpu, args.tool_gpu, getattr(args, "vlm_gpus", None))
        env["AGENT_LLM_MODEL_PATH"] = str(model_path)
        if getattr(args, "instructsam_backend", None):
            env["TERRABOX_INSTRUCTSAM_BACKEND"] = args.instructsam_backend
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated script before launching DPO eval rollout.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrabox standalone DPO baseline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pairs = sub.add_parser("prepare-pairs", help="Build (prompt, chosen=gold, rejected=rollout) pairs")
    p_pairs.add_argument("--experiment", required=True)
    p_pairs.add_argument("--strict-data", default=DEFAULT_DATA)
    p_pairs.add_argument("--rollout-dir", nargs="+", required=True,
                         help="一个或多个 rollout 结果目录/jsonl(ReAct/Reflection 缓存),按 task_id 配 rejected")
    p_pairs.add_argument("--seed", type=int, default=42)
    p_pairs.add_argument("--train-start", type=int, default=216)
    p_pairs.add_argument("--train-limit", type=int, default=None)
    p_pairs.add_argument("--val-start", type=int, default=0)
    p_pairs.add_argument("--val-limit", type=int, default=216)
    p_pairs.add_argument("--rejected-policy", default="below-gold",
                         choices=["below-gold", "failed-only", "all"])
    p_pairs.add_argument("--max-rollout-f1", type=float, default=1.0)
    p_pairs.add_argument("--max-obs-chars", type=int, default=1500)
    p_pairs.set_defaults(func=cmd_prepare_pairs)

    p_train = sub.add_parser("train", help="Write or launch QLoRA DPO training")
    p_train.add_argument("--experiment", required=True)
    p_train.add_argument("--train-file")
    p_train.add_argument("--val-file")
    p_train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--output-model-dir")
    p_train.add_argument("--cuda-visible-devices", default="0")
    p_train.add_argument("--beta", type=float, default=0.1)
    p_train.add_argument("--max-length", type=int, default=8192)
    p_train.add_argument("--max-prompt-length", type=int, default=4096)
    p_train.add_argument("--num-train-epochs", type=float, default=1.0)
    p_train.add_argument("--max-steps", type=int)
    p_train.add_argument("--learning-rate", type=float, default=5e-6)
    p_train.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_train.add_argument("--lora-rank", type=int, default=16)
    p_train.add_argument("--save-steps", type=int, default=50)
    p_train.add_argument("--save-total-limit", type=int, default=2)
    p_train.add_argument("--eval-steps", type=int, default=50)
    p_train.add_argument("--save-merged-model", action="store_true", default=True)
    p_train.add_argument("--no-save-merged-model", dest="save_merged_model", action="store_false")
    p_train.add_argument("--launch", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("rollout", help="Evaluate a DPO checkpoint through real Terrabox rollout")
    p_eval.add_argument("--experiment", required=True)
    p_eval.add_argument("--task-file")
    p_eval.add_argument("--output-dir")
    p_eval.add_argument("--model-path")
    p_eval.add_argument("--port", type=int, default=9100)
    p_eval.add_argument("--agent-gpu", default=0)
    p_eval.add_argument("--tool-gpu", default=1)
    p_eval.add_argument("--vlm-gpus", default="2")
    p_eval.add_argument("--instructsam-backend", default="service")
    p_eval.add_argument("--start-index", type=int)
    p_eval.add_argument("--limit", type=int)
    p_eval.add_argument("--max-iterations", type=int, default=15)
    p_eval.add_argument("--launch", action="store_true")
    p_eval.set_defaults(func=cmd_rollout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
