"""Single-GPU QLoRA GRPO runner for strict ExperienceEvo RL pilots.

This entrypoint is intentionally separate from the veRL runner. The current
24GB RTX 3090 environment can run Qwen2.5-3B QLoRA forward/backward, while the
veRL colocated-vLLM/FSDP path has a larger synchronization memory peak. This
runner uses TRL/PEFT/bitsandbytes on exactly the visible CUDA device and does
not execute Terrabox tools or call external LLM APIs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .data_builder import scan_forbidden_payload
from .reward_fn import compute_score, parse_tool_calls


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXP_ROOT = REPO_ROOT / "tmp/experience_evo_rl"
DEFAULT_MODEL_PATH = Path("/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct")


def _exp_dir(name: str) -> Path:
    return DEFAULT_EXP_ROOT / name


def _read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _messages_to_prompt(tokenizer: Any, messages: Any) -> str:
    if isinstance(messages, list):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    return str(messages or "")


def _normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content") or completion)
    return str(completion or "")


def build_dataset(rows: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    flat_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        policy = row.get("reward_model", {}).get("ground_truth", "{}")
        flat_rows.append(
            {
                "prompt": _messages_to_prompt(tokenizer, row.get("prompt")),
                "policy_json": policy,
                "data_source": row.get("data_source", "oea_experience_evo_rl_strict"),
                "row_index": idx,
            }
        )
    leak_scan = scan_forbidden_payload(flat_rows)
    if not leak_scan["ok"]:
        raise RuntimeError(f"Forbidden strict no-label fields found in flattened rows: {leak_scan}")
    return Dataset.from_list(flat_rows)


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        payload = {"time": time.time(), "step": state.global_step, **logs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_reward_traces(path: Path, out_path: Path) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
    parsed = [t for t in traces if int(t.get("num_calls") or 0) > 0]
    tools: dict[str, int] = {}
    for trace in parsed:
        for tool in trace.get("called_tools") or []:
            tools[str(tool)] = tools.get(str(tool), 0) + 1
    scores = [float(t.get("score")) for t in traces if isinstance(t.get("score"), (int, float))]
    summary = {
        "trace_path": str(path),
        "n_traces": len(traces),
        "parsed": len(parsed),
        "parse_rate": (len(parsed) / len(traces)) if traces else None,
        "score_mean": (sum(scores) / len(scores)) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "top_tools": sorted(tools.items(), key=lambda kv: kv[1], reverse=True)[:20],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict single-GPU QLoRA GRPO with TRL")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    args = parser.parse_args()

    exp_dir = _exp_dir(args.experiment)
    train_file = Path(args.train_file or exp_dir / "verl_data" / "train.jsonl")
    val_file = Path(args.val_file or exp_dir / "verl_data" / "val.jsonl")
    output_dir = Path(args.output_dir or exp_dir / "qlora_grpo")
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reward_trace_path = metrics_dir / "trl_grpo_reward_traces.jsonl"
    log_path = metrics_dir / "trl_grpo_logs.jsonl"
    for path in (reward_trace_path, log_path):
        if path.exists():
            path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_rows = _read_jsonl(train_file, limit=args.train_limit)
    val_rows = _read_jsonl(val_file, limit=args.eval_limit)
    train_dataset = build_dataset(train_rows, tokenizer)
    eval_dataset = build_dataset(val_rows, tokenizer) if val_rows else None
    uses_experience_evo_reward = any(
        "experience_evo" in str(
            row.get("reward_model", {}).get("ground_truth", "")
        )
        for row in train_rows[: min(len(train_rows), 20)]
    )

    def reward_func(completions, policy_json=None, data_source=None, **kwargs):  # noqa: ANN001
        rewards: list[float] = []
        gts = policy_json or ["{}"] * len(completions)
        sources = data_source or ["oea_experience_evo_rl_strict"] * len(completions)
        for completion, gt, source in zip(completions, gts, sources):
            rewards.append(
                compute_score(
                    str(source),
                    _normalize_completion(completion),
                    str(gt),
                    trace_path=str(reward_trace_path),
                )
            )
        return rewards

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        use_vllm=False,
        beta=0.0,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "local_files_only": True,
        },
    )

    run_config = {
        "experiment": args.experiment,
        "model_path": str(args.model_path),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "output_dir": str(output_dir),
        "train_rows": len(train_dataset),
        "val_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "strict_nolabel": True,
        "uses_experience_evo_reward": uses_experience_evo_reward,
        "uses_external_api": False,
        "executes_terrabox_tools": False,
    }
    (exp_dir / "qlora_grpo_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer = GRPOTrainer(
        model=str(args.model_path),
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[JsonlMetricsCallback(log_path)],
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    summary = summarize_reward_traces(reward_trace_path, metrics_dir / "trl_grpo_reward_summary.json")
    print(json.dumps({"done": True, "summary": summary, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
