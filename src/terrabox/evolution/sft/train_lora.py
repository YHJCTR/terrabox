"""QLoRA SFT training entrypoint.

This file imports heavy training libraries only inside ``main`` so unit tests can
exercise command construction without requiring the training stack.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _messages_to_text(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("task_id") or row.get("id") or f"row_{index}")


def token_length_stats(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
    split_name: str,
) -> dict[str, Any]:
    lengths: list[int] = []
    overlong: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = _messages_to_text(tokenizer, row.get("messages", []) or [])
        encoded = tokenizer(text, add_special_tokens=False)
        length = len(encoded.input_ids)
        lengths.append(length)
        if length > max_seq_length:
            overlong.append(
                {
                    "row_index": index,
                    "task_id": _row_id(row, index),
                    "source": row.get("source"),
                    "task_type": row.get("task_type"),
                    "tokens": length,
                }
            )
    sorted_lengths = sorted(lengths)

    def percentile(value: float) -> int:
        if not sorted_lengths:
            return 0
        pos = int((len(sorted_lengths) - 1) * value)
        return sorted_lengths[pos]

    return {
        "split_name": split_name,
        "num_rows": len(rows),
        "max_seq_length": max_seq_length,
        "min_tokens": min(lengths) if lengths else 0,
        "p50_tokens": percentile(0.50),
        "p90_tokens": percentile(0.90),
        "p95_tokens": percentile(0.95),
        "p99_tokens": percentile(0.99),
        "max_tokens": max(lengths) if lengths else 0,
        "overlong_count": len(overlong),
        "overlong_examples": overlong[:20],
    }


def assert_no_overlength_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
    split_name: str,
) -> dict[str, Any]:
    """Fail before SFTTrainer can silently truncate tool catalog/messages."""
    stats = token_length_stats(rows, tokenizer, max_seq_length=max_seq_length, split_name=split_name)
    if stats["overlong_count"]:
        examples = ", ".join(
            f"{item['task_id']}={item['tokens']} tokens" for item in stats["overlong_examples"][:5]
        )
        raise ValueError(
            f"SFT {split_name} contains {stats['overlong_count']} rows longer than "
            f"max_seq_length={max_seq_length}; longest={stats['max_tokens']} tokens. "
            f"Examples: {examples}. Increase max_seq_length or shorten the prompt/tool catalog; "
            "silent truncation is disabled."
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QLoRA SFT adapter on Terrabox chat JSONL")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-merged-model", action="store_true")
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise RuntimeError(
            "SFT training requires the unsloth environment with datasets, trl, and unsloth installed."
        ) from exc

    train_rows = _load_rows(args.train_file)
    eval_rows = _load_rows(args.val_file)
    preflight_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    train_length_stats = assert_no_overlength_rows(
        train_rows,
        preflight_tokenizer,
        max_seq_length=args.max_seq_length,
        split_name="train",
    )
    eval_length_stats = assert_no_overlength_rows(
        eval_rows,
        preflight_tokenizer,
        max_seq_length=args.max_seq_length,
        split_name="val",
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def convert(row: dict[str, Any]) -> dict[str, str]:
        return {"text": _messages_to_text(tokenizer, row.get("messages", []) or [])}

    train_dataset = Dataset.from_list([convert(row) for row in train_rows])
    eval_dataset = Dataset.from_list([convert(row) for row in eval_rows])
    train_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
    )
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if args.save_merged_model:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    metadata = {
        "base_model": args.model_path,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if args.save_merged_model else None,
        "serving_note": "Terrabox Docker/vLLM rollout should use merged_dir, not adapter_dir.",
        "length_stats": {
            "train": train_length_stats,
            "val": eval_length_stats,
            "truncation_policy": "fail_before_training_if_any_row_exceeds_max_seq_length",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
