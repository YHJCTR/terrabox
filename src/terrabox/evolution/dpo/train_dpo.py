"""QLoRA DPO training entrypoint (trl DPOTrainer + unsloth).

Consumes ``(prompt, chosen, rejected)`` JSONL produced by ``pair_builder`` and
trains a LoRA adapter on the text Qwen3-8B. Heavy libs are imported inside
``main`` so command construction stays testable without the training stack.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QLoRA DPO adapter on Terrabox preference pairs")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL regularisation strength.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-merged-model", action="store_true", default=True)
    parser.add_argument("--no-save-merged-model", dest="save_merged_model", action="store_false")
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import DPOTrainer, DPOConfig
    except ImportError as exc:
        raise RuntimeError(
            "DPO training requires the unsloth environment with datasets, trl, and unsloth installed."
        ) from exc

    train_rows = _load_rows(args.train_file)
    eval_rows = _load_rows(args.val_file)

    def to_pref(row: dict[str, Any]) -> dict[str, str]:
        return {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}

    train_dataset = Dataset.from_list([to_pref(r) for r in train_rows])
    eval_dataset = Dataset.from_list([to_pref(r) for r in eval_rows]) if eval_rows else None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        # ref_model=None: with a PEFT/LoRA policy, DPOTrainer derives the frozen
        # reference by disabling the adapter — no second copy of the 8B weights.
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
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
        "method": "dpo",
        "beta": args.beta,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if args.save_merged_model else None,
        "serving_note": "Terrabox/vLLM rollout should mount merged_dir as /model, not adapter_dir.",
        "num_train_pairs": len(train_rows),
        "num_val_pairs": len(eval_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dpo_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
