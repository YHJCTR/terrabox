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

# Defensive CPU-thread caps (applied at import, before torch/numpy spin up their
# thread pools). Without these, intra-op math libs and the rust tokenizer fan out
# to ALL 56 cores — a CPU power spike that can trip a marginal shared breaker.
# setdefault so the launch script's explicit values still win.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    # RAYON_NUM_THREADS caps the rust threadpool (safetensors writer during the
    # merged-model save) which otherwise fans out to all 56 cores at save time.
    os.environ.setdefault(_var, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
    drop: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Guard against SFTTrainer silently truncating tool catalog/messages.

    Default: FAIL if any row exceeds max_seq_length (silent truncation would
    corrupt the gold tool calls / catalog). With ``drop=True``: instead DROP the
    overlength rows (logged) and keep the rest verbatim — used to align with the
    rollout regime (no lossy per-content compaction; rows that can't fit the
    window are removed rather than truncated mid-tool-call). Returns (kept_rows, stats).
    """
    stats = token_length_stats(rows, tokenizer, max_seq_length=max_seq_length, split_name=split_name)
    if stats["overlong_count"]:
        if drop:
            kept = [
                row for row in rows
                if len(tokenizer(_messages_to_text(tokenizer, row.get("messages", []) or []),
                                 add_special_tokens=False).input_ids) <= max_seq_length
            ]
            print(
                f"[SFT {split_name}] dropping {stats['overlong_count']}/{len(rows)} rows over "
                f"max_seq_length={max_seq_length} (longest={stats['max_tokens']} tokens) — "
                f"kept {len(kept)} verbatim.",
                flush=True,
            )
            stats["dropped_overlong"] = stats["overlong_count"]
            return kept, stats
        examples = ", ".join(
            f"{item['task_id']}={item['tokens']} tokens" for item in stats["overlong_examples"][:5]
        )
        raise ValueError(
            f"SFT {split_name} contains {stats['overlong_count']} rows longer than "
            f"max_seq_length={max_seq_length}; longest={stats['max_tokens']} tokens. "
            f"Examples: {examples}. Increase max_seq_length, shorten the prompt/tool catalog, "
            "or pass --drop-overlength; silent truncation is disabled."
        )
    return rows, stats


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
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save a training checkpoint every N steps (mid-training ckpt).")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="Max number of checkpoints to keep.")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Run validation every N steps (only if --eval-strategy steps).")
    parser.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps",
                        help="Mid-training validation strategy. Use 'no' to disable in-loop "
                             "eval entirely: the eval pass is an uninterrupted forward-only "
                             "burst (194 val rows, ~12min) that ran the GPU flat-out and "
                             "coincided with a breaker trip even though 5.5h of stop-and-go "
                             "training at the same 220W cap was fine. With 'no', evaluate "
                             "offline from a saved checkpoint instead.")
    parser.add_argument("--rest-every-steps", type=int, default=0,
                        help="Every N optimizer steps, idle the GPU for --rest-sec (deep "
                             "cooldown) to let a marginal shared breaker's bimetal fully cool. "
                             "0 = off. Pairs well with --save-steps so a checkpoint lands just "
                             "before each rest.")
    parser.add_argument("--rest-sec", type=float, default=600.0,
                        help="Seconds to idle the GPU at each --rest-every-steps boundary.")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-merged-model", action="store_true")
    # veRL standard: compute loss only on assistant turns (mask system/user/
    # observation). veRL's MultiTurnSFTDataset does this; mirror it here so the
    # unsloth backend trains the same objective. Default ON.
    parser.add_argument("--mask-prompt", dest="mask_prompt", action="store_true", default=True,
                        help="Only train on assistant responses (assistant-only loss; veRL-aligned).")
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false")
    parser.add_argument("--drop-overlength", action="store_true",
                        help="Drop rows exceeding max_seq_length (verbatim-keep the rest) instead of "
                             "erroring. Aligns with rollout: no lossy mid-tool-call truncation.")
    parser.add_argument("--cooldown-sec", type=float, default=0.0,
                        help="Sleep N seconds after EACH optimizer step (GPU drops to idle), to lower "
                             "duty-cycle / average power. 0 = off (default; use nvidia-smi -pl instead "
                             "when you have root).")
    parser.add_argument("--dataset-num-proc", type=int, default=8,
                        help="CPU processes for dataset tokenization. unsloth defaults to ALL cores "
                             "(~60), which pins the whole CPU at 100%% during the load phase and can "
                             "trip a marginal breaker BEFORE training even starts. Keep this low.")
    parser.add_argument("--temp-target", type=float, default=0.0,
                        help="If >0, after each step poll GPU temperature and sleep until it drops "
                             "below this (C). Keeps the card cool AND lowers average power/duty-cycle, "
                             "which is the real lever on a marginal thermal-magnetic breaker. "
                             "Use ~80 when you cannot lower the power limit with root.")
    parser.add_argument("--temp-resume", type=float, default=0.0,
                        help="Resume training once temp falls to this (C). Default = temp-target - 8 "
                             "(hysteresis, avoids thrashing on the throttle boundary).")
    parser.add_argument("--temp-poll-sec", type=float, default=3.0,
                        help="Seconds to sleep between temperature polls while throttling.")
    parser.add_argument("--temp-max-wait", type=float, default=120.0,
                        help="Hard cap (s) on a single throttle wait, so a stuck/hot card can't "
                             "stall the run forever.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Auto-resume from the latest checkpoint in output-dir if present (default on).")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    try:
        import time
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from datasets import Dataset
        from transformers import AutoTokenizer, TrainerCallback
        from transformers.trainer_utils import get_last_checkpoint
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
    train_rows, train_length_stats = assert_no_overlength_rows(
        train_rows,
        preflight_tokenizer,
        max_seq_length=args.max_seq_length,
        split_name="train",
        drop=args.drop_overlength,
    )
    eval_rows, eval_length_stats = assert_no_overlength_rows(
        eval_rows,
        preflight_tokenizer,
        max_seq_length=args.max_seq_length,
        split_name="val",
        drop=args.drop_overlength,
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
    # Skip building/passing the eval set entirely when in-loop eval is disabled,
    # so no forward-only eval burst ever runs (see --eval-strategy).
    eval_dataset = (
        Dataset.from_list([convert(row) for row in eval_rows])
        if args.eval_strategy != "no" else None
    )
    train_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        # Eval OOM guard: at max_seq_length=13312 a default eval batch keeps the
        # full LM-head logits (151936 vocab x seq x bf16), which previously OOM'd
        # at step-50 eval ("Tried to allocate 11.25 GiB"). batch=1 + loss-only
        # (drop logits/labels) keeps eval within the same budget as training.
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        save_total_limit=args.save_total_limit,
        bf16=True,
        report_to=[],
        packing=False,
        # Cap CPU parallelism so the load-phase tokenization doesn't pin all cores
        # at 100% (a CPU power spike that can trip a marginal breaker before any
        # GPU step runs). See --dataset-num-proc.
        dataset_num_proc=args.dataset_num_proc,
        dataloader_num_workers=2,
    )
    # Also cap intra-op CPU threads (matmul/tokenizer) as a second guard.
    try:
        import torch as _torch
        _torch.set_num_threads(min(8, args.dataset_num_proc))
    except Exception:
        pass
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
    )
    if args.mask_prompt:
        # Qwen3 ChatML turn markers. Masks everything except assistant spans, so
        # loss is computed on assistant responses only (handles multi-turn ReAct
        # trajectories — every assistant turn is supervised, prompts/observations
        # are ignored). Matches veRL's assistant-only objective.
        # IMPORTANT: train_on_responses_only runs its OWN dataset.map, and if
        # num_proc is not passed it defaults to min(cpu_count()+4, 64) ≈ 60 on
        # this 56-core box — pinning ALL cores at 100% (a large CPU power spike
        # during the load phase). SFTConfig.dataset_num_proc only governs the
        # earlier tokenize map, NOT this one. Pass num_proc explicitly so this
        # map is capped too.
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            num_proc=args.dataset_num_proc,
        )

    # No-root power workaround: sleep after each optimizer step so the GPU drops
    # to idle between steps. This lowers the duty cycle / average current (like
    # the bursty ReAct rollout that never tripped) and lets a marginal shared
    # breaker cool, avoiding the I²t trip that a continuous 100%% SFT load causes.
    if args.cooldown_sec > 0 or args.temp_target > 0 or args.rest_every_steps > 0:
        import os as _os
        import subprocess as _sp

        # Resolve the PHYSICAL gpu index nvidia-smi should query. Inside the
        # process the device is cuda:0, but nvidia-smi sees physical indices, so
        # map through CUDA_VISIBLE_DEVICES (e.g. "2" -> query -i 2).
        _vis = _os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        _gpu_idx = _vis if _vis else "0"
        _temp_resume = args.temp_resume if args.temp_resume > 0 else max(0.0, args.temp_target - 8.0)

        def _gpu_pw_temp():
            # One nvidia-smi call -> (power_draw_W, temp_C); None on failure.
            try:
                out = _sp.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits", "-i", _gpu_idx],
                    stderr=_sp.DEVNULL, timeout=10,
                )
                p, t = out.decode().strip().splitlines()[0].split(",")
                return float(p), float(t)
            except Exception:
                return None, None

        class _ThrottleCallback(TrainerCallback):
            # Fixed cooldown lowers duty-cycle every step; temp-throttle adds extra
            # idle whenever the card runs hot. Both reduce average (RMS) current,
            # letting a marginal shared breaker's bimetal cool and avoiding the I²t
            # trip that a continuous 100% SFT load causes.
            def on_step_end(self, targs, state, control, **kw):
                # Per-step telemetry (captured at step end, before any sleep, so
                # GPU power is still near its active level) — gives a load trace
                # right up to the moment of a breaker trip, to settle whether CPU
                # or GPU power is the driver. CPU side = 1-min loadavg (cheap, no
                # extra process); on this 56-core box loadavg>>16 would mean a CPU
                # spike, loadavg~1-3 means the trip is GPU/electrical, not CPU.
                # cuda.synchronize() first: PyTorch CUDA is async, so without it
                # this callback can run while the step's kernels are still queued,
                # making the power/temp reading (and duty-cycle estimate) wrong.
                # Draining the queue gives an accurate end-of-step reading.
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.synchronize()
                except Exception:
                    pass
                pw, t = _gpu_pw_temp()
                try:
                    la1 = _os.getloadavg()[0]
                except Exception:
                    la1 = -1.0
                pw_s = f"{pw:.0f}W" if pw is not None else "?W"
                t_s = f"{t:.0f}C" if t is not None else "?C"
                print(f"[telemetry] step={state.global_step} gpu{_gpu_idx} "
                      f"{pw_s} {t_s} cpu_load1={la1:.1f}", flush=True)
                if args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)
                if args.temp_target > 0:
                    waited = 0.0
                    _, t = _gpu_pw_temp()
                    if t is not None and t > args.temp_target:
                        print(f"[SFT] gpu{_gpu_idx} {t:.0f}C > {args.temp_target:.0f}C, "
                              f"throttling until <= {_temp_resume:.0f}C", flush=True)
                    while t is not None and t > _temp_resume and waited < args.temp_max_wait:
                        time.sleep(args.temp_poll_sec)
                        waited += args.temp_poll_sec
                        _, t = _gpu_pw_temp()
                # Periodic deep rest: every N steps, idle the GPU for a long stretch
                # so a marginal shared breaker's bimetal fully cools. A checkpoint
                # lands just before this when rest_every_steps == save_steps multiple.
                if args.rest_every_steps > 0 and state.global_step % args.rest_every_steps == 0:
                    pw0, t0 = _gpu_pw_temp()
                    print(f"[SFT] deep rest {args.rest_sec:.0f}s at step={state.global_step} "
                          f"(gpu{_gpu_idx} {pw0}W {t0}C before rest)", flush=True)
                    time.sleep(args.rest_sec)
                    pw1, t1 = _gpu_pw_temp()
                    print(f"[SFT] resumed after rest (gpu{_gpu_idx} {pw1}W {t1}C)", flush=True)

        trainer.add_callback(_ThrottleCallback())
        msg = []
        if args.cooldown_sec > 0:
            msg.append(f"cooldown {args.cooldown_sec}s/step")
        if args.temp_target > 0:
            msg.append(f"temp-throttle target {args.temp_target:.0f}C "
                       f"(resume <= {_temp_resume:.0f}C, gpu{_gpu_idx})")
        print(f"[SFT] throttle enabled: {', '.join(msg)}.", flush=True)

    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    # Breakpoint-resume: continue from the latest checkpoint in output_dir if one
    # exists (e.g. after a power trip). save_steps controls checkpoint frequency.
    resume_ckpt = None
    if args.resume and output_dir.exists():
        resume_ckpt = get_last_checkpoint(str(output_dir))
        if resume_ckpt:
            print(f"[SFT] resuming from checkpoint: {resume_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
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
