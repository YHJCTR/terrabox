"""ms-swift backend helpers for current-harness SFT cold start.

The SFT runner writes shell scripts instead of starting long jobs by default.
This module keeps the Swift-specific command surface in one place so the
legacy Unsloth/TRL path can remain available for debugging without being the
default recommendation.
"""
from __future__ import annotations

from pathlib import Path


DEFAULT_SWIFT_DIR = "/data1/yuhongjie2/ms-swift"


def build_swift_sft_command(
    *,
    train_file: str | Path,
    val_file: str | Path | None,
    model_path: str | Path,
    model_type: str | None,
    template: str | None,
    output_dir: str | Path,
    python_executable: str,
    swift_dir: str | Path = DEFAULT_SWIFT_DIR,
    max_steps: int | None = None,
    num_train_epochs: float = 1.0,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    lora_rank: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 4096,
    packing: bool = False,
    packing_strategy: str = "sequential",
    packing_num_proc: int = 1,
    truncation_strategy: str = "delete",
    dataset_num_proc: int = 1,
    dataloader_num_workers: int = 1,
    dataloader_persistent_workers: bool = False,
    save_steps: int = 200,
    save_total_limit: int = 2,
    eval_steps: int = 200,
    eval_strategy: str = "steps",
    logging_steps: int = 5,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    report_to: str = "tensorboard",
    save_only_model: bool = True,
    use_logits_to_keep: bool = True,
    torch_dtype: str = "bfloat16",
    tuner_backend: str = "peft",
    target_modules: str = "all-linear",
    seed: int = 42,
    data_seed: int = 42,
    split_dataset_ratio: float = 0.0,
    load_from_cache_file: bool = True,
    check_model: bool = False,
    resume_from_checkpoint: str | Path | None = None,
    resume_only_model: bool = False,
    quant_method: str | None = None,
    quant_bits: int | None = None,
) -> list[str]:
    """Build an ms-swift SFT command.

    Defaults are intentionally conservative for a 24GB 3090 lane: single GPU,
    LoRA, assistant-only Swift template loss, low dataset workers, explicit
    deletion of overlength samples, and model-only checkpoints by default.
    """
    sft_entry = Path(swift_dir) / "swift" / "cli" / "sft.py"
    cmd = [
        python_executable,
        str(sft_entry),
        "--model",
        str(model_path),
        "--dataset",
        str(train_file),
        "--output_dir",
        str(output_dir),
        "--tuner_type",
        "lora",
        "--tuner_backend",
        tuner_backend,
        "--target_modules",
        target_modules,
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--lora_dropout",
        str(lora_dropout),
        "--torch_dtype",
        torch_dtype,
        "--num_train_epochs",
        str(num_train_epochs),
        "--learning_rate",
        str(learning_rate),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--max_length",
        str(max_length),
        "--truncation_strategy",
        truncation_strategy,
        "--dataset_num_proc",
        str(dataset_num_proc),
        "--dataloader_num_workers",
        str(dataloader_num_workers),
        "--dataloader_persistent_workers",
        str(dataloader_persistent_workers).lower(),
        "--save_steps",
        str(save_steps),
        "--save_total_limit",
        str(save_total_limit),
        "--eval_steps",
        str(eval_steps),
        "--eval_strategy",
        eval_strategy,
        "--logging_steps",
        str(logging_steps),
        "--warmup_ratio",
        str(warmup_ratio),
        "--lr_scheduler_type",
        lr_scheduler_type,
        "--report_to",
        report_to,
        "--loss_scale",
        "default",
        "--seed",
        str(seed),
        "--data_seed",
        str(data_seed),
        "--split_dataset_ratio",
        str(split_dataset_ratio),
        "--load_from_cache_file",
        str(load_from_cache_file).lower(),
        "--check_model",
        str(check_model).lower(),
    ]
    if model_type:
        cmd.extend(["--model_type", model_type])
    if template:
        cmd.extend(["--template", template])
    if val_file is not None:
        cmd.extend(["--val_dataset", str(val_file)])
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if packing:
        cmd.extend([
            "--packing",
            "true",
            "--packing_strategy",
            packing_strategy,
            "--packing_num_proc",
            str(packing_num_proc),
            "--padding_free",
            "true",
        ])
    if save_only_model:
        cmd.extend(["--save_only_model", "true"])
    if use_logits_to_keep:
        cmd.extend(["--use_logits_to_keep", "true"])
    if resume_from_checkpoint is not None:
        cmd.extend(["--resume_from_checkpoint", str(resume_from_checkpoint)])
        if resume_only_model:
            cmd.extend(["--resume_only_model", "true"])
    if quant_method is not None:
        cmd.extend(["--quant_method", quant_method])
        if quant_bits is not None:
            cmd.extend(["--quant_bits", str(quant_bits)])
    return cmd


def build_swift_export_command(
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    python_executable: str,
    swift_dir: str | Path = DEFAULT_SWIFT_DIR,
    model_path: str | Path | None = None,
    model_type: str | None = None,
    template: str | None = None,
    merge_lora: bool = True,
    torch_dtype: str = "bfloat16",
    exist_ok: bool = True,
) -> list[str]:
    """Build an ms-swift export command for LoRA → HF merged model."""
    export_entry = Path(swift_dir) / "swift" / "cli" / "export.py"
    cmd = [
        python_executable,
        str(export_entry),
        "--adapters",
        str(checkpoint_dir),
        "--output_dir",
        str(output_dir),
        "--torch_dtype",
        torch_dtype,
    ]
    if model_path is not None:
        cmd.extend(["--model", str(model_path)])
    if model_type:
        cmd.extend(["--model_type", model_type])
    if template:
        cmd.extend(["--template", template])
    if merge_lora:
        cmd.extend(["--merge_lora", "true"])
    if exist_ok:
        cmd.extend(["--exist_ok", "true"])
    return cmd
