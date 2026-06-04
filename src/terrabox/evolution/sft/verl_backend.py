"""veRL FSDP backend helpers for Terrabox SFT.

The Unsloth path is useful for small smoke runs, but it does not perform true
multi-GPU FSDP. This module prepares veRL-compatible Parquet data and builds the
torchrun command used by ``verl.trainer.sft_trainer``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chat_row_to_verl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one Terrabox chat SFT row to veRL MultiTurnSFTDataset format."""
    return {
        "messages": list(row.get("messages", []) or []),
        "task_id": row.get("task_id") or row.get("id"),
        "source": row.get("source"),
        "task_type": row.get("task_type"),
        "extra_info": {
            "question": row.get("question"),
            "images": list(row.get("images", []) or []),
            "data_files": list(row.get("data_files", []) or []),
            "expected_tools": list(row.get("expected_tools", []) or []),
            "sft_compaction": row.get("sft_compaction"),
        },
    }


def write_verl_sft_parquet(jsonl_path: str | Path, output_path: str | Path) -> int:
    """Write veRL SFT Parquet from Terrabox chat JSONL."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Preparing veRL SFT data requires pandas/pyarrow in the active environment.") from exc

    rows = [chat_row_to_verl_row(row) for row in _load_jsonl(jsonl_path)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return len(rows)


def build_verl_sft_command(
    *,
    verl_dir: str | Path,
    train_file: str | Path,
    val_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    nproc_per_node: int = 4,
    max_length: int = 13312,
    max_token_len_per_gpu: int = 13312,
    micro_batch_size_per_gpu: int = 1,
    train_batch_size: int = 4,
    total_epochs: int = 1,
    learning_rate: float = 2e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_targets: str = "all-linear",
    sequence_parallel_size: int = 1,
    param_offload: bool = False,
    optimizer_offload: bool = True,
    activation_offload: bool = False,
    use_torch_compile: bool = True,
    save_hf_model: bool = False,
    save_freq: str = "after_each_epoch",
    test_freq: str = "after_each_epoch",
    max_ckpt_to_keep: int = 2,
) -> list[str]:
    """Build a veRL FSDP SFT torchrun command.

    ``data.truncation=error`` keeps the no-silent-truncation contract. The
    default checkpoint saves FSDP shards only; full HuggingFace conversion is
    intentionally opt-in because it gathers the 8B model on CPU.
    """
    normalized_model_path = str(model_path).rstrip("/")
    checkpoint_contents = "['model','extra']"
    if save_hf_model:
        checkpoint_contents = "['hf_model','model','extra']"
    return [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={train_batch_size}",
        f"data.micro_batch_size_per_gpu={micro_batch_size_per_gpu}",
        "data.messages_key=messages",
        "data.ignore_input_ids_mismatch=True",
        f"data.max_length={max_length}",
        "data.truncation=error",
        "data.use_dynamic_bsz=True",
        f"data.max_token_len_per_gpu={max_token_len_per_gpu}",
        "data.pad_mode=no_padding",
        "data.num_workers=8",
        f"optim.lr={learning_rate}",
        "engine=fsdp",
        "engine.strategy=fsdp",
        "engine.dtype=bfloat16",
        f"engine.param_offload={param_offload}",
        f"engine.optimizer_offload={optimizer_offload}",
        f"engine.use_torch_compile={use_torch_compile}",
        "engine.reshard_after_forward=True",
        f"engine.ulysses_sequence_parallel_size={sequence_parallel_size}",
        f"model.path={normalized_model_path}",
        "model.trust_remote_code=True",
        "model.use_remove_padding=True",
        "model.enable_gradient_checkpointing=True",
        f"model.enable_activation_offload={activation_offload}",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        f"model.target_modules={lora_targets}",
        f"checkpoint.save_contents={checkpoint_contents}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.project_name=terrabox-sft",
        "trainer.experiment_name=terrabox-sft-verl",
        "trainer.logger=['console']",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.max_ckpt_to_keep={max_ckpt_to_keep}",
        "trainer.resume_mode=auto",
        f"trainer.n_gpus_per_node={nproc_per_node}",
        "trainer.nnodes=1",
    ]


def latest_hf_model_path(checkpoint_dir: str | Path) -> Path | None:
    """Return the newest veRL HuggingFace checkpoint directory, if present."""
    root = Path(checkpoint_dir)
    candidates = sorted(
        root.glob("global_step_*/huggingface"),
        key=lambda p: int(p.parent.name.rsplit("_", 1)[-1]) if p.parent.name.rsplit("_", 1)[-1].isdigit() else -1,
    )
    return candidates[-1] if candidates else None
