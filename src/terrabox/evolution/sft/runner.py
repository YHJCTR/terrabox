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
from typing import Any

from ..ReAct.runner import build_rollout_env
from ..reflection.data_split import load_shuffled_samples, sample_slice
from .data_adapter import build_replayed_sft_dataset, write_chat_jsonl, write_eval_task_file
from .swift_backend import DEFAULT_SWIFT_DIR, build_swift_export_command, build_swift_sft_command
from .verl_backend import build_verl_sft_command, write_verl_sft_parquet


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
# Aligned, de-collapsed strict data (44 tools) — the SAME source ReAct/Reflection
# now evaluate on. (Pre-alignment data lived in data/newdata; use fixdata_decollapse
# so the SFT model is trained on the same canonical, executable tool schema.)
DEFAULT_DATA = "data/fixdata_decollapse/sft_train_strict.jsonl"
DEFAULT_MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
DEFAULT_QWEN25_3B_MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct"
DEFAULT_CURRENT_HARNESS_SFT_TRAIN = "data/oea_current_harness_sft/swift_train_messages.jsonl"
DEFAULT_CURRENT_HARNESS_SFT_EVAL = "data/oea_current_harness_sft/swift_eval_messages.jsonl"
DEFAULT_UNSLOTH_PYTHON = "/home/yuhongjie/miniconda3/envs/unsloth/bin/python"
DEFAULT_UNSLOTH_BIN = "/home/yuhongjie/miniconda3/envs/unsloth/bin"
DEFAULT_VERL_DIR = "/data1/yuhongjie2/verl"
LOCAL_PYTHON_DEPS = REPO_ROOT / "tmp" / "python_deps"


def agent_rl_sft_dir(experiment: str) -> Path:
    return REPO_ROOT / "tmp" / "agent_rl_runs" / "sft" / experiment


def _swift_env(*, swift_dir: Path, cuda_visible_devices: str | None = None) -> dict[str, str]:
    """Environment shared by Swift SFT/export scripts and launch-time checks."""
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PATH"] = f"{DEFAULT_UNSLOTH_BIN}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{LOCAL_PYTHON_DEPS}:{swift_dir}:{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _run_json_probe(cmd: list[str], *, env: dict[str, str]) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail or f"Probe failed with exit code {proc.returncode}: {' '.join(cmd)}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover - defensive CLI diagnostics
        raise RuntimeError(f"Probe did not return JSON. stdout={proc.stdout!r} stderr={proc.stderr!r}") from exc


def _swift_length_preflight(
    *,
    train_file: str | Path,
    val_file: str | Path | None,
    model_path: str | Path,
    max_length: int,
    report_path: str | Path,
    allow_overlength: bool,
) -> dict[str, Any]:
    """Audit Swift chat rows with the same tokenizer before training.

    Swift's ``truncation_strategy=delete`` can otherwise remove long replay
    episodes without a visible failure.  We record the distribution and fail
    by default so a caller must explicitly choose a larger window or opt into
    the truncation policy.
    """
    if max_length <= 0:
        raise ValueError("--max-length must be positive")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Swift length preflight requires transformers in the active environment") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load the local tokenizer for Swift length preflight: {model_path}. "
            "Use a local Qwen checkpoint or explicitly pass --skip-length-preflight for diagnostics."
        ) from exc

    def read_rows(path: str | Path | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        rows: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows

    def row_tokens(row: dict[str, Any]) -> int:
        messages = row.get("messages") or []
        if hasattr(tokenizer, "apply_chat_template"):
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if isinstance(encoded, dict):
                encoded = encoded.get("input_ids") or []
            return len(encoded)
        rendered = "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}" for message in messages
        )
        return len(tokenizer(rendered, add_special_tokens=False).input_ids)

    def summarize(path: str | Path | None, split_name: str) -> dict[str, Any]:
        rows = read_rows(path)
        lengths: list[int] = []
        overlong: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            length = row_tokens(row)
            lengths.append(length)
            if length > max_length:
                overlong.append(
                    {
                        "row_index": index,
                        "task_id": str(row.get("task_id") or row.get("id") or f"row_{index}"),
                        "tokens": length,
                    }
                )
        ordered = sorted(lengths)

        def percentile(fraction: float) -> int:
            if not ordered:
                return 0
            return ordered[int((len(ordered) - 1) * fraction)]

        return {
            "split_name": split_name,
            "path": str(path) if path else "",
            "num_rows": len(rows),
            "min_tokens": min(lengths) if lengths else 0,
            "p50_tokens": percentile(0.50),
            "p90_tokens": percentile(0.90),
            "p95_tokens": percentile(0.95),
            "p99_tokens": percentile(0.99),
            "max_tokens": max(lengths) if lengths else 0,
            "max_length": max_length,
            "overlong_count": len(overlong),
            "overlong_examples": overlong[:20],
        }

    report: dict[str, Any] = {
        "model_path": str(model_path),
        "max_length": max_length,
        "allow_overlength": allow_overlength,
        "train": summarize(train_file, "train"),
        "val": summarize(val_file, "val") if val_file else None,
    }
    report["overlong_total"] = int(report["train"]["overlong_count"])
    if report.get("val"):
        report["overlong_total"] += int(report["val"]["overlong_count"])
    report["status"] = "allowed" if report["overlong_total"] and allow_overlength else (
        "ok" if not report["overlong_total"] else "error"
    )
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["overlong_total"] and not allow_overlength:
        raise ValueError(
            f"Swift SFT length preflight found {report['overlong_total']} rows over "
            f"max_length={max_length}. Increase --max-length (replayed OEA data usually needs "
            "8192–10240) or explicitly pass --allow-overlength after reviewing "
            f"{report_file}; no rows were silently deleted."
        )
    return report


def _swift_preflight(
    *,
    python_executable: str,
    swift_dir: str | Path,
    cuda_visible_devices: str,
    torch_dtype: str,
    packing: bool,
) -> dict[str, object]:
    """Fail fast before Swift starts tokenization/model loading."""
    swift_path = Path(swift_dir)
    sft_entry = swift_path / "swift" / "cli" / "sft.py"
    export_entry = swift_path / "swift" / "cli" / "export.py"
    if not sft_entry.exists():
        raise FileNotFoundError(f"Missing ms-swift sft.py under: {swift_path}")
    if not export_entry.exists():
        raise FileNotFoundError(f"Missing ms-swift export.py under: {swift_path}")

    env = _swift_env(swift_dir=swift_path, cuda_visible_devices=cuda_visible_devices)
    torch_probe = _run_json_probe(
        [
            python_executable,
            "-c",
            (
                "import json, torch; "
                "available=torch.cuda.is_available(); "
                "count=torch.cuda.device_count() if available else 0; "
                "names=[torch.cuda.get_device_name(i) for i in range(count)] if available else []; "
                "bf16=bool(torch.cuda.is_bf16_supported()) if available else False; "
                "print(json.dumps({'cuda_available': available, 'device_count': count, "
                "'device_names': names, 'bf16_supported': bf16}))"
            ),
        ],
        env=env,
    )
    if not torch_probe.get("cuda_available"):
        raise RuntimeError(
            "Swift SFT requires a visible CUDA GPU, but torch.cuda.is_available() is false. "
            "Check full-access/tmux/container GPU visibility before launching."
        )
    if int(torch_probe.get("device_count") or 0) < 1:
        raise RuntimeError("CUDA is visible but no devices are exposed after CUDA_VISIBLE_DEVICES filtering.")

    requested = torch_dtype.lower()
    if requested == "auto":
        resolved_dtype = "bfloat16" if torch_probe.get("bf16_supported") else "float16"
    elif requested in {"bf16", "bfloat16"}:
        if not torch_probe.get("bf16_supported"):
            raise RuntimeError("Requested bfloat16, but this GPU/PyTorch setup does not support bf16; use --torch-dtype float16.")
        resolved_dtype = "bfloat16"
    elif requested in {"fp16", "float16"}:
        resolved_dtype = "float16"
    elif requested in {"fp32", "float32"}:
        resolved_dtype = "float32"
    else:
        resolved_dtype = torch_dtype

    flash_probe = _run_json_probe(
        [
            python_executable,
            "-c",
            (
                "import importlib.util, json; "
                "print(json.dumps({'flash_attn_available': importlib.util.find_spec('flash_attn') is not None}))"
            ),
        ],
        env=env,
    )
    if packing and not flash_probe.get("flash_attn_available"):
        raise RuntimeError("Swift packing was requested, but flash_attn is not importable. Disable --packing or install flash-attn.")

    return {
        "cuda_visible_devices": cuda_visible_devices,
        "torch": torch_probe,
        "flash_attn_available": bool(flash_probe.get("flash_attn_available")),
        "requested_torch_dtype": torch_dtype,
        "resolved_torch_dtype": resolved_dtype,
        "packing": packing,
        "swift_dir": str(swift_path),
    }


def cmd_swift_preflight(args: argparse.Namespace) -> None:
    report = _swift_preflight(
        python_executable=args.python_executable,
        swift_dir=args.swift_dir,
        cuda_visible_devices=args.cuda_visible_devices,
        torch_dtype=args.torch_dtype,
        packing=args.packing,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


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
    system_prompt_file = data_dir / "system_prompt.txt"
    system_prompt = ""
    if train_file.exists():
        first_line = train_file.read_text(encoding="utf-8").splitlines()
        if first_line:
            try:
                first_row = json.loads(first_line[0])
                system_prompt = str((first_row.get("messages") or [{}])[0].get("content") or "")
            except (json.JSONDecodeError, AttributeError):
                system_prompt = ""
    if not system_prompt and val_file.exists():
        first_line = val_file.read_text(encoding="utf-8").splitlines()
        if first_line:
            try:
                first_row = json.loads(first_line[0])
                system_prompt = str((first_row.get("messages") or [{}])[0].get("content") or "")
            except (json.JSONDecodeError, AttributeError):
                system_prompt = ""
    system_prompt_file.write_text(system_prompt, encoding="utf-8")
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
        "system_prompt_file": str(system_prompt_file),
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
    eval_strategy: str = "steps",
    dataset_num_proc: int = 8,
    chat_format: str = "auto",
    preflight_only: bool = False,
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
        "--eval-strategy",
        str(eval_strategy),
        "--dataset-num-proc",
        str(dataset_num_proc),
        "--chat-format",
        str(chat_format),
    ]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if save_merged_model:
        cmd.append("--save-merged-model")
    if drop_overlength:
        cmd.append("--drop-overlength")
    if preflight_only:
        cmd.append("--preflight-only")
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
    tool_protocol: str = "sft-json",
    sft_system_prompt_file: str | Path | None = None,
    agent_context_length: int = 24576,
    max_completion_tokens: int = 4096,
    workers: int = 1,
    python_executable: str | None = None,
) -> list[str]:
    """Build a prompt-only rollout command for a trained SFT checkpoint."""
    if tool_protocol not in {"native", "sft-json"}:
        raise ValueError(f"Unsupported SFT rollout tool protocol: {tool_protocol}")
    if tool_protocol == "sft-json" and not sft_system_prompt_file:
        raise ValueError("sft-json rollout requires --sft-system-prompt-file")
    if workers < 1:
        raise ValueError("rollout workers must be >= 1")
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
        "--workers",
        str(workers),
        "--tool-protocol",
        tool_protocol,
        "--agent-context-length",
        str(agent_context_length),
        "--max-completion-tokens",
        str(max_completion_tokens),
        "--use-docker",
        "--resume",
    ]
    if sft_system_prompt_file:
        cmd.extend(["--sft-system-prompt-file", str(sft_system_prompt_file)])
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


def cmd_prepare_replayed_data(args: argparse.Namespace) -> None:
    """Build JSON-actions SFT data from completed real Terrabox replays."""
    manifest = build_replayed_sft_dataset(
        args.source_data,
        args.replay_dir,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


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
        eval_strategy=args.eval_strategy,
        dataset_num_proc=args.dataset_num_proc,
        chat_format=args.chat_format,
        preflight_only=args.preflight_only,
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


def cmd_train_swift(args: argparse.Namespace) -> None:
    """Write or launch current-harness SFT with ms-swift as the primary backend."""
    exp_dir = experiment_dir(args.experiment)
    run_root = Path(args.run_dir or agent_rl_sft_dir(args.experiment))
    model_dir = Path(args.output_model_dir or run_root / "swift")
    train_file = Path(args.train_file)
    val_file = Path(args.val_file) if args.val_file else None
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if val_file is not None and not val_file.exists():
        raise FileNotFoundError(f"Missing validation file: {val_file}")
    swift_dir = Path(args.swift_dir)
    if not (swift_dir / "swift" / "cli" / "sft.py").exists():
        raise FileNotFoundError(f"Missing ms-swift source checkout or sft.py under: {swift_dir}")
    run_root.mkdir(parents=True, exist_ok=True)
    if getattr(args, "skip_length_preflight", False):
        length_preflight: dict[str, Any] = {
            "status": "skipped",
            "reason": "--skip-length-preflight",
            "model_path": str(args.model_path),
            "max_length": args.max_length,
        }
        (run_root / "swift_length_preflight.json").write_text(
            json.dumps(length_preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        length_preflight = _swift_length_preflight(
            train_file=train_file,
            val_file=val_file,
            model_path=args.model_path,
            max_length=args.max_length,
            report_path=run_root / "swift_length_preflight.json",
            allow_overlength=getattr(args, "allow_overlength", False),
        )
    if args.skip_preflight:
        requested_dtype = args.torch_dtype.lower()
        resolved_torch_dtype = "bfloat16" if requested_dtype == "auto" else args.torch_dtype
        preflight: dict[str, object] = {
            "skipped": True,
            "requested_torch_dtype": args.torch_dtype,
            "resolved_torch_dtype": resolved_torch_dtype,
            "reason": "--skip-preflight",
        }
    else:
        preflight = _swift_preflight(
            python_executable=args.python_executable,
            swift_dir=swift_dir,
            cuda_visible_devices=args.cuda_visible_devices,
            torch_dtype=args.torch_dtype,
            packing=args.packing,
        )
        resolved_torch_dtype = str(preflight["resolved_torch_dtype"])

    cmd = build_swift_sft_command(
        train_file=train_file,
        val_file=val_file,
        model_path=args.model_path,
        model_type=args.model_type,
        template=args.template,
        output_dir=model_dir,
        python_executable=args.python_executable,
        swift_dir=swift_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        packing=args.packing,
        packing_strategy=args.packing_strategy,
        packing_num_proc=args.packing_num_proc,
        truncation_strategy=args.truncation_strategy,
        dataset_num_proc=args.dataset_num_proc,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        save_only_model=args.save_only_model,
        use_logits_to_keep=args.use_logits_to_keep,
        torch_dtype=resolved_torch_dtype,
        tuner_backend=args.tuner_backend,
        target_modules=args.target_modules,
        seed=args.seed,
        data_seed=args.data_seed,
        split_dataset_ratio=args.split_dataset_ratio,
        load_from_cache_file=args.load_from_cache_file,
        check_model=args.check_model,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_only_model=args.resume_only_model,
        quant_method=args.quant_method,
        quant_bits=args.quant_bits,
    )
    script = exp_dir / "run_swift_sft_train.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "backend": "ms-swift",
        "experiment": args.experiment,
        "run_root": str(run_root),
        "train_file": str(train_file),
        "val_file": str(val_file) if val_file else None,
        "model_path": str(args.model_path),
        "model_type": args.model_type,
        "template": args.template,
        "output_model_dir": str(model_dir),
        "swift_dir": str(swift_dir),
        "cuda_visible_devices": args.cuda_visible_devices,
        "max_length": args.max_length,
        "packing": args.packing,
        "packing_strategy": args.packing_strategy,
        "truncation_strategy": args.truncation_strategy,
        "dataset_num_proc": args.dataset_num_proc,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_persistent_workers": args.dataloader_persistent_workers,
        "requested_torch_dtype": args.torch_dtype,
        "resolved_torch_dtype": resolved_torch_dtype,
        "preflight": preflight,
        "length_preflight": length_preflight,
        "allow_overlength": getattr(args, "allow_overlength", False),
        "assistant_only_loss": "swift loss_scale=default; tool/tool_response tokens are excluded by Swift template",
    }
    (run_root / "swift_sft_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        f"export PATH={DEFAULT_UNSLOTH_BIN}:$PATH\n"
        f"export PYTHONPATH={LOCAL_PYTHON_DEPS}:{swift_dir}:{REPO_ROOT / 'src'}:${{PYTHONPATH:-}}\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "export TOKENIZERS_PARALLELISM=false\n"
        "export HF_HUB_OFFLINE=1\n"
        "export TRANSFORMERS_OFFLINE=1\n"
        + (
            " ".join(
                shlex.quote(str(part))
                for part in [
                    args.python_executable,
                    "-m",
                    "terrabox.evolution.sft.runner",
                    "swift-preflight",
                    "--swift-dir",
                    str(swift_dir),
                    "--python-executable",
                    args.python_executable,
                    "--torch-dtype",
                    resolved_torch_dtype,
                ]
                + (["--packing"] if args.packing else [])
            )
            + " --cuda-visible-devices \"$CUDA_VISIBLE_DEVICES\""
            + f" 2>&1 | tee {shlex.quote(str(run_root / 'swift_sft_preflight.log'))}\n"
            if not args.skip_preflight
            else ""
        )
        + " ".join(shlex.quote(str(part)) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(run_root / 'swift_sft_train.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(json.dumps({"script": str(script), "run_root": str(run_root), "model_dir": str(model_dir)}, ensure_ascii=False, indent=2))
    if args.launch:
        env = _swift_env(swift_dir=swift_dir, cuda_visible_devices=args.cuda_visible_devices)
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated Swift SFT script before launching.")


def _find_latest_swift_checkpoint(model_dir: Path) -> Path | None:
    """Return the newest checkpoint-* under a Swift output directory."""
    candidates: list[tuple[int, Path]] = []
    for child in model_dir.rglob("checkpoint-*"):
        if not child.is_dir():
            continue
        try:
            step = int(child.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if (child / "adapter_config.json").exists() or any(child.glob("*.safetensors")):
            candidates.append((step, child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def cmd_export_swift(args: argparse.Namespace) -> None:
    """Write or launch Swift LoRA export to a ReAct/vLLM-servable HF directory."""
    exp_dir = experiment_dir(args.experiment)
    run_root = Path(args.run_dir or agent_rl_sft_dir(args.experiment))
    model_dir = Path(args.model_dir or run_root / "swift")
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        latest = _find_latest_swift_checkpoint(model_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint-* with adapter files found under {model_dir}")
        checkpoint_dir = latest
    target_dir = Path(args.target_dir or run_root / "merged_hf" / checkpoint_dir.name)
    swift_dir = Path(args.swift_dir)
    if not (swift_dir / "swift" / "cli" / "export.py").exists():
        raise FileNotFoundError(f"Missing ms-swift export.py under: {swift_dir}")
    cmd = build_swift_export_command(
        checkpoint_dir=checkpoint_dir,
        output_dir=target_dir,
        python_executable=args.python_executable,
        swift_dir=swift_dir,
        model_path=args.model_path,
        model_type=args.model_type,
        template=args.template,
        merge_lora=not args.no_merge_lora,
        torch_dtype=args.torch_dtype,
        exist_ok=args.exist_ok,
    )
    script = exp_dir / "run_swift_export.sh"
    run_root.mkdir(parents=True, exist_ok=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        f"export PATH={DEFAULT_UNSLOTH_BIN}:$PATH\n"
        f"export PYTHONPATH={LOCAL_PYTHON_DEPS}:{swift_dir}:{REPO_ROOT / 'src'}:${{PYTHONPATH:-}}\n"
        "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}\n"
        "export HF_HUB_OFFLINE=1\n"
        "export TRANSFORMERS_OFFLINE=1\n"
        + " ".join(shlex.quote(str(part)) for part in cmd)
        + f" 2>&1 | tee {shlex.quote(str(run_root / 'swift_export.log'))}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(json.dumps({
        "checkpoint_dir": str(checkpoint_dir),
        "target_dir": str(target_dir),
        "script": str(script),
    }, ensure_ascii=False, indent=2))
    if args.launch:
        env = os.environ.copy()
        env["PATH"] = f"{DEFAULT_UNSLOTH_BIN}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{LOCAL_PYTHON_DEPS}:{swift_dir}:{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        subprocess.run(["bash", str(script)], cwd=REPO_ROOT, env=env, check=True)
    else:
        print("Review the generated Swift export script before launching.")


def cmd_rollout(args: argparse.Namespace) -> None:
    exp_dir = experiment_dir(args.experiment)
    out_dir = Path(args.output_dir or exp_dir / "eval")
    task_file = Path(args.task_file or exp_dir / "eval_tasks.json")
    model_path = Path(args.model_path or MODULE_ROOT / "model" / args.experiment / "merged")
    tool_protocol = getattr(args, "tool_protocol", "sft-json")
    prompt_file = getattr(args, "sft_system_prompt_file", "") or ""
    if tool_protocol == "sft-json" and not prompt_file:
        candidates = [
            exp_dir / "sft_data" / "system_prompt.txt",
            exp_dir / "system_prompt.txt",
            Path(args.output_dir).parent / "system_prompt.txt" if args.output_dir else Path(),
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                prompt_file = str(candidate)
                break
        if not prompt_file:
            raise FileNotFoundError(
                "SFT JSON-actions rollout requires a training system prompt. Pass "
                "--sft-system-prompt-file or run prepare-data/prepare-replayed-data first."
            )
    cmd = build_sft_rollout_command(
        task_file=task_file,
        experiment=f"{args.experiment}_sft_eval",
        output_dir=out_dir,
        model_path=model_path,
        port=args.port,
        start_index=args.start_index,
        limit=args.limit,
        max_iterations=args.max_iterations,
        tool_protocol=tool_protocol,
        sft_system_prompt_file=prompt_file or None,
        agent_context_length=getattr(args, "agent_context_length", 24576),
        max_completion_tokens=getattr(args, "max_completion_tokens", 4096),
        workers=1,
    )
    run_script = exp_dir / "run_sft_eval.sh"
    run_script.parent.mkdir(parents=True, exist_ok=True)
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
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

    p_replayed = sub.add_parser(
        "prepare-replayed-data",
        help="Build assistant-only JSON-actions SFT data from real gold-replay results",
    )
    p_replayed.add_argument("--source-data", required=True, help="Selected OEA train JSONL")
    p_replayed.add_argument("--replay-dir", required=True, help="gold-replay output directory")
    p_replayed.add_argument("--output-dir", required=True, help="独立 SFT 数据目录")
    p_replayed.add_argument("--validation-fraction", type=float, default=0.1)
    p_replayed.add_argument("--seed", type=int, default=42)
    p_replayed.add_argument("--overwrite", action="store_true",
                            help="显式覆盖输出目录中已有的 replayed SFT 文件")
    p_replayed.set_defaults(func=cmd_prepare_replayed_data)

    p_train = sub.add_parser("train", help="Legacy: write or launch Unsloth/TRL QLoRA SFT training")
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
    p_train.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    p_train.add_argument("--dataset-num-proc", type=int, default=8)
    p_train.add_argument("--chat-format", choices=["auto", "tokenizer", "current_harness"], default="auto")
    p_train.add_argument("--preflight-only", action="store_true")
    p_train.add_argument("--save-merged-model", action="store_true", default=True)
    p_train.add_argument("--no-save-merged-model", dest="save_merged_model", action="store_false")
    p_train.add_argument("--drop-overlength", action="store_true",
                         help="超长行丢弃而非报错(verbatim 保留其余,与 rollout 对齐,不做有损截断)")
    p_train.add_argument("--launch", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_swift_preflight = sub.add_parser("swift-preflight", help="Check CUDA, dtype, flash-attn and Swift CLI before Swift SFT")
    p_swift_preflight.add_argument("--swift-dir", default=DEFAULT_SWIFT_DIR)
    p_swift_preflight.add_argument("--python-executable", default=DEFAULT_UNSLOTH_PYTHON)
    p_swift_preflight.add_argument("--cuda-visible-devices", default="2")
    p_swift_preflight.add_argument("--torch-dtype", default="auto")
    p_swift_preflight.add_argument("--packing", action="store_true", default=False)
    p_swift_preflight.set_defaults(func=cmd_swift_preflight)

    p_swift = sub.add_parser("train-swift", help="Primary: write or launch ms-swift LoRA SFT training")
    p_swift.add_argument("--experiment", required=True)
    p_swift.add_argument("--train-file", default=DEFAULT_CURRENT_HARNESS_SFT_TRAIN)
    p_swift.add_argument("--val-file", default=DEFAULT_CURRENT_HARNESS_SFT_EVAL)
    p_swift.add_argument("--model-path", default=DEFAULT_QWEN25_3B_MODEL_PATH)
    p_swift.add_argument("--model-type", default="qwen2")
    p_swift.add_argument("--template", default="qwen2_5")
    p_swift.add_argument("--run-dir", help="Run root (default: tmp/agent_rl_runs/sft/<experiment>)")
    p_swift.add_argument("--output-model-dir", help="Swift checkpoint output dir (default: <run-dir>/swift)")
    p_swift.add_argument("--swift-dir", default=DEFAULT_SWIFT_DIR)
    p_swift.add_argument("--python-executable", default=DEFAULT_UNSLOTH_PYTHON)
    p_swift.add_argument("--cuda-visible-devices", default="2")
    p_swift.add_argument("--max-steps", type=int)
    p_swift.add_argument("--num-train-epochs", type=float, default=1.0)
    p_swift.add_argument("--learning-rate", type=float, default=1e-4)
    p_swift.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_swift.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p_swift.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_swift.add_argument("--lora-rank", type=int, default=8)
    p_swift.add_argument("--lora-alpha", type=int, default=32)
    p_swift.add_argument("--lora-dropout", type=float, default=0.05)
    p_swift.add_argument("--max-length", type=int, default=4096)
    p_swift.add_argument(
        "--allow-overlength",
        action="store_true",
        help="允许 Swift 按 --truncation-strategy 处理超长样本；默认训练前发现超长行直接失败",
    )
    p_swift.add_argument(
        "--skip-length-preflight",
        action="store_true",
        help="仅用于诊断：跳过 tokenizer 长度预检，可能重新引入静默删除/截断风险",
    )
    p_swift.add_argument(
        "--packing",
        action="store_true",
        default=False,
        help="启用 Swift packing；需要可用的 flash-attn，默认关闭以保证单卡稳定性",
    )
    p_swift.add_argument("--no-packing", dest="packing", action="store_false")
    p_swift.add_argument("--packing-strategy", choices=["binpack", "sequential"], default="sequential")
    p_swift.add_argument("--packing-num-proc", type=int, default=1)
    p_swift.add_argument("--truncation-strategy", choices=["delete", "left", "right", "split"], default="delete")
    p_swift.add_argument("--dataset-num-proc", type=int, default=1)
    p_swift.add_argument("--dataloader-num-workers", type=int, default=1)
    p_swift.add_argument("--dataloader-persistent-workers", action="store_true", default=False)
    p_swift.add_argument("--no-dataloader-persistent-workers", dest="dataloader_persistent_workers", action="store_false")
    p_swift.add_argument("--save-steps", type=int, default=200)
    p_swift.add_argument("--save-total-limit", type=int, default=2)
    p_swift.add_argument("--eval-steps", type=int, default=200)
    p_swift.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    p_swift.add_argument("--logging-steps", type=int, default=5)
    p_swift.add_argument("--warmup-ratio", type=float, default=0.03)
    p_swift.add_argument("--lr-scheduler-type", default="cosine")
    p_swift.add_argument("--report-to", default="tensorboard")
    p_swift.add_argument("--tuner-backend", choices=["peft", "unsloth"], default="peft")
    p_swift.add_argument("--target-modules", default="all-linear")
    p_swift.add_argument("--torch-dtype", default="auto", help="auto selects bfloat16 when supported, otherwise float16")
    p_swift.add_argument("--seed", type=int, default=42)
    p_swift.add_argument("--data-seed", type=int, default=42)
    p_swift.add_argument("--split-dataset-ratio", type=float, default=0.0)
    p_swift.add_argument("--load-from-cache-file", action="store_true", default=True)
    p_swift.add_argument("--no-load-from-cache-file", dest="load_from_cache_file", action="store_false")
    p_swift.add_argument("--check-model", action="store_true", default=False)
    p_swift.add_argument("--save-only-model", action="store_true", default=True)
    p_swift.add_argument("--no-save-only-model", dest="save_only_model", action="store_false")
    p_swift.add_argument("--use-logits-to-keep", action="store_true", default=True)
    p_swift.add_argument("--no-use-logits-to-keep", dest="use_logits_to_keep", action="store_false")
    p_swift.add_argument("--resume-from-checkpoint")
    p_swift.add_argument("--resume-only-model", action="store_true", default=False)
    p_swift.add_argument("--quant-method")
    p_swift.add_argument("--quant-bits", type=int)
    p_swift.add_argument("--skip-preflight", action="store_true", help="Only for script generation/debug; launch normally keeps preflight enabled")
    p_swift.add_argument("--launch", action="store_true")
    p_swift.set_defaults(func=cmd_train_swift)

    p_swift_export = sub.add_parser("export-swift", help="Export Swift LoRA checkpoint to merged HF model")
    p_swift_export.add_argument("--experiment", required=True)
    p_swift_export.add_argument("--run-dir", help="Run root (default: tmp/agent_rl_runs/sft/<experiment>)")
    p_swift_export.add_argument("--model-dir", help="Swift checkpoint dir (default: <run-dir>/swift)")
    p_swift_export.add_argument("--checkpoint-dir", help="Specific checkpoint-* dir (default: newest under model-dir)")
    p_swift_export.add_argument("--target-dir", help="Merged HF output dir (default: <run-dir>/merged_hf/<checkpoint>)")
    p_swift_export.add_argument("--model-path", default=DEFAULT_QWEN25_3B_MODEL_PATH)
    p_swift_export.add_argument("--model-type", default="qwen2")
    p_swift_export.add_argument("--template", default="qwen2_5")
    p_swift_export.add_argument("--swift-dir", default=DEFAULT_SWIFT_DIR)
    p_swift_export.add_argument("--python-executable", default=DEFAULT_UNSLOTH_PYTHON)
    p_swift_export.add_argument("--torch-dtype", default="bfloat16")
    p_swift_export.add_argument("--no-merge-lora", action="store_true")
    p_swift_export.add_argument("--exist-ok", action="store_true", default=True)
    p_swift_export.add_argument("--no-exist-ok", dest="exist_ok", action="store_false")
    p_swift_export.add_argument("--launch", action="store_true")
    p_swift_export.set_defaults(func=cmd_export_swift)

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
    p_eval.add_argument(
        "--tool-protocol",
        choices=["native", "sft-json"],
        default="sft-json",
        help="SFT 默认使用 JSON-actions；只有原生 tool-call checkpoint 才选 native",
    )
    p_eval.add_argument("--sft-system-prompt-file", help="训练时使用的 system prompt.txt")
    p_eval.add_argument("--agent-context-length", type=int, default=24576)
    p_eval.add_argument("--max-completion-tokens", type=int, default=4096)
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
