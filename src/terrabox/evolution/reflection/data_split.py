"""Shared deterministic shuffle and task-slice utilities for reflection experiments."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..ReAct.data_adapter import samples_to_tasks
from ..full_shared.sft_schema import FullSFTSample, load_sft_samples


# OEA full OpenEarth split (23 OE tools). train.jsonl is the reflection TRAIN pool;
# the fixed TEST set is data/oea_full_sft/openearth_test_tasks.json (prompt-only,
# already produced by the ReAct experiment) — pass it directly as the eval --task-file.
DEFAULT_STRICT_DATA = "data/oea_full_sft/openearth/train.jsonl"
DEFAULT_SHUFFLED_DATA = "data/oea_full_sft/openearth/train_shuffled_seed42.jsonl"


def load_shuffled_samples(
    path: str | Path = DEFAULT_STRICT_DATA,
    *,
    seed: int = 42,
    limit: int | None = None,
) -> list[FullSFTSample]:
    """Load strict SFT samples and return a deterministic shuffled order."""
    samples = load_sft_samples(path)
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:limit] if limit is not None else samples


def write_shuffled_jsonl(
    input_path: str | Path = DEFAULT_STRICT_DATA,
    output_path: str | Path = DEFAULT_SHUFFLED_DATA,
    *,
    seed: int = 42,
) -> int:
    """Write a deterministic shuffled copy of the strict SFT JSONL rows."""
    samples = load_shuffled_samples(input_path, seed=seed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.raw, ensure_ascii=False) + "\n")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "shuffle_seed": seed,
        "num_samples": len(samples),
    }
    out.with_suffix(out.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(samples)


def sample_slice(samples: list[FullSFTSample], *, start: int = 0, limit: int | None = None) -> list[FullSFTSample]:
    """Return a stable sample slice."""
    end = None if limit is None else start + limit
    return samples[start:end]


def write_task_slice(
    strict_data: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    start: int = 0,
    limit: int | None = None,
    split_name: str = "split",
) -> int:
    """Write a prompt-only rollout task JSON for one shuffled data slice."""
    samples = sample_slice(load_shuffled_samples(strict_data, seed=seed), start=start, limit=limit)
    tasks = samples_to_tasks(samples)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": str(strict_data),
            "split_name": split_name,
            "shuffle_seed": seed,
            "slice_start": start,
            "slice_limit": limit,
            "num_tasks": len(tasks),
            "format": "terrabox_rollout_tasks",
            "gold_leakage_policy": "messages/gold_tool_calls/ground_truth omitted from model-facing task file",
        },
        "tasks": tasks,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(tasks)


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
