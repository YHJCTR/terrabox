"""Convert strict SFT rows into prompt-only rollout tasks.

The SFT rows contain gold assistant/tool messages. ReAct rollout must not feed
those messages to the model, so this adapter emits only task-facing fields plus
``expected_tools`` for offline metrics.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..full_shared.sft_schema import FullSFTSample


def sample_to_task(sample: FullSFTSample) -> dict[str, Any]:
    """Return a task dict accepted by ``scripts/run_trajectory_experiment.py``."""
    task: dict[str, Any] = {
        "task_id": sample.task_id,
        "id": sample.task_id,
        "source": sample.source,
        "task_type": sample.task_type,
        "question": sample.question,
        "images": list(sample.images),
        "data_files": list(sample.data_files),
        "expected_tools": list(sample.tool_sequence),
    }
    if sample.data_dir:
        task["data_dir"] = sample.data_dir
    return task


def samples_to_tasks(samples: Iterable[FullSFTSample]) -> list[dict[str, Any]]:
    return [sample_to_task(sample) for sample in samples]


def write_task_file(
    samples: Iterable[FullSFTSample],
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> int:
    """Write rollout task JSON and return the number of tasks written."""
    tasks = samples_to_tasks(samples)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": str(source_path) if source_path else None,
            "num_tasks": len(tasks),
            "format": "terrabox_rollout_tasks",
            "gold_leakage_policy": "messages/gold_tool_calls/ground_truth omitted from model-facing task file",
        },
        "tasks": tasks,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(tasks)

