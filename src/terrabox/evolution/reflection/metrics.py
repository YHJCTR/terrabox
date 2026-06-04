"""Reflection experiment metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ReAct.metrics import write_metrics as write_rollout_metrics
from .data_split import iter_jsonl
from .memory import ReflectionMemoryBank


def retrieval_metrics(task_file: str | Path, bank: ReflectionMemoryBank, *, top_k: int = 5) -> dict[str, Any]:
    payload = json.loads(Path(task_file).read_text(encoding="utf-8"))
    tasks = payload.get("tasks", payload)
    counts = []
    top_tools: dict[str, int] = {}
    for task in tasks:
        retrieved = bank.retrieve(str(task.get("question", "")), top_k=top_k)
        counts.append(len(retrieved))
        for entry in retrieved:
            for tool in entry.tools_called or entry.expected_tools:
                top_tools[tool] = top_tools.get(tool, 0) + 1
    return {
        "task_file": str(task_file),
        "num_tasks": len(tasks),
        "top_k": top_k,
        "avg_retrieved_reflections": sum(counts) / len(counts) if counts else 0.0,
        "min_retrieved_reflections": min(counts) if counts else 0,
        "max_retrieved_reflections": max(counts) if counts else 0,
        "retrieved_tool_counts": dict(sorted(top_tools.items(), key=lambda item: item[1], reverse=True)),
    }


def write_reflection_metrics(
    *,
    rollout_dir: str | Path,
    memory_path: str | Path,
    task_file: str | Path | None = None,
    output_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    bank = ReflectionMemoryBank(memory_path)
    metrics = {
        "rollout": write_rollout_metrics(rollout_dir, Path(output_path).parent / "rollout_metrics.json"),
        "memory": bank.stats(),
    }
    if task_file is not None:
        metrics["retrieval"] = retrieval_metrics(task_file, bank, top_k=top_k)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
