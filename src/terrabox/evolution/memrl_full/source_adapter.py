"""Adapters from Terrabox SFT/rollout data to MemRL source records."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..full_shared.sft_schema import load_sft_samples
from .trajectory_formatter import format_real_trajectory, format_sft_trajectory


@dataclass
class MemRLSourceRecord:
    task_id: str
    task_description: str
    trajectory: str
    success: bool
    reward: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_sft_as_memrl_records(
    path: str | Path,
    *,
    limit: int | None = None,
    expected_tool_overrides: dict[str, list[str]] | None = None,
) -> list[MemRLSourceRecord]:
    samples = load_sft_samples(path, limit=limit)
    records: list[MemRLSourceRecord] = []
    for sample in samples:
        source = sample.source or "unknown"
        sft_tool_sequence = sample.tool_sequence
        tool_sequence = list(
            (expected_tool_overrides or {}).get(sample.task_id) or sft_tool_sequence
        )
        records.append(
            MemRLSourceRecord(
                task_id=sample.task_id,
                task_description=sample.question,
                trajectory=format_sft_trajectory(sample),
                success=True,
                reward=1.0,
                metadata={
                    "source_benchmark": f"terrabox_{source}_sft",
                    "source": source,
                    "task_type": sample.task_type,
                    "expected_tools": tool_sequence,
                    "tool_sequence": tool_sequence,
                    "sft_tool_sequence": sft_tool_sequence,
                    "images": sample.images,
                    "data_files": sample.data_files,
                    "origin": "sft",
                },
            )
        )
    return records


def _success_from_rollout(row: dict[str, Any]) -> bool:
    if "real_success" in row:
        return bool(row["real_success"])
    if "success" in row:
        return bool(row["success"])
    metrics = row.get("metrics") or {}
    return float(metrics.get("f1", row.get("f1", 0.0)) or 0.0) > 0.0


def _reward_from_rollout(row: dict[str, Any], success: bool) -> float:
    if success:
        return 1.0
    metrics = row.get("metrics") or {}
    try:
        return max(0.0, min(1.0, float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _failure_type(row: dict[str, Any]) -> str | None:
    if _success_from_rollout(row):
        return None
    if not row.get("conversation_history"):
        return f"{row.get('status', 'unknown')}_empty_conversation"
    if row.get("has_tool_oom"):
        return "tool_oom"
    if row.get("error"):
        return "exception"
    return str(row.get("status", "failed"))


def load_trajectories_as_memrl_records(
    path: str | Path,
    *,
    limit: int | None = None,
    min_f1: float | None = None,
    include_empty_failures: bool = False,
) -> list[MemRLSourceRecord]:
    records: list[MemRLSourceRecord] = []
    for row in _iter_jsonl(path):
        metrics = row.get("metrics") or {}
        f1 = float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)
        success = _success_from_rollout(row)
        if min_f1 is not None and f1 < min_f1:
            continue
        if not success and not include_empty_failures and not row.get("conversation_history"):
            continue
        source = str(row.get("source") or "unknown")
        failure_type = _failure_type(row)
        metadata = {
            "source_benchmark": f"terrabox_{source}_rollout",
            "source": source,
            "task_type": row.get("task_type", "unknown"),
            "expected_tools": row.get("expected_tools", []),
            "tool_sequence": row.get("tool_sequence") or row.get("tools_called") or row.get("tool_calls") or [],
            "status": row.get("status", "unknown"),
            "metrics": metrics,
            "tokens": row.get("tokens", {}),
            "origin": "rollout",
        }
        if failure_type:
            metadata["failure_type"] = failure_type
        records.append(
            MemRLSourceRecord(
                task_id=str(row.get("task_id") or f"trajectory_{len(records)}"),
                task_description=str(row.get("question") or row.get("query") or ""),
                trajectory=format_real_trajectory(row),
                success=success,
                reward=_reward_from_rollout(row, success),
                metadata=metadata,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def write_memrl_records_jsonl(records: list[MemRLSourceRecord], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
