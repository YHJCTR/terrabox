"""Adapters from Terrabox SFT/rollout data to MemRL source records."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..full_shared.sft_schema import load_sft_samples
from .trajectory_formatter import format_real_trajectory, format_sft_trajectory


_INFRA_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "quota",
    "billing",
    "payment",
    "oom",
    "out of memory",
    "cuda",
    "connection",
    "network",
    "provider",
    "context length",
    "docker",
    "service health",
)


@dataclass
class MemRLSourceRecord:
    task_id: str
    task_description: str
    trajectory: str
    success: bool
    reward: float
    metadata: dict[str, Any]

    def to_dict(self, *, strict_nolabel: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if strict_nolabel:
            payload.pop("task_id", None)
        return payload


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


def _contains_infra_failure(row: dict[str, Any]) -> bool:
    if row.get("has_tool_oom") or row.get("system_limitation_acknowledged"):
        return True
    if str(row.get("status") or "") == "completed":
        return False
    text = json.dumps(
        {
            "status": row.get("status"),
            "error": row.get("error"),
            "error_type": row.get("error_type"),
            "failure_reason": row.get("failure_reason"),
        },
        ensure_ascii=False,
    ).lower()
    return any(marker in text for marker in _INFRA_MARKERS)


def _strict_reward_from_rollout(row: dict[str, Any]) -> float:
    if bool(row.get("has_tool_error")):
        return 0.15
    completed = str(row.get("status") or "") == "completed"
    final = str(
        row.get("final_answer")
        or row.get("final_answer_full")
        or row.get("final_answer_preview")
        or row.get("final")
        or ""
    ).strip()
    tools = row.get("tool_sequence") or row.get("tools_called") or row.get("tool_calls") or []
    if completed and final and tools:
        return 0.85
    if completed and final:
        return 0.70
    if completed and tools:
        return 0.55
    return 0.30


def _strict_success_from_rollout(row: dict[str, Any]) -> bool:
    return _strict_reward_from_rollout(row) >= 0.8


def _redact_text(value: Any, limit: int = 360) -> str:
    text = str(value or "")
    text = re.sub(r"\[[^\]]*(?:image|file)s?\s*:\s*[^\]]+\]", "<artifact_reference>", text, flags=re.I)
    text = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", text)
    text = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_reference>", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<artifact_reference>", text, flags=re.I)
    text = re.sub(
        r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)+\b",
        "<named_area>",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[: limit - 3].rstrip() + "..." if len(text) > limit else text


def _reward_from_rollout(row: dict[str, Any], success: bool) -> float:
    if success:
        return 1.0
    metrics = row.get("metrics") or {}
    try:
        return max(0.0, min(1.0, float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _failure_type(row: dict[str, Any], *, strict_nolabel: bool = False) -> str | None:
    if _strict_success_from_rollout(row) if strict_nolabel else _success_from_rollout(row):
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
    strict_nolabel: bool = False,
) -> list[MemRLSourceRecord]:
    records: list[MemRLSourceRecord] = []
    if strict_nolabel and min_f1 is not None:
        raise ValueError("strict_nolabel MemRL records cannot be filtered by metrics/F1")
    for row in _iter_jsonl(path):
        if strict_nolabel and _contains_infra_failure(row):
            continue
        if strict_nolabel:
            success = _strict_success_from_rollout(row)
            reward = _strict_reward_from_rollout(row)
        else:
            metrics = row.get("metrics") or {}
            f1 = float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)
            success = _success_from_rollout(row)
            if min_f1 is not None and f1 < min_f1:
                continue
            reward = _reward_from_rollout(row, success)
        if not success and not include_empty_failures and not row.get("conversation_history"):
            continue
        source = str(row.get("source") or "unknown")
        failure_type = _failure_type(row, strict_nolabel=strict_nolabel)
        if strict_nolabel:
            metadata = {
                "source_benchmark": f"terrabox_{source}_rollout",
                "source": source,
                "tool_sequence": row.get("tool_sequence") or row.get("tools_called") or row.get("tool_calls") or [],
                "status": row.get("status", "unknown"),
                "origin": "rollout_strict_nolabel",
                "strict_nolabel": True,
                "reward_basis": "completed_final_tools_and_observed_tool_error_only",
            }
        else:
            metrics = row.get("metrics") or {}
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
                task_description=(
                    _redact_text(row.get("question") or row.get("query") or "")
                    if strict_nolabel else str(row.get("question") or row.get("query") or "")
                ),
                trajectory=format_real_trajectory(row, strict_nolabel=strict_nolabel),
                success=success,
                reward=reward,
                metadata=metadata,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def write_memrl_records_jsonl(
    records: list[MemRLSourceRecord],
    path: str | Path,
    *,
    strict_nolabel: bool = False,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(strict_nolabel=strict_nolabel), ensure_ascii=False) + "\n")
