"""Build structured reflection memories from real rollout trajectories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .memory import ReflectionEntry


def load_rollout_rows(trajectory_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(trajectory_dir)
    full = root / "trajectories_full.jsonl"
    compact = root / "trajectories.jsonl"
    rows: list[dict[str, Any]] = []
    path = full if full.exists() else compact
    if path.exists():
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
        return rows
    results_dir = root / "results"
    if results_dir.exists():
        for result_path in sorted(results_dir.glob("*.json")):
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    return rows


def error_types_from_row(row: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [
            str(row.get("error", "")),
            str(row.get("final_answer", "")),
            str(row.get("final_answer_full", "")),
        ]
    ).lower()
    errors: list[str] = []
    if row.get("has_tool_oom") or "out of memory" in text or "tool_oom" in text:
        errors.append("tool_oom")
    if row.get("has_tool_error") or "tool execution error" in text:
        errors.append("tool_error")
    if "timeout" in text or "timed out" in text:
        errors.append("timeout")
    if "file not found" in text or "no such file" in text:
        errors.append("file_not_found")
    if not row.get("tools_called") and not row.get("tool_calls"):
        errors.append("no_tool_call")
    return sorted(set(errors))


def build_reflection(row: dict[str, Any], *, low_f1_threshold: float = 0.8) -> ReflectionEntry:
    metrics = row.get("metrics") or {}
    f1 = float(row.get("f1", metrics.get("f1", 0.0)) or 0.0)
    called = list(row.get("tools_called") or row.get("tool_calls") or row.get("tool_sequence") or [])
    expected = list(row.get("expected_tools") or [])
    errors = error_types_from_row(row)
    status = str(row.get("status", "unknown"))

    if f1 >= low_f1_threshold and not errors:
        kind = "success"
        reflection = (
            "A similar task worked when the agent followed this tool flow: "
            f"{' -> '.join(called) if called else 'no tool call'}. "
            "Reuse the pattern only when the task requirements match."
        )
    else:
        kind = "failure"
        missing = [tool for tool in expected if tool not in called]
        extra = [tool for tool in called if tool not in expected]
        parts = [
            f"Previous similar task had tool-match F1={f1:.2f} and status={status}.",
        ]
        if missing:
            parts.append("Missing expected tools: " + ", ".join(missing[:8]) + ".")
        if extra:
            parts.append("Possibly unnecessary tools: " + ", ".join(extra[:8]) + ".")
        if errors:
            parts.append("Observed execution issues: " + ", ".join(errors) + ".")
        parts.append("Before answering, choose tools deliberately and avoid repeating the same failed pattern.")
        reflection = " ".join(parts)

    return ReflectionEntry(
        task_id=str(row.get("task_id", "")),
        question=str(row.get("question") or row.get("query") or ""),
        task_type=str(row.get("task_type", "unknown")),
        kind=kind,
        reflection=reflection,
        tools_called=called,
        expected_tools=expected,
        f1=f1,
        error_types=errors,
        status=status,
        source=str(row.get("source", "unknown")),
    )


def build_reflections_from_rollout(
    trajectory_dir: str | Path,
    *,
    low_f1_threshold: float = 0.8,
    include_success: bool = True,
) -> list[ReflectionEntry]:
    entries: list[ReflectionEntry] = []
    for row in load_rollout_rows(trajectory_dir):
        entry = build_reflection(row, low_f1_threshold=low_f1_threshold)
        if include_success or entry.kind != "success":
            entries.append(entry)
    return entries
