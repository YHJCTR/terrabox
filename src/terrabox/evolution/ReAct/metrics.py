"""Metrics aggregation for ReAct real rollout outputs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ERROR_MARKERS = {
    "tool_oom": ("tool_oom", "CUDA out of memory", "OutOfMemoryError", "out of memory"),
    "file_not_found": ("FileNotFoundError", "No such file or directory", "file not found"),
    "timeout": ("TimeoutError", "timed out", "timeout"),
    "schema": ("ValidationError", "missing required", "unknown_current_args", "schema"),
}


def load_results(trajectory_dir: str | Path) -> list[dict[str, Any]]:
    """Load full trajectory rows or per-task result JSON files."""
    root = Path(trajectory_dir)
    rows: list[dict[str, Any]] = []
    full_path = root / "trajectories_full.jsonl"
    if full_path.exists():
        with full_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    results_dir = root / "results"
    if results_dir.exists():
        for path in sorted(results_dir.glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _text_for_errors(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("error", "")),
        str(row.get("final_answer", "")),
        str(row.get("final_answer_full", "")),
    ]
    for message in row.get("conversation_history", []) or []:
        pieces.append(str(message.get("content", "")))
    return "\n".join(pieces)


def _bucket_errors(row: dict[str, Any]) -> list[str]:
    text = _text_for_errors(row)
    buckets = []
    for name, markers in ERROR_MARKERS.items():
        if any(marker.lower() in text.lower() for marker in markers):
            buckets.append(name)
    if row.get("has_tool_error") and not buckets:
        buckets.append("tool_error")
    return buckets


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    source_counts = Counter(str(row.get("source", "unknown")) for row in rows)
    task_type_counts = Counter(str(row.get("task_type", "unknown")) for row in rows)
    error_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    token_totals = Counter()
    f1_scores: list[float] = []
    exact_matches = 0
    no_tool_calls = 0
    success_count = 0
    real_success_count = 0
    verified_task_success_counts: Counter[str] = Counter()

    for row in rows:
        tools = row.get("tools_called") or row.get("tool_calls") or row.get("tool_sequence") or []
        if not tools:
            no_tool_calls += 1
        for tool in tools:
            tool_counts[str(tool)] += 1
        metrics = row.get("metrics", {}) or {}
        f1 = float(row.get("f1", metrics.get("f1", 0)) or 0)
        f1_scores.append(f1)
        if metrics.get("exact_match"):
            exact_matches += 1
        if row.get("success"):
            success_count += 1
        if row.get("real_success", row.get("success", False)):
            real_success_count += 1
        verified_task_success_counts[str(row.get("verified_task_success"))] += 1
        for bucket in _bucket_errors(row):
            error_counts[bucket] += 1
        for key, value in (row.get("tokens", {}) or {}).items():
            token_totals[str(key)] += int(value or 0)

    return {
        "total": total,
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "task_type_counts": dict(task_type_counts),
        "success_count": success_count,
        "real_success_count": real_success_count,
        "success_rate": success_count / total if total else 0.0,
        "real_success_rate": real_success_count / total if total else 0.0,
        "success_metric_basis": (
            "rollout proxy from status/F1/tool-sequence fields; not semantic task correctness"
        ),
        "verified_task_success_counts": dict(verified_task_success_counts),
        "task_success_basis": "not_auto_verifiable",
        "avg_f1": sum(f1_scores) / total if total else 0.0,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "no_tool_call_count": no_tool_calls,
        "no_tool_call_rate": no_tool_calls / total if total else 0.0,
        "error_counts": dict(error_counts),
        "top_tools": tool_counts.most_common(30),
        "token_totals": dict(token_totals),
    }


def write_metrics(trajectory_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    rows = load_results(trajectory_dir)
    metrics = aggregate_results(rows)
    out = Path(output_path) if output_path else Path(trajectory_dir) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
