"""Metrics for dynamic GRPO reward traces."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_reward_traces(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_reward_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    scores = [float(row.get("score", 0) or 0) for row in rows]
    tool_success_rates = [
        float((row.get("summary") or {}).get("tool_success_rate", 0) or 0)
        for row in rows
    ]
    all_tools_succeeded = sum(
        1 for row in rows if (row.get("summary") or {}).get("all_tools_succeeded")
    )
    verified = Counter(
        str((row.get("summary") or {}).get("verified_task_success"))
        for row in rows
    )
    errors: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    for row in rows:
        for tool in row.get("called_tools", []) or []:
            tools[str(tool)] += 1
        for error_type in (row.get("summary") or {}).get("error_types", []) or []:
            errors[str(error_type)] += 1
    return {
        "total_reward_evaluations": total,
        "avg_reward": sum(scores) / total if total else 0.0,
        "avg_tool_success_rate": sum(tool_success_rates) / total if total else 0.0,
        "all_tools_succeeded_count": all_tools_succeeded,
        "all_tools_succeeded_rate": all_tools_succeeded / total if total else 0.0,
        "verified_task_success_counts": dict(verified),
        "task_success_note": "verified_task_success=None means the task was not automatically verifiable",
        "error_counts": dict(errors),
        "top_tools": tools.most_common(30),
    }


def write_reward_metrics(trace_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = load_reward_traces(trace_path)
    metrics = summarize_reward_traces(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics

