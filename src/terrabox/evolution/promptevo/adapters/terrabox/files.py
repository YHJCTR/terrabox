"""File helpers for Terrabox/OEA promptevo experiments."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


TERRABOX_ADAPTER_DIR = os.path.dirname(__file__)
DEFAULT_TERRABOX_EXPERIMENTS_DIR = os.path.join(TERRABOX_ADAPTER_DIR, "experiments")
DEFAULT_OEA_TASK_FILE = "data/oea_full_sft/openearth_test_tasks.json"
DEFAULT_BASE_EXPERIMENT = "oe_full_react_offline"
DEFAULT_MODE = "standard"


def trajectory_dir(experiment: str, mode: str = DEFAULT_MODE) -> str:
    return os.path.join("tmp", "trajectories", experiment, mode)


def results_dir(experiment: str, mode: str = DEFAULT_MODE) -> str:
    return os.path.join(trajectory_dir(experiment, mode), "results")


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def load_tasks(path: str | Path) -> list[dict]:
    data = read_json(path)
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Expected list or {{'tasks': [...]}} in {path}")
    for i, task in enumerate(tasks):
        task.setdefault("task_id", task.get("id", f"task_{i}"))
        task.setdefault("id", task["task_id"])
        task.setdefault("question", task.get("prompt", ""))
        task.setdefault("images", [])
        task.setdefault("expected_tools", [])
    return tasks


def load_result_rows(results_path: str | Path, task_order: Iterable[dict] | None = None) -> list[dict]:
    path = Path(results_path)
    order: dict[str, int] = {}
    if task_order is not None:
        order = {
            str(task.get("task_id") or task.get("id") or f"task_{i}"): i
            for i, task in enumerate(task_order)
        }
    rows: list[dict] = []
    for result_file in path.glob("*.json"):
        try:
            row = read_json(result_file)
        except Exception:
            continue
        row.setdefault("task_id", result_file.stem)
        rows.append(row)
    rows.sort(key=lambda r: (order.get(str(r.get("task_id")), len(order)), str(r.get("task_id"))))
    return rows


def rebuild_trajectory_files(
    experiment: str = DEFAULT_BASE_EXPERIMENT,
    task_file: str = DEFAULT_OEA_TASK_FILE,
    mode: str = DEFAULT_MODE,
) -> dict:
    """Rebuild report.json and trajectories*.jsonl from authoritative results/.

    `results/<task_id>.json` is the source of truth for OEA rollouts. This
    helper only refreshes derived files for promptevo sampling and human
    inspection; it does not modify per-task results.
    """

    out_dir = Path(trajectory_dir(experiment, mode))
    res_dir = out_dir / "results"
    tasks = load_tasks(task_file)
    rows = load_result_rows(res_dir, tasks)
    if not rows:
        raise RuntimeError(f"No results found under {res_dir}")

    compact_path = out_dir / "trajectories.jsonl"
    with compact_path.open("w", encoding="utf-8") as f:
        for row in rows:
            traj = {
                "task_id": row["task_id"],
                "source": row.get("source", "unknown"),
                "query": row.get("question", ""),
                "tool_sequence": row.get("tool_calls_deduped", []),
                "tools_called": row.get("tool_calls", []),
                "expected_tools": row.get("expected_tools", []),
                "f1": row.get("metrics", {}).get("f1", 0),
                "reward": 1.0 if row.get("success") else 0.0,
                "task_type": row.get("task_type", "general"),
                "status": row.get("status", "unknown"),
                "real_success": row.get("real_success", row.get("success", False)),
                "system_limitation_acknowledged": row.get("system_limitation_acknowledged", False),
                "has_tool_error": row.get("has_tool_error", False),
                "has_tool_oom": row.get("has_tool_oom", False),
            }
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    full_path = out_dir / "trajectories_full.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for row in rows:
            traj = {
                "task_id": row["task_id"],
                "source": row.get("source", "unknown"),
                "question": row.get("question", ""),
                "expected_tools": row.get("expected_tools", []),
                "tool_sequence": row.get("tool_calls_deduped", []),
                "tools_called": row.get("tool_calls", []),
                "metrics": row.get("metrics", {}),
                "status": row.get("status", "unknown"),
                "success": row.get("success", False),
                "real_success": row.get("real_success", row.get("success", False)),
                "system_limitation_acknowledged": row.get("system_limitation_acknowledged", False),
                "has_tool_error": row.get("has_tool_error", False),
                "has_tool_oom": row.get("has_tool_oom", False),
                "tokens": row.get("tokens", {}),
                "llm_calls": row.get("llm_calls", 0),
                "time": row.get("time", 0),
                "final_answer": row.get("final_answer_full", ""),
                "conversation_history": row.get("conversation_history", []),
            }
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    total_tokens = {
        key: sum((row.get("tokens") or {}).get(key, 0) for row in rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    report = {
        "experiment": experiment,
        "mode": mode,
        "task_file": task_file,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_tasks": len(rows),
        "success_count": sum(1 for row in rows if row.get("success")),
        "success_rate": sum(1 for row in rows if row.get("success")) / len(rows),
        "total_tokens": total_tokens,
        "avg_f1": sum(row.get("metrics", {}).get("f1", 0) for row in rows) / len(rows),
        "status_distribution": dict(Counter(row.get("status", "unknown") for row in rows)),
        "unique_tools_called": sorted(set(tool for row in rows for tool in row.get("tool_calls_deduped", []))),
        "derived_from_results": True,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "n": len(rows),
        "trajectory_dir": str(out_dir),
        "results_dir": str(res_dir),
        "trajectories": str(compact_path),
        "trajectories_full": str(full_path),
        "report": str(report_path),
    }
