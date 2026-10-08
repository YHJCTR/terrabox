"""Public-view dataset exports for shared Terrabox agentic RL runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def swift_rows_from_verl_rows(
    rows: Iterable[dict[str, Any]],
    *,
    method: str,
    tool_catalog_path: str | Path,
    artifact_root: str | Path,
    trace_path: str | Path,
    raw_observation_trace_path: str | Path,
    max_turns: int = 12,
) -> list[dict[str, Any]]:
    """Convert existing strict veRL public rows into Swift Gym-env rows."""
    converted: list[dict[str, Any]] = []
    for row in rows:
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        sample_index = extra.get("sample_index", len(converted))
        converted.append(
            {
                "messages": row.get("prompt") or row.get("messages") or [],
                "sample_index": sample_index,
                "method": method,
                "question": _last_user_content(row.get("prompt") or row.get("messages") or []),
                "data_files": list(extra.get("data_files") or []),
                "data_dir": str(extra.get("data_dir") or ""),
                "tool_catalog": str(tool_catalog_path),
                "env_config": {
                    "name": "terrabox_oea",
                    "method": method,
                    "sample_index": sample_index,
                    "images": list(extra.get("images") or []),
                    "data_files": list(extra.get("data_files") or []),
                    "data_dir": str(extra.get("data_dir") or ""),
                    "tool_catalog": str(tool_catalog_path),
                    "artifact_root": str(artifact_root),
                    "trace_path": str(trace_path),
                    "raw_observation_trace_path": str(raw_observation_trace_path),
                    "max_turns": max_turns,
                },
            }
        )
    return converted


def _last_user_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""
