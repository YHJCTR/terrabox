"""Prepare strict SFT data for parameter-training baselines.

Training data is allowed to contain gold assistant messages because that is the
supervision signal. Evaluation task files must stay prompt-only and omit gold
messages/calls so rollout remains comparable with ReAct and Reflection.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..ReAct.data_adapter import samples_to_tasks
from ..full_shared.sft_schema import FullSFTSample


def _shorten_text(value: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    head_chars = max(1, int(max_chars * 0.65))
    tail_chars = max(1, max_chars - head_chars)
    omitted = len(value) - head_chars - tail_chars
    return (
        value[:head_chars]
        + f"\n...[SFT_COMPACTED omitted_chars={omitted}]...\n"
        + value[-tail_chars:]
    )


def _compact_value(value: Any, *, max_list_items: int, max_string_chars: int) -> Any:
    if isinstance(value, str):
        return _shorten_text(value, max_chars=max_string_chars)
    if isinstance(value, list):
        if len(value) > max_list_items:
            keep = max(1, max_list_items // 2)
            return {
                "__sft_compacted_list__": True,
                "total_items": len(value),
                "head": [
                    _compact_value(item, max_list_items=max_list_items, max_string_chars=max_string_chars)
                    for item in value[:keep]
                ],
                "tail": [
                    _compact_value(item, max_list_items=max_list_items, max_string_chars=max_string_chars)
                    for item in value[-keep:]
                ],
            }
        return [
            _compact_value(item, max_list_items=max_list_items, max_string_chars=max_string_chars)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, max_list_items=max_list_items, max_string_chars=max_string_chars)
            for key, item in value.items()
        }
    return value


def _compact_system_catalog(content: str) -> tuple[str, bool]:
    marker = "Tool catalog:\n"
    if marker not in content:
        return content, False
    prefix, raw_catalog = content.split(marker, 1)
    try:
        catalog = json.loads(raw_catalog)
    except json.JSONDecodeError:
        return content, False
    compact_catalog = []
    for tool in catalog:
        params = tool.get("parameters") or {}
        properties = params.get("properties") or {}
        compact_catalog.append(
            {
                "slug": tool.get("slug"),
                "function_name": tool.get("function_name"),
                "description": tool.get("description"),
                "required": params.get("required", []),
                "parameters": sorted(str(name) for name in properties),
            }
        )
    return prefix + marker + json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":")), True


def compact_messages_for_sft(
    messages: list[dict[str, Any]],
    *,
    compact_system_catalog: bool = True,
    max_observation_chars: int = 1200,
    max_string_chars: int = 800,
    max_list_items: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compact long SFT contexts without dropping the tool catalog.

    The goal is to keep every tool visible while preventing huge file lists or
    tool outputs from dominating QLoRA memory. This is a training-only transform;
    rollout evaluation still uses prompt-only task files.
    """
    compacted: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "enabled": True,
        "system_catalog_compacted": False,
        "observation_messages_compacted": 0,
        "assistant_messages_compacted": 0,
        "original_chars": 0,
        "compacted_chars": 0,
        "max_observation_chars": max_observation_chars,
        "max_string_chars": max_string_chars,
        "max_list_items": max_list_items,
    }

    for message in messages:
        role = str(message.get("role", ""))
        original_content = str(message.get("content", ""))
        content = original_content
        stats["original_chars"] += len(original_content)

        if role == "system" and compact_system_catalog:
            content, did_compact = _compact_system_catalog(content)
            stats["system_catalog_compacted"] = bool(stats["system_catalog_compacted"] or did_compact)
        elif role == "user" and content.startswith("OBSERVATION:"):
            shortened = _shorten_text(content, max_chars=max_observation_chars)
            if shortened != content:
                stats["observation_messages_compacted"] += 1
                content = shortened
        elif role == "assistant":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and "actions" in payload:
                # CRITICAL for tool-use SFT: the assistant action `arguments` are
                # the learning target — never compact them (compacting a long
                # file-list arg into a {__sft_compacted_list__} placeholder would
                # teach the model to emit a broken, non-executable tool call).
                # Only shorten the free-text `thought`.
                thought = payload.get("thought")
                if isinstance(thought, str):
                    new_thought = _shorten_text(thought, max_chars=max_string_chars)
                    if new_thought != thought:
                        payload = dict(payload)
                        payload["thought"] = new_thought
                        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                        stats["assistant_messages_compacted"] += 1
            else:
                # Final answer / non-action text → safe to shorten.
                shortened = _shorten_text(content, max_chars=max_string_chars)
                if shortened != content:
                    stats["assistant_messages_compacted"] += 1
                    content = shortened

        stats["compacted_chars"] += len(content)
        new_message = dict(message)
        new_message["content"] = content
        compacted.append(new_message)

    stats["saved_chars"] = stats["original_chars"] - stats["compacted_chars"]
    return compacted, stats


def sample_to_chat_row(
    sample: FullSFTSample,
    *,
    compact_long_context: bool = False,
    compact_system_catalog: bool = True,
    max_observation_chars: int = 1200,
    max_string_chars: int = 800,
    max_list_items: int = 8,
) -> dict[str, Any]:
    """Return one chat SFT row in a trainer-friendly JSONL format."""
    messages = list(sample.messages)
    compaction: dict[str, Any] = {"enabled": False}
    if compact_long_context:
        messages, compaction = compact_messages_for_sft(
            messages,
            compact_system_catalog=compact_system_catalog,
            max_observation_chars=max_observation_chars,
            max_string_chars=max_string_chars,
            max_list_items=max_list_items,
        )
    return {
        "task_id": sample.task_id,
        "source": sample.source,
        "task_type": sample.task_type,
        "question": sample.question,
        "images": list(sample.images),
        "data_files": list(sample.data_files),
        "data_dir": sample.data_dir,
        "messages": messages,
        "expected_tools": list(sample.tool_sequence),
        "gold_tool_calls": list(sample.gold_tool_calls),
        "ground_truth": sample.ground_truth,
        "sft_compaction": compaction,
    }


def write_chat_jsonl(
    samples: Iterable[FullSFTSample],
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    split_name: str = "train",
    compact_long_context: bool = False,
    compact_system_catalog: bool = True,
    max_observation_chars: int = 1200,
    max_string_chars: int = 800,
    max_list_items: int = 8,
) -> int:
    """Write chat SFT JSONL and a small sidecar metadata file."""
    rows = [
        sample_to_chat_row(
            sample,
            compact_long_context=compact_long_context,
            compact_system_catalog=compact_system_catalog,
            max_observation_chars=max_observation_chars,
            max_string_chars=max_string_chars,
            max_list_items=max_list_items,
        )
        for sample in samples
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": str(source_path) if source_path else None,
        "split_name": split_name,
        "num_rows": len(rows),
        "format": "terrabox_chat_sft_jsonl",
        "gold_usage_policy": "messages/gold_tool_calls are allowed only for SFT training, not rollout eval",
        "compact_long_context": compact_long_context,
        "compact_system_catalog": compact_system_catalog,
        "max_observation_chars": max_observation_chars,
        "max_string_chars": max_string_chars,
        "max_list_items": max_list_items,
    }
    out.with_suffix(out.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(rows)


def write_eval_task_file(
    samples: Iterable[FullSFTSample],
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    split_name: str = "eval",
) -> int:
    """Write prompt-only rollout tasks for evaluating an SFT checkpoint."""
    tasks = samples_to_tasks(samples)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": str(source_path) if source_path else None,
            "split_name": split_name,
            "num_tasks": len(tasks),
            "format": "terrabox_rollout_tasks",
            "gold_leakage_policy": "messages/gold_tool_calls/ground_truth omitted from model-facing task file",
        },
        "tasks": tasks,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(tasks)
