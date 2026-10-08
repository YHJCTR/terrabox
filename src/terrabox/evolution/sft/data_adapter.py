"""Prepare strict SFT data for parameter-training baselines.

Training data is allowed to contain gold assistant messages because that is the
supervision signal. Evaluation task files must stay prompt-only and omit gold
messages/calls so rollout remains comparable with ReAct and Reflection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..ReAct.data_adapter import samples_to_tasks
from ..full_shared.sft_schema import FullSFTSample


def _read_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _replay_is_usable(result: dict[str, Any], expected_tools: list[str]) -> tuple[bool, str]:
    """Return whether a gold replay is safe to turn into SFT supervision.

    A teacher-forced replay validates the executable tool protocol; it does not
    validate the natural-language answer.  We therefore require a complete,
    error-free, order-preserving tool trace here and leave answer correctness
    to the source ground truth and the later rollout judge.
    """
    if str(result.get("status") or "") != "completed":
        return False, str((result.get("replay_meta") or {}).get("failure_type") or "replay_not_completed")
    observations = result.get("replay_observations") or []
    called = [str(item.get("tool") or "") for item in observations if isinstance(item, dict)]
    if called != [str(tool) for tool in expected_tools]:
        return False, "tool_sequence_mismatch"
    if any(bool(item.get("is_error")) for item in observations if isinstance(item, dict)):
        return False, "tool_error_observation"
    if not observations:
        return False, "empty_replay_observations"
    return True, "usable"


def _canonicalize_replay_observation(text: str, result: dict[str, Any]) -> str:
    """Remove run-specific replay paths while retaining real tool evidence."""
    value = str(text or "")
    meta = result.get("replay_meta") or {}
    replacements: list[tuple[str, str]] = []
    for item in meta.get("alias_captures") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        alias = str(item.get("alias") or "")
        if path and alias:
            replacements.append((path, alias))
    # Replace longer paths first so a nested artifact path cannot partially
    # consume a more specific alias replacement.
    for path, alias in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        value = value.replace(path, alias)
    artifact_dir = str(meta.get("artifact_dir") or "")
    if artifact_dir:
        value = value.replace(artifact_dir, "<current_artifact_dir>")
    # The replay may expose the artifact index path in a diagnostic field.
    value = re.sub(r"/[^\s\"']*/artifact_index\.json", "<artifact_index>", value)
    return value


def _with_replayed_observations(source: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Merge full real observations into an OEA JSON-actions SFT row."""
    row = copy.deepcopy(source)
    replay_observations = [
        item for item in (replay.get("replay_observations") or []) if isinstance(item, dict)
    ]
    obs_index = 0
    messages = row.get("messages") or []
    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "")
        if not content.startswith("OBSERVATION:"):
            continue
        if obs_index >= len(replay_observations):
            break
        prefix = "OBSERVATION:\n"
        suffix = ""
        # Preserve the source protocol's short request after each observation;
        # only the observation body is replaced by actual Terrabox output.
        marker = "\nPlease summarize"
        marker_pos = content.find(marker)
        if marker_pos >= 0:
            suffix = content[marker_pos:]
        actual = _canonicalize_replay_observation(
            str(replay_observations[obs_index].get("content") or ""), replay
        )
        message["content"] = prefix + actual + suffix
        obs_index += 1

    # The original OEA rows often encode the final turn as an empty-actions
    # object without final_answer.  That is ambiguous to the SFT rollout loop
    # (it looks like the initial planning turn), so attach the labelled answer
    # only to the final assistant turn.  Earlier empty-actions planning turns
    # remain unchanged.
    assistant_indices = [
        index for index, message in enumerate(messages)
        if str(message.get("role") or "") == "assistant"
    ]
    if assistant_indices:
        final_index = assistant_indices[-1]
        final_message = messages[final_index]
        try:
            payload = json.loads(str(final_message.get("content") or ""))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and not payload.get("actions") and "final_answer" not in payload:
            answer = source.get("ground_truth")
            if answer is not None:
                payload["final_answer"] = str(answer)
                final_message["content"] = json.dumps(payload, ensure_ascii=False)
    row["messages"] = messages
    row["sft_replay"] = {
        "method": "gold_teacher_forced_replay",
        "source_task_id": str(source.get("id") or source.get("task_id") or ""),
        "replay_status": replay.get("status"),
        "replay_observation_count": len(replay_observations),
        "answer_source": "source_ground_truth_only",
        "absolute_paths_removed_from_observations": True,
    }
    # Keep only the model-facing messages in the output writer; these fields
    # are useful to audits before writing but must never be passed to rollout.
    return row


def build_replayed_sft_dataset(
    source_data: str | Path,
    replay_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a repeatable SFT set backed by actual Terrabox tool replays.

    ``source_data`` may be a selected OEA train JSONL.  Only rows with a full,
    order-preserving replay are retained.  The output contains plain
    JSON-actions ``messages`` compatible with the existing SFT adapter; no
    gold fields are included in the model-facing JSONL.
    """
    source_rows = _read_jsonl_dicts(source_data)
    results_dir = Path(replay_dir) / "results"
    output = Path(output_dir)
    managed_outputs = [
        output / "train.jsonl",
        output / "val.jsonl",
        output / "system_prompt.txt",
        output / "manifest.json",
    ]
    existing_outputs = [path for path in managed_outputs if path.exists()]
    if existing_outputs and not overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite replayed SFT outputs in {output}: {names}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)
    usable: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for index, source in enumerate(source_rows):
        task_id = str(source.get("id") or source.get("task_id") or f"row_{index}")
        result_path = results_dir / f"{task_id}.json"
        if not result_path.exists():
            rejected["missing_replay_result"] = rejected.get("missing_replay_result", 0) + 1
            continue
        try:
            replay = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            rejected["invalid_replay_json"] = rejected.get("invalid_replay_json", 0) + 1
            continue
        expected = [str(tool) for tool in (source.get("expected_tools") or [])]
        ok, reason = _replay_is_usable(replay, expected)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        usable.append(_with_replayed_observations(source, replay))

    rng = random.Random(seed)
    rng.shuffle(usable)
    val_count = int(round(len(usable) * max(0.0, min(0.5, validation_fraction))))
    val_rows = usable[:val_count]
    train_rows = usable[val_count:]

    def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"messages": row.get("messages") or []}, ensure_ascii=False) + "\n")

    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    write_rows(train_path, train_rows)
    write_rows(val_path, val_rows)
    system_prompt = ""
    if train_rows:
        system_prompt = str((train_rows[0].get("messages") or [{}])[0].get("content") or "")
    (output / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    manifest = {
        "format": "terrabox_json_actions_sft_replayed",
        "source_data": str(source_data),
        "replay_dir": str(replay_dir),
        "output_dir": str(output),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_rows": len(source_rows),
        "usable_rows": len(usable),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "rejected_rows": sum(rejected.values()),
        "rejection_reasons": rejected,
        "observation_policy": "actual Terrabox gold-replay observations; run-specific paths canonicalized to aliases",
        "answer_policy": "source ground_truth is supervised only; rollout eval uses prompt-only test tasks",
        "loss_policy": "assistant JSON action/final-answer spans only; user observations are context",
        "train_file": str(train_path),
        "val_file": str(val_path),
        "system_prompt_file": str(output / "system_prompt.txt"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


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
