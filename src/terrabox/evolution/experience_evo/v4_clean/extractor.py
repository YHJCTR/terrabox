"""Label-free event extraction for the v4-clean experience store.

This adapter intentionally consumes only an executed rollout's question, input
files, tool calls, tool observations, and observable execution errors.  It
does not retain benchmark task ids/types, gold tools, answers, or evaluation
metrics in the event index.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..transition_extractor import load_rollout_rows
from ..v2.extractor import extract_transition_events_from_row, intent_signature
from ..v2.models import TransitionEvent


def _event_id(event: TransitionEvent) -> str:
    """Create a content id without using a benchmark task identifier."""
    payload = {
        "intent": event.intent_signature,
        "step": event.step_index,
        "tool": event.tool,
        "input": event.input_product_state,
        "target": event.target_product_state,
        "args": event.args_summary,
        "observation": event.observation_summary,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:18]


def extract_clean_transition_events_from_row(
    row: dict, *, source_name: str = ""
) -> list[TransitionEvent]:
    """Extract one rollout after replacing all dataset-label fields.

    The v2 extractor is reused for artifact-state and local-evidence parsing.
    Its output is immediately rebuilt so no original task id or task type is
    persisted.  The intent is inferred from task text alone; ``general`` is a
    constant schema value, never a dataset bucket.
    """
    question = str(row.get("question") or row.get("query") or "")
    intent, hint = intent_signature(question, "general")
    raw_events = extract_transition_events_from_row(row, source_name=source_name)
    events: list[TransitionEvent] = []
    for event in raw_events:
        event.task_id = "rollout"
        event.task_type = "general"
        event.intent_signature = intent
        event.task_hint = hint
        event.event_id = _event_id(event)
        events.append(event)
    return events


def extract_clean_transition_events(
    paths: Iterable[str | Path],
    *,
    source_name: str = "",
    completed_only: bool = False,
    max_tasks: int | None = None,
) -> list[TransitionEvent]:
    """Extract label-free events from rollout result files."""
    events: list[TransitionEvent] = []
    duplicate_ids: Counter[str] = Counter()
    used_tasks = 0
    for row in load_rollout_rows(paths):
        status = str(row.get("status") or "").lower()
        if completed_only and status not in {"completed", "completed_with_recovery"}:
            continue
        task_events = extract_clean_transition_events_from_row(row, source_name=source_name)
        if not task_events:
            continue
        for event in task_events:
            duplicate_ids[event.event_id] += 1
            # The counter is an extraction-local occurrence number, not a task
            # identifier or retrieval feature. It only keeps SQLite primary keys unique.
            if duplicate_ids[event.event_id] > 1:
                event.event_id = _event_id(event) + f"_{duplicate_ids[event.event_id]}"
            events.append(event)
        used_tasks += 1
        if max_tasks is not None and used_tasks >= max_tasks:
            break
    return events
