"""Build MemRL-style episodic memories from full SFT samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..full_shared.sft_schema import FullSFTSample
from ..shared.storage import SQLiteMemoryStore


def _keywords(sample: FullSFTSample) -> list[str]:
    words = []
    for token in sample.question.lower().replace("/", " ").replace("_", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if len(cleaned) > 3 and cleaned not in words:
            words.append(cleaned)
    return words[:24]


def build_memory_records(samples: list[FullSFTSample]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        records.append({
            "id": sample.task_id,
            "task_id": sample.task_id,
            "intent": {
                "source": sample.source,
                "task_type": sample.task_type,
                "keywords": _keywords(sample),
            },
            "experience": {
                "question": sample.question,
                "tool_sequence": sample.tool_sequence,
                "tool_calls": sample.gold_tool_calls,
                "final_answer": sample.ground_truth,
                "images": sample.images,
                "data_files": sample.data_files,
            },
            "utility": 1.0 if sample.executable else 0.5,
            "q_visits": 1,
        })
    return records


def write_memory_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def populate_sqlite_memory(
    records: list[dict[str, Any]],
    db_path: str | Path,
    *,
    reset: bool = False,
) -> int:
    store = SQLiteMemoryStore(str(db_path))
    if reset:
        store.clear()
    for record in records:
        store.insert(record)
    return store.count()
