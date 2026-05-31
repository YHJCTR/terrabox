"""Reader for Terrabox OpenEarth/EarthBench full SFT JSONL data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..shared.trajectory import Trajectory, Turn


@dataclass
class FullSFTSample:
    """One SFT sample aligned to the current Terrabox tool interface."""

    task_id: str
    source: str
    task_type: str
    question: str
    images: list[str]
    data_files: list[str]
    messages: list[dict[str, Any]]
    gold_tool_calls: list[dict[str, Any]]
    ground_truth: str | None = None
    data_dir: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_sequence(self) -> list[str]:
        return [
            str(call.get("tool"))
            for call in self.gold_tool_calls
            if call.get("tool")
        ]

    @property
    def executable(self) -> bool:
        return bool(self.gold_tool_calls) and all(
            bool(call.get("is_executable_under_current_schema", True))
            for call in self.gold_tool_calls
        )

    def to_trajectory(self) -> Trajectory:
        turns: list[Turn] = []
        for message in self.messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role == "system":
                continue
            if role in {"user", "assistant"}:
                turns.append(Turn(role=role, content=content))
        for call in self.gold_tool_calls:
            turns.append(
                Turn(
                    role="tool",
                    content="",
                    tool_name=call.get("tool"),
                    tool_args=call.get("arguments") or {},
                )
            )
        return Trajectory(
            task_id=self.task_id,
            question=self.question,
            images=self.images,
            turns=turns,
            tools_called=self.tool_sequence,
            expected_tools=self.tool_sequence,
            final_answer=self.ground_truth or "",
            success=True,
            source=self.source,
            task_type=self.task_type,
            metadata={
                "data_files": self.data_files,
                "data_dir": self.data_dir,
                "sft_messages": self.messages,
                "gold_tool_calls": self.gold_tool_calls,
            },
        )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sample_from_row(row: dict[str, Any]) -> FullSFTSample:
    return FullSFTSample(
        task_id=str(row.get("id") or row.get("task_id") or ""),
        source=str(row.get("source") or "unknown"),
        task_type=str(row.get("task_type") or "unknown"),
        question=str(row.get("question") or ""),
        images=[str(p) for p in row.get("images", []) or []],
        data_files=[str(p) for p in row.get("data_files", []) or []],
        data_dir=row.get("data_dir"),
        ground_truth=row.get("ground_truth"),
        messages=list(row.get("messages", []) or []),
        gold_tool_calls=list(row.get("gold_tool_calls", []) or []),
        raw=row,
    )


def load_sft_samples(path: str | Path, *, limit: int | None = None) -> list[FullSFTSample]:
    """Load full SFT samples from JSONL.

    The default input is ``data/newdata/sft_train_strict.jsonl``.
    """
    source = Path(path)
    samples: list[FullSFTSample] = []
    for row in _iter_jsonl(source):
        samples.append(_sample_from_row(row))
        if limit is not None and len(samples) >= limit:
            break
    return samples
