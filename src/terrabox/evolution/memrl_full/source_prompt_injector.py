"""Prompt augmenter backed by memrl_full source memory index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..shared.prompt_builder import PromptAugmenter
from .source_memory_service import LiteMemRLSourceService


class MemRLSourcePromptInjector(PromptAugmenter):
    def __init__(self, store_dir: str | Path, top_k: int = 5, threshold: float = 0.0):
        self.store_dir = Path(store_dir)
        self.top_k = top_k
        self.threshold = threshold
        self._service = LiteMemRLSourceService(self.store_dir)

    def augment(self, user_query: str, **kwargs) -> str:
        records = self._service.retrieve(user_query, top_k=self.top_k, threshold=self.threshold)
        if not records:
            return ""
        success_blocks: list[str] = []
        failure_blocks: list[str] = []
        for record in records:
            block = self._format_memory(record)
            if record.get("success"):
                success_blocks.append(block)
            else:
                failure_blocks.append(block)

        blocks = [
            "## Relevant MemRL Source Memories",
            (
                "In addition to the current task, you have the following memories from past "
                "Terrabox runs. Use them only if they are relevant."
            ),
        ]
        if success_blocks:
            blocks.append(
                "--- SUCCESSFUL MEMORIES (examples to follow) ---\n"
                + "\n\n".join(success_blocks)
            )
        if failure_blocks:
            blocks.append(
                "--- FAILED MEMORIES (examples to avoid or learn from) ---\n"
                + "\n\n".join(failure_blocks)
            )
        return "\n".join(blocks)

    def _format_memory(self, record: dict[str, Any]) -> str:
        meta = record.get("metadata", {})
        tools = " -> ".join(str(t) for t in meta.get("tool_sequence", []))
        trajectory = str(record.get("trajectory", "")).strip()
        if len(trajectory) > 1200:
            trajectory = trajectory[:1200] + "\n...[truncated]"
        header = [
            f"Task: {record.get('task_description', '')}",
            f"Outcome: {'success' if record.get('success') else 'failure'}, utility={float(record.get('utility', 0.0)):.2f}",
        ]
        if tools:
            header.append(f"Tool sequence: {tools}")
        failure_type = meta.get("failure_type")
        if failure_type:
            header.append(f"Failure type: {failure_type}")
        return "\n".join(header) + "\n\nArchived Trajectory:\n" + trajectory


def load_source_memory_index(store_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(store_dir) / "memory_index.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
