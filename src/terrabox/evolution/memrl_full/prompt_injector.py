"""Prompt augmenter backed by memrl_full SQLite memories."""
from __future__ import annotations

import json
from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter
from ..shared.storage import SQLiteMemoryStore


class MemRLFullPromptInjector(PromptAugmenter):
    def __init__(self, memory_db: str, top_k: int = 5):
        self.memory_db = memory_db
        self.top_k = top_k
        self._store = SQLiteMemoryStore(memory_db)

    def augment(self, user_query: str, **kwargs) -> str:
        memories = self._store.get_all(limit=2000)
        q = set(user_query.lower().split())
        scored = []
        for memory in memories:
            intent = memory.get("intent", {})
            exp = memory.get("experience", {})
            keywords = set(intent.get("keywords", []))
            sequence = exp.get("tool_sequence", [])
            overlap = len(q & keywords)
            score = overlap + float(memory.get("utility", 0.0))
            scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        blocks = []
        for _, memory in scored[: self.top_k]:
            exp = memory.get("experience", {})
            blocks.append(
                "- Task: {question}\n  Tools: {tools}\n  Utility: {utility:.2f}".format(
                    question=str(exp.get("question", ""))[:220],
                    tools=" -> ".join(exp.get("tool_sequence", [])),
                    utility=float(memory.get("utility", 0.0)),
                )
            )
        if not blocks:
            return ""
        return "## Relevant MemRL Full Experiences\n" + "\n".join(blocks)


def load_memories_preview(memory_db: str, limit: int = 5) -> list[dict]:
    if not Path(memory_db).exists():
        return []
    return SQLiteMemoryStore(memory_db).get_all(limit=limit)
