"""Reflection memory bank with small, deterministic lexical retrieval."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1}


@dataclass
class ReflectionEntry:
    task_id: str
    question: str
    task_type: str
    kind: str
    reflection: str
    tools_called: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    f1: float = 0.0
    error_types: list[str] = field(default_factory=list)
    status: str = "unknown"
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionEntry":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReflectionMemoryBank:
    """JSONL-backed reflection memory with top-k lexical retrieval."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: list[ReflectionEntry] = []
        if self.path.exists():
            self.load()

    def load(self) -> list[ReflectionEntry]:
        self.entries = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(ReflectionEntry.from_dict(json.loads(line)))
        return self.entries

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def add(self, entry: ReflectionEntry) -> None:
        self.entries.append(entry)

    def extend(self, entries: list[ReflectionEntry]) -> None:
        self.entries.extend(entries)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[ReflectionEntry]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return self.entries[:top_k]

        scored: list[tuple[float, int, ReflectionEntry]] = []
        for index, entry in enumerate(self.entries):
            e_tokens = tokenize(entry.question)
            e_tokens.update(tokenize(entry.reflection))
            e_tokens.update(tokenize(" ".join(entry.expected_tools + entry.tools_called)))
            overlap = len(q_tokens & e_tokens)
            union = len(q_tokens | e_tokens) or 1
            jaccard = overlap / union
            quality = 0.1 * float(entry.f1)
            failure_bonus = 0.05 if entry.kind in {"failure", "recovery"} else 0.0
            score = jaccard + quality + failure_bonus
            if score > 0 or overlap > 0:
                scored.append((score, -index, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:top_k]]

    def stats(self) -> dict[str, Any]:
        kinds = Counter(entry.kind for entry in self.entries)
        statuses = Counter(entry.status for entry in self.entries)
        expected = Counter(tool for entry in self.entries for tool in entry.expected_tools)
        called = Counter(tool for entry in self.entries for tool in entry.tools_called)
        errors = Counter(error for entry in self.entries for error in entry.error_types)
        f1_values = [entry.f1 for entry in self.entries]
        return {
            "memory_path": str(self.path),
            "total_reflections": len(self.entries),
            "kind_counts": dict(kinds),
            "status_counts": dict(statuses),
            "avg_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
            "f1_std": math.sqrt(sum((value - (sum(f1_values) / len(f1_values))) ** 2 for value in f1_values) / len(f1_values))
            if f1_values
            else 0.0,
            "top_expected_tools": expected.most_common(30),
            "top_called_tools": called.most_common(30),
            "error_counts": dict(errors),
        }
