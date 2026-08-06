"""Persistent store for ExpeL principles."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass


@dataclass
class Principle:
    text: str
    score: float = 1.0
    support: int = 1


class PrincipleBank:
    """JSON-backed principle store.

    Layout:
      {
        "general": [{"text": ..., "score": ..., "support": ...}],
        "task_specific": {"task_type": [...]},
        "mistakes": [{...}]
      }
    """

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.path = os.path.join(store_dir, "principles.json")
        self.data = {
            "general": [],
            "task_specific": {},
            "mistakes": [],
            "successful_episodes": [],
            "manifest": {},
        }
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.data.setdefault("general", [])
            self.data.setdefault("task_specific", {})
            self.data.setdefault("mistakes", [])
            self.data.setdefault("successful_episodes", [])
            self.data.setdefault("manifest", {})

    def save(self) -> None:
        os.makedirs(self.store_dir, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=".principles.", suffix=".json", dir=self.store_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, self.path)

    @staticmethod
    def _entry(text: str, score: float = 1.0, support: int = 1, source_tasks: list[str] | None = None) -> dict:
        return {
            "text": text.strip(),
            "score": float(score),
            "support": int(support),
            "source_tasks": source_tasks or [],
        }

    def add_general(self, text: str, score: float = 1.0, source_task: str | None = None) -> None:
        self.data["general"].append(
            self._entry(text, score=score, support=1, source_tasks=[source_task] if source_task else [])
        )

    def add_task_specific(self, task_type: str, text: str, score: float = 1.0, source_task: str | None = None) -> None:
        if task_type not in self.data["task_specific"]:
            self.data["task_specific"][task_type] = []
        self.data["task_specific"][task_type].append(
            self._entry(text, score=score, support=1, source_tasks=[source_task] if source_task else [])
        )

    def add_mistake(self, text: str, score: float = 1.0, source_task: str | None = None) -> None:
        self.data["mistakes"].append(
            self._entry(text, score=score, support=1, source_tasks=[source_task] if source_task else [])
        )

    def add_successful_episode(self, episode: dict) -> None:
        """Store a short, anonymized successful episode for ExpeL few-shot retrieval."""
        episodes = self.data.setdefault("successful_episodes", [])
        episode_id = str(episode.get("source_id") or "")
        if episode_id and any(str(item.get("source_id") or "") == episode_id for item in episodes):
            return
        episodes.append(episode)

    def merge_duplicates(self) -> None:
        """Merge duplicate texts by summing support and score."""
        def _merge(entries: list[dict]) -> list[dict]:
            merged: dict[str, dict] = {}
            for e in entries:
                text = (e.get("text") or "").strip()
                if not text:
                    continue
                if text not in merged:
                    merged[text] = self._entry(
                        text=text,
                        score=float(e.get("score", 1.0)),
                        support=int(e.get("support", 1)),
                        source_tasks=list(e.get("source_tasks", [])),
                    )
                else:
                    merged[text]["score"] += float(e.get("score", 1.0))
                    merged[text]["support"] += int(e.get("support", 1))
                    merged[text]["source_tasks"].extend(e.get("source_tasks", []))
                seen_sources = set()
                unique_sources = []
                for source_task in merged[text]["source_tasks"]:
                    if source_task and source_task not in seen_sources:
                        seen_sources.add(source_task)
                        unique_sources.append(source_task)
                merged[text]["source_tasks"] = unique_sources
            out = list(merged.values())
            out.sort(key=lambda x: (x.get("support", 0), x.get("score", 0.0)), reverse=True)
            return out

        self.data["general"] = _merge(self.data.get("general", []))
        self.data["mistakes"] = _merge(self.data.get("mistakes", []))
        for task_type, entries in list(self.data.get("task_specific", {}).items()):
            self.data["task_specific"][task_type] = _merge(entries)
