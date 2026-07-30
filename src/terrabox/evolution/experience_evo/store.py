"""JSONL + SQLite storage for ExperienceEvo."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .schemas import ExperienceEntry, TransitionRecord


def _json_default(value: Any) -> str:
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "and",
    "any",
    "are",
    "between",
    "both",
    "can",
    "could",
    "each",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "most",
    "near",
    "need",
    "other",
    "should",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "use",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
    "would",
}

_DOMAIN_ALIASES = {
    "closest": {"nearest", "distance", "route", "proximity", "compute_route_dist"},
    "nearest": {"closest", "distance", "route", "proximity", "compute_route_dist"},
    "distance": {"closest", "nearest", "route", "compute_route_dist"},
    "fire": {"poi", "pois", "station", "add_pois_layer", "amenity"},
    "police": {"poi", "pois", "station", "add_pois_layer", "amenity"},
    "hospital": {"poi", "pois", "add_pois_layer", "amenity"},
    "school": {"poi", "pois", "add_pois_layer", "amenity"},
    "restaurant": {"poi", "pois", "add_pois_layer", "amenity"},
    "station": {"poi", "pois", "add_pois_layer", "amenity"},
    "boundary": {"area", "gpkg", "get_area_boundary"},
    "park": {"area", "boundary", "gpkg", "get_area_boundary"},
    "national": {"area", "boundary", "gpkg", "get_area_boundary"},
    "count": {"perception_counts", "detect", "detection"},
    "detect": {"perception_counts", "count", "detection"},
    "segment": {"mask", "perception", "sam", "sam2"},
    "change": {"index_change", "compute_index_change", "difference"},
    "ndvi": {"index", "add_index_layer", "compute_index_change"},
    "nbr": {"index", "add_index_layer", "compute_index_change"},
    "ndbi": {"index", "add_index_layer", "compute_index_change"},
}


def _tokens(text: str) -> set[str]:
    raw: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_]{3,}", (text or "").lower()):
        raw.add(token)
        if "_" in token:
            raw.update(part for part in token.split("_") if len(part) >= 3)
    tokens = {token for token in raw if token not in _STOPWORDS}
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_DOMAIN_ALIASES.get(token, set()))
    return expanded


class ExperienceEvoStore:
    """A small portable store with JSONL as source of truth and SQLite mirror."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.transitions_path = self.store_dir / "transitions.jsonl"
        self.experiences_path = self.store_dir / "experiences.jsonl"
        self.manifest_path = self.store_dir / "manifest.json"
        self.sqlite_path = self.store_dir / "experience_evo.sqlite"

    def load_transitions(self) -> list[TransitionRecord]:
        return [TransitionRecord.from_dict(row) for row in _read_jsonl(self.transitions_path)]

    def write_transitions(self, transitions: list[TransitionRecord]) -> None:
        rows = [item.to_dict() for item in transitions]
        _write_jsonl(self.transitions_path, rows)
        self._write_sqlite(transitions=transitions, experiences=None)

    def load_experiences(self) -> list[ExperienceEntry]:
        return [ExperienceEntry.from_dict(row) for row in _read_jsonl(self.experiences_path)]

    def write_experiences(self, experiences: list[ExperienceEntry]) -> None:
        rows = [item.to_dict() for item in experiences]
        _write_jsonl(self.experiences_path, rows)
        self._write_sqlite(transitions=None, experiences=experiences)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        current = self.manifest()
        current.update(manifest)
        self.manifest_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def stats(self) -> dict[str, Any]:
        transitions = self.load_transitions()
        experiences = self.load_experiences()
        return {
            "store_dir": str(self.store_dir),
            "transitions": len(transitions),
            "infra_transitions": sum(1 for t in transitions if getattr(t, "infra_error", False)),
            "experience_risk_transitions": sum(1 for t in transitions if t.risk > 0),
            "experiences": len(experiences),
            "transition_tools": Counter(t.tool for t in transitions).most_common(20),
            "experience_levels": dict(Counter(e.level for e in experiences)),
            "experience_tools": Counter(e.tool or "(signature)" for e in experiences).most_common(20),
            "manifest": self.manifest(),
        }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        level: str | None = None,
        task_type: str | None = None,
        min_q: float = -1.0,
        max_risk: float = 1.0,
    ) -> list[ExperienceEntry]:
        q_tokens = _tokens(query)
        scored: list[tuple[float, ExperienceEntry]] = []
        for entry in self.load_experiences():
            if level and entry.level != level:
                continue
            if task_type and entry.task_type not in {task_type, "general", "unknown"}:
                continue
            if entry.q < min_q or entry.risk > max_risk:
                continue
            e_tokens = _tokens(entry.search_text)
            overlap = len(q_tokens & e_tokens)
            if not overlap and q_tokens:
                # Keep low-Q zero-overlap memories out unless the store is tiny.
                continue
            support = min(entry.n, 10) * 0.05
            signature_match = 0
            if entry.input_signature:
                signature_match += len(q_tokens & _tokens(entry.input_signature))
            if entry.output_signature:
                signature_match += len(q_tokens & _tokens(entry.output_signature))
            score = overlap * 2.5 + signature_match * 1.0 + entry.q * 1.0 + support - entry.risk * 2.0
            if entry.level == "signature":
                score += 0.75
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[ExperienceEntry] = []
        seen: set[str] = set()
        for _, entry in scored:
            if entry.experience_id in seen:
                continue
            seen.add(entry.experience_id)
            out.append(entry)
            if len(out) >= top_k:
                break
        return out

    def _write_sqlite(
        self,
        *,
        transitions: list[TransitionRecord] | None,
        experiences: list[ExperienceEntry] | None,
    ) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    transition_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    source TEXT,
                    task_type TEXT,
                    tool TEXT,
                    input_signature TEXT,
                    output_signature TEXT,
                    reward REAL,
                    risk REAL,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    level TEXT,
                    task_type TEXT,
                    tool TEXT,
                    input_signature TEXT,
                    output_signature TEXT,
                    q REAL,
                    n INTEGER,
                    risk REAL,
                    status TEXT,
                    search_text TEXT,
                    payload TEXT
                )
                """
            )
            if transitions is not None:
                conn.execute("DELETE FROM transitions")
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO transitions
                    (transition_id, task_id, source, task_type, tool, input_signature,
                     output_signature, reward, risk, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.transition_id,
                            item.task_id,
                            item.source,
                            item.task_type,
                            item.tool,
                            item.input_signature,
                            item.output_signature,
                            item.reward,
                            item.risk,
                            json.dumps(item.to_dict(), ensure_ascii=False, default=_json_default),
                        )
                        for item in transitions
                    ],
                )
            if experiences is not None:
                conn.execute("DELETE FROM experiences")
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO experiences
                    (experience_id, level, task_type, tool, input_signature, output_signature,
                     q, n, risk, status, search_text, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.experience_id,
                            item.level,
                            item.task_type,
                            item.tool,
                            item.input_signature,
                            item.output_signature,
                            item.q,
                            item.n,
                            item.risk,
                            item.status,
                            item.search_text,
                            json.dumps(item.to_dict(), ensure_ascii=False, default=_json_default),
                        )
                        for item in experiences
                    ],
                )
            conn.commit()
        finally:
            conn.close()
