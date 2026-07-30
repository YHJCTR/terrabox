"""JSONL + SQLite storage for ExperienceEvo v2 families."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ....agent.artifacts.signatures import state_overlap
from ..store import _tokens
from .models import TransitionEvent, TransitionFamily


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


class ExperienceEvoV2Store:
    """Portable v2 store with JSONL source of truth and SQLite mirror."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.events_path = self.store_dir / "events_v2.jsonl"
        self.families_path = self.store_dir / "families_v2.jsonl"
        self.manifest_path = self.store_dir / "manifest.json"
        self.sqlite_path = self.store_dir / "experience_evo_v2.sqlite"

    def load_events(self) -> list[TransitionEvent]:
        return [TransitionEvent.from_dict(row) for row in _read_jsonl(self.events_path)]

    def write_events(self, events: list[TransitionEvent]) -> None:
        _write_jsonl(self.events_path, [event.to_dict() for event in events])
        self._write_sqlite(events=events, families=None)

    def load_families(self) -> list[TransitionFamily]:
        return [TransitionFamily.from_dict(row) for row in _read_jsonl(self.families_path)]

    def write_families(self, families: list[TransitionFamily]) -> None:
        _write_jsonl(self.families_path, [family.to_dict() for family in families])
        self._write_sqlite(events=None, families=families)

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, manifest: dict[str, Any], *, replace: bool = False) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        current = {} if replace else self.manifest()
        current.update(manifest)
        self.manifest_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def stats(self) -> dict[str, Any]:
        events = self.load_events()
        families = self.load_families()
        policies = [policy for family in families for policy in family.tool_policies]
        return {
            "store_dir": str(self.store_dir),
            "schema_version": 2,
            "events": len(events),
            "infra_events": sum(1 for event in events if event.evidence.infra_error),
            "risk_events": sum(1 for event in events if event.evidence.risk_observed > 0),
            "families": len(families),
            "family_status": dict(Counter(family.status for family in families)),
            "event_tools": Counter(event.tool for event in events).most_common(20),
            "policy_tools": Counter(policy.tool for policy in policies).most_common(20),
            "manifest": self.manifest(),
        }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        task_type: str | None = None,
        current_product_state: list[str] | None = None,
        min_q: float = 0.0,
        max_risk: float = 0.75,
    ) -> list[TransitionFamily]:
        q_tokens = _tokens(query)
        current_state = [str(item) for item in (current_product_state or [])]
        scored: list[tuple[float, TransitionFamily]] = []

        for family in self.load_families():
            exp = family.product_experience
            if task_type and family.task_type not in {task_type, "general", "unknown"}:
                continue
            if exp.q < min_q or exp.risk > max_risk:
                continue

            tool_text = " ".join(
                " ".join(
                    [
                        policy.tool,
                        policy.experience,
                        " ".join(policy.parameter_binding_rules),
                        " ".join(policy.output_contract),
                        " ".join(policy.post_checks),
                    ]
                )
                for policy in family.tool_policies
            )
            f_tokens = _tokens(family.search_text + "\n" + tool_text)
            overlap = len(q_tokens & f_tokens)
            if q_tokens and not overlap:
                continue

            state_score = 0.0
            if current_state:
                state_score = state_overlap(current_state, family.input_product_state) * 2.0
            support = min(exp.n, 10) * 0.05
            status_bonus = 0.5 if family.status == "positive" else 0.0
            score = overlap * 2.5 + state_score + exp.q + support + status_bonus - exp.risk * 2.0
            scored.append((score, family))

        scored.sort(key=lambda item: item[0], reverse=True)
        output: list[TransitionFamily] = []
        seen: set[str] = set()
        for _, family in scored:
            if family.family_id in seen:
                continue
            seen.add(family.family_id)
            output.append(family)
            if len(output) >= top_k:
                break
        return output

    def _write_sqlite(
        self,
        *,
        events: list[TransitionEvent] | None,
        families: list[TransitionFamily] | None,
    ) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events_v2 (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    source TEXT,
                    task_type TEXT,
                    intent_signature TEXT,
                    tool TEXT,
                    reward REAL,
                    risk REAL,
                    infra_error INTEGER,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS families_v2 (
                    family_id TEXT PRIMARY KEY,
                    task_type TEXT,
                    intent_signature TEXT,
                    input_product_state TEXT,
                    target_product_state TEXT,
                    q REAL,
                    n INTEGER,
                    risk REAL,
                    status TEXT,
                    search_text TEXT,
                    payload TEXT
                )
                """
            )
            if events is not None:
                conn.execute("DELETE FROM events_v2")
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO events_v2
                    (event_id, task_id, source, task_type, intent_signature, tool,
                     reward, risk, infra_error, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event.event_id,
                            event.task_id,
                            event.source,
                            event.task_type,
                            event.intent_signature,
                            event.tool,
                            event.evidence.reward,
                            event.evidence.risk_observed,
                            1 if event.evidence.infra_error else 0,
                            json.dumps(event.to_dict(), ensure_ascii=False, default=_json_default),
                        )
                        for event in events
                    ],
                )
            if families is not None:
                conn.execute("DELETE FROM families_v2")
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO families_v2
                    (family_id, task_type, intent_signature, input_product_state,
                     target_product_state, q, n, risk, status, search_text, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            family.family_id,
                            family.task_type,
                            family.intent_signature,
                            json.dumps(family.input_product_state, ensure_ascii=False),
                            json.dumps(family.target_product_state, ensure_ascii=False),
                            family.product_experience.q,
                            family.product_experience.n,
                            family.product_experience.risk,
                            family.status,
                            family.search_text,
                            json.dumps(family.to_dict(), ensure_ascii=False, default=_json_default),
                        )
                        for family in families
                    ],
                )
            conn.commit()
        finally:
            conn.close()
