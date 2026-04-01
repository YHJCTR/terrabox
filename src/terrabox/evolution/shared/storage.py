"""Persistent storage backends for evolution artifacts."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Optional


class JSONSkillStore:
    """Thread-safe JSON file store for skill records.

    File format: {"skills": [SkillRecord, ...], "metadata": {...}}

    Each SkillRecord:
        id, category, content, tags, performance_delta,
        created_at, use_count, source_tasks
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            self._write({"skills": [], "metadata": {"created_at": time.time()}})

    def save(self, skill: dict) -> str:
        """Save a new skill; return skill_id."""
        with self._lock:
            data = self._read()
            skill_id = skill.get("id") or str(uuid.uuid4())
            skill["id"] = skill_id
            skill.setdefault("created_at", time.time())
            skill.setdefault("use_count", 0)
            data["skills"].append(skill)
            self._write(data)
        return skill_id

    def save_if_unique(self, skill: dict) -> Optional[str]:
        """Save skill only if content hash is new; return skill_id or None if duplicate."""
        content = skill.get("content", "")
        content_hash = self._hash(content)
        with self._lock:
            data = self._read()
            # Check for existing skill with same content hash
            for existing in data["skills"]:
                if existing.get("_content_hash") == content_hash:
                    return None   # duplicate — skip
            skill_id = skill.get("id") or str(uuid.uuid4())
            skill["id"] = skill_id
            skill["_content_hash"] = content_hash
            skill.setdefault("created_at", time.time())
            skill.setdefault("use_count", 0)
            data["skills"].append(skill)
            self._write(data)
        return skill_id

    def clear(self) -> None:
        """Remove all skills (for fresh re-runs)."""
        with self._lock:
            self._write({"skills": [], "metadata": {"created_at": time.time(), "cleared_at": time.time()}})

    @staticmethod
    def _hash(text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

    def load_all(self) -> list[dict]:
        with self._lock:
            return list(self._read().get("skills", []))

    def update(self, skill_id: str, updates: dict) -> bool:
        with self._lock:
            data = self._read()
            for skill in data["skills"]:
                if skill.get("id") == skill_id:
                    skill.update(updates)
                    skill["updated_at"] = time.time()
                    self._write(data)
                    return True
        return False

    def delete(self, skill_id: str) -> bool:
        with self._lock:
            data = self._read()
            before = len(data["skills"])
            data["skills"] = [s for s in data["skills"] if s.get("id") != skill_id]
            if len(data["skills"]) < before:
                self._write(data)
                return True
        return False

    def count(self) -> int:
        return len(self.load_all())

    def _read(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"skills": [], "metadata": {}}

    def _write(self, data: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


class SQLiteMemoryStore:
    """SQLite store for MemRL episodic memories.

    Schema:
        id TEXT PRIMARY KEY,
        intent TEXT,          -- JSON-encoded structured intent
        experience TEXT,      -- JSON-encoded trajectory summary
        utility REAL,         -- Q-value; higher = more useful
        q_visits INTEGER,
        embedding TEXT,       -- optional JSON float list
        created_at REAL,
        updated_at REAL
    """

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        task_id TEXT,
        intent TEXT NOT NULL,
        experience TEXT NOT NULL,
        utility REAL DEFAULT 0.5,
        q_visits INTEGER DEFAULT 0,
        embedding TEXT,
        created_at REAL,
        updated_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_utility ON memories(utility DESC);
    CREATE INDEX IF NOT EXISTS idx_task_id ON memories(task_id);
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._lock = threading.RLock()
        with self._conn() as conn:
            for stmt in self._CREATE_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()

    def insert(self, memory: dict) -> str:
        memory_id = memory.get("id") or str(uuid.uuid4())
        task_id = memory.get("task_id")
        now = time.time()
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (id, task_id, intent, experience, utility, q_visits, embedding, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        task_id,
                        json.dumps(memory.get("intent", {})),
                        json.dumps(memory.get("experience", {})),
                        memory.get("utility", 0.5),
                        memory.get("q_visits", 0),
                        json.dumps(memory.get("embedding")) if memory.get("embedding") else None,
                        memory.get("created_at", now),
                        now,
                    ),
                )
                conn.commit()
        return memory_id

    def exists_by_task_id(self, task_id: str) -> bool:
        """Check if a memory with this task_id already exists."""
        if not task_id:
            return False
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE task_id=? LIMIT 1", (task_id,)
            ).fetchone()
        return row is not None

    def clear(self) -> None:
        """Delete all memories (for fresh re-runs)."""
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM memories")
                conn.commit()

    def update_q_value(self, memory_id: str, new_utility: float, visits: int) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE memories SET utility=?, q_visits=?, updated_at=? WHERE id=?",
                    (new_utility, visits, time.time(), memory_id),
                )
                conn.commit()

    def get(self, memory_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all(self, limit: int = 10000) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY utility DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("intent", "experience", "embedding"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


class FileEpisodeStore:
    """Append-only JSONL store for raw episode trajectories."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()

    def append(self, episode: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def exists_by_task_id(self, task_id: str) -> bool:
        """Check if an episode with this task_id was already stored (scans file)."""
        if not task_id or not os.path.exists(self.path):
            return False
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ep = json.loads(line)
                        if ep.get("task_id") == task_id:
                            return True
                    except json.JSONDecodeError:
                        pass
        return False

    def get_all_task_ids(self) -> set[str]:
        """Return the set of all stored task_ids (for batch dedup check)."""
        ids: set[str] = set()
        if not os.path.exists(self.path):
            return ids
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tid = json.loads(line).get("task_id")
                        if tid:
                            ids.add(tid)
                    except json.JSONDecodeError:
                        pass
        return ids

    def clear(self) -> None:
        """Delete all episodes (for fresh re-runs)."""
        with self._lock:
            if os.path.exists(self.path):
                os.remove(self.path)

    def load_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        episodes = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return episodes

    def load_since(self, timestamp: float) -> list[dict]:
        return [e for e in self.load_all() if e.get("timestamp", 0) >= timestamp]

    def count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        count = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
