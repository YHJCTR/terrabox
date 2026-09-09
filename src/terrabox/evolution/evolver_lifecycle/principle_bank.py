"""Persistent EvolveR-style principle bank.

The schema mirrors the lightweight objects used by the official EvolveR
ExperienceManager while removing OEA labels, task identifiers, gold answers,
expected tool traces, and final-answer facts in strict mode.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


_STOPWORDS = {
    "about",
    "after",
    "all",
    "and",
    "answer",
    "area",
    "based",
    "before",
    "between",
    "calculate",
    "compute",
    "does",
    "from",
    "generate",
    "given",
    "image",
    "into",
    "need",
    "provide",
    "show",
    "that",
    "the",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
}


def tokenize(text: object) -> set[str]:
    value = str(text or "").lower()
    raw = re.findall(r"[a-z0-9_\.\-]{3,}|[\u4e00-\u9fff]{2,}", value)
    out: set[str] = set()
    for token in raw:
        token = token.strip("._-")
        if not token or token in _STOPWORDS:
            continue
        out.add(token)
        if "." in token:
            out.update(part for part in token.split(".") if len(part) >= 3 and part not in _STOPWORDS)
        if "_" in token:
            out.update(part for part in token.split("_") if len(part) >= 3 and part not in _STOPWORDS)
    return out


@dataclass
class ExperiencePrinciple:
    principle_id: str
    type: str
    description: str
    structure: list[Any] = field(default_factory=list)
    metric_score: float = 1.0
    usage_count: int = 0
    success_count: int = 0
    successful_trajectory_ids: list[str] = field(default_factory=list)
    failed_trajectory_ids: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["type"] = "cautionary" if self.type == "cautionary" else "guiding"
        record["description"] = self.description.strip()
        return record


@dataclass
class RetrievedExperiencePackage:
    principle: ExperiencePrinciple
    similarity_score: float
    positive_examples: list[dict[str, Any]] = field(default_factory=list)
    negative_examples: list[dict[str, Any]] = field(default_factory=list)


class EvolveRPrincipleBank:
    """JSON-backed store for EvolveR-style principles and compact trajectories."""

    PRINCIPLE_FILE = "principles.jsonl"
    TRAJECTORY_FILE = "trajectories.jsonl"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.principle_path = self.store_dir / self.PRINCIPLE_FILE
        self.trajectory_path = self.store_dir / self.TRAJECTORY_FILE
        self.manifest_path = self.store_dir / self.MANIFEST_FILE
        self.principles: list[ExperiencePrinciple] = []
        self.trajectories: dict[str, dict[str, Any]] = {}
        self.manifest: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        self.principles = []
        self.trajectories = {}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.principle_path.exists():
            with self.principle_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    self.principles.append(
                        ExperiencePrinciple(
                            principle_id=str(row.get("principle_id") or f"principle-{len(self.principles) + 1:05d}"),
                            type=str(row.get("type") or "guiding"),
                            description=str(row.get("description") or ""),
                            structure=list(row.get("structure") or []),
                            metric_score=float(row.get("metric_score") or 0.0),
                            usage_count=int(row.get("usage_count") or 0),
                            success_count=int(row.get("success_count") or 0),
                            successful_trajectory_ids=[str(x) for x in row.get("successful_trajectory_ids") or []],
                            failed_trajectory_ids=[str(x) for x in row.get("failed_trajectory_ids") or []],
                        )
                    )
        if self.trajectory_path.exists():
            with self.trajectory_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("trajectory_id"):
                        self.trajectories[str(row["trajectory_id"])] = row

    def save(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_principles = tempfile.mkstemp(prefix=".principles.", suffix=".jsonl", dir=self.store_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for principle in self.principles:
                f.write(json.dumps(principle.to_record(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_principles, self.principle_path)

        fd, tmp_trajectories = tempfile.mkstemp(prefix=".trajectories.", suffix=".jsonl", dir=self.store_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for trajectory in self.trajectories.values():
                f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_trajectories, self.trajectory_path)

        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_trajectory(self, trajectory: dict[str, Any]) -> None:
        trajectory_id = str(trajectory.get("trajectory_id") or f"traj-{len(self.trajectories) + 1:05d}")
        trajectory = dict(trajectory)
        trajectory["trajectory_id"] = trajectory_id
        trajectory["golden_answer"] = ""
        self.trajectories[trajectory_id] = trajectory

    def add_principle(self, principle: ExperiencePrinciple) -> None:
        if not principle.description.strip():
            return
        self.principles.append(principle)

    def merge_duplicates(self) -> None:
        merged: dict[tuple[str, str], ExperiencePrinciple] = {}
        for principle in self.principles:
            key = (principle.type, re.sub(r"\s+", " ", principle.description.strip().lower()))
            if key not in merged:
                merged[key] = principle
                continue
            target = merged[key]
            target.metric_score += principle.metric_score
            target.usage_count += principle.usage_count
            target.success_count += principle.success_count
            target.successful_trajectory_ids = _unique(target.successful_trajectory_ids + principle.successful_trajectory_ids)
            target.failed_trajectory_ids = _unique(target.failed_trajectory_ids + principle.failed_trajectory_ids)
            if not target.structure and principle.structure:
                target.structure = principle.structure
        out = list(merged.values())
        out.sort(key=lambda p: (p.metric_score, len(p.successful_trajectory_ids) + len(p.failed_trajectory_ids)), reverse=True)
        for idx, principle in enumerate(out, 1):
            principle.principle_id = f"principle-{idx:05d}"
        self.principles = out

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        similarity_fn: Callable[[str, ExperiencePrinciple], float] | None = None,
    ) -> list[RetrievedExperiencePackage]:
        if not self.principles or top_k <= 0:
            return []
        q_tokens = tokenize(query)
        scored: list[tuple[float, ExperiencePrinciple]] = []
        for principle in self.principles:
            text = principle.description + " " + json.dumps(principle.structure, ensure_ascii=False)
            overlap = len(q_tokens & tokenize(text))
            semantic = 0.0
            if similarity_fn is not None:
                try:
                    semantic = float(similarity_fn(query, principle))
                except Exception:
                    semantic = 0.0
            score = overlap * 1.5 + principle.metric_score + semantic * 10.0
            if overlap == 0 and score < 1.5:
                continue
            scored.append((score, principle))
        if not scored:
            scored = [(principle.metric_score, principle) for principle in self.principles]
        scored.sort(key=lambda item: item[0], reverse=True)

        packages: list[RetrievedExperiencePackage] = []
        for score, principle in scored[:top_k]:
            positives = [self.trajectories[tid] for tid in principle.successful_trajectory_ids[:1] if tid in self.trajectories]
            negatives = [self.trajectories[tid] for tid in principle.failed_trajectory_ids[:1] if tid in self.trajectories]
            packages.append(
                RetrievedExperiencePackage(
                    principle=principle,
                    similarity_score=score,
                    positive_examples=positives,
                    negative_examples=negatives,
                )
            )
        return packages


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
