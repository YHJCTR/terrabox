"""JSONL-backed Memento-style case bank."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "about",
    "after",
    "all",
    "and",
    "answer",
    "area",
    "based",
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


def normalize_tools(tools: object) -> set[str]:
    if not tools:
        return set()
    if isinstance(tools, dict):
        tools = tools.values()
    if isinstance(tools, (str, bytes)):
        tools = [tools]
    out: set[str] = set()
    for item in tools:  # type: ignore[assignment]
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("slug") or item.get("tool") or "")
        else:
            name = str(getattr(item, "name", "") or getattr(item, "slug", ""))
        if name:
            out.add(name)
    return out


@dataclass
class MemoryCase:
    id: str
    task_id: str
    task_type: str
    state: str
    action: list[str]
    reward: float
    outcome: str
    lesson: str
    keywords: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    final_answer_excerpt: str = ""
    tool_error: bool = False

    def to_record(self, *, strict_nolabel: bool = False) -> dict[str, Any]:
        """Serialize a case without benchmark-only fields in strict mode."""
        record = asdict(self)
        if strict_nolabel:
            for key in ("task_id", "task_type", "final_answer_excerpt"):
                record.pop(key, None)
        return record

    def prompt_summary(self, max_tools: int = 10) -> str:
        flow = " -> ".join(self.action[:max_tools]) if self.action else "no tool call"
        return (
            f"Case {self.id} reward={self.reward:.2f} outcome={self.outcome}\n"
            f"State: {self.state}\n"
            f"Action/tool plan: {flow}\n"
            f"Lesson: {self.lesson}"
        )


class CaseBank:
    """Load, save, and retrieve Memento-style cases."""

    CASE_FILE = "cases.jsonl"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.CASE_FILE
        self.manifest_path = self.store_dir / self.MANIFEST_FILE
        self.cases: list[MemoryCase] = []
        self.manifest: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        self.cases = []
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.manifest = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                self.cases.append(
                    MemoryCase(
                        id=str(row.get("id") or f"case-{len(self.cases) + 1:05d}"),
                        task_id=str(row.get("task_id") or "unknown"),
                        task_type=str(row.get("task_type") or "unknown"),
                        state=str(row.get("state") or ""),
                        action=[str(x) for x in row.get("action") or row.get("tools") or []],
                        reward=float(row.get("reward") or 0.0),
                        outcome=str(row.get("outcome") or "unknown"),
                        lesson=str(row.get("lesson") or ""),
                        keywords=[str(x) for x in row.get("keywords") or []],
                        tools=[str(x) for x in row.get("tools") or row.get("action") or []],
                        final_answer_excerpt=str(row.get("final_answer_excerpt") or ""),
                        tool_error=bool(row.get("tool_error")),
                    )
                )

    def save(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        strict_nolabel = bool(self.manifest.get("strict_nolabel", False))
        with self.path.open("w", encoding="utf-8") as f:
            for case in self.cases:
                f.write(json.dumps(case.to_record(strict_nolabel=strict_nolabel), ensure_ascii=False) + "\n")
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def retrieve(
        self,
        query: str,
        *,
        task_type: str | None = None,
        available_tools: object = None,
        top_k: int = 5,
        include_negative: bool = True,
        similarity_fn: Any | None = None,
    ) -> list[tuple[float, MemoryCase]]:
        if not self.cases or top_k <= 0:
            return []
        # Memento's non-parametric retriever ranks the complete memory pool
        # from the natural-language task. ``task_type`` is benchmark metadata
        # in OEA, not an observation available to a normal agent, so it must
        # not narrow or rank the runtime candidate set.
        del task_type, available_tools
        q_tokens = tokenize(query)
        candidate_cases = self.cases
        scored: list[tuple[float, MemoryCase]] = []
        for case in candidate_cases:
            c_tokens = set(case.keywords) | tokenize(case.state) | tokenize(case.lesson) | tokenize(" ".join(case.action))
            overlap = len(q_tokens & c_tokens)
            reward_bonus = case.reward * 3.0
            negative_bonus = 1.0 if include_negative and case.reward < 0.4 and overlap else 0.0
            negative_penalty = 3.0 if case.reward < 0.5 else 0.0
            semantic = 0.0
            if similarity_fn is not None:
                try:
                    semantic = float(similarity_fn(query, case))
                except Exception:
                    semantic = 0.0
            score = overlap * 2.0 + reward_bonus + negative_bonus + semantic * 8.0 - negative_penalty
            if overlap == 0 and score < 2.5:
                continue
            scored.append((score, case))
        if not scored:
            scored = [(case.reward, case) for case in candidate_cases]
        scored.sort(key=lambda item: (item[0], item[1].reward), reverse=True)

        positive_candidates: list[tuple[float, MemoryCase]] = []
        negative_candidates: list[tuple[float, MemoryCase]] = []
        seen: set[str] = set()
        for score, case in scored:
            key = f"{case.state}|{' '.join(case.action[:8])}"
            if key in seen:
                continue
            seen.add(key)
            if case.reward >= 0.5 or not include_negative:
                positive_candidates.append((score, case))
            else:
                negative_candidates.append((score, case))

        out = positive_candidates[:top_k]
        if include_negative and len(out) < top_k and negative_candidates:
            out.append(negative_candidates[0])
        if not out:
            out = negative_candidates[:top_k]
        return out
