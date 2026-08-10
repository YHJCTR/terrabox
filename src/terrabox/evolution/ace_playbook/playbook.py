"""JSON-backed ACE-style playbook store.

The official ACE project represents evolved context as playbook bullets with
helpful/harmful counters. This module keeps the same memory shape while using a
Terrabox-native, dependency-light retriever for OEA live evaluation.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "analysis",
    "answer",
    "area",
    "based",
    "between",
    "calculate",
    "compute",
    "create",
    "does",
    "during",
    "each",
    "from",
    "generate",
    "given",
    "have",
    "image",
    "into",
    "location",
    "map",
    "need",
    "provide",
    "show",
    "task",
    "the",
    "that",
    "their",
    "there",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(text: object) -> set[str]:
    """Return coarse lexical tokens for cheap, deterministic retrieval."""
    value = str(text or "").lower()
    raw = re.findall(r"[a-z0-9_\.\-]{3,}|[\u4e00-\u9fff]{2,}", value)
    tokens: set[str] = set()
    for token in raw:
        token = token.strip("._-")
        if not token or token in _STOPWORDS:
            continue
        tokens.add(token)
        if "." in token:
            tokens.update(part for part in token.split(".") if len(part) >= 3 and part not in _STOPWORDS)
        if "_" in token:
            tokens.update(part for part in token.split("_") if len(part) >= 3 and part not in _STOPWORDS)
    return tokens


def normalize_tools(tools: object) -> set[str]:
    """Normalize available tool payloads from rollout kwargs."""
    out: set[str] = set()
    if not tools:
        return out
    if isinstance(tools, dict):
        tools = tools.values()
    if isinstance(tools, (str, bytes)):
        tools = [tools]
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
class PlaybookBullet:
    id: str
    text: str
    helpful: int = 0
    harmful: int = 0
    support: int = 0
    task_types: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_tasks: list[str] = field(default_factory=list)
    section: str = "tool_flow"

    @property
    def prior_score(self) -> float:
        return math.log1p(max(self.support, self.helpful)) + self.helpful * 0.25 - self.harmful * 0.4

    def as_prompt_line(self) -> str:
        return f"[{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.text}"


class ACEPlaybook:
    """Persistent ACE-style playbook with deterministic retrieval."""

    FILE_NAME = "playbook.json"


    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.manifest: dict[str, Any] = {}
        self.bullets: list[PlaybookBullet] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.manifest = dict(data.get("manifest") or {})
        self.bullets = []
        for row in data.get("bullets") or []:
            if not isinstance(row, dict) or not row.get("text"):
                continue
            self.bullets.append(
                PlaybookBullet(
                    id=str(row.get("id") or f"ace-{len(self.bullets) + 1:05d}"),
                    text=str(row.get("text") or "").strip(),
                    helpful=int(row.get("helpful") or 0),
                    harmful=int(row.get("harmful") or 0),
                    support=int(row.get("support") or row.get("helpful") or 0),
                    task_types=[str(x) for x in row.get("task_types") or []],
                    tools=[str(x) for x in row.get("tools") or []],
                    keywords=[str(x) for x in row.get("keywords") or []],
                    source_tasks=[str(x) for x in row.get("source_tasks") or []],
                    section=str(row.get("section") or "tool_flow"),
                )
            )

    def save(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": self.manifest,
            "bullets": [asdict(bullet) for bullet in self.bullets],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write_text_playbook()

    def write_text_playbook(self) -> None:
        """Write a human-readable ACE playbook in official line format."""
        grouped: dict[str, list[PlaybookBullet]] = {}
        for bullet in self.bullets:
            grouped.setdefault(bullet.section or "tool_flow", []).append(bullet)
        lines: list[str] = ["# ACE-style Terrabox Playbook"]
        for section, bullets in sorted(grouped.items()):
            lines.append("")
            lines.append(f"## {section}")
            lines.extend(bullet.as_prompt_line() for bullet in bullets)
        (self.store_dir / "playbook.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def as_text(self, *, max_bullets: int | None = None) -> str:
        """Return the playbook in ACE's human-readable bullet format."""
        grouped: dict[str, list[PlaybookBullet]] = {}
        bullets = self.bullets if max_bullets is None else self.bullets[-max_bullets:]
        for bullet in bullets:
            grouped.setdefault(bullet.section or "tool_flow", []).append(bullet)
        lines: list[str] = ["# ACE-style Terrabox Playbook"]
        for section, section_bullets in sorted(grouped.items()):
            lines.append("")
            lines.append(f"## {section}")
            lines.extend(bullet.as_prompt_line() for bullet in section_bullets)
        return "\n".join(lines).strip()

    def format_bullets(self, bullet_ids: list[str] | set[str]) -> str:
        """Format selected bullets for ACE Reflector tagging."""
        wanted = {str(bullet_id) for bullet_id in bullet_ids if str(bullet_id).strip()}
        lines = [bullet.as_prompt_line() for bullet in self.bullets if bullet.id in wanted]
        return "\n".join(lines) if lines else "(No playbook bullets were used.)"

    def update_bullet_counts(self, bullet_tags: object) -> dict[str, int]:
        """Apply ACE-style helpful/harmful/neutral tags to bullet counters.

        Official ACE updates playbook counters after the Reflector tags bullets
        used by the Generator. This store keeps the same counter semantics while
        allowing the OEA adapter to tag retrieved playbook bullets.
        """
        if not isinstance(bullet_tags, list):
            return {"helpful": 0, "harmful": 0, "neutral": 0, "unknown": 0}
        by_id = {bullet.id: bullet for bullet in self.bullets}
        counts = {"helpful": 0, "harmful": 0, "neutral": 0, "unknown": 0}
        for item in bullet_tags:
            if not isinstance(item, dict):
                counts["unknown"] += 1
                continue
            bullet_id = str(item.get("id") or item.get("bullet") or item.get("bullet_id") or "").strip()
            tag = str(item.get("tag") or "neutral").strip().lower()
            bullet = by_id.get(bullet_id)
            if bullet is None or tag not in {"helpful", "harmful", "neutral"}:
                counts["unknown"] += 1
                continue
            if tag == "helpful":
                bullet.helpful += 1
            elif tag == "harmful":
                bullet.harmful += 1
            counts[tag] += 1
        return counts

    def retrieve(
        self,
        query: str,
        *,
        task_type: str | None = None,
        available_tools: object = None,
        top_k: int = 5,
    ) -> list[tuple[float, PlaybookBullet]]:
        if not self.bullets or top_k <= 0:
            return []
        q_tokens = tokenize(" ".join([str(task_type or ""), query]))
        avail = normalize_tools(available_tools)
        candidate_bullets = self.bullets
        if task_type:
            exact = [bullet for bullet in self.bullets if task_type in set(bullet.task_types)]
            if exact:
                candidate_bullets = exact

        scored: list[tuple[float, PlaybookBullet]] = []
        for bullet in candidate_bullets:
            b_tokens = set(bullet.keywords) | tokenize(bullet.text) | tokenize(" ".join(bullet.tools))
            overlap = len(q_tokens & b_tokens)
            type_bonus = 4.0 if task_type and task_type in set(bullet.task_types) else 0.0
            tool_bonus = len(avail & set(bullet.tools)) * 0.35 if avail else 0.0
            caution_bonus = 1.0 if bullet.section == "caution" and overlap else 0.0
            score = bullet.prior_score + overlap * 2.2 + type_bonus + tool_bonus + caution_bonus
            if overlap == 0 and type_bonus == 0 and score < 2.0:
                continue
            scored.append((score, bullet))
        if not scored:
            scored = [(bullet.prior_score, bullet) for bullet in candidate_bullets]
        scored.sort(key=lambda item: (item[0], item[1].helpful, item[1].support), reverse=True)

        out: list[tuple[float, PlaybookBullet]] = []
        seen_text: set[str] = set()
        for score, bullet in scored:
            if bullet.text in seen_text:
                continue
            seen_text.add(bullet.text)
            out.append((score, bullet))
            if len(out) >= top_k:
                break
        return out
