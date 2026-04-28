"""ExpeL prompt injector: retrieve and inject principle rules."""
from __future__ import annotations

import re

from ..shared.prompt_builder import PromptAugmenter
from .principle_bank import PrincipleBank


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]{3,}", (text or "").lower()))


def _rank_by_overlap(query: str, candidates: list[dict], top_k: int) -> list[str]:
    q = _tokens(query)
    scored: list[tuple[float, str]] = []
    for item in candidates:
        text = item.get("text", "")
        base = float(item.get("score", 1.0))
        overlap = len(q.intersection(_tokens(text)))
        score = base + overlap * 2.0
        if text:
            scored.append((score, text))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    seen = set()
    for _, text in scored:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= top_k:
            break
    return out


class ExpeLPromptInjector(PromptAugmenter):
    """Inject concise principles distilled from historical trajectories."""

    def __init__(self, bank: PrincipleBank, top_k: int = 5):
        self._bank = bank
        self.top_k = top_k

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        general = _rank_by_overlap(
            user_query,
            self._bank.data.get("general", []),
            top_k=self.top_k,
        )

        task_entries = self._bank.data.get("task_specific", {}).get(task_type, [])
        task_rules = _rank_by_overlap(user_query, task_entries, top_k=max(2, self.top_k // 2))

        mistakes = _rank_by_overlap(
            user_query,
            self._bank.data.get("mistakes", []),
            top_k=max(2, self.top_k // 2),
        )

        augmentation = ""
        augmentation += self._format_skill_block(general, "ExpeL Principles")
        augmentation += self._format_skill_block(task_rules, "Task-Specific ExpeL Principles")
        augmentation += self._format_skill_block(mistakes, "ExpeL Cautions")

        if not augmentation.strip():
            return self.BASE_SYSTEM
        return self.BASE_SYSTEM + augmentation
