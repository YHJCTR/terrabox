"""ExpeL prompt injector: retrieve and inject principle rules."""
from __future__ import annotations

import re
from collections.abc import Callable

from ..shared.prompt_builder import PromptAugmenter
from .principle_bank import PrincipleBank

SimilarityFn = Callable[[str, str], float]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]{3,}", (text or "").lower()))


def _rank_principles(
    query: str,
    candidates: list[dict],
    top_k: int,
    similarity_fn: SimilarityFn | None = None,
) -> list[str]:
    q = _tokens(query)
    scored: list[tuple[float, str]] = []
    for item in candidates:
        text = item.get("text", "")
        if not text:
            continue
        base = float(item.get("score", 1.0))
        support = int(item.get("support", 1))
        overlap = len(q.intersection(_tokens(text)))
        semantic = 0.0
        if similarity_fn is not None:
            try:
                semantic = float(similarity_fn(query, text))
            except Exception:
                semantic = 0.0
        score = base + support * 0.25 + overlap * 2.0 + semantic * 10.0
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

    def __init__(
        self,
        bank: PrincipleBank,
        top_k: int = 5,
        similarity_fn: SimilarityFn | None = None,
    ):
        self._bank = bank
        self.top_k = top_k
        self._similarity_fn = similarity_fn

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        similarity_fn = kwargs.get("similarity_fn") or self._similarity_fn

        general = _rank_principles(
            user_query,
            self._bank.data.get("general", []),
            top_k=self.top_k,
            similarity_fn=similarity_fn,
        )

        task_entries = self._bank.data.get("task_specific", {}).get(task_type, [])
        task_rules = _rank_principles(
            user_query,
            task_entries,
            top_k=max(1, self.top_k),
            similarity_fn=similarity_fn,
        )

        mistakes = _rank_principles(
            user_query,
            self._bank.data.get("mistakes", []),
            top_k=max(2, self.top_k // 2),
            similarity_fn=similarity_fn,
        )

        augmentation = ""
        augmentation += self._format_skill_block(general, "ExpeL Principles")
        augmentation += self._format_skill_block(task_rules, "Task-Specific ExpeL Principles")
        augmentation += self._format_skill_block(mistakes, "ExpeL Cautions")

        if not augmentation.strip():
            return self.BASE_SYSTEM
        return self.BASE_SYSTEM + augmentation
