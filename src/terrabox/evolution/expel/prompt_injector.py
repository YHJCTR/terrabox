"""ExpeL prompt injector: retrieve and inject principle rules."""
from __future__ import annotations

import re
from collections.abc import Callable

from ..shared.prompt_builder import PromptAugmenter
from .principle_bank import PrincipleBank
from .semantic_retriever import QwenEmbeddingIndex

SimilarityFn = Callable[[str, str], float]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z_]{3,}|[\u4e00-\u9fff]{2,}", (text or "").lower())
    return {w for w in words if w not in {"the", "and", "for", "with", "from", "this", "that"}}


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
        semantic_index: QwenEmbeddingIndex | None = None,
    ):
        self._bank = bank
        self.top_k = top_k
        self._similarity_fn = similarity_fn
        self._semantic_index = semantic_index

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        similarity_fn = kwargs.get("similarity_fn") or self._similarity_fn
        if similarity_fn is None and self._semantic_index is not None:
            similarity_fn = self._semantic_index.similarity

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

        episodes = self._retrieve_episodes(user_query, task_type, top_k=1, similarity_fn=similarity_fn)
        if episodes:
            lines = ["\n\n## Retrieved Successful ExpeL Episode"]
            for episode in episodes:
                lines.append(f"Task pattern: {episode.get('task_pattern', 'unknown')}")
                lines.append(f"Successful tool plan: {' -> '.join(episode.get('tool_sequence', []))}")
                if episode.get("steps"):
                    lines.append("Relevant steps:")
                    lines.extend(f"- {step}" for step in episode["steps"][:6])
            augmentation += "\n".join(lines)

        if not augmentation.strip():
            return self.BASE_SYSTEM
        return self.BASE_SYSTEM + augmentation

    def _retrieve_episodes(self, query: str, task_type: str, top_k: int, similarity_fn: SimilarityFn | None = None) -> list[dict]:
        episodes = self._bank.data.get("successful_episodes", [])
        q = _tokens(f"{task_type} {query}")
        scored = []
        for episode in episodes:
            text = " ".join([
                str(episode.get("task_pattern", "")),
                str(episode.get("task_type", "")),
                " ".join(episode.get("tool_sequence", [])),
                " ".join(episode.get("steps", [])),
            ])
            overlap = len(q & _tokens(text))
            type_bonus = 4.0 if episode.get("task_type") == task_type else 0.0
            semantic = 0.0
            if similarity_fn is not None:
                try:
                    semantic = float(similarity_fn(query, text))
                except Exception:
                    semantic = 0.0
            scored.append((type_bonus + overlap * 2.0 + semantic * 10.0, episode))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [episode for score, episode in scored[:top_k] if score > 0 or len(scored) == 1]
