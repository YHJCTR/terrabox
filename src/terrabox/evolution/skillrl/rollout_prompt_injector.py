"""Prompt injection for the strict rollout-only SkillRL adaptation."""
from __future__ import annotations

from ..shared.prompt_builder import PromptAugmenter
from .rollout_semantic_retriever import SkillRLRolloutEmbeddingIndex
from .skill_bank import HierarchicalSkillBank


class RolloutSkillRLPromptInjector(PromptAugmenter):
    """Inject compact hierarchical skills retrieved from actual train rollouts."""

    def __init__(self, store_dir: str, top_k: int = 4):
        self.bank = HierarchicalSkillBank(store_dir)
        self.index = SkillRLRolloutEmbeddingIndex(store_dir, required=True)
        self.top_k = top_k

    def _retrieve(self, tier: str, skills: list[dict], query: str, limit: int) -> list[dict]:
        scored: list[tuple[float, dict]] = []
        available = set(query.lower().split())
        for skill in skills:
            content = str(skill.get("content") or "")
            if not content:
                continue
            semantic = self.index.similarity(query, tier, skill)
            tags = {str(item).lower() for item in skill.get("tags") or []}
            lexical = len(available.intersection(tags))
            scored.append((semantic * 10.0 + lexical * 0.25, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:limit]]

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        query = f"{task_type}\n{user_query}"
        general = self._retrieve("general", self.bank.general.load_all(), query, max(1, self.top_k // 2))
        specific = self._retrieve("specific", self.bank.specific.load_all(), query, self.top_k)
        mistakes = self._retrieve("mistakes", self.bank.mistakes.load_all(), query, max(1, self.top_k // 2))
        if not (general or specific or mistakes):
            return self.BASE_SYSTEM
        lines = [
            "\n\n## SkillRL-Style Retrieved Skills",
            "These are compact strategies distilled from imperfect train rollouts. Use them as conditional guidance, verify every step against the current tool schema and observations, and do not copy any historical facts or artifacts.",
        ]
        for heading, skills in (("General strategies", general), ("Relevant task skills", specific), ("Failure patterns to avoid", mistakes)):
            if not skills:
                continue
            lines.append(f"\n{heading}:")
            for skill in skills:
                lines.append(f"- {skill.get('content', '')}")
        return self.BASE_SYSTEM + "\n".join(lines)
