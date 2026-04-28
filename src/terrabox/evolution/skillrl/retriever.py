"""SkillRetriever: BM25-based retrieval over the HierarchicalSkillBank."""
from __future__ import annotations

import logging
from typing import Optional

from .skill_bank import HierarchicalSkillBank

logger = logging.getLogger(__name__)


class SkillRetriever:
    """Retrieve relevant skills for a query using BM25 or keyword overlap.

    Uses rank_bm25 if available; falls back to keyword overlap scoring.
    Two-stage per tier: (1) filter by task_type, (2) BM25 score.
    """

    def __init__(self, bank: HierarchicalSkillBank):
        self._bank = bank
        self._bm25_available = self._check_bm25()

    def retrieve(
        self,
        query: str,
        task_type: str = "unknown",
        top_k: int = 3,
    ) -> dict[str, list[str]]:
        """Return top-k skill texts per tier.

        Returns:
            {"general": [...], "specific": [...], "mistakes": [...]}
        """
        if self._bm25_available:
            return {
                # task_type filter removed: stored labels (e.g. "ind_nbr", "type30")
                # never match inferred eval labels ("change_detection"), so filtering
                # silently empties the specific tier. Use full BM25 instead.
                "general":  self._bm25_retrieve(self._bank.general.load_all(), query, top_k),
                "specific": self._bm25_retrieve(self._bank.specific.load_all(), query, top_k),
                "mistakes": self._bm25_retrieve(self._bank.mistakes.load_all(), query, top_k),
            }
        else:
            return self._bank.retrieve(query, task_type, top_k)

    def _bm25_retrieve(self, skills: list[dict], query: str, top_k: int) -> list[str]:
        """Score skills using BM25; return top-k content strings."""
        if not skills:
            return []
        try:
            from rank_bm25 import BM25Okapi
            corpus = [s.get("content", "") for s in skills]
            tokenized = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(zip(scores, skills), key=lambda x: x[0], reverse=True)
            return [s.get("content", "") for _, s in ranked[:top_k] if s.get("content")]
        except Exception as e:
            logger.debug(f"BM25 retrieval error: {e}, falling back to keyword overlap")
            return self._keyword_retrieve(skills, query, top_k)

    def _keyword_retrieve(self, skills: list[dict], query: str, top_k: int) -> list[str]:
        q_words = set(query.lower().split())
        scored = []
        for skill in skills:
            content = skill.get("content", "")
            words = set(content.lower().split())
            score = len(q_words & words)
            scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s.get("content", "") for _, s in scored[:top_k] if s.get("content")]

    @staticmethod
    def _check_bm25() -> bool:
        try:
            import rank_bm25  # noqa: F401
            return True
        except ImportError:
            logger.info("rank_bm25 not installed; using keyword overlap retrieval")
            return False
