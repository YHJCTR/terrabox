"""CriticRetriever: BM25-based retrieval over CriticSkillBank."""
from __future__ import annotations

import logging

from .bank import CriticSkillBank

logger = logging.getLogger(__name__)


class CriticRetriever:
    """Retrieve relevant critic skills for a query using BM25 or keyword overlap.

    Two-stage: (1) filter by task_type if available, (2) BM25 score on content.
    """

    def __init__(self, bank: CriticSkillBank):
        self._bank = bank
        self._bm25_ok = self._check_bm25()

    def retrieve(self, query: str, task_type: str = "unknown", top_k: int = 3) -> list[str]:
        """Return top-k critic skill content strings for the given query."""
        skills = self._bank.load_all()
        if not skills:
            return []

        # Filter by task_type first if we have a match
        if task_type and task_type != "unknown":
            typed = [s for s in skills if s.get("task_type") == task_type]
            if typed:
                skills = typed

        if self._bm25_ok:
            return self._bm25(skills, query, top_k)
        return self._keyword(skills, query, top_k)

    def _bm25(self, skills: list[dict], query: str, top_k: int) -> list[str]:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [s.get("content", "") for s in skills]
            tokenized = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(zip(scores, skills), key=lambda x: x[0], reverse=True)
            return [s.get("content", "") for _, s in ranked[:top_k] if s.get("content")]
        except Exception as e:
            logger.debug(f"BM25 error: {e}, falling back to keyword")
            return self._keyword(skills, query, top_k)

    def _keyword(self, skills: list[dict], query: str, top_k: int) -> list[str]:
        q_words = set(query.lower().split())
        scored = sorted(
            skills,
            key=lambda s: len(q_words & set(s.get("content", "").lower().split())),
            reverse=True,
        )
        return [s.get("content", "") for s in scored[:top_k] if s.get("content")]

    @staticmethod
    def _check_bm25() -> bool:
        try:
            import rank_bm25  # noqa: F401
            return True
        except ImportError:
            return False
