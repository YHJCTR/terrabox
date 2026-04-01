"""TwoPhaseRetriever: semantic relevance → Q-value ranking for MemRL."""
from __future__ import annotations

from .episodic_memory import EpisodicMemory
from .intent_parser import IntentParser


class TwoPhaseRetriever:
    """Two-phase memory retrieval from MemRL paper (arXiv 2601.03192).

    Phase 1 — Semantic filtering:
        Score all stored memories by intent similarity to query intent.
        Keep top CANDIDATE_K candidates.

    Phase 2 — Q-value ranking:
        Among candidates, rank by learned utility (Q-value).
        Return top final_k.

    This design: (a) avoids retrieving semantically irrelevant memories,
    (b) favors memories that have proven useful in past episodes.
    The Q-values evolve via Bellman updates in bellman_updater.py.
    """

    CANDIDATE_K = 20

    def __init__(self, intent_parser: IntentParser):
        self._intent_parser = intent_parser

    def retrieve(
        self,
        memory: EpisodicMemory,
        query_intent: dict,
        final_k: int = 5,
    ) -> list[dict]:
        """Full two-phase retrieval.

        Args:
            memory: EpisodicMemory store to query.
            query_intent: Structured intent dict from IntentParser.
            final_k: Number of memories to return.

        Returns:
            List of top-final_k memory dicts, sorted by utility descending.
        """
        # Phase 1: filter by intent similarity
        candidates = memory.retrieve_candidates(query_intent, top_k=self.CANDIDATE_K)
        if not candidates:
            return []

        # Phase 2: rank by Q-value
        candidate_ids = [m["id"] for m in candidates if "id" in m]
        top = memory.get_top_by_utility(candidate_ids, top_k=final_k)
        return top
