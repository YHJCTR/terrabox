"""MemRLPromptInjector: inject retrieved episodic memories into system prompt."""
from __future__ import annotations

from typing import Optional

from ..shared.prompt_builder import PromptAugmenter
from .bellman_updater import BellmanUpdater
from .episodic_memory import EpisodicMemory
from .intent_parser import IntentParser
from .retriever import TwoPhaseRetriever


class MemRLPromptInjector(PromptAugmenter):
    """Augment the system prompt with top-k retrieved episodic memories.

    Injected section format:
    ## Relevant Past Experiences (Memory)
    Experience 1 (utility=0.87, task: poi_routing):
      - Used tools: osm_gis.get_area_boundary → osm_gis.add_pois_layer
      - Key insight: Always buffer area by 3000m before querying POIs

    The injector also handles online Bellman updates after each episode.
    Retrieved memory IDs are stored in _last_retrieved_ids for this purpose.
    """

    def __init__(
        self,
        memory: EpisodicMemory,
        retriever: TwoPhaseRetriever,
        intent_parser: IntentParser,
        bellman_updater: Optional[BellmanUpdater] = None,
        top_k: int = 5,
    ):
        self._memory = memory
        self._retriever = retriever
        self._intent_parser = intent_parser
        self._bellman = bellman_updater or BellmanUpdater()
        self.top_k = top_k
        self._last_retrieved_ids: list[str] = []
        self._last_retrieved_entries: list[dict] = []

    def augment(
        self,
        user_query: str,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """Return system prompt augmented with relevant episodic memories."""
        intent = self._intent_parser.parse(user_query, images)
        memories = self._retriever.retrieve(self._memory, intent, final_k=self.top_k)

        self._last_retrieved_ids = [m.get("id", "") for m in memories]
        self._last_retrieved_entries = memories

        if not memories:
            return self.BASE_SYSTEM

        return self.BASE_SYSTEM + self._format_memory_block(memories)

    def record_outcome(self, reward: float = 0.5, **kwargs) -> None:
        """Update Q-values for last retrieved memories after observing episode reward.

        Call this after each agent episode completes to enable online learning.

        Args:
            reward: Episode reward in [0, 1] (e.g. tool-match F1 score).
            **kwargs: Ignored extra kwargs (query, tools_called) for API compatibility.
        """
        for mem_entry, mem_id in zip(
            self._last_retrieved_entries, self._last_retrieved_ids
        ):
            if mem_id:
                new_utility = self._bellman.update(mem_entry, reward)
                self._memory.update_utility(mem_id, new_utility)
