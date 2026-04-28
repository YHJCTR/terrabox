"""RewardEvo PromptAugmenter: Integrate LLM-judged experiences into agent prompts."""
from __future__ import annotations

import logging
from typing import Optional

from ..memrl.episodic_memory import EpisodicMemory
from ..shared.llm_client import EvolutionLLMClient
from ..shared.prompt_builder import PromptAugmenter

logger = logging.getLogger(__name__)


class RewardEvoPromptInjector(PromptAugmenter):
    """Augment agent prompts with LLM-judged high-quality memory trajectories.

    Unlike standard MemRL which uses Q-values learned over time, RewardEvo uses
    LLM judgment to immediately identify and bootstrap high-quality examples.
    """

    def __init__(
        self,
        memory_db: str,
        llm_client: Optional[EvolutionLLMClient] = None,
        top_k: int = 3,
    ):
        """
        Args:
            memory_db: Path to MemRL SQLite database (populated by SelfConsistentLabeler)
            llm_client: For optional online re-judging
            top_k: Number of high-quality examples to inject
        """
        self._memory = EpisodicMemory(memory_db, llm_client=llm_client)
        self._llm = llm_client or EvolutionLLMClient()
        self._top_k = top_k

    def augment(self, user_query: str) -> str:
        """Retrieve LLM-judged high-quality examples and create system prompt."""
        # Phase 1: Intent-based retrieval
        query_intent = {"task_type": self._extract_task_type(user_query)}
        candidates = self._memory.retrieve_candidates(query_intent, top_k=20)

        if not candidates:
            logger.warning("No judged memories found; using empty prompt")
            return ""

        # Phase 2: Q-value ranking (memories with high utility come first)
        top_memories = self._memory.get_top_by_utility(
            [m["id"] for m in candidates],
            top_k=self._top_k,
        )

        # Phase 3: Format as examples with quality markers
        examples_text = self._format_examples(top_memories)

        # Phase 4: Create augmented system prompt
        system_prompt = f"""You are a geospatial AI agent. Use the following high-quality examples (LLM-verified) to guide your tool selection:

{examples_text}

When solving a similar task:
1. Identify which example is most relevant
2. Follow the tool sequence pattern, adjusting for differences
3. Prefer tool combinations that appeared in high-quality examples"""

        logger.info(f"RewardEvo augmented prompt with {len(top_memories)} examples")
        return system_prompt

    def record_outcome(
        self,
        memory_id: str,
        is_success: bool,
        new_utility: Optional[float] = None,
    ) -> None:
        """Online: update memory utility after observing agent outcome.

        Args:
            memory_id: Memory to update
            is_success: Did using this memory lead to success?
            new_utility: If provided, use this; else auto-compute
        """
        if new_utility is None:
            # Bellman-style update: if success, boost utility
            new_utility = 0.8 if is_success else 0.3

        self._memory.update_utility(memory_id, new_utility)
        logger.info(f"Updated memory {memory_id} utility to {new_utility:.2f}")

    @staticmethod
    def _extract_task_type(query: str) -> str:
        """Heuristic task_type extraction from query text."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["flood", "water", "inundation"]):
            return "flood_detection"
        if any(kw in query_lower for kw in ["building", "damage", "destroyed"]):
            return "earthquake_damage"
        if any(kw in query_lower for kw in ["fire", "burn", "wildfire"]):
            return "fire_detection"
        if any(kw in query_lower for kw in ["landslide", "slope"]):
            return "landslide"
        if any(kw in query_lower for kw in ["drought", "dry", "moisture"]):
            return "drought_monitoring"

        return "generic_geospatial"

    def _format_examples(self, memories: list[dict]) -> str:
        """Format memories as markdown examples with quality indicators."""
        examples = []

        for i, mem in enumerate(memories, 1):
            utility = mem.get("utility", 0.5)
            quality_indicator = "⭐⭐⭐" if utility >= 0.8 else "⭐⭐" if utility >= 0.5 else "⭐"

            experience = mem.get("experience", {})
            question = experience.get("question", "")
            tool_seq = experience.get("tool_sequence", [])
            insight = experience.get("insight", "")

            example_text = f"""{quality_indicator} Example {i} (LLM-judged quality: {utility:.2f}):
- Question: {question[:150]}...
- Tool sequence: {" → ".join(tool_seq[:5]) if tool_seq else "none"}
- Key insight: {insight[:100]}"""

            examples.append(example_text)

        return "\n\n".join(examples) if examples else "(no high-quality examples available)"
