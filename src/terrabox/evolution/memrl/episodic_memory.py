"""EpisodicMemory: Intent-Experience-Utility (IEU) memory store for MemRL."""
from __future__ import annotations

import time
import uuid
from typing import Optional

from ..shared.storage import SQLiteMemoryStore
from ..shared.trajectory import Trajectory
from .intent_parser import IntentParser


class EpisodicMemory:
    """Intent-Experience-Utility (IEU) memory.

    Each memory entry:
    {
        id: str,
        intent: dict,           # structured intent (task_type, domain, ...)
        experience: dict,       # trajectory summary (tool_sequence, insight, ...)
        utility: float,         # Q-value; higher = more useful
        q_visits: int,
        created_at: float,
        updated_at: float,
    }

    The IEU framework (from MemRL paper) organizes memory around experiences
    that generated environmental feedback. Q-values track which strategies
    are genuinely useful over time via Bellman updates.
    """

    def __init__(self, db_path: str, llm_client=None):
        self._store = SQLiteMemoryStore(db_path)
        self._intent_parser = IntentParser()
        self._llm = llm_client   # optional; used to extract insights

    def add_memory(
        self,
        trajectory: Trajectory,
        intent: Optional[dict] = None,
        initial_utility: float = 0.5,
        skip_if_exists: bool = True,
    ) -> Optional[str]:
        """Store a new episodic memory; return memory_id, or None if skipped.

        Args:
            skip_if_exists: If True (default), skip if a memory with the same
                            task_id already exists — prevents duplicates on re-run.
        """
        if skip_if_exists and trajectory.task_id:
            if self._store.exists_by_task_id(trajectory.task_id):
                return None   # already stored; skip

        if intent is None:
            intent = self._intent_parser.parse(trajectory.question, trajectory.images)

        experience = self._summarize_trajectory(trajectory)
        memory_id = str(uuid.uuid4())

        self._store.insert({
            "id": memory_id,
            "task_id": trajectory.task_id,
            "intent": intent,
            "experience": experience,
            "utility": initial_utility,
            "q_visits": 0,
            "created_at": time.time(),
        })
        return memory_id

    def clear(self) -> None:
        """Delete all memories (for fresh re-runs)."""
        self._store.clear()

    def retrieve_candidates(self, query_intent: dict, top_k: int = 20) -> list[dict]:
        """Phase 1: return top_k candidates ranked by intent similarity."""
        all_memories = self._store.get_all(limit=5000)
        scored = []
        for mem in all_memories:
            sim = self._intent_parser.similarity(query_intent, mem.get("intent", {}))
            scored.append((sim, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def get_top_by_utility(self, candidate_ids: list[str], top_k: int = 5) -> list[dict]:
        """Phase 2: rank candidates by Q-value utility, return top_k."""
        candidates = [m for m in self._store.get_all() if m.get("id") in set(candidate_ids)]
        candidates.sort(key=lambda m: m.get("utility", 0.0), reverse=True)
        return candidates[:top_k]

    def update_utility(self, memory_id: str, new_utility: float) -> None:
        """Update Q-value for a memory entry."""
        mem = self._store.get(memory_id)
        if mem is None:
            return
        visits = mem.get("q_visits", 0) + 1
        self._store.update_q_value(memory_id, new_utility, visits)

    def count(self) -> int:
        return self._store.count()

    def _summarize_trajectory(self, trajectory: Trajectory) -> dict:
        """Build the experience dict stored for each memory."""
        # Extract key insight via LLM if available
        insight = ""
        if self._llm and trajectory.tools_called:
            try:
                insight = self._llm.extract_key_insight(
                    trajectory.question,
                    trajectory.tools_called,
                    trajectory.final_answer,
                )
            except Exception:
                pass

        return {
            "question": trajectory.question[:200],
            "tool_sequence": trajectory.tools_called,
            "trajectory_length": len(trajectory.turns),
            "final_answer_quality": 1.0 if trajectory.success else 0.0,
            "key_insights": insight,
            "task_type": trajectory.task_type,
            "source": trajectory.source,
        }
