"""Self-Navigating: experience pool and hybrid policy for AgentEvolver.

From: AgentEvolver: Towards Efficient Self-Evolving Agent System (arXiv 2511.10395)

Self-Navigating improves exploration efficiency by maintaining an experience pool
that captures cross-task insights and implements a hybrid policy combining
exploitation (similar past successes) and exploration (novel tasks).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..shared.storage import FileEpisodeStore
from ..shared.trajectory import EpisodeResult

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.3  # BM25 score below = novel task → explore


class ExperiencePool:
    """Store past agent episodes with per-task-type indexing.

    Each stored episode:
    {
        task_id: str,
        question: str,
        task_type: str,
        tool_sequence: list[str],
        f1: float,
        reward: float,
        timestamp: float,
        credit_annotated: bool,
    }

    Backed by FileEpisodeStore (append-only JSONL).
    """

    def __init__(self, store_path: str):
        self._store = FileEpisodeStore(store_path)
        self._index: dict[str, list[dict]] = {}   # task_type → episodes
        self._loaded = False

    def add_episode(self, result: EpisodeResult, skip_if_exists: bool = True) -> bool:
        """Add episode; returns False if skipped as duplicate (same task_id)."""
        traj = result.trajectory
        if skip_if_exists and traj.task_id:
            self._ensure_loaded()
            if traj.task_id in self._get_all_task_ids():
                return False
        episode = {
            "task_id": traj.task_id,
            "question": traj.question[:200],
            "task_type": traj.task_type,
            "tool_sequence": traj.tools_called,
            "f1": result.tool_f1,
            "reward": result.reward,
            "timestamp": time.time(),
            "credit_annotated": False,
        }
        self._store.append(episode)
        self._index.setdefault(traj.task_type, []).append(episode)
        return True

    def _get_all_task_ids(self) -> set[str]:
        return {ep.get("task_id", "") for eps in self._index.values() for ep in eps}

    def clear(self) -> None:
        """Clear all episodes (for fresh re-runs)."""
        self._store.clear()
        self._index.clear()
        self._loaded = False

    def get_similar(self, question: str, task_type: str, top_k: int = 5) -> list[dict]:
        """Return top-k similar past episodes by BM25/keyword match."""
        self._ensure_loaded()
        candidates = self._index.get(task_type, []) + self._index.get("unknown", [])
        if not candidates:
            return []

        try:
            from rank_bm25 import BM25Okapi
            corpus = [e.get("question", "") for e in candidates]
            tokenized = [doc.lower().split() for doc in corpus]
            # BM25Okapi crashes with ZeroDivisionError if all tokens have zero IDF
            # (e.g. all docs are empty). Guard by checking non-empty token lists.
            non_empty = [t for t in tokenized if t]
            if not non_empty:
                raise ImportError("empty corpus")
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.lower().split())
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [ep for _, ep in ranked[:top_k]]
        except ImportError:
            # Keyword overlap fallback
            q_words = set(question.lower().split())
            scored = []
            for ep in candidates:
                words = set(ep.get("question", "").lower().split())
                score = len(q_words & words)
                scored.append((score, ep))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [ep for _, ep in scored[:top_k]]

    def get_high_reward(self, task_type: str, top_k: int = 3) -> list[dict]:
        """Return top-k high-reward episodes for a task type."""
        self._ensure_loaded()
        candidates = self._index.get(task_type, [])
        if not candidates:
            # Fall back to all types
            candidates = [ep for eps in self._index.values() for ep in eps]
        return sorted(candidates, key=lambda e: e.get("reward", 0), reverse=True)[:top_k]

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            episodes = self._store.load_all()
            for ep in episodes:
                task_type = ep.get("task_type", "unknown")
                self._index.setdefault(task_type, []).append(ep)
            self._loaded = True

    def size(self) -> int:
        return self._store.count()


class HybridPolicy:
    """Exploration guidance using experience pool.

    Hybrid = exploitation + exploration:
    - Exploitation: similar past successes → recommend their tool sequence
    - Exploration:  novel tasks (low similarity) → suggest diverse tool combos

    Outputs a navigation hint string injected into the system prompt.
    """

    EXPLOIT_THRESHOLD = _SIMILARITY_THRESHOLD

    def get_navigation_hint(
        self,
        question: str,
        task_type: str,
        pool: ExperiencePool,
    ) -> str:
        """Return navigation hint for system prompt injection."""
        similar = pool.get_similar(question, task_type, top_k=5)
        high_reward = pool.get_high_reward(task_type, top_k=3)

        if similar:
            return self._exploit(similar, high_reward)
        else:
            available_tools = self._get_available_tools()
            return self._explore(question, available_tools)

    def _exploit(self, similar: list[dict], high_reward: list[dict]) -> str:
        lines = ["\n\n## Navigation Guidance (Based on Similar Past Tasks)"]

        # Best tool sequence from high-reward episodes
        if high_reward:
            best = high_reward[0]
            seq = best.get("tool_sequence", [])
            if seq:
                lines.append(f"Recommended tool sequence (reward={best.get('reward', 0):.2f}): "
                              f"{' → '.join(seq)}")

        # Aggregate common tool patterns from similar episodes
        tool_counts: dict[str, int] = {}
        for ep in similar:
            for tool in ep.get("tool_sequence", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        if tool_counts:
            top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"Frequently used tools in similar tasks: "
                         f"{', '.join(t for t, _ in top_tools)}")

        return "\n".join(lines)

    def _explore(self, question: str, available_tools: list[str]) -> str:
        lines = ["\n\n## Navigation Guidance (Novel Task — Exploratory)"]
        q_lower = question.lower()

        # Simple rule-based suggestions for novel tasks
        if "image" in q_lower or "satellite" in q_lower or "aerial" in q_lower:
            lines.append("This appears to be an image analysis task. "
                         "Start with vlm_analyze for scene understanding, "
                         "then use remotesam_segment if object detection is needed.")
        elif "boundary" in q_lower or "area" in q_lower or "poi" in q_lower:
            lines.append("This appears to be a geospatial query. "
                         "Start with osm_gis.get_area_boundary to get the region, "
                         "then use relevant analysis tools.")
        elif any(idx in q_lower for idx in ["ndvi", "ndwi", "nbr", "index"]):
            lines.append("This appears to be a raster index calculation. "
                         "Start with osm_gis.get_area_boundary for the region, "
                         "then use georaster.calculate_index.")
        else:
            lines.append("Novel task type. Consider starting with a boundary query "
                         "if a location is mentioned, or vlm_analyze if images are provided.")

        return "\n".join(lines)

    def _get_available_tools(self) -> list[str]:
        try:
            from ...core.registry import registry
            return [spec.slug for spec in registry.list_tools()]
        except Exception:
            return []
