"""HierarchicalSkillBank: three-tier skill storage for SkillRL.

From: SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement
      Learning (arXiv 2602.08234)

Three tiers:
  Tier 1 — General skills: universal strategies applicable across geo tasks
  Tier 2 — Task-specific skills: heuristics keyed to specific task types
  Tier 3 — Mistakes: failure patterns with avoidance strategies
"""
from __future__ import annotations

import os
import re
from typing import Optional

from ..shared.storage import JSONSkillStore


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks from skill content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class HierarchicalSkillBank:
    """Three-tier hierarchical skill library.

    Each tier backed by a separate JSONSkillStore file.

    Skill record schema:
    {
        id: str,
        content: str,           # the skill text for injection
        tags: list[str],        # keywords for retrieval
        performance_delta: float, # observed F1 improvement
        task_type: str,         # for task-specific skills
        failed_tool: str,       # for mistakes: which tool caused the failure
        created_at: float,
        use_count: int,
        source_tasks: list[str],
    }
    """

    def __init__(self, store_dir: str):
        os.makedirs(store_dir, exist_ok=True)
        self.general = JSONSkillStore(os.path.join(store_dir, "general_skills.json"))
        self.specific = JSONSkillStore(os.path.join(store_dir, "task_skills.json"))
        self.mistakes = JSONSkillStore(os.path.join(store_dir, "mistakes.json"))

    def add_general_skill(
        self,
        content: str,
        tags: list[str],
        performance_delta: float = 0.0,
        source_tasks: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save general skill; returns skill_id or None if duplicate content."""
        return self.general.save_if_unique({
            "content": content,
            "tags": tags,
            "performance_delta": performance_delta,
            "task_type": "general",
            "source_tasks": source_tasks or [],
        })

    def add_task_skill(
        self,
        task_type: str,
        content: str,
        performance_delta: float = 0.0,
        source_tasks: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save task-specific skill; returns skill_id or None if duplicate content."""
        return self.specific.save_if_unique({
            "content": content,
            "tags": [task_type],
            "performance_delta": performance_delta,
            "task_type": task_type,
            "source_tasks": source_tasks or [],
        })

    def add_mistake(
        self,
        pattern: str,
        failed_tool: str,
        lesson: str,
        source_tasks: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save mistake pattern; returns skill_id or None if duplicate content."""
        content = f"Mistake: {pattern}\nLesson: {lesson}"
        return self.mistakes.save_if_unique({
            "content": content,
            "pattern": pattern,
            "lesson": lesson,
            "failed_tool": failed_tool,
            "tags": [failed_tool],
            "performance_delta": -1.0,
            "source_tasks": source_tasks or [],
        })

    def clear(self) -> None:
        """Clear all three tiers (for fresh re-runs)."""
        self.general.clear()
        self.specific.clear()
        self.mistakes.clear()

    def retrieve(
        self,
        query: str,
        task_type: str,
        top_k: int = 3,
    ) -> dict[str, list[str]]:
        """Return top-k skill texts per tier for injection.

        Returns:
            {
                "general": [skill_text, ...],
                "specific": [skill_text, ...],
                "mistakes": [skill_text, ...],
            }
        """
        return {
            "general": self._retrieve_from(self.general, query, top_k),
            "specific": self._retrieve_from(
                self.specific, query, top_k, task_type_filter=task_type
            ),
            "mistakes": self._retrieve_from(self.mistakes, query, top_k),
        }

    def get_performance_baseline(self) -> float:
        """Average performance_delta across all skills — baseline for drift detection."""
        all_skills = self.general.load_all() + self.specific.load_all()
        if not all_skills:
            return 0.0
        deltas = [s.get("performance_delta", 0.0) for s in all_skills]
        return sum(deltas) / len(deltas)

    def counts(self) -> dict[str, int]:
        return {
            "general": self.general.count(),
            "specific": self.specific.count(),
            "mistakes": self.mistakes.count(),
        }

    def _retrieve_from(
        self,
        store: JSONSkillStore,
        query: str,
        top_k: int,
        task_type_filter: Optional[str] = None,
    ) -> list[str]:
        skills = store.load_all()
        if not skills:
            return []

        if task_type_filter and task_type_filter != "unknown":
            type_matched = [s for s in skills if s.get("task_type") == task_type_filter]
            if type_matched:
                skills = type_matched

        # Score by keyword overlap with query
        q_words = set(query.lower().split())
        scored = []
        for skill in skills:
            tags = set(t.lower() for t in skill.get("tags", []))
            content_words = set(skill.get("content", "").lower().split())
            overlap = len(q_words & (tags | content_words))
            # Also consider performance_delta as a secondary signal
            delta = skill.get("performance_delta", 0.0)
            score = overlap + max(0.0, delta)
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, s in scored[:top_k]:
            content = _strip_think_tags(s.get("content", ""))
            if content:
                results.append(content)
        return results
