"""LLM-based ExpeL distiller for extracting reusable principles from trajectories."""
from __future__ import annotations

import logging
import re

from ..shared.llm_client import EvolutionLLMClient
from ..shared.trajectory import Trajectory
from .principle_bank import PrincipleBank

logger = logging.getLogger(__name__)

_SUCCESS_LIMIT = 48
_FAILURE_LIMIT = 48


class ExpeLDistiller:
    """Distill trajectories into ExpeL-style principles using LLM.

    Design goal: higher fidelity to ExpeL's "experience -> principle" paradigm,
    while adapting to Terrabox data and tool slugs.
    """

    def __init__(self, bank: PrincipleBank, llm_client: EvolutionLLMClient):
        self._bank = bank
        self._llm = llm_client

    def distill_batch(self, trajectories: list[Trajectory], limit: int | None = None) -> dict:
        successes = [t for t in trajectories if t.tools_called and t.success]
        failures = [t for t in trajectories if t.tools_called and not t.success]

        if limit is not None:
            successes = successes[:limit]
            failures = failures[:limit]
        else:
            successes = successes[:_SUCCESS_LIMIT]
            failures = failures[:_FAILURE_LIMIT]

        created = {"general": 0, "task": 0, "mistakes": 0}

        for traj in successes:
            try:
                general, task_specific = self._extract_from_success(traj)
                for p in general:
                    self._bank.add_general(p, score=1.0, source_task=traj.task_id)
                    created["general"] += 1
                for p in task_specific:
                    self._bank.add_task_specific(traj.task_type or "unknown", p, score=1.0, source_task=traj.task_id)
                    created["task"] += 1
            except Exception as exc:
                logger.warning("ExpeL success distillation failed for %s: %s", traj.task_id, exc)

        for traj in failures:
            try:
                mistakes = self._extract_from_failure(traj)
                for p in mistakes:
                    self._bank.add_mistake(p, score=1.0, source_task=traj.task_id)
                    created["mistakes"] += 1
            except Exception as exc:
                logger.warning("ExpeL failure distillation failed for %s: %s", traj.task_id, exc)

        self._bank.merge_duplicates()
        self._bank.save()
        return created

    def _extract_from_success(self, traj: Trajectory) -> tuple[list[str], list[str]]:
        system = (
            "You are extracting reusable tool-calling principles for an autonomous geospatial agent. "
            "Focus on sequencing, preconditions, and argument discipline."
        )
        prompt = f"""Given a successful trajectory, extract principles.

Task type: {traj.task_type}
Question: {traj.question}
Tools used (ordered): {' -> '.join(traj.tools_called)}
Final answer: {traj.final_answer[:400]}

Return strict JSON:
{{
  "general_principles": ["...", "..."],
  "task_specific_principles": ["...", "..."]
}}

Constraints:
- Each principle is one sentence.
- Must mention at least one concrete tool slug when possible.
- Avoid copying task-specific numeric values.
"""
        data = self._llm.call_json(prompt, system=system, max_tokens=900)
        if not isinstance(data, dict):
            text = self._llm.call(prompt, system=system, max_tokens=700)
            return self._fallback_split(text)

        general = [s.strip() for s in data.get("general_principles", []) if isinstance(s, str) and s.strip()]
        task_specific = [s.strip() for s in data.get("task_specific_principles", []) if isinstance(s, str) and s.strip()]
        return general[:4], task_specific[:4]

    def _extract_from_failure(self, traj: Trajectory) -> list[str]:
        error_msgs = [t.tool_result for t in traj.turns if t.is_error and t.tool_result]
        error_msg = error_msgs[0][:400] if error_msgs else "No explicit error message available"

        system = (
            "You are extracting concise failure-avoidance principles for tool-calling agents. "
            "Prefer actionable statements that reduce repeated tool misuse."
        )
        prompt = f"""Given a failed trajectory, extract mistake-avoidance principles.

Task type: {traj.task_type}
Question: {traj.question}
Tools used (ordered): {' -> '.join(traj.tools_called)}
Observed failure: {error_msg}

Return strict JSON:
{{
  "mistake_principles": ["...", "..."]
}}

Constraints:
- One sentence each.
- Include what to avoid and a safer alternative when possible.
- Mention concrete tool slugs when relevant.
"""
        data = self._llm.call_json(prompt, system=system, max_tokens=700)
        if isinstance(data, dict):
            vals = data.get("mistake_principles", [])
            return [s.strip() for s in vals if isinstance(s, str) and s.strip()][:4]

        text = self._llm.call(prompt, system=system, max_tokens=500)
        _, mistakes = self._fallback_split(text)
        return mistakes[:4]

    @staticmethod
    def _fallback_split(text: str) -> tuple[list[str], list[str]]:
        lines = [re.sub(r"^[-*\d.\s]+", "", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            return [], []
        mid = max(1, len(lines) // 2)
        return lines[:mid], lines[mid:]
