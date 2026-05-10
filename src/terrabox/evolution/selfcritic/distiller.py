"""SelfCriticDistiller: generate counterfactual optimal chains from successful trajectories.

For each successful trajectory (F1 ≥ threshold), ask the LLM:
  "This chain succeeded. Were any steps redundant? What is the optimal chain?"

The LLM-suggested optimal chain is stored as a critic skill.
This is complementary to SkillRL's failure lessons:
  failure → "what to avoid"
  success → "what was done"  (SkillRL general tier)
  success → "how to do it better"  (this module, new)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..shared.llm_client import EvolutionLLMClient
from ..shared.trajectory import EpisodeResult
from .bank import CriticSkillBank

logger = logging.getLogger(__name__)

_MIN_CHAIN_LENGTH = 3   # chains shorter than this aren't worth critiquing
_DEFAULT_MIN_F1 = 0.8   # only critique high-quality successes; set 0.0 for SFT data

_SYSTEM = (
    "You are an expert geospatial analyst reviewing tool chains for efficiency. "
    "Be concise and respond only in valid JSON."
)

_PROMPT = """Review this successful geospatial agent tool chain for optimality.

Task: {question}
Task type: {task_type}
Tool chain executed ({n_steps} steps): {chain}

Identify redundant or mergeable steps, then propose the most efficient chain.
Respond ONLY in valid JSON (no markdown, no text outside the JSON):
{{
  "redundant_steps": ["tool_slug"],
  "optimal_chain": ["tool_a", "tool_b"],
  "critique": "one or two sentences explaining the improvement"
}}

Rules:
- redundant_steps must be a subset of the original chain (exact slug match)
- optimal_chain must achieve the same goal with fewer or better-ordered steps
- If already optimal, set redundant_steps=[] and optimal_chain=same as input
- Only use tool slugs present in the original chain"""


class SelfCriticDistiller:
    """Critique successful trajectories and extract optimal chains."""

    def __init__(
        self,
        bank: CriticSkillBank,
        llm_client: EvolutionLLMClient,
        min_f1: float = _DEFAULT_MIN_F1,
    ):
        self._bank = bank
        self._llm = llm_client
        self._min_f1 = min_f1

    def distill_batch(self, episodes: list[EpisodeResult]) -> int:
        """Critique all eligible episodes; return count of new skills created."""
        eligible = [
            ep for ep in episodes
            if ep.tool_f1 >= self._min_f1
            and len(ep.trajectory.tools_called) >= _MIN_CHAIN_LENGTH
        ]
        logger.info(
            f"[SelfCritic] {len(eligible)}/{len(episodes)} episodes eligible "
            f"(F1≥{self._min_f1}, chain≥{_MIN_CHAIN_LENGTH})"
        )
        created = 0
        for episode in eligible:
            try:
                n = self._critique_one(episode)
                created += n
            except Exception as e:
                logger.warning(
                    f"[SelfCritic] Failed for {episode.trajectory.task_id}: {e}"
                )
        logger.info(f"[SelfCritic] Critic skills created: {created}")
        return created

    def _critique_one(self, episode: EpisodeResult) -> int:
        traj = episode.trajectory
        prompt = _PROMPT.format(
            question=traj.question[:400],
            task_type=traj.task_type,
            n_steps=len(traj.tools_called),
            chain=" → ".join(traj.tools_called),
        )
        raw = self._llm.call(
            prompt,
            system=_SYSTEM,
            max_tokens=400,
            enable_thinking=False,  # avoid Qwen3 thinking truncation on short outputs
        )
        parsed = _parse_json(raw)
        if not parsed:
            logger.debug(f"[SelfCritic] JSON parse failed: {raw[:120]}")
            return 0

        optimal = parsed.get("optimal_chain", [])
        redundant = parsed.get("redundant_steps", [])
        critique = parsed.get("critique", "").strip()

        if not optimal or not critique:
            return 0
        # Already optimal → skip
        if optimal == traj.tools_called and not redundant:
            return 0
        # Validate: redundant steps must be in original chain
        original_set = set(traj.tools_called)
        redundant = [t for t in redundant if t in original_set]

        skill_id = self._bank.add(
            task_type=traj.task_type,
            original_chain=traj.tools_called,
            optimal_chain=optimal,
            critique=critique,
            redundant_steps=redundant,
            source_tasks=[traj.task_id],
        )
        return 1 if skill_id else 0


def _parse_json(raw: str) -> Optional[dict]:
    """Robustly parse JSON from LLM output, tolerating markdown fences."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return data if "optimal_chain" in data else None
    except json.JSONDecodeError:
        return None
