"""LLM-based relation extraction for SkillGraph in GraphSkillEvo.

When adding a new skill to the graph, use LLM to discover its relationships
with existing skills (precedes, enables, conflicts, etc.).
"""
from __future__ import annotations

import logging
from typing import Optional

from ..shared.llm_client import EvolutionLLMClient

logger = logging.getLogger(__name__)


class LLMRelationExtractor:
    """Extract skill-to-skill relations using LLM reasoning."""

    def __init__(self, llm_client: Optional[EvolutionLLMClient] = None):
        self._llm = llm_client or EvolutionLLMClient()

    def extract_relations(
        self,
        new_skill: str,
        existing_skills: list[tuple[str, str]],  # [(skill_id, skill_text), ...]
        min_confidence: float = 0.7,
    ) -> list[dict]:
        """Analyze new_skill and find relations to existing skills.

        Returns list of:
            {
                "type": "precedes|enables|conflicts_with|generalizes|alternative_to",
                "target_id": skill_id,
                "confidence": 0.0–1.0,
                "reasoning": "explanation",
            }
        """
        if not existing_skills:
            return []

        # Batch relations: ask LLM to compare new_skill against top-k existing
        top_skills = existing_skills[:10]  # Limit for efficiency

        system = "You are a geospatial skill relation expert."
        prompt = f"""Analyze the following new geospatial skill and determine its relationships with existing skills.

NEW SKILL:
"{new_skill}"

EXISTING SKILLS:
{self._format_skills_for_llm(top_skills)}

For each existing skill, determine if there are relationships (precedes, enables, conflicts_with, generalizes, alternative_to).

Respond in JSON array format:
[
  {{
    "target_index": 0,  # index in EXISTING SKILLS list
    "relation_type": "precedes",  # or other types
    "confidence": 0.85,
    "reasoning": "brief explanation"
  }},
  ...
]

Only include relations with confidence >= 0.7. Be conservative."""

        result = self._llm.call_json(prompt, system=system)

        if result is None:
            logger.warning("LLM relation extraction failed")
            return []

        # Parse LLM response and map indices to skill_ids
        # Result can be a list directly or a dict with "relations" key
        relations = []
        items = result if isinstance(result, list) else result.get("relations", [])
        for rel in items:
            try:
                target_idx = int(rel.get("target_index", -1))
                if 0 <= target_idx < len(top_skills):
                    target_id, _ = top_skills[target_idx]
                    confidence = float(rel.get("confidence", 0.5))

                    if confidence >= min_confidence:
                        relations.append({
                            "type": rel.get("relation_type", "alternative_to"),
                            "target_id": target_id,
                            "confidence": confidence,
                            "reasoning": rel.get("reasoning", ""),
                        })
            except (ValueError, KeyError, IndexError) as e:
                logger.debug(f"Failed to parse relation: {e}")

        logger.info(f"Extracted {len(relations)} relations for new skill")
        return relations

    def extract_conflict_warnings(
        self,
        skill: str,
        all_skills: list[str],
    ) -> list[str]:
        """Identify skills that might conflict with the given skill."""
        system = "You are a geospatial skill compatibility expert."
        prompt = f"""Does the following skill conflict with any of these existing skills?

NEW SKILL: "{skill}"

EXISTING SKILLS:
{chr(10).join(f"- {s}" for s in all_skills[:15])}

List any skills (by their text) that would conflict or produce incorrect results if used together with the new skill.
Respond as a JSON array of skill indices or texts that conflict.

Example: ["skill_3", "skill_7"]

Be conservative - only flag clear conflicts."""

        result = self._llm.call_json(prompt, system=system)

        if result is None:
            return []

        conflicts = result if isinstance(result, list) else result.get("conflicts", [])
        return [str(c) for c in conflicts[:5]]  # Limit to top 5

    @staticmethod
    def _format_skills_for_llm(skills: list[tuple[str, str]]) -> str:
        """Format skill list for LLM prompt."""
        lines = []
        for i, (skill_id, skill_text) in enumerate(skills):
            lines.append(f"{i}. [{skill_id}] {skill_text[:150]}")
        return "\n".join(lines)
