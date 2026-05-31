"""Prompt augmenter for SkillRL full skillbank JSON."""
from __future__ import annotations

import json
from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter


class SkillRLFullPromptInjector(PromptAugmenter):
    def __init__(self, skillbank_path: str, top_k: int = 5):
        self.skillbank_path = skillbank_path
        self.top_k = top_k

    def _load(self) -> dict:
        path = Path(self.skillbank_path)
        if not path.exists():
            return {"general_skills": [], "task_specific_skills": {}, "common_mistakes": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def augment(self, user_query: str, task_type: str | None = None, **kwargs) -> str:
        bank = self._load()
        q = set(user_query.lower().split())
        candidates = []
        for skill in bank.get("general_skills", []):
            candidates.append(("general", skill))
        specific = bank.get("task_specific_skills", {})
        if task_type:
            candidates.extend((task_type, s) for s in specific.get(f"task_type:{task_type}", []))
        for key, skills in specific.items():
            candidates.extend((key, s) for s in skills)

        scored = []
        for tier, skill in candidates:
            text = " ".join(str(skill.get(k, "")) for k in ("title", "principle", "when_to_apply"))
            overlap = len(q & set(text.lower().split()))
            scored.append((overlap + float(skill.get("support", 0)) / 1000.0, tier, skill))
        scored.sort(key=lambda item: item[0], reverse=True)

        lines = []
        for _, tier, skill in scored[: self.top_k]:
            lines.append(
                f"- [{tier}] {skill.get('title', 'Skill')}: {skill.get('principle', '')}"
            )
        if not lines:
            return ""
        return "## Relevant SkillRL Full Skills\n" + "\n".join(lines)
