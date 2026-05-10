"""SelfCriticPromptInjector: inject optimal chain recommendations into system prompt."""
from __future__ import annotations

from ..shared.prompt_builder import PromptAugmenter
from .bank import CriticSkillBank
from .retriever import CriticRetriever


class SelfCriticPromptInjector(PromptAugmenter):
    """Inject self-critic distilled optimal chains into the agent system prompt.

    Appends one section after the base ReAct prompt:
      ## Optimized Tool Chains (Self-Critic)
        - Task: <task_type> | Optimal: tool_a → tool_b → tool_c
          Critique: "..."

    Only injects skills where optimal_chain differs from original_chain.
    """

    def __init__(self, bank: CriticSkillBank, retriever: CriticRetriever, top_k: int = 3):
        self._bank = bank
        self._retriever = retriever
        self.top_k = top_k

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        skills = self._retriever.retrieve(user_query, task_type=task_type, top_k=self.top_k)
        if not skills:
            return self.BASE_SYSTEM

        lines = ["\n\n## Optimized Tool Chains (Self-Critic)"]
        lines.append(
            "The following are more efficient tool chains identified by self-critique "
            "of past successful trajectories:\n"
        )
        for skill_text in skills:
            # Parse out the structured fields for clean formatting
            lines.append(self._format_skill(skill_text))

        return self.BASE_SYSTEM + "\n".join(lines)

    @staticmethod
    def _format_skill(content: str) -> str:
        """Reformat raw skill content into a compact injection line."""
        task_type = optimal = critique = ""
        for line in content.splitlines():
            if line.startswith("Task type:"):
                task_type = line.split(":", 1)[1].strip()
            elif line.startswith("Optimized chain:"):
                optimal = line.split(":", 1)[1].strip()
            elif line.startswith("Critique:"):
                critique = line.split(":", 1)[1].strip()
        if optimal:
            result = f"- Task: {task_type} | Optimal: {optimal}"
            if critique:
                result += f'\n  Critique: "{critique}"'
            return result
        return f"- {content[:200]}"
