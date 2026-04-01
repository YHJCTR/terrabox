"""SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning.

Paper: arXiv 2602.08234

Key components:
- HierarchicalSkillBank: three-tier skill library (general/task-specific/mistakes)
- ExperienceDistiller: trajectory → skill extraction via LLM
- SkillRetriever: BM25-based skill retrieval
- SkillEvolver: monitors F1, triggers recursive skill evolution on drift
- SkillRLPromptInjector: injects retrieved skills into system prompt
"""
from .distiller import ExperienceDistiller
from .evolver import SkillEvolver
from .prompt_injector import SkillRLPromptInjector
from .retriever import SkillRetriever
from .skill_bank import HierarchicalSkillBank

__all__ = [
    "ExperienceDistiller",
    "SkillEvolver",
    "SkillRLPromptInjector",
    "SkillRetriever",
    "HierarchicalSkillBank",
]
