"""EvoSkill: Automated Skill Discovery for Multi-Agent Systems.

Paper: arXiv 2603.02766

Three-agent architecture for failure-driven skill evolution:
- BaseAgent: executes tasks, generates trajectories
- ProposerAgent: analyzes failures, identifies needed capabilities
- SkillBuilderAgent: generates structured SkillModule from proposals
- ParetoManager: maintains Pareto-optimal skill frontier
- CrossDomainTransfer: transfers skills across geospatial domains
"""
from .agents import BaseAgent, ProposerAgent, SkillBuilderAgent
from .pareto_manager import ParetoManager
from .prompt_injector import EvoSkillPromptInjector
from .skill_module import SkillModule
from .transfer_engine import CrossDomainTransfer

__all__ = [
    "BaseAgent",
    "ProposerAgent",
    "SkillBuilderAgent",
    "ParetoManager",
    "EvoSkillPromptInjector",
    "SkillModule",
    "CrossDomainTransfer",
]
