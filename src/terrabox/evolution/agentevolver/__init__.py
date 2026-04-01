"""AgentEvolver: Towards Efficient Self-Evolving Agent System.

Paper: arXiv 2511.10395

Three synergistic self-evolution mechanisms:
1. Self-Questioning (TaskGenerator): mine templates, generate synthetic tasks
2. Self-Navigating (ExperiencePool + HybridPolicy): experience-guided exploration
3. Self-Attributing (ADCAGRPOAttributor): per-step credit assignment

No weight updates — all improvements via prompt augmentation.
"""
from .self_attributing import ADCAGRPOAttributor
from .self_navigating import ExperiencePool, HybridPolicy
from .self_questioning import TaskGenerator
from .trainer import AgentEvolverTrainer
from .prompt_injector import AgentEvolverPromptInjector

__all__ = [
    "ADCAGRPOAttributor",
    "ExperiencePool",
    "HybridPolicy",
    "TaskGenerator",
    "AgentEvolverTrainer",
    "AgentEvolverPromptInjector",
]
