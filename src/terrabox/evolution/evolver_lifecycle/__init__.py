"""EvolveR lifecycle-style adapted baseline for Terrabox/OEA.

This module implements a strict no-label, frozen-agent adaptation of EvolveR's
experience lifecycle: rollout trajectories are distilled into guiding and
cautionary principles, then semantically retrieved and injected at inference
time.  It is not the full veRL/GRPO training stack from the official repo.
"""

from .principle_bank import EvolveRPrincipleBank, ExperiencePrinciple, RetrievedExperiencePackage
from .prompt_injector import EvolveRLifecyclePromptInjector

__all__ = [
    "EvolveRPrincipleBank",
    "ExperiencePrinciple",
    "RetrievedExperiencePackage",
    "EvolveRLifecyclePromptInjector",
]
