"""RewardEvo: Self-Bootstrapping Reward Model via LLM-as-Judge.

Novel self-evolution approach that addresses the "Evaluation Supervision Bottleneck":
all existing methods (MemRL, SkillRL, CausalEvo) require ground-truth expected_tools
or human-labeled rewards. RewardEvo uses LLM to judge trajectory quality without labels.

Core components:
  1. LLMJudge: Compare trajectory pairs using geospatial reasoning
  2. SelfConsistentLabeler: Pseudo-label trajectories by self-consistent voting
  3. RewardEvoPromptInjector: Inject LLM-judged examples into prompts

Paper idea: "RewardEvo: Self-Bootstrapping Reward Estimation for Annotation-Free Agent Self-Evolution"
"""
from .llm_judge import LLMJudge
from .prompt_injector import RewardEvoPromptInjector
from .self_consistent_labeler import SelfConsistentLabeler

__all__ = ["LLMJudge", "SelfConsistentLabeler", "RewardEvoPromptInjector"]
