"""Shared infrastructure for Terrabox agentic RL experiments.

This package is intentionally method-neutral.  Concrete methods such as pure
GRPO, ExperienceEvo+RL, or future QNR/Quse variants should import these helpers
instead of duplicating tool execution, rewards, and trace handling.
"""

from .env import TerraboxAgentEnv, TerraboxEnvConfig
from .rewards import episode_reward, status_from_observation, step_reward

__all__ = [
    "TerraboxAgentEnv",
    "TerraboxEnvConfig",
    "episode_reward",
    "status_from_observation",
    "step_reward",
]
