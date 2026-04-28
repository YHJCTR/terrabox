"""CausalPolicyEvo: causal policy-state evolution for tool-calling agents.

Moves beyond pure skill/memory retrieval by maintaining a lightweight policy
state that captures:
  - tool priors      (which tools should start a task)
  - transition priors (what tool should likely follow next)
  - stop guidance    (when further tool calls are likely unnecessary)
  - recovery guidance (what to try when the current plan is weak)

The initial state is bootstrapped from trajectory statistics (CCA + CTFM +
sequential pattern mining), then optionally refined with LLM-generated policy
updates from failed cases.
"""

from .policy_state import PolicyState
from .prompt_injector import CausalPolicyEvoPromptInjector

__all__ = ["PolicyState", "CausalPolicyEvoPromptInjector"]
