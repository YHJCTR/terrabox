"""BellmanUpdater: non-parametric RL updates on episodic memory Q-values.

From MemRL (arXiv 2601.03192): instead of updating model weights, we apply
Bellman-style temporal difference updates directly to the utility (Q-value)
of episodic memory entries. This enables runtime self-evolution without
any gradient computation or weight modification.
"""
from __future__ import annotations


class BellmanUpdater:
    """Compute updated utility for a memory entry after observing a reward.

    Update rule (simplified Bellman without successor transitions):
        U(m) ← U(m) + α * (r - U(m))
    where r = observed reward from using memory m in the current episode.

    The learning rate α decays with visits to stabilize over time:
        α = max(ALPHA_MIN, ALPHA_0 / (1 + decay_rate * visits))

    This mirrors the paper's approach of non-parametric RL on episodic
    memory: the LLM backbone is frozen, and only the utility scores
    evolve in response to environmental feedback.
    """

    GAMMA = 0.9           # discount (not used in simplified single-step form)
    ALPHA_0 = 0.3         # initial learning rate
    ALPHA_MIN = 0.05      # minimum learning rate
    DECAY_RATE = 0.1      # learning rate decay per visit

    def update(self, memory_entry: dict, observed_reward: float) -> float:
        """Compute new utility value.

        Args:
            memory_entry: dict with "utility" and "q_visits" fields.
            observed_reward: float in [0, 1] — the reward observed when
                             this memory was used in the current episode.

        Returns:
            The updated utility value (float).
        """
        visits = memory_entry.get("q_visits", 0)
        current_u = memory_entry.get("utility", 0.5)
        alpha = max(self.ALPHA_MIN, self.ALPHA_0 / (1 + self.DECAY_RATE * visits))
        new_u = current_u + alpha * (observed_reward - current_u)
        return float(max(0.0, min(1.0, new_u)))

    def batch_update(
        self,
        memory_entries: list[dict],
        observed_reward: float,
    ) -> list[float]:
        """Update all retrieved memory entries with the same episode reward."""
        return [self.update(m, observed_reward) for m in memory_entries]
