"""Self-Attributing: fine-grained credit assignment for AgentEvolver.

From: AgentEvolver: Towards Efficient Self-Evolving Agent System (arXiv 2511.10395)

ADCA-GRPO (Advantage-Discounted Credit Assignment with Group Relative Policy
Optimization) assigns differentiated rewards to each step of a long trajectory,
identifying which tool calls genuinely contributed to task success.

In prompt-only mode (no weight updates), this credit information is used to:
1. Annotate experience pool entries with per-step credit
2. Generate credit-aware skill hints (highlight high-credit patterns)
3. Warn about low-credit / harmful tool call patterns
"""
from __future__ import annotations

import math
from typing import Optional

from ..shared.trajectory import EpisodeResult, StepCredit, Trajectory


class ADCAGRPOAttributor:
    """Compute per-step credit for tool calls in a trajectory.

    Step reward formula (ADCA):
        r_t = R_terminal * γ^(T-t) + Δf1_t
    where:
        R_terminal = terminal episode reward (tool-match F1)
        γ          = discount factor (0.95)
        T          = total number of tool calls
        t          = step index (0-based)
        Δf1_t      = change in partial tool-match F1 after step t

    Partial F1 is computed against expected_tools at each step,
    measuring how much the current call contributed to coverage.
    """

    GAMMA = 0.95

    def attribute(
        self,
        trajectory: Trajectory,
        expected_tools: Optional[list[str]] = None,
    ) -> list[StepCredit]:
        """Compute per-step credit for each tool call.

        Args:
            trajectory: Agent trajectory with tools_called.
            expected_tools: Ground truth tool list; falls back to trajectory.expected_tools.

        Returns:
            List of StepCredit objects, one per tool call.
        """
        if expected_tools is None:
            expected_tools = trajectory.expected_tools

        tool_calls = trajectory.tools_called
        T = len(tool_calls)
        if T == 0:
            return []

        R_terminal = trajectory.success and bool(expected_tools) and \
            self._partial_f1(tool_calls, expected_tools) or 0.5

        credits: list[StepCredit] = []
        prev_f1 = 0.0

        for t, tool_slug in enumerate(tool_calls):
            tools_so_far = tool_calls[: t + 1]
            current_f1 = self._partial_f1(tools_so_far, expected_tools) if expected_tools else 0.5

            delta_f1 = current_f1 - prev_f1
            discounted_terminal = R_terminal * (self.GAMMA ** (T - 1 - t))
            step_reward = discounted_terminal + delta_f1

            cumulative = sum(
                R_terminal * (self.GAMMA ** (T - 1 - i)) + (
                    self._partial_f1(tool_calls[:i+1], expected_tools) -
                    self._partial_f1(tool_calls[:i], expected_tools)
                    if expected_tools else 0.5
                )
                for i in range(t + 1)
            )

            if step_reward > 0.3:
                contribution = "positive"
            elif step_reward < -0.1:
                contribution = "negative"
            else:
                contribution = "neutral"

            credits.append(StepCredit(
                step=t,
                tool_slug=tool_slug,
                args_summary="",
                step_reward=float(step_reward),
                cumulative_reward=float(cumulative),
                contribution=contribution,
            ))
            prev_f1 = current_f1

        return credits

    def annotate_experience(
        self,
        trajectory: Trajectory,
        credits: list[StepCredit],
    ) -> dict:
        """Build annotated experience record for storage in ExperiencePool."""
        positive_tools = [c.tool_slug for c in credits if c.contribution == "positive"]
        negative_tools = [c.tool_slug for c in credits if c.contribution == "negative"]
        avg_reward = sum(c.step_reward for c in credits) / len(credits) if credits else 0.0

        return {
            "task_id": trajectory.task_id,
            "question": trajectory.question[:200],
            "task_type": trajectory.task_type,
            "tool_sequence": trajectory.tools_called,
            "high_credit_tools": positive_tools,
            "low_credit_tools": negative_tools,
            "avg_step_reward": float(avg_reward),
            "total_steps": len(credits),
        }

    def generate_credit_hint(self, annotated_experiences: list[dict]) -> str:
        """Generate navigation hints from credit-annotated experiences."""
        if not annotated_experiences:
            return ""

        # Aggregate tool credit scores
        tool_pos: dict[str, int] = {}
        tool_neg: dict[str, int] = {}
        for exp in annotated_experiences:
            for tool in exp.get("high_credit_tools", []):
                tool_pos[tool] = tool_pos.get(tool, 0) + 1
            for tool in exp.get("low_credit_tools", []):
                tool_neg[tool] = tool_neg.get(tool, 0) + 1

        lines = []
        if tool_pos:
            top_pos = sorted(tool_pos.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("\n\n## Credit-Aware Tool Guidance")
            lines.append("High-value tools (proven effective): "
                         + ", ".join(t for t, _ in top_pos))

        if tool_neg:
            top_neg = sorted(tool_neg.items(), key=lambda x: x[1], reverse=True)[:3]
            if not lines:
                lines.append("\n\n## Credit-Aware Tool Guidance")
            lines.append("Low-value tools (often unhelpful in similar tasks): "
                         + ", ".join(t for t, _ in top_neg))

        return "\n".join(lines)

    def _partial_f1(self, tools_so_far: list[str], expected: list[str]) -> float:
        """F1 for partial tool match at intermediate step."""
        if not expected:
            return 0.5
        predicted = set(tools_so_far)
        exp_set = set(expected)
        tp = len(predicted & exp_set)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(exp_set) if exp_set else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
