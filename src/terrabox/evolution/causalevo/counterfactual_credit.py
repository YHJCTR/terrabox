"""Counterfactual Credit Attribution (CCA) for CausalEvo.

Two variants:
  Statistical CCA (default, fast):
    When training data has failure examples:
      CCA(tool) = P(success | tool called) - P(success | tool not called)
    When all training examples are successes (e.g. OpenEarth expert demos),
    falls back to "necessity score" based on how often removing a tool from
    the sequence would break the downstream dependency chain:
      necessity(tool) = (mean_position_score × coverage_rate)
    where coverage_rate = fraction of trajectories containing the tool.
    No LLM calls required.

  LLM CCA (accurate, slow, opt-in):
    For each step t, asks the LLM: "If this tool was skipped/replaced,
    how critically would the outcome change?"

Higher CCA score → tool is more causally necessary for task success.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from ..shared.trajectory import Trajectory

logger = logging.getLogger(__name__)


def compute_statistical_cca(trajectories: list[Trajectory]) -> dict[str, float]:
    """Statistical CCA scores for each tool across all trajectories.

    Handles two data regimes:

    Regime A — mixed success/failure (ideal):
        raw_cca(tool) = P(success | tool called) - P(success | tool not called)
        Normalized to [0, 1].

    Regime B — all-success demos (e.g. OpenEarth expert trajectories):
        The success signal carries no variance, so we use a necessity proxy:
          necessity(tool) = coverage_rate × position_score
        where:
          coverage_rate   = fraction of episodes containing the tool  [0, 1]
          position_score  = 1 - mean_normalized_position  (earlier = more critical)
        Tools appearing early in almost every trajectory score highest.
        This reflects the intuition that "setup" tools (boundary query,
        raster fetch) are prerequisites: skipping them breaks downstream steps.

    Returns:
        dict[tool_slug → cca_score]   scores in [0, 1]
    """
    all_tools: set[str] = set()
    for traj in trajectories:
        all_tools.update(traj.tools_called)

    if not all_tools:
        return {}

    # Collect per-tool stats
    present_success: dict[str, int] = {t: 0 for t in all_tools}
    present_total: dict[str, int] = {t: 0 for t in all_tools}
    absent_success: dict[str, int] = {t: 0 for t in all_tools}
    absent_total: dict[str, int] = {t: 0 for t in all_tools}
    positions: dict[str, list[float]] = {t: [] for t in all_tools}
    n_trajs = len(trajectories)

    for traj in trajectories:
        called = set(traj.tools_called)
        seq = traj.tools_called
        T = len(seq)
        win = 1 if traj.success else 0
        for tool in all_tools:
            if tool in called:
                present_success[tool] += win
                present_total[tool] += 1
                # normalized position of first occurrence
                idx = seq.index(tool)
                positions[tool].append(idx / max(T - 1, 1))
            else:
                absent_success[tool] += win
                absent_total[tool] += 1

    # Detect regime: any failures?
    total_failures = sum(1 for t in trajectories if not t.success)

    scores: dict[str, float] = {}

    if total_failures > 0:
        # Regime A: success/failure variance available
        for tool in all_tools:
            p_in = present_success[tool] / present_total[tool] if present_total[tool] else 0.5
            p_out = absent_success[tool] / absent_total[tool] if absent_total[tool] else 0.5
            raw = p_in - p_out
            scores[tool] = max(0.0, min(1.0, (raw + 1.0) / 2.0))
        logger.info(f"CCA regime A (success/failure): {total_failures} failures in corpus")
    else:
        # Regime B: all-success demos — use necessity proxy
        for tool in all_tools:
            coverage = present_total[tool] / n_trajs           # [0, 1]
            pos_list = positions[tool]
            mean_pos = sum(pos_list) / len(pos_list) if pos_list else 0.5
            position_score = 1.0 - mean_pos                    # earlier → higher
            scores[tool] = coverage * position_score
        logger.info(
            f"CCA regime B (all-success demos, necessity proxy): "
            f"{n_trajs} trajectories, {len(all_tools)} tools"
        )

    return scores


def compute_step_cca(
    trajectory: Trajectory,
    global_cca: dict[str, float],
) -> list[float]:
    """Per-step CCA scores for a single trajectory.

    Uses the global_cca table as base, with a small position bonus for
    earlier steps (boundary/setup tools tend to appear first and are critical).
    """
    tools = trajectory.tools_called
    if not tools:
        return []
    T = len(tools)
    scores = []
    for t, tool in enumerate(tools):
        base = global_cca.get(tool, 0.5)
        # Modest position bonus: first tool slightly more critical
        pos_bonus = 0.05 * (1.0 - t / max(T - 1, 1))
        scores.append(min(1.0, base + pos_bonus))
    return scores


def compute_llm_cca(
    trajectory: Trajectory,
    llm_client,
) -> list[float]:
    """LLM-based counterfactual CCA — accurate but requires one LLM call per step.

    For each step, prompts:
      "If tool X at step t was SKIPPED, how critically would the outcome change?"
    Score 1.0 = absolutely critical, 0.0 = redundant.

    Falls back to 0.5 on any parsing error.
    """
    tools = trajectory.tools_called
    if not tools:
        return []

    outcome = "SUCCESS" if trajectory.success else "FAILURE"
    seq_str = " → ".join(tools)
    scores: list[float] = []

    for t, tool in enumerate(tools):
        try:
            prompt = (
                f"Analyze a geospatial agent trajectory.\n"
                f"Task: {trajectory.question[:300]}\n"
                f"Tool sequence executed: {seq_str}\n"
                f"Final outcome: {outcome}\n\n"
                f"Counterfactual question:\n"
                f"If step {t + 1} (tool: `{tool}`) was SKIPPED or replaced "
                f"with a different tool, how much would the outcome change?\n\n"
                f"Score:\n"
                f"  1.0 = absolutely critical (skipping would very likely cause failure)\n"
                f"  0.5 = helpful but not critical\n"
                f"  0.0 = redundant (skipping would not affect outcome)\n\n"
                f'Respond ONLY as JSON: {{"score": <float 0-1>, "reason": "<brief>"}}'
            )
            result = llm_client.call_json(prompt)
            score = float(result.get("score", 0.5))
            scores.append(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.debug(f"LLM CCA failed at step {t} ({tool}): {e}")
            scores.append(0.5)

    return scores
