"""Method-neutral rewards for true-tool Terrabox agentic RL."""
from __future__ import annotations

import json
from typing import Any


def status_from_observation(text: str) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    if not raw:
        return "empty"
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            structured_status = str(payload.get("status") or "").lower()
            if structured_status in {"error", "failed", "failure", "timeout"}:
                return "error"
            if structured_status in {"empty", "none"}:
                return "empty"
    except (TypeError, ValueError):
        pass
    error_markers = (
        "tool execution error",
        "tool execution timeout",
        "traceback",
        "error:",
        "exception",
        "missing required",
        "invalid argument",
        "file not found",
        "no such file",
    )
    if any(marker in lowered for marker in error_markers):
        return "error"
    return "success"


def step_reward(status: str, text: str) -> float:
    if status == "success":
        reward = 0.35
        lowered = str(text or "").lower()
        artifact_markers = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gpkg", "artifact", "output_path")
        if any(marker in lowered for marker in artifact_markers):
            reward += 0.15
        return min(0.6, reward)
    if status == "empty":
        return -0.2
    return -0.6


def episode_reward(tool_calls: list[dict[str, Any]], num_turns: int, response_tokens: int) -> tuple[float, dict[str, Any]]:
    total_calls = len(tool_calls)
    success_calls = sum(1 for call in tool_calls if call.get("status") == "success")
    error_calls = sum(1 for call in tool_calls if call.get("status") == "error")
    empty_calls = sum(1 for call in tool_calls if call.get("status") == "empty")
    artifact_calls = sum(
        1
        for call in tool_calls
        if any(
            marker in str(call.get("observation_preview") or "").lower()
            for marker in (".png", ".jpg", ".tif", ".gpkg", "artifact")
        )
    )

    if total_calls == 0:
        # In OEA, tasks are tool-use tasks.  A no-tool final answer is normally
        # an early-stop failure, and in GRPO it must not become an attractive
        # local optimum compared with attempting tools.  This remains label-free:
        # it uses only observable behavior, not expected/gold tools.
        reward = -1.0
    else:
        success_rate = success_calls / total_calls
        reward = -0.15 + 0.95 * success_rate - 0.45 * error_calls - 0.20 * empty_calls
        reward += min(0.20, 0.05 * artifact_calls)
        if total_calls > 8:
            reward -= min(0.30, 0.04 * (total_calls - 8))
        if num_turns > 14:
            reward -= 0.10
        if response_tokens > 0:
            reward -= min(0.10, response_tokens / 20000.0)
    reward = max(-1.0, min(1.0, float(reward)))
    breakdown = {
        "score": reward,
        "tool_calls": total_calls,
        "successful_tool_calls": success_calls,
        "error_tool_calls": error_calls,
        "empty_tool_calls": empty_calls,
        "artifact_calls": artifact_calls,
        "num_turns": num_turns,
        "response_tokens": response_tokens,
    }
    return reward, breakdown
