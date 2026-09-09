"""Terrabox online agent loop for veRL GRPO.

This loop subclasses veRL's generic ``ToolAgentLoop`` and adds two Terrabox-
specific pieces that the stock loop does not provide:

1. episode-level reward derived from real tool execution observations;
2. JSONL traces suitable for later plotting and audit.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


def _append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_reward(tool_calls: list[dict[str, Any]], num_turns: int, response_tokens: int) -> tuple[float, dict[str, Any]]:
    total_calls = len(tool_calls)
    success_calls = sum(1 for call in tool_calls if call.get("status") == "success")
    error_calls = sum(1 for call in tool_calls if call.get("status") == "error")
    empty_calls = sum(1 for call in tool_calls if call.get("status") == "empty")
    artifact_calls = sum(
        1
        for call in tool_calls
        if any(marker in str(call.get("observation_preview") or "").lower() for marker in (".png", ".jpg", ".tif", ".gpkg", "artifact"))
    )

    if total_calls == 0:
        reward = -0.8
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


@register("terrabox_tool_agent")
class TerraboxToolAgentLoop(ToolAgentLoop):
    """veRL tool-agent loop with Terrabox episode reward and traces."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.time()
        output = await super().run(sampling_params, **kwargs)
        tool_calls = list(output.extra_fields.get("terrabox_tool_calls") or [])
        reward, breakdown = _episode_reward(tool_calls, output.num_turns, len(output.response_ids))
        output.reward_score = reward
        output.extra_fields["reward_extra_info"] = breakdown
        output.extra_fields["terrabox_online_rl"] = {
            "elapsed_sec": round(time.time() - started, 3),
            "sample_index": (kwargs.get("extra_info") or {}).get("sample_index"),
            "strict_nolabel": (kwargs.get("extra_info") or {}).get("strict_nolabel"),
        }
        trace_path = os.environ.get("TERRABOX_ONLINE_RL_TRACE_PATH")
        _append_jsonl(
            trace_path,
            {
                "sample_index": (kwargs.get("extra_info") or {}).get("sample_index"),
                "reward": reward,
                "reward_breakdown": breakdown,
                "tool_calls": tool_calls,
                "num_turns": output.num_turns,
                "response_tokens": len(output.response_ids),
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        return output
