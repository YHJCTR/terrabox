"""Terrabox online agent loop for veRL GRPO.

This loop subclasses veRL's generic ``ToolAgentLoop`` and adds two Terrabox-
specific pieces that the stock loop does not provide:

1. episode-level reward derived from real tool execution observations;
2. JSONL traces suitable for later plotting and audit.
"""
from __future__ import annotations

import os
import time
from typing import Any

from terrabox.evolution.agent_rl.observability import append_jsonl, trim_process_memory
from terrabox.evolution.agent_rl.rewards import episode_reward
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


@register("terrabox_tool_agent")
class TerraboxToolAgentLoop(ToolAgentLoop):
    """veRL tool-agent loop with Terrabox episode reward and traces."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        started = time.time()
        output = await super().run(sampling_params, **kwargs)
        tool_calls = list(output.extra_fields.get("terrabox_tool_calls") or [])
        reward, breakdown = episode_reward(tool_calls, output.num_turns, len(output.response_ids))
        output.reward_score = reward
        output.extra_fields["reward_extra_info"] = breakdown
        output.extra_fields["terrabox_online_rl"] = {
            "elapsed_sec": round(time.time() - started, 3),
            "sample_index": (kwargs.get("extra_info") or {}).get("sample_index"),
            "strict_nolabel": (kwargs.get("extra_info") or {}).get("strict_nolabel"),
        }
        trace_path = os.environ.get("TERRABOX_ONLINE_RL_TRACE_PATH")
        append_jsonl(
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
        # Tool observations can contain large JSON/artifact metadata.  The
        # loop process is long-lived, so explicitly collect after each episode
        # and trim libc arenas when available; this does not alter the output.
        trim_process_memory()
        return output
