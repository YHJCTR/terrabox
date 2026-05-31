"""Format Terrabox SFT and rollout records as MemRL-readable trajectories."""
from __future__ import annotations

import json
from typing import Any

from ..full_shared.sft_schema import FullSFTSample


def _shorten(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def format_sft_trajectory(sample: FullSFTSample, *, max_observation_chars: int = 2000) -> str:
    """Turn a strict SFT sample into the trajectory text MemRL expects."""
    parts = [
        "TASK:",
        sample.question,
        "",
        "GOLD TOOL CALLS:",
    ]
    for idx, call in enumerate(sample.gold_tool_calls, start=1):
        parts.extend(
            [
                f"Step {idx} Tool:",
                str(call.get("tool", "")),
                "Arguments:",
                _shorten(call.get("arguments", {}), max_observation_chars),
                "",
            ]
        )

    parts.append("MESSAGES:")
    for idx, message in enumerate(sample.messages, start=1):
        role = message.get("role", "")
        if role == "system":
            continue
        parts.extend(
            [
                f"Message {idx} [{role}]:",
                _shorten(message.get("content", ""), max_observation_chars),
                "",
            ]
        )

    parts.extend(
        [
            "FINAL ANSWER:",
            sample.ground_truth or "",
            "",
            "OUTCOME:",
            "success=True, reward=1.0",
        ]
    )
    return "\n".join(parts)


def format_real_trajectory(row: dict[str, Any], *, max_observation_chars: int = 2000) -> str:
    """Turn a saved Terrabox rollout row into a MemRL trajectory string."""
    metrics = row.get("metrics") or {}
    success = bool(row.get("real_success", row.get("success", False)))
    parts = [
        "TASK:",
        str(row.get("question") or row.get("query") or ""),
        "",
        "EXPECTED TOOLS:",
        " -> ".join(str(t) for t in row.get("expected_tools", []) or []),
        "",
        "TOOLS CALLED:",
        " -> ".join(str(t) for t in (row.get("tool_sequence") or row.get("tools_called") or [])),
        "",
        "EXECUTION:",
    ]
    history = row.get("conversation_history") or []
    if history:
        for idx, message in enumerate(history, start=1):
            role = message.get("role", "")
            content = message.get("content", message)
            parts.extend(
                [
                    f"Turn {idx} [{role}]:",
                    _shorten(content, max_observation_chars),
                    "",
                ]
            )
    else:
        parts.extend(
            [
                "No conversation history was saved for this rollout.",
                f"status={row.get('status', 'unknown')}",
                f"error={row.get('error', '')}",
                "",
            ]
        )

    parts.extend(
        [
            "FINAL ANSWER:",
            str(row.get("final_answer") or row.get("final") or ""),
            "",
            "OUTCOME:",
            (
                f"success={success}, status={row.get('status', 'unknown')}, "
                f"f1={metrics.get('f1', row.get('f1', 0.0))}"
            ),
        ]
    )
    return "\n".join(parts)

