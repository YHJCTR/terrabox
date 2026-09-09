"""Format Terrabox SFT and rollout records as MemRL-readable trajectories."""
from __future__ import annotations

import json
import re
from typing import Any

from ..full_shared.sft_schema import FullSFTSample


def _shorten(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _redact_visible_text(value: Any, max_chars: int) -> str:
    """Remove benchmark-only identifiers and brittle task facts from memory text."""
    text = _shorten(value, max_chars)
    text = re.sub(r"\[[^\]]*(?:image|file)s?\s*:\s*[^\]]+\]", "<artifact_reference>", text, flags=re.I)
    text = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", text)
    text = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_reference>", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<artifact_reference>", text, flags=re.I)
    text = re.sub(
        r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)+\b",
        "<named_area>",
        text,
    )
    text = re.sub(
        r"\b(?:in|near|around|at|within|from)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*\b",
        lambda match: match.group(0).rsplit(" ", 1)[0] + " <named_area>",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:lat|lon|latitude|longitude)\s*[:=]?\s*[-+]?\d+(?:\.\d+)?", "<named_area>", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


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


def format_real_trajectory(
    row: dict[str, Any],
    *,
    max_observation_chars: int = 2000,
    strict_nolabel: bool = False,
) -> str:
    """Turn a saved Terrabox rollout row into a MemRL trajectory string."""
    metrics = row.get("metrics") or {}
    success = bool(row.get("real_success", row.get("success", False)))
    task_text = str(row.get("question") or row.get("query") or "")
    if strict_nolabel:
        task_text = _redact_visible_text(task_text, max_observation_chars)
        tools = row.get("tool_sequence") or row.get("tools_called") or row.get("tool_calls") or []
        final_answer = str(
            row.get("final_answer")
            or row.get("final_answer_full")
            or row.get("final_answer_preview")
            or row.get("final")
            or ""
        ).strip()
        redacted_final = _redact_visible_text(final_answer, max_observation_chars) if final_answer else ""
        parts = [
            "TASK:",
            task_text,
            "",
            "TOOLS CALLED:",
            " -> ".join(str(t) for t in tools),
            "",
            "EXECUTION:",
        ]
        history = row.get("conversation_history") or []
        if history:
            for idx, message in enumerate(history, start=1):
                role = message.get("role", "") if isinstance(message, dict) else "unknown"
                content = message.get("content", message) if isinstance(message, dict) else message
                redacted_content = _redact_visible_text(content, max_observation_chars)
                if redacted_final and redacted_content == redacted_final:
                    continue
                parts.extend(
                    [
                        f"Turn {idx} [{role}]:",
                        redacted_content,
                        "",
                    ]
                )
        else:
            parts.extend(
                [
                    "No conversation history was saved for this rollout.",
                    f"status={row.get('status', 'unknown')}",
                    "",
                ]
            )
        parts.extend(
            [
                "OUTCOME:",
                (
                    f"completed={str(row.get('status') or '') == 'completed'}, "
                    f"has_tool_error={bool(row.get('has_tool_error'))}"
                ),
            ]
        )
        return "\n".join(parts)
    parts = [
        "TASK:",
        task_text,
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
