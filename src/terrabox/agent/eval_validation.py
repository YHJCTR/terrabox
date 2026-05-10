"""Automatic validation for real agent rollout trajectories.

This module validates a completed token-eval run without treating gold tool
chains as the success criterion.  Gold/expected tools are recorded as a weak
reference signal; the final success label is based on execution health plus a
judge assessment of whether the final answer is supported by tool evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


SUCCESS_LABELS = {"success"}
FAILURE_LABELS = {"agent_failed", "tool_failed", "source_gold_invalid", "needs_review"}


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def canonical_tool_match(called: list[str], expected: list[str]) -> dict[str, Any]:
    called_set = set(called)
    expected_set = set(expected)
    if not called_set and not expected_set:
        precision = recall = f1 = 1.0
    else:
        overlap = called_set & expected_set
        precision = len(overlap) / len(called_set) if called_set else 0.0
        recall = len(overlap) / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_tool_precision": round(precision, 6),
        "gold_tool_recall": round(recall, 6),
        "gold_tool_f1": round(f1, 6),
        "used_as_success_criterion": False,
    }


def infer_execution_status(result: dict[str, Any]) -> str:
    final = str(result.get("final") or "")
    final_clean = strip_think(final)
    status = str(result.get("status") or "")
    trace = result.get("tool_trace", []) or []
    has_tool_error = bool(result.get("has_tool_error")) or any(
        isinstance(step, dict) and step.get("status") == "error"
        for step in trace
    )
    if final.startswith("ERROR:"):
        if "timeout" in final.lower():
            return "timeout"
        if "max" in final.lower() and "step" in final.lower():
            return "max_steps"
        return "failed"
    if not final_clean:
        return "empty_final"
    if status in {"completed", "completed_with_recovery"}:
        return "completed_with_recovery" if has_tool_error else "completed"
    if has_tool_error:
        return "tool_error"
    if status in {"incomplete", "empty_final", "failed"}:
        return status
    return "completed"


def _compact_observation(step: dict[str, Any], *, max_chars: int) -> str:
    text = step.get("observation")
    if text is None and step.get("observation_path"):
        try:
            text = Path(step["observation_path"]).read_text(encoding="utf-8")
        except OSError:
            text = None
    if text is None:
        text = step.get("observation_preview", "")
    text = str(text)
    if len(text) > max_chars:
        return text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"
    return text


def compact_tool_evidence(result: dict[str, Any], *, max_steps: int, max_observation_chars: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for step in (result.get("tool_trace", []) or [])[-max_steps:]:
        if not isinstance(step, dict):
            continue
        evidence.append({
            "step": step.get("step"),
            "tool": step.get("tool"),
            "args": step.get("args", {}),
            "status": step.get("status"),
            "observation": _compact_observation(step, max_chars=max_observation_chars),
        })
    return evidence


def parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = strip_think(text)
    if cleaned.startswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines() if not line.startswith("```")).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def normalize_answer_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"correct", "incorrect", "unsupported", "partial", "uncertain"}:
        return status
    return "uncertain"


def derive_trajectory_label(execution_status: str, answer_status: str, has_tool_error: bool) -> str:
    if answer_status == "correct" and execution_status in {"completed", "completed_with_recovery", "tool_error"}:
        return "success"
    if answer_status == "unsupported":
        return "agent_failed" if not has_tool_error else "tool_failed"
    if execution_status in {"timeout", "tool_error"}:
        return "tool_failed"
    if execution_status in {"empty_final", "max_steps", "incomplete", "failed"}:
        return "agent_failed"
    if answer_status in {"incorrect", "partial"}:
        return "agent_failed"
    return "needs_review"


def validate_result_with_llm(
    *,
    llm,
    payload: dict[str, Any],
    mode: str,
    result: dict[str, Any],
    max_evidence_steps: int = 12,
    max_observation_chars: int = 1800,
) -> dict[str, Any]:
    """Return a structured validation dict for one mode result."""
    expected = payload.get("canonical_expected_tools") or payload.get("expected_tools") or []
    called = result.get("tool_calls", []) or []
    tool_plan = canonical_tool_match(called, expected)
    execution_status = infer_execution_status(result)
    has_tool_error = bool(result.get("has_tool_error"))
    evidence = compact_tool_evidence(
        result,
        max_steps=max_evidence_steps,
        max_observation_chars=max_observation_chars,
    )
    judge_input = {
        "dataset": payload.get("dataset"),
        "task_id": payload.get("task_id"),
        "mode": mode,
        "question": payload.get("question"),
        "images": payload.get("images", []),
        "final_answer": result.get("final", ""),
        "execution_status": execution_status,
        "tool_calls": called,
        "tool_evidence": evidence,
        "expected_tools_reference_only": expected,
        "tool_plan_reference_only": tool_plan,
    }
    system = (
        "You are an independent evaluator for a real tool-using agent rollout. "
        "Judge whether the final answer is supported by the concrete tool observations. "
        "Do not treat expected_tools as the answer or as a success criterion; they are only a weak reference. "
        "Return ONLY JSON with keys: answer_status, judge_confidence, judge_reason, failure_type. "
        "answer_status must be one of: correct, incorrect, unsupported, partial, uncertain. "
        "failure_type should be one of: none, wrong_tool, wrong_args, tool_runtime_error, unsupported_answer, no_answer, source_problem, uncertain."
    )
    user = (
        "Evaluate this rollout. Mark correct only when the answer directly addresses the question "
        "and its key claims are grounded in tool observations.\n\n"
        f"{json.dumps(judge_input, ensure_ascii=False)}"
    )
    raw = ""
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = str(getattr(response, "content", response))
        parsed = parse_judge_json(raw)
    except Exception as exc:
        parsed = {
            "answer_status": "uncertain",
            "judge_confidence": 0.0,
            "judge_reason": f"judge_error: {type(exc).__name__}: {exc}",
            "failure_type": "uncertain",
        }
        raw = parsed["judge_reason"]

    answer_status = normalize_answer_status(parsed.get("answer_status"))
    label = derive_trajectory_label(execution_status, answer_status, has_tool_error)
    try:
        confidence = float(parsed.get("judge_confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    failure_type = str(parsed.get("failure_type") or "uncertain").strip() or "uncertain"
    if label == "success":
        failure_type = "none"
    return {
        "execution_status": execution_status,
        "answer_status": answer_status,
        "trajectory_label": label,
        "success": label in SUCCESS_LABELS,
        "tool_plan_status": tool_plan,
        "judge_confidence": confidence,
        "judge_reason": str(parsed.get("judge_reason") or "")[:4000],
        "failure_type": failure_type,
        "judge_raw": raw[:8000],
    }
