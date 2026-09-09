"""Strict process reward for ExperienceEvo-guided GRPO.

The reward is intentionally static: it scores the model's proposed tool action
against public tool schemas and rollout-derived ExperienceEvo policy summaries.
It does not execute tools and does not read benchmark gold labels or expected
tool sequences.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any


ACTION_PATTERN = re.compile(
    r"Action:\s*(?P<tool>[^\n]+)\s*\nAction Input:\s*(?P<args>\{.*?\})(?=\nAction:|\Z)",
    re.DOTALL,
)
_TRACE_LOCK = threading.Lock()


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _load_policy(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        data = json.loads(ground_truth or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _balanced_json_objects(text: str) -> list[str]:
    """Return balanced top-level JSON object substrings in left-to-right order.

    Model rollouts often emit two JSON objects back-to-back, or a valid object
    followed by clipped text. A greedy ``{.*}`` extraction rejects those cases.
    This scanner extracts the first balanced object without treating braces
    inside strings as structure.
    """
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def parse_tool_calls(solution_str: str) -> list[dict[str, Any]]:
    text = _strip_think(solution_str)
    if not text:
        return []
    # Strict training target: the completion must be one complete top-level JSON
    # object. Do not salvage nested action snippets from clipped/malformed JSON,
    # otherwise the model can receive high reward for outputs the runtime cannot
    # safely execute as a single process-action decision.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    if any(key in data for key in ("tool", "tool_name", "name")):
        data = {"actions": [data]}
    elif isinstance(data.get("action"), dict):
        data = {"actions": [data["action"]]}
    actions = data.get("actions")
    if not isinstance(actions, list):
        return []
    calls = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        tool = action.get("tool") or action.get("tool_name") or action.get("name")
        arguments = (
            action.get("arguments")
            or action.get("args")
            or action.get("tool_input")
            or action.get("parameters")
            or {}
        )
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"_parse_error": arguments}
            arguments = parsed_arguments
        calls.append({"tool": str(tool or ""), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def _schema_arg_score(args: dict[str, Any], schema: dict[str, Any]) -> float:
    if not isinstance(args, dict) or "_parse_error" in args:
        return 0.0
    if not isinstance(schema, dict):
        return 0.8
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if required:
        present = sum(1 for key in required if key in args and args.get(key) not in (None, ""))
        required_score = present / len(required)
    else:
        required_score = 1.0
    type_checks = []
    for key, value in args.items():
        spec = properties.get(key) if isinstance(properties, dict) else None
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        typ = spec.get("type")
        ok = True
        if typ == "string":
            ok = isinstance(value, str)
        elif typ == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif typ == "number":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif typ == "array":
            ok = isinstance(value, list)
        elif typ == "object":
            ok = isinstance(value, dict)
        elif typ == "boolean":
            ok = isinstance(value, bool)
        type_checks.append(1.0 if ok else 0.0)
    type_score = sum(type_checks) / len(type_checks) if type_checks else 1.0
    return 0.75 * required_score + 0.25 * type_score


def _write_trace(payload: dict[str, Any], trace_path: str | None) -> None:
    if not trace_path:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _repeat_penalty(called: list[str]) -> float:
    if len(called) <= 1:
        return 0.0
    repeats = len(called) - len(set(called))
    return min(0.4, repeats * 0.15)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    trace_path: str | None = None,
) -> float:
    policy = _load_policy(ground_truth)
    recommended_for_trace = [p for p in policy.get("recommended_tools") or [] if isinstance(p, dict)]
    calls = parse_tool_calls(solution_str)
    solution_preview = _strip_think(solution_str)[:500]
    if not calls:
        score = -1.0
        _write_trace(
            {
                "data_source": data_source,
                "num_calls": 0,
                "score": score,
                "reason": "no_parseable_tool_call",
                "policy_source": str(policy.get("policy_source") or ""),
                "recommended_count": len(recommended_for_trace),
                "solution_preview": solution_preview,
            },
            trace_path or os.environ.get("TERRABOX_EXPEVO_RL_REWARD_TRACE_PATH"),
        )
        return score

    allowed = set(str(x) for x in policy.get("allowed_tools") or [])
    schemas = policy.get("tool_schemas") if isinstance(policy.get("tool_schemas"), dict) else {}
    recommended = recommended_for_trace
    rec_by_tool: dict[str, list[dict[str, Any]]] = {}
    for rec in recommended:
        tool = str(rec.get("tool") or "")
        if tool:
            rec_by_tool.setdefault(tool, []).append(rec)

    called = [str(call.get("tool") or "") for call in calls]
    valid_tools = sum(1 for tool in called if tool and (not allowed or tool in allowed)) / len(calls)
    schema_scores = [_schema_arg_score(call.get("arguments") if isinstance(call.get("arguments"), dict) else {}, schemas.get(str(call.get("tool") or ""), {})) for call in calls]
    arg_score = sum(schema_scores) / len(schema_scores)

    align_scores: list[float] = []
    risk_penalties: list[float] = []
    for rank, call in enumerate(calls):
        tool = str(call.get("tool") or "")
        recs = rec_by_tool.get(tool, [])
        if not recs:
            align_scores.append(0.0)
            risk_penalties.append(0.0)
            continue
        rec = max(recs, key=lambda r: float(r.get("q") or 0.0) - float(r.get("risk") or 0.0))
        q = max(0.0, min(1.0, float(rec.get("q") or 0.0)))
        risk = max(0.0, min(1.0, float(rec.get("risk") or 0.0)))
        support = min(1.0, (float(rec.get("n") or 0.0) / 20.0) ** 0.5)
        rank_bonus = 1.0 if rank == 0 else 0.85
        align_scores.append(rank_bonus * (0.75 * q + 0.25 * support))
        risk_penalties.append(0.4 * risk)
    experience_alignment = sum(align_scores) / len(align_scores)
    risk_penalty = sum(risk_penalties) / len(risk_penalties)
    repetition_penalty = _repeat_penalty(called)
    if len(calls) == 1:
        action_count_score = 1.0
        extra_call_penalty = 0.0
    else:
        # The pilot trains a single process-action policy: one JSON object with
        # exactly one tool action. Without an explicit count term, two valid
        # schema-compatible actions can receive the same reward as the intended
        # one-action output and teach the wrong behavior.
        action_count_score = 0.0
        extra_call_penalty = min(0.7, 0.35 * abs(len(calls) - 1))

    if recommended:
        raw = 0.25 * valid_tools + 0.20 * arg_score + 0.40 * experience_alignment + 0.15 * action_count_score
    else:
        # Pure GRPO/schema baseline: no ExperienceEvo policy is provided, so the
        # reward should measure only parseability, public tool legality, and
        # public schema compatibility. This keeps the baseline comparable instead
        # of implicitly assigning zero to the ExperienceEvo alignment term.
        raw = 0.45 * valid_tools + 0.35 * arg_score + 0.20 * action_count_score
    raw -= risk_penalty + repetition_penalty + extra_call_penalty
    score = max(-1.0, min(1.0, 2 * raw - 1.0))
    _write_trace(
        {
            "data_source": data_source,
            "strict_nolabel": bool(policy.get("strict_nolabel")),
            "num_calls": len(calls),
            "called_tools": called,
            "valid_tool_score": round(valid_tools, 4),
            "arg_score": round(arg_score, 4),
            "action_count_score": round(action_count_score, 4),
            "experience_alignment": round(experience_alignment, 4),
            "risk_penalty": round(risk_penalty, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "extra_call_penalty": round(extra_call_penalty, 4),
            "policy_source": str(policy.get("policy_source") or ""),
            "recommended_count": len(recommended),
            "score": round(float(score), 6),
            "solution_preview": solution_preview,
        },
        trace_path or os.environ.get("TERRABOX_EXPEVO_RL_REWARD_TRACE_PATH"),
    )
    return float(score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test strict ExperienceEvo RL reward")
    parser.add_argument("--solution-str", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()
    print(compute_score("oea_experience_evo_rl_strict", args.solution_str, args.ground_truth))
