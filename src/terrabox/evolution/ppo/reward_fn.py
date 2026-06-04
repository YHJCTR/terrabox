"""GRPO reward for Terrabox tool-use completions.

The dynamic mode executes parsed tool calls through Terrabox's real tool
executor. It intentionally does not claim task-level success unless a separate
verifier is available.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ACTION_PATTERN = re.compile(
    r"Action:\s*(?P<tool>[^\n]+)\s*\nAction Input:\s*(?P<args>\{.*?\})(?=\nAction:|\Z)",
    re.DOTALL,
)
_TRACE_LOCK = threading.Lock()


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _load_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        data = json.loads(ground_truth or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_tool_calls(solution_str: str) -> list[dict[str, Any]]:
    """Parse JSON-action or ReAct text completions into tool calls."""
    text = _strip_think(solution_str)
    if not text:
        return []

    candidates = [text]
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match and json_match.group(0) != text:
        candidates.append(json_match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        actions = data.get("actions") if isinstance(data, dict) else None
        if isinstance(actions, list):
            calls = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                tool = action.get("tool") or action.get("name")
                if tool:
                    calls.append(
                        {
                            "tool": str(tool),
                            "arguments": action.get("arguments") or action.get("args") or {},
                        }
                    )
            return calls

    calls = []
    for match in ACTION_PATTERN.finditer(text):
        args_text = match.group("args")
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {"_parse_error": args_text}
        calls.append({"tool": match.group("tool").strip(), "arguments": args})
    return calls


def _json_payload_from_error_text(result: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", result or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_tool_result(result: Any) -> dict[str, Any]:
    """Classify a single real tool output.

    This is a tool-execution judgment only. It is not a task-success judgment.
    """
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    lowered = text.lower()
    payload = _json_payload_from_error_text(text)
    payload_status = str(payload.get("status", "")).lower()
    error_type = payload.get("error_type")

    error_markers = (
        "tool execution error:",
        '"status": "error"',
        '"status":"error"',
        "traceback",
        "exception",
        "outofmemoryerror",
        "cuda out of memory",
        "timed out",
        "timeout",
    )
    failed = payload_status == "error" or any(marker in lowered for marker in error_markers)
    if failed and not error_type:
        if "out of memory" in lowered or "outofmemory" in lowered:
            error_type = "tool_oom"
        elif "timeout" in lowered or "timed out" in lowered:
            error_type = "tool_timeout"
        else:
            error_type = "tool_error"
    return {
        "tool_success": not failed and bool(str(text).strip()),
        "error_type": error_type,
        "raw_status": payload.get("status"),
    }


def summarize_executed_trace(executed: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(executed)
    success_count = sum(1 for step in executed if step.get("tool_success"))
    error_types = [
        str(step.get("error_type"))
        for step in executed
        if step.get("error_type")
    ]
    return {
        "num_tool_calls": total,
        "tool_success_count": success_count,
        "tool_success_rate": success_count / total if total else 0.0,
        "all_tools_succeeded": bool(total) and success_count == total,
        "error_types": error_types,
        "verified_task_success": None,
        "task_success_basis": "not_auto_verifiable",
    }


@contextmanager
def task_data_env(extra_info: dict[str, Any] | None):
    keys = (
        "TERRABOX_TASK_DATA_DIR",
        "TERRABOX_TASK_DATA_FILES",
        "TERRABOX_ARTIFACT_OUTPUT_DIR",
        "TERRABOX_ARTIFACT_INDEX_PATH",
    )
    previous = {key: os.environ.get(key) for key in keys}
    info = extra_info or {}
    try:
        data_dir = str(info.get("data_dir") or "").strip()
        data_files = [str(path) for path in info.get("data_files", []) or [] if path]
        if data_dir:
            os.environ["TERRABOX_TASK_DATA_DIR"] = data_dir
        if data_files:
            os.environ["TERRABOX_TASK_DATA_FILES"] = json.dumps(data_files, ensure_ascii=False)
        artifact_dir = str(info.get("artifact_dir") or os.environ.get("TERRABOX_GRPO_ARTIFACT_DIR") or "").strip()
        if artifact_dir:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            os.environ["TERRABOX_ARTIFACT_OUTPUT_DIR"] = artifact_dir
            os.environ.setdefault("TERRABOX_ARTIFACT_INDEX_PATH", str(Path(artifact_dir) / "artifact_index.json"))
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute_tool_call(call: dict[str, Any], extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one parsed tool call through the Terrabox runtime."""
    tool = str(call.get("tool") or "")
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if not tool:
        return {
            "tool": tool,
            "arguments": arguments,
            "result": "Tool execution error: missing tool name",
            "tool_success": False,
            "error_type": "missing_tool_name",
        }

    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import registry
    from terrabox.agent.tool_executor import AgentToolExecutor

    if not registry.list_toolkits():
        load_builtin_toolkits()
    if registry.get_tool(tool) is None or registry.get_handler(tool) is None:
        return {
            "tool": tool,
            "arguments": arguments,
            "result": f"Tool execution error: Tool not found: {tool}",
            "tool_success": False,
            "error_type": "tool_not_found",
        }

    with task_data_env(extra_info):
        result = AgentToolExecutor.execute(tool, arguments, user=None)
    classification = classify_tool_result(result)
    return {
        "tool": tool,
        "arguments": arguments,
        "result": result,
        **classification,
    }


def execute_tool_trace(
    calls: list[dict[str, Any]],
    extra_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [execute_tool_call(call, extra_info) for call in calls]


def _write_reward_trace(payload: dict[str, Any], trace_path: str | None) -> None:
    if not trace_path:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _tool_f1(called: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    called_set = set(called)
    if not expected_set:
        return 1.0 if not called_set else 0.0
    if not called_set:
        return 0.0
    tp = len(called_set & expected_set)
    precision = tp / len(called_set)
    recall = tp / len(expected_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _order_score(called: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0 if not called else 0.0
    if not called:
        return 0.0
    index = 0
    matched = 0
    for tool in called:
        while index < len(expected) and expected[index] != tool:
            index += 1
        if index < len(expected):
            matched += 1
            index += 1
    return matched / len(expected)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
    execute_tools: bool = False,
    trace_path: str | None = None,
) -> float:
    """veRL-compatible reward function.

    With ``execute_tools=True``, parsed tool calls are executed in Terrabox and
    the reward uses real tool success/failure. Task-level success remains
    unverified unless an external evaluator is added.
    """
    truth = _load_ground_truth(ground_truth)
    expected = [str(t) for t in truth.get("expected_tools", []) if t]
    calls = parse_tool_calls(solution_str)
    if not calls:
        return -1.0

    called = [str(call.get("tool")) for call in calls if call.get("tool")]
    f1 = _tool_f1(called, expected)
    order = _order_score(called, expected)
    valid_args = sum(isinstance(call.get("arguments"), dict) and "_parse_error" not in call.get("arguments", {}) for call in calls)
    arg_score = valid_args / len(calls) if calls else 0.0
    centered_policy_score = (0.7 * f1 + 0.3 * order) * 2 - 1

    malformed_penalty = -0.2 if len(called) != len(calls) else 0.0
    forbidden_penalty = 0.0
    allowed = set((extra_info or {}).get("allowed_tools") or [])
    if allowed:
        forbidden = sum(1 for tool in called if tool not in allowed)
        forbidden_penalty = -0.5 * forbidden / len(called) if called else 0.0

    if execute_tools:
        started = time.time()
        trace = execute_tool_trace(calls, extra_info)
        summary = summarize_executed_trace(trace)
        execution_score = summary["tool_success_rate"] * 2 - 1
        score = 0.5 * execution_score + 0.4 * centered_policy_score + 0.1 * arg_score
        score += malformed_penalty + forbidden_penalty
        _write_reward_trace(
            {
                "task_id": truth.get("task_id") or (extra_info or {}).get("task_id"),
                "data_source": data_source,
                "called_tools": called,
                "expected_tools": expected,
                "tool_f1": f1,
                "order_score": order,
                "score": float(max(-1.0, min(1.0, score))),
                "elapsed_seconds": round(time.time() - started, 3),
                "summary": summary,
                "trace": trace,
                "task_success_note": "verified_task_success is intentionally unknown without a task evaluator",
            },
            trace_path or os.environ.get("TERRABOX_GRPO_REWARD_TRACE_PATH"),
        )
        return float(max(-1.0, min(1.0, score)))

    score = alpha * f1 + beta * order + gamma * arg_score + malformed_penalty + forbidden_penalty
    return float(max(-1.0, min(1.0, score)))


if __name__ == "__main__":  # pragma: no cover - manual smoke helper
    import argparse

    parser = argparse.ArgumentParser(description="Test Terrabox GRPO reward")
    parser.add_argument("--solution-str", required=True)
    parser.add_argument("--ground-truth", default='{"expected_tools": []}')
    args = parser.parse_args()
    print(compute_score("terrabox_grpo", args.solution_str, args.ground_truth))
