"""Trajectory conversion for tau2-bench result files."""
from __future__ import annotations

import json
import os
from typing import Iterable

from ...interfaces import Step, Trace
from ...text_utils import strip_think
from .files import _iter_result_paths, _load_results_like


def _message_text(message: dict) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return strip_think(content)
    return strip_think(json.dumps(content, ensure_ascii=False))


def _tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


def _steps_from_messages(messages: list[dict]) -> list[Step]:
    steps: list[Step] = []
    for msg in messages:
        role = msg.get("role")
        text = _message_text(msg)
        if role == "assistant":
            calls = _tool_calls(msg)
            if calls:
                for call in calls:
                    function = call.get("function")
                    if isinstance(function, dict):
                        name = call.get("name") or function.get("name")
                    else:
                        name = call.get("name") or function
                    args = call.get("arguments") or call.get("args") or {}
                    steps.append(Step(role="assistant", text=text, tool=str(name or ""), args=args))
            else:
                steps.append(Step(role="assistant", text=text))
        elif role in {"user", "system"}:
            steps.append(Step(role=role, text=text))
        else:
            steps.append(Step(role="tool", text=text, errored=bool(msg.get("error"))))
    return steps


def _reward_info(sim: dict) -> dict:
    info = sim.get("reward_info") or {}
    breakdown = info.get("reward_breakdown") or {}
    db = info.get("db_check") or {}
    action_checks = info.get("action_checks") or []
    nl_assertions = info.get("nl_assertions") or []
    communicate = info.get("communicate_checks") or []
    return {
        "reward": float(info.get("reward") or 0.0),
        "db_reward": float(db.get("db_reward") if db.get("db_reward") is not None else breakdown.get("DB") or 0.0),
        "db_match": bool(db.get("db_match")) if db else None,
        "action_reward": float(breakdown.get("ACTION") or 0.0),
        "communicate_reward": float(breakdown.get("COMMUNICATE") if breakdown.get("COMMUNICATE") is not None else 0.0),
        "nl_reward": float(breakdown.get("NL_ASSERTION") if breakdown.get("NL_ASSERTION") is not None else 0.0),
        "n_action_checks": len(action_checks) if isinstance(action_checks, list) else 0,
        "n_nl_assertions": len(nl_assertions) if isinstance(nl_assertions, list) else 0,
        "n_communicate_checks": len(communicate) if isinstance(communicate, list) else 0,
        "reward_basis": info.get("reward_basis") or [],
    }


def _sim_task_id(sim: dict, source: str, domain: str = "") -> str:
    trial = sim.get("trial")
    task_id = sim.get("task_id") or sim.get("id") or "unknown_task"
    prefix = domain or os.path.basename(source)
    if trial is None:
        return f"{prefix}::{task_id}"
    return f"{prefix}::{task_id}::trial{trial}"


def _task_lookup(tasks: list[dict]) -> dict[str, dict]:
    return {str(t.get("id")): t for t in tasks if isinstance(t, dict)}


def _task_query(task: dict) -> str:
    scenario = task.get("user_scenario") or {}
    if not isinstance(scenario, dict):
        return str(scenario)
    instructions = scenario.get("instructions") or ""
    if isinstance(instructions, dict):
        return str(instructions.get("reason_for_call") or "")
    return str(instructions)


def _final_assistant(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return _message_text(msg)
    return ""


class Tau2TrajectorySource:
    """Read tau2 Results files as generic promptevo traces."""

    def __init__(self, results_path_fn=lambda exp: exp):
        self._results_path = results_path_fn

    def traces(self, experiment: str) -> Iterable[Trace]:
        for path in _iter_result_paths(self._results_path(experiment)):
            meta, simulations = _load_results_like(path)
            tasks = _task_lookup(meta.get("tasks") or [])
            domain = (((meta.get("info") or {}).get("environment_info") or {}).get("domain_name") or "")
            for sim in simulations:
                reward = _reward_info(sim)
                task = tasks.get(str(sim.get("task_id"))) or {}
                query = _task_query(task)
                yield Trace(
                    task_id=_sim_task_id(sim, path, domain),
                    query=str(query),
                    steps=_steps_from_messages(sim.get("messages") or []),
                    success=reward["reward"] >= 1.0 - 1e-6,
                    final_answer=_final_assistant(sim.get("messages") or []),
                    raw={"path": path, "simulation": sim, "task": task, "reward": reward},
                )
