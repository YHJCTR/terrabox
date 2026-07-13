"""Metric provider for tau2-bench Results files."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Optional

from ...interfaces import MetricSpec, TaskMetric
from .files import _iter_result_paths, _load_results_like, _read_json
from .traces import _reward_info, _sim_task_id, _task_lookup, _tool_calls


class Tau2MetricProvider:
    """Compute tau2 reward and trajectory metrics from Results files."""

    def __init__(self, results_path_fn=lambda exp: exp):
        self._results_path = results_path_fn

    def per_task(self, experiment: str) -> dict[str, TaskMetric]:
        out: dict[str, TaskMetric] = {}
        for path in _iter_result_paths(self._results_path(experiment)):
            meta, simulations = _load_results_like(path)
            tasks = _task_lookup(meta.get("tasks") or [])
            domain = (((meta.get("info") or {}).get("environment_info") or {}).get("domain_name") or "")
            for sim in simulations:
                reward = _reward_info(sim)
                messages = sim.get("messages") or []
                task = tasks.get(str(sim.get("task_id"))) or {}
                tool_calls = [c for msg in messages for c in _tool_calls(msg)]
                termination = str(sim.get("termination_reason") or "")
                success = reward["reward"] >= 1.0 - 1e-6
                flags = {
                    "reward_failure": not success,
                    "db_failure": reward["db_reward"] < 1.0 if reward["db_match"] is not None else False,
                    "communicate_failure": reward["communicate_reward"] < 1.0 and "COMMUNICATE" in reward["reward_basis"],
                    "action_failure": reward["action_reward"] < 1.0 and "ACTION" in reward["reward_basis"],
                    "max_steps": termination == "max_steps",
                    "error_termination": "error" in termination,
                    "infrastructure_error": termination == "infrastructure_error",
                }
                extra = {
                    "path": path,
                    "domain": domain,
                    "task_id": str(sim.get("task_id")),
                    "trial": sim.get("trial"),
                    "reward": reward["reward"],
                    "db_reward": reward["db_reward"],
                    "db_match": reward["db_match"],
                    "action_reward": reward["action_reward"],
                    "communicate_reward": reward["communicate_reward"],
                    "nl_reward": reward["nl_reward"],
                    "reward_basis": reward["reward_basis"],
                    "termination_reason": termination,
                    "duration": float(sim.get("duration") or 0),
                    "agent_cost": float(sim.get("agent_cost") or 0),
                    "user_cost": float(sim.get("user_cost") or 0),
                    "n_messages": len(messages),
                    "n_tool_calls": len(tool_calls),
                    "n_action_checks": reward["n_action_checks"],
                    "n_nl_assertions": reward["n_nl_assertions"],
                    "n_communicate_checks": reward["n_communicate_checks"],
                    "task_purpose": (task.get("description") or {}).get("purpose"),
                }
                task_key = _sim_task_id(sim, path, domain)
                out[task_key] = TaskMetric(
                    task_id=task_key,
                    success=success,
                    tool_f1=reward["reward"],
                    failure_flags=flags,
                    extra=extra,
                )
        if out:
            return out
        return self._per_task_from_submission(experiment)

    def _per_task_from_submission(self, experiment: str) -> dict[str, TaskMetric]:
        out = {}
        for path in _iter_result_paths(self._results_path(experiment)):
            data = _read_json(path)
            results = data.get("results") or {}
            for domain, vals in results.items():
                if not isinstance(vals, dict):
                    continue
                pass_1 = float(vals.get("pass_1") or 0) / 100.0
                task_id = f"{os.path.basename(path)}::{domain}"
                out[task_id] = TaskMetric(
                    task_id=task_id,
                    success=pass_1 >= 1.0 - 1e-6,
                    tool_f1=pass_1,
                    failure_flags={"summary_only": True},
                    extra={"path": path, "domain": domain, "reward": pass_1, **vals},
                )
        return out

    def aggregate(self, experiment: str, task_ids: Optional[list[str]] = None) -> dict[str, Any]:
        metrics = self.per_task(experiment)
        if task_ids is not None:
            keep = set(task_ids)
            metrics = {k: v for k, v in metrics.items() if k in keep}
        vals = list(metrics.values())
        n = len(vals) or 1
        domains = Counter(str(m.extra.get("domain") or "") for m in vals)
        terminations = Counter(str(m.extra.get("termination_reason") or "") for m in vals)

        def avg(name: str) -> float:
            return sum(float(m.extra.get(name) or 0) for m in vals) / n

        def rate(flag: str) -> float:
            return sum(m.failure_flags.get(flag, False) for m in vals) / n

        domain_rewards = []
        per_domain = {}
        for domain in domains:
            subset = [m for m in vals if str(m.extra.get("domain") or "") == domain]
            if subset:
                score = sum(float(m.extra.get("reward") or 0) for m in subset) / len(subset)
                domain_rewards.append(score)
                if domain:
                    per_domain[domain] = {"n": len(subset), "avg_reward": score}
        return {
            "n": len(vals),
            "success_rate": sum(m.success for m in vals) / n,
            "tool_f1": sum(m.tool_f1 for m in vals) / n,
            "avg_reward": avg("reward"),
            "avg_db_reward": avg("db_reward"),
            "avg_action_reward": avg("action_reward"),
            "avg_communicate_reward": avg("communicate_reward"),
            "avg_nl_reward": avg("nl_reward"),
            "db_failure_rate": rate("db_failure"),
            "action_failure_rate": rate("action_failure"),
            "communicate_failure_rate": rate("communicate_failure"),
            "max_steps_rate": rate("max_steps"),
            "error_termination_rate": rate("error_termination"),
            "infrastructure_error_rate": rate("infrastructure_error"),
            "avg_messages": avg("n_messages"),
            "avg_tool_calls": avg("n_tool_calls"),
            "avg_duration_s": avg("duration"),
            "avg_agent_cost": avg("agent_cost"),
            "avg_user_cost": avg("user_cost"),
            "n_domains": len([d for d in domains if d]),
            "n_termination_reasons": len([t for t in terminations if t]),
            "macro_domain_reward": sum(domain_rewards) / len(domain_rewards) if domain_rewards else 0.0,
            "worst_domain_reward": min(domain_rewards) if domain_rewards else 0.0,
            "per_domain": per_domain,
            "termination_reasons": dict(terminations),
        }

    def metric_specs(self) -> list[MetricSpec]:
        return [
            MetricSpec("success_rate", "Fraction of simulations with full reward.", "higher_better"),
            MetricSpec("avg_reward", "Mean tau2 final reward.", "higher_better"),
            MetricSpec("avg_db_reward", "Mean database/end-state reward component.", "higher_better"),
            MetricSpec("avg_action_reward", "Mean action reward component when used.", "higher_better"),
            MetricSpec("avg_communicate_reward", "Mean required-communication reward component.", "higher_better"),
            MetricSpec("avg_nl_reward", "Mean natural-language assertion reward component.", "higher_better"),
            MetricSpec("db_failure_rate", "DB/end-state component failed.", "lower_better"),
            MetricSpec("action_failure_rate", "Required action component failed.", "lower_better"),
            MetricSpec("communicate_failure_rate", "Required communication component failed.", "lower_better"),
            MetricSpec("max_steps_rate", "Simulation ended due to max steps.", "lower_better"),
            MetricSpec("error_termination_rate", "Simulation ended with an error.", "lower_better"),
            MetricSpec("infrastructure_error_rate", "Runs that failed because of infrastructure or serving errors, not task behavior.", "lower_better"),
            MetricSpec("avg_messages", "Average conversation messages.", "neutral"),
            MetricSpec("avg_tool_calls", "Average agent tool calls.", "neutral"),
            MetricSpec("macro_domain_reward", "Mean reward averaged equally over domains.", "higher_better"),
            MetricSpec("worst_domain_reward", "Worst domain-level mean reward.", "higher_better"),
        ]
