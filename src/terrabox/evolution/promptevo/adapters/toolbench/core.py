"""ToolBench adapter for promptevo.

This module reads ToolBench rollout outputs and exposes them through the
domain-neutral promptevo Protocols. It intentionally does not import ToolBench
source code, so it can be committed with promptevo and pointed at any local
ToolBench checkout or result directory.

Expected ToolBench result files are the JSON files produced by
``toolbench/inference/qa_pipeline.py``. The adapter supports both CoT-style
``trys[].chain`` outputs and DFS/DFSDT ``tree`` / ``compare_candidates``
outputs; when ``answer_generation.train_messages`` is available, it is used as
the most faithful linear trace.
"""
from __future__ import annotations

import ast
import csv
import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ...interfaces import MetricSpec, Step, TaskMetric, Trace


DEFAULT_STABLE_TOOLBENCH_ROOT = "/data1/yuhongjie2/StepTool/stabletoolbench"
DEFAULT_TOOLBENCH_ROOT = "/data1/yuhongjie2/ToolBench"
DEFAULT_STABLE_TOOLBENCH_PROMPT_PATH = os.path.join(
    DEFAULT_STABLE_TOOLBENCH_ROOT,
    "toolbench",
    "inference",
    "Prompts",
    "ReAct_prompts.py",
)

_ERROR_CODES = {
    1: "hallucinated_function",
    2: "invalid_input",
    4: "give_up",
    5: "timeout",
    6: "api_not_working",
    7: "not_subscribed",
    8: "unauthorized",
    9: "too_many_requests",
    10: "rate_limit",
    11: "message_error",
    12: "request_error",
}

_TRANSIENT_CODES = {5, 9, 10, 12}


def _default_results_dir(experiment: str) -> str:
    return experiment


def _iter_result_files(results_dir: str) -> Iterable[str]:
    patterns = ["*.json", "*/*.json"]
    seen: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(results_dir, pattern)):
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                yield path


def _task_id_from_path(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    # ToolBench filenames are usually <query_id>_<method>.json.
    return name.split("_", 1)[0] if "_" in name else name


def _safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _load_python_string_constant(path: str, name: str) -> str:
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source, filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise ValueError(f"{name} in {path} is not a string")
        return value.strip()
    raise ValueError(f"{name} not found in {path}")


def _parse_args(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        return text


def _tool_error(text: str) -> bool:
    low = text.lower()
    return any(
        key in low
        for key in (
            '"error": "api not working',
            '"error": "unauthorized',
            '"error": "unsubscribed',
            '"error": "too many requests',
            '"error": "rate limit',
            '"error": "message error',
            "no such function name",
            "must have",
            "not a valid choice",
            "request invalid",
            "timeout",
        )
    )


def _extract_query(data: dict) -> str:
    ag = data.get("answer_generation") or {}
    return str(ag.get("query") or data.get("query") or "")


def _extract_final_answer(data: dict) -> str:
    ag = data.get("answer_generation") or {}
    return str(ag.get("final_answer") or "")


def _last_train_messages(data: dict) -> list[dict]:
    ag = data.get("answer_generation") or {}
    batches = ag.get("train_messages") or []
    if not batches:
        return []
    last = batches[-1]
    return last if isinstance(last, list) else []


def _steps_from_messages(messages: list[dict]) -> list[Step]:
    steps: list[Step] = []
    for msg in messages:
        role = str(msg.get("role") or "").lower()
        content = str(msg.get("content") or "")
        if role == "system":
            continue
        if role == "user":
            steps.append(Step(role="user", text=content))
        elif role == "assistant":
            call = msg.get("function_call") or {}
            if call:
                steps.append(
                    Step(
                        role="assistant",
                        text=content,
                        tool=str(call.get("name") or ""),
                        args=_parse_args(call.get("arguments")),
                    )
                )
            else:
                steps.append(Step(role="assistant", text=content))
        elif role == "function":
            steps.append(Step(role="tool", text=content, errored=_tool_error(content)))
    return steps


def _linear_nodes_from_result(data: dict) -> list[dict]:
    for candidate in data.get("compare_candidates") or []:
        if isinstance(candidate, list) and candidate:
            return candidate
    for attempt in data.get("trys") or []:
        chain = attempt.get("chain")
        if isinstance(chain, list) and chain:
            return chain
    return []


def _steps_from_nodes(nodes: list[dict]) -> list[Step]:
    steps: list[Step] = []
    pending_action: Optional[str] = None
    for node in nodes:
        node_type = str(node.get("node_type") or "")
        desc = str(node.get("description") or "")
        if node_type == "Thought":
            steps.append(Step(role="assistant", text=desc))
        elif node_type == "Action":
            pending_action = desc
        elif node_type == "Action Input":
            code = node.get("observation_code")
            errored = isinstance(code, int) and code not in (0, 3)
            steps.append(
                Step(
                    role="assistant",
                    tool=pending_action or "",
                    args=_parse_args(desc),
                    errored=errored,
                )
            )
            if "observation" in node:
                steps.append(
                    Step(
                        role="tool",
                        text=str(node.get("observation") or ""),
                        errored=errored or _tool_error(str(node.get("observation") or "")),
                    )
                )
            pending_action = None
    return steps


def _walk_tree(node: dict) -> Iterable[dict]:
    yield node
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from _walk_tree(child)


def _all_nodes(data: dict) -> list[dict]:
    tree = ((data.get("tree") or {}).get("tree") or {})
    if isinstance(tree, dict) and tree:
        return list(_walk_tree(tree))
    nodes = []
    for candidate in data.get("compare_candidates") or []:
        if isinstance(candidate, list):
            nodes.extend(x for x in candidate if isinstance(x, dict))
    for attempt in data.get("trys") or []:
        chain = attempt.get("chain") or []
        if isinstance(chain, list):
            nodes.extend(x for x in chain if isinstance(x, dict))
    return nodes


def _called_tools_from_steps(steps: list[Step]) -> list[str]:
    return [s.tool for s in steps if s.role == "assistant" and s.tool]


def _max_repeat(seq: list[str]) -> int:
    best = cur = 0
    prev = None
    for item in seq:
        cur = cur + 1 if item == prev else 1
        prev = item
        best = max(best, cur)
    return best


def _method_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_", 1)[1] if "_" in stem else ""


def _load_tool_eval_labels(path: str) -> dict[str, bool]:
    """Load optional ToolEval pass labels from JSON or TSV/CSV.

    Accepted JSON shapes:
      {"123": true}
      {"123": {"pass_rate_label": "passed"}}
      {"123": {"passed": 3, "failed": 1}}

    Accepted tabular columns include query_id plus pass_rate_label/is_solved.
    """
    if not path:
        return {}
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        out = {}
        for key, value in data.items():
            if isinstance(value, bool):
                out[str(key)] = value
            elif isinstance(value, dict):
                if "pass_rate_label" in value:
                    out[str(key)] = str(value["pass_rate_label"]).lower() == "passed"
                elif "passed" in value and "failed" in value:
                    out[str(key)] = int(value["passed"]) > int(value["failed"])
                elif "is_solved" in value:
                    out[str(key)] = bool(value["is_solved"])
        return out

    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            qid = row.get("query_id") or row.get("id") or row.get("task_id")
            if not qid:
                continue
            if "pass_rate_label" in row:
                out[str(qid)] = str(row["pass_rate_label"]).lower() == "passed"
            elif "is_solved" in row:
                out[str(qid)] = str(row["is_solved"]).lower() in ("true", "1", "yes", "solved")
    return out


@dataclass
class ToolBenchPaths:
    """Path configuration for ToolBench experiments."""

    results_dir: str
    tool_eval_labels: str = ""


class ToolBenchPromptStore:
    """Versioned static prompts for ToolBench.

    ``base_prompt_path`` should point at a plain-text copy of ToolBench's static
    system prompt. The adapter deliberately avoids editing ToolBench source.
    """

    def __init__(
        self,
        versions_dir: str = "evolution_store/promptevo/toolbench/versions",
        base_prompt_path: str = "",
        source_prompt_path: str = DEFAULT_STABLE_TOOLBENCH_PROMPT_PATH,
        source_constant: str = "FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION",
    ):
        self.versions_dir = versions_dir
        self.base_prompt_path = base_prompt_path
        self.source_prompt_path = source_prompt_path
        self.source_constant = source_constant

    def _path(self, version: str) -> str:
        return os.path.join(self.versions_dir, f"{version}.txt")

    def load(self, version: str) -> str:
        if version in ("base", "orig", "original"):
            if self.base_prompt_path:
                with open(self.base_prompt_path, encoding="utf-8") as f:
                    return f.read().strip()
            if self.source_prompt_path:
                return _load_python_string_constant(self.source_prompt_path, self.source_constant)
            raise ValueError(
                "base_prompt_path or source_prompt_path is required to load the ToolBench base prompt"
            )
        with open(self._path(version), encoding="utf-8") as f:
            return f.read().strip()

    def save(self, version: str, prompt: str, meta: dict) -> str:
        os.makedirs(self.versions_dir, exist_ok=True)
        prompt_path = self._path(version)
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        with open(prompt_path.replace(".txt", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return os.path.abspath(prompt_path)


class ToolBenchTrajectorySource:
    """Read ToolBench rollout JSON files as promptevo ``Trace`` objects."""

    def __init__(self, results_dir_fn=_default_results_dir):
        self._dir = results_dir_fn

    def _to_trace(self, path: str, data: dict, success: bool) -> Trace:
        steps = _steps_from_messages(_last_train_messages(data))
        if not steps:
            steps = _steps_from_nodes(_linear_nodes_from_result(data))
        return Trace(
            task_id=_task_id_from_path(path),
            query=_extract_query(data),
            steps=steps,
            success=success,
            final_answer=_extract_final_answer(data),
            raw=data,
        )

    def traces(self, experiment: str) -> Iterable[Trace]:
        results_dir = self._dir(experiment)
        for path in _iter_result_files(results_dir):
            data = _safe_load_json(path)
            if not data:
                continue
            ag = data.get("answer_generation") or {}
            success = bool(ag.get("valid_data") or data.get("win"))
            yield self._to_trace(path, data, success)


class ToolBenchMetricProvider:
    """Compute prompt-evolution metrics from ToolBench rollout JSON files.

    If ``tool_eval_label_fn`` is provided and returns a ToolEval label file for
    an experiment, ``success`` and ``pass_rate`` use those labels. Otherwise
    success falls back to ToolBench's structural ``valid_data`` / ``win`` flag,
    which means the agent reached ``Finish(give_answer)`` rather than a judged
    task pass.
    """

    def __init__(
        self,
        results_dir_fn=_default_results_dir,
        tool_eval_label_fn=None,
    ):
        self._dir = results_dir_fn
        self._labels = tool_eval_label_fn or (lambda _experiment: "")

    def _metric_from_file(self, path: str, data: dict, labels: dict[str, bool]) -> TaskMetric:
        task_id = _task_id_from_path(path)
        ag = data.get("answer_generation") or {}
        steps = _steps_from_messages(_last_train_messages(data))
        if not steps:
            steps = _steps_from_nodes(_linear_nodes_from_result(data))
        tools = _called_tools_from_steps(steps)
        nodes = _all_nodes(data)
        codes = [
            n.get("observation_code")
            for n in nodes
            if isinstance(n.get("observation_code"), int)
        ]
        finish_calls = sum(1 for t in tools if t == "Finish")
        hallucinated = sum(1 for c in codes if c == 1)
        invalid_input = sum(1 for c in codes if c == 2)
        transient = sum(1 for c in codes if c in _TRANSIENT_CODES)
        error_calls = sum(1 for c in codes if c not in (0, 3))
        structural_success = bool(ag.get("valid_data") or data.get("win"))
        judged = labels.get(task_id)
        success = structural_success if judged is None else judged
        max_repeat = _max_repeat([t for t in tools if t != "Finish"])
        n_tool_calls = len([t for t in tools if t != "Finish"])
        tree = data.get("tree") or {}

        flags = {
            "no_finish": finish_calls == 0,
            "give_up": str(ag.get("finish_type") or "").lower() == "give_up" or 4 in codes,
            "hallucinated_function": hallucinated > 0,
            "invalid_input": invalid_input > 0,
            "transient_error": transient > 0,
            "tool_error": error_calls > 0,
            "repeat_call": max_repeat >= 4,
            "over_calling": n_tool_calls > 8,
        }
        extra = {
            "method": _method_from_path(path),
            "structural_success": structural_success,
            "judged_success": judged,
            "n_tool_calls": n_tool_calls,
            "finish_calls": finish_calls,
            "query_count": int(ag.get("query_count") or 0),
            "total_tokens": int(ag.get("total_tokens") or 0),
            "tree_size": int(tree.get("size") or len(nodes) or 0),
            "tree_max_depth": int(tree.get("max_length") or 0),
            "max_repeat": max_repeat,
            "hallucinated_calls": hallucinated,
            "invalid_input_calls": invalid_input,
            "transient_error_calls": transient,
            "error_calls": error_calls,
            "observation_codes": Counter(codes),
        }
        # ToolBench has no gold tool list in rollout output; use structural
        # success/pass labels as the primary scalar expected by promptevo.
        return TaskMetric(
            task_id=task_id,
            success=success,
            tool_f1=1.0 if success else 0.0,
            failure_flags=flags,
            extra=extra,
        )

    def per_task(self, experiment: str) -> dict[str, TaskMetric]:
        labels = _load_tool_eval_labels(self._labels(experiment))
        out: dict[str, TaskMetric] = {}
        for path in _iter_result_files(self._dir(experiment)):
            data = _safe_load_json(path)
            if not data:
                continue
            metric = self._metric_from_file(path, data, labels)
            out[metric.task_id] = metric
        return out

    def aggregate(self, experiment: str, task_ids: Optional[list[str]] = None) -> dict[str, Any]:
        metrics = self.per_task(experiment)
        if task_ids is not None:
            keep = set(task_ids)
            metrics = {k: v for k, v in metrics.items() if k in keep}
        n = len(metrics) or 1
        vals = list(metrics.values())

        def rate_flag(name: str) -> float:
            return sum(m.failure_flags.get(name, False) for m in vals) / n

        def avg_extra(name: str) -> float:
            return sum(float(m.extra.get(name) or 0.0) for m in vals) / n

        judged = [m for m in vals if m.extra.get("judged_success") is not None]
        agg = {
            "n": len(vals),
            "success_rate": sum(m.success for m in vals) / n,
            "tool_f1": sum(m.tool_f1 for m in vals) / n,
            "structural_success_rate": sum(m.extra.get("structural_success", False) for m in vals) / n,
            "pass_rate": (sum(m.success for m in judged) / len(judged)) if judged else None,
            "avg_tool_calls": avg_extra("n_tool_calls"),
            "avg_query_count": avg_extra("query_count"),
            "avg_total_tokens": avg_extra("total_tokens"),
            "avg_tree_size": avg_extra("tree_size"),
            "avg_tree_max_depth": avg_extra("tree_max_depth"),
            "avg_max_repeat": avg_extra("max_repeat"),
            "no_finish_rate": rate_flag("no_finish"),
            "give_up_rate": rate_flag("give_up"),
            "hallucinated_function_rate": rate_flag("hallucinated_function"),
            "invalid_input_rate": rate_flag("invalid_input"),
            "tool_error_rate": rate_flag("tool_error"),
            "transient_error_rate": rate_flag("transient_error"),
            "repeat_call_rate": rate_flag("repeat_call"),
            "over_calling_rate": rate_flag("over_calling"),
        }
        return agg

    def metric_specs(self) -> list[MetricSpec]:
        return [
            MetricSpec("success_rate", "Primary success rate. Uses ToolEval labels when provided; otherwise uses ToolBench structural valid_data/win.", "higher_better"),
            MetricSpec("pass_rate", "ToolEval judged pass rate when a ToolEval label file is supplied.", "higher_better"),
            MetricSpec("structural_success_rate", "Rate of rollouts that reached Finish(give_answer); not a semantic correctness metric.", "higher_better"),
            MetricSpec("avg_tool_calls", "Average non-Finish tool calls per task; high values often indicate over-exploration.", "lower_better"),
            MetricSpec("avg_query_count", "Average LLM calls per task.", "lower_better"),
            MetricSpec("avg_total_tokens", "Average generated token count reported by ToolBench.", "lower_better"),
            MetricSpec("avg_tree_size", "Average DFS tree size; high values indicate expensive search.", "lower_better"),
            MetricSpec("avg_tree_max_depth", "Average maximum DFS tree depth.", "lower_better"),
            MetricSpec("avg_max_repeat", "Average longest consecutive repeated non-Finish tool call.", "lower_better"),
            MetricSpec("no_finish_rate", "Tasks where the agent never called Finish.", "lower_better"),
            MetricSpec("give_up_rate", "Tasks ending in Finish(give_up_and_restart) or prune/give-up status.", "lower_better"),
            MetricSpec("hallucinated_function_rate", "Tasks with at least one nonexistent function call.", "lower_better"),
            MetricSpec("invalid_input_rate", "Tasks with at least one invalid action input or malformed Finish input.", "lower_better"),
            MetricSpec("tool_error_rate", "Tasks with any non-success ToolBench observation status.", "lower_better"),
            MetricSpec("transient_error_rate", "Tasks affected by timeout/rate-limit/request transient statuses.", "lower_better"),
            MetricSpec("repeat_call_rate", "Tasks with four or more consecutive calls to the same non-Finish tool.", "lower_better"),
            MetricSpec("over_calling_rate", "Tasks with more than eight non-Finish tool calls.", "lower_better"),
        ]


def is_transient_trace(trace: Trace) -> bool:
    """Return True when all observed tool errors are transient ToolBench errors."""
    codes = []
    for node in _all_nodes(trace.raw):
        code = node.get("observation_code")
        if isinstance(code, int) and code not in (0, 3):
            codes.append(code)
    return bool(codes) and all(code in _TRANSIENT_CODES for code in codes)


def extract_prompt_from_toolbench_result(path: str) -> str:
    """Extract the system prompt from a ToolBench result file.

    This is useful for creating the plain-text ``base_prompt_path`` consumed by
    ``ToolBenchPromptStore`` without importing ToolBench.
    """
    data = _safe_load_json(path)
    if not data:
        raise ValueError(f"not a valid ToolBench result JSON: {path}")
    for msg in _last_train_messages(data):
        if str(msg.get("role") or "").lower() == "system":
            return str(msg.get("content") or "").strip()
    raise ValueError(f"no system message found in ToolBench result: {path}")


def query_id_from_filename(path: str) -> str:
    """Public helper for callers that need ToolBench query ids."""
    return _task_id_from_path(path)


def method_from_filename(path: str) -> str:
    """Public helper for callers that need ToolBench method names."""
    return _method_from_path(path)


def make_toolbench_components(
    results_dir: str,
    *,
    tool_eval_labels: str = "",
    base_prompt_path: str = "",
    source_prompt_path: str = DEFAULT_STABLE_TOOLBENCH_PROMPT_PATH,
    versions_dir: str = "evolution_store/promptevo/toolbench/versions",
):
    """Build ToolBench prompt, trace, and metric adapter components.

    ToolBench rollout is intentionally not launched here. Run ToolBench or
    StableToolBench with their official pipeline, then point this factory at
    the saved JSON result directory and optional ToolEval labels.
    """
    prompts = ToolBenchPromptStore(
        versions_dir=versions_dir,
        base_prompt_path=base_prompt_path,
        source_prompt_path=source_prompt_path,
    )
    traces = ToolBenchTrajectorySource(results_dir_fn=lambda _exp: results_dir)
    metrics = ToolBenchMetricProvider(
        results_dir_fn=lambda _exp: results_dir,
        tool_eval_label_fn=lambda _exp: tool_eval_labels,
    )
    return prompts, traces, metrics
