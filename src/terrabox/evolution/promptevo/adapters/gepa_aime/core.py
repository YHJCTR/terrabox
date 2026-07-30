"""AIME prompt-optimization adapter inspired by GEPA's public example.

This adapter intentionally keeps the benchmark tiny and prompt-only:
- tasks come from the same cached HF datasets used by ``gepa.examples.aime``;
- the optimized slot is only the static math system prompt;
- rollout uses an injected OpenAI-compatible provider via Terrabox's
  ``make_llm_client`` abstraction;
- metrics are strict final-answer extraction against AIME integer answers.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from terrabox.agent.llm_provider import is_retryable_remote_error, make_llm_client
from terrabox.evolution.promptevo.interfaces import MetricSpec, Step, TaskMetric, Trace


DEFAULT_STATIC_PROMPT = (
    "You are a helpful assistant. Answer the question. "
    "Put your final answer in the format '### <answer>'"
)
DEFAULT_VERSIONS_DIR = "evolution_store/promptevo/gepa_aime/versions"
DEFAULT_EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
DEFAULT_HF_HOME = "/data1/yuhongjie2/hf_cache"
DEFAULT_HF_DATASETS_CACHE = "/data1/yuhongjie2/hf_datasets_cache"


@dataclass(frozen=True)
class AIMETask:
    task_id: str
    problem: str
    answer: str
    solution: str = ""
    source: str = ""


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.is_dir():
        target = target / "results.jsonl"
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _ensure_hf_env(offline: bool = True) -> None:
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE)
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_aime_tasks(split: str = "train", *, offline: bool = True) -> list[AIMETask]:
    """Load the same AIME data splits as GEPA's packaged example.

    GEPA's example shuffles 90 historical AIME examples with seed 0, uses the
    first half as train and the second half as val, then uses MathArena AIME
    2025 as test. The test split is not repeated here by default; repeated
    evaluation can be simulated by increasing trials outside this adapter.
    """

    _ensure_hf_env(offline=offline)
    from datasets import load_dataset

    split = split.lower()
    if split in {"train", "val", "validation"}:
        ds = load_dataset("AI-MO/aimo-validation-aime")["train"]
        rows = [
            AIMETask(
                task_id=f"aime_train_{row['id']}",
                problem=str(row["problem"]),
                answer=str(row["answer"]),
                solution=str(row.get("solution") or ""),
                source="AI-MO/aimo-validation-aime",
            )
            for row in ds
        ]
        random.Random(0).shuffle(rows)
        mid = len(rows) // 2
        return rows[:mid] if split == "train" else rows[mid:]
    if split == "test":
        ds = load_dataset("MathArena/aime_2025")["train"]
        return [
            AIMETask(
                task_id=f"aime_2025_{row['problem_idx']}",
                problem=str(row["problem"]),
                answer=str(row["answer"]),
                solution="",
                source="MathArena/aime_2025",
            )
            for row in ds
        ]
    raise ValueError(f"unknown AIME split: {split}")


def experiment_dir(name: str, output_dir: str = DEFAULT_EXPERIMENTS_DIR) -> str:
    return str(Path(output_dir, name).resolve())


def results_path(experiment: str) -> str:
    return str(Path(experiment_dir(experiment), "results.jsonl"))


def metrics_path(experiment: str) -> str:
    return str(Path(experiment_dir(experiment), "metrics.json"))


def _extract_answer(text: str) -> str:
    """Extract the final AIME answer from model text.

    Preferred contract is ``### <answer>``. Fallback extracts the last integer,
    because models often give a long derivation followed by the integer.
    """

    text = str(text or "")
    marked = re.findall(r"###\s*(-?\d+)", text)
    if marked:
        return marked[-1].lstrip("+")
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", text)
    if boxed:
        return boxed[-1].lstrip("+")
    nums = re.findall(r"(?<![\w.])-?\d+(?![\w.])", text)
    return nums[-1].lstrip("+") if nums else ""


def _correct(pred: str, gold: str) -> bool:
    return _extract_answer(pred) == str(gold).strip().lstrip("+")


class GEPAAIMEPromptStore:
    def __init__(self, versions_dir: str = DEFAULT_VERSIONS_DIR, base_prompt: str = DEFAULT_STATIC_PROMPT):
        self.versions_dir = versions_dir
        self.base_prompt = base_prompt

    def _path(self, version: str) -> str:
        return os.path.join(self.versions_dir, f"{version}.txt")

    def load(self, version: str) -> str:
        if version in {"base", "orig", "original"}:
            return self.base_prompt
        with open(self._path(version), encoding="utf-8") as f:
            return f.read().strip()

    def save(self, version: str, prompt: str, meta: dict) -> str:
        os.makedirs(self.versions_dir, exist_ok=True)
        path = self._path(version)
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        with open(path.replace(".txt", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return str(Path(path).resolve())


class GEPAAIMETrajectorySource:
    def __init__(self, result_path_fn=results_path):
        self._result_path = result_path_fn

    def traces(self, experiment: str) -> Iterable[Trace]:
        for row in _read_jsonl(self._result_path(experiment)):
            task_id = str(row.get("task_id") or "")
            prompt = str(row.get("problem") or "")
            output = str(row.get("output") or "")
            error = str(row.get("error") or "")
            steps = [Step(role="user", text=prompt)]
            if output:
                steps.append(Step(role="assistant", text=output))
            if error:
                steps.append(Step(role="assistant", text=error, errored=True))
            yield Trace(
                task_id=task_id,
                query=prompt,
                steps=steps,
                success=bool(row.get("correct")),
                final_answer=output,
                raw=row,
            )


class GEPAAIMEMetricProvider:
    def __init__(self, result_path_fn=results_path):
        self._result_path = result_path_fn

    def per_task(self, experiment: str) -> dict[str, TaskMetric]:
        out: dict[str, TaskMetric] = {}
        for row in _read_jsonl(self._result_path(experiment)):
            task_id = str(row.get("task_id") or "")
            if not task_id:
                continue
            success = bool(row.get("correct"))
            out[task_id] = TaskMetric(
                task_id=task_id,
                success=success,
                tool_f1=1.0 if success else 0.0,
                failure_flags={
                    "format_error": bool(row.get("status") == "completed" and not row.get("pred_answer")),
                    "provider_error": bool(row.get("status") == "provider_error"),
                    "runtime_error": bool(row.get("status") not in {"completed", "provider_error"}),
                },
                extra={
                    "pred_answer": row.get("pred_answer"),
                    "gold_answer": row.get("answer"),
                    "status": row.get("status"),
                },
            )
        return out

    def aggregate(self, experiment: str, task_ids: Optional[list[str]] = None) -> dict[str, Any]:
        metrics = self.per_task(experiment)
        if task_ids is not None:
            keep = {str(task_id) for task_id in task_ids}
            metrics = {key: value for key, value in metrics.items() if key in keep}
        rows = list(metrics.values())
        n = len(rows)
        total = n or 1
        return {
            "n": n,
            "accuracy": sum(row.success for row in rows) / total,
            "success_rate": sum(row.success for row in rows) / total,
            "format_error_rate": sum(row.failure_flags.get("format_error", False) for row in rows) / total,
            "provider_error_rate": sum(row.failure_flags.get("provider_error", False) for row in rows) / total,
            "runtime_error_rate": sum(row.failure_flags.get("runtime_error", False) for row in rows) / total,
            "tool_f1": sum(row.tool_f1 for row in rows) / total,
        }

    def metric_specs(self) -> list[MetricSpec]:
        return [
            MetricSpec("accuracy", "Exact AIME final-answer accuracy; higher is better.", "higher_better"),
            MetricSpec("success_rate", "Alias of exact-answer accuracy; higher is better.", "higher_better"),
            MetricSpec("format_error_rate", "Fraction of completed answers without an extractable integer; lower is better.", "lower_better"),
            MetricSpec("provider_error_rate", "External API/provider failure rate; lower is better.", "lower_better"),
            MetricSpec("runtime_error_rate", "Non-provider runtime failure rate; lower is better.", "lower_better"),
            MetricSpec("tool_f1", "PromptEvo compatibility alias equal to exact correctness for this no-tool benchmark.", "higher_better"),
        ]


class GEPAAIMERolloutRunner:
    def __init__(
        self,
        prompts: GEPAAIMEPromptStore | None = None,
        output_dir: str = DEFAULT_EXPERIMENTS_DIR,
        provider: str = "longcat",
        split: str = "train",
        limit: int = 0,
        max_tokens: int = 2048,
        offline: bool = True,
        retries: int = 3,
    ):
        self.prompts = prompts or GEPAAIMEPromptStore()
        self.output_dir = output_dir
        self.provider = provider
        self.split = split
        self.limit = limit
        self.max_tokens = max_tokens
        self.offline = offline
        self.retries = retries

    def _tasks(self, task_ids: Optional[list[str]] = None) -> list[AIMETask]:
        tasks = load_aime_tasks(self.split, offline=self.offline)
        if self.limit:
            tasks = tasks[: self.limit]
        if task_ids:
            keep = {str(task_id) for task_id in task_ids}
            tasks = [task for task in tasks if task.task_id in keep]
        return tasks

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        return self.run_prompt(prompt, experiment, task_ids=task_ids or None)

    def run_version(self, version: str, experiment: str, task_ids: Optional[list[str]] = None) -> str:
        return self.run_prompt(self.prompts.load(version), experiment, task_ids=task_ids)

    def run_prompt(self, prompt: str, experiment: str, task_ids: Optional[list[str]] = None) -> str:
        run_dir = Path(self.output_dir, experiment).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        result_file = run_dir / "results.jsonl"
        meta_file = run_dir / "meta.json"
        prompt_file = run_dir / "system_prompt.txt"
        prompt_file.write_text(prompt.strip() + "\n", encoding="utf-8")

        tasks = self._tasks(task_ids)
        existing = {str(row.get("task_id")) for row in _read_jsonl(result_file)}
        client = make_llm_client(self.provider)
        _write_json(
            meta_file,
            {
                "experiment": experiment,
                "provider": self.provider,
                "split": self.split,
                "limit": self.limit,
                "max_tokens": self.max_tokens,
                "total_tasks": len(tasks),
                "prompt_path": str(prompt_file),
                "source": "GEPA AIME cached datasets",
            },
        )

        for index, task in enumerate(tasks, start=1):
            if task.task_id in existing:
                continue
            user_prompt = (
                f"Solve this AIME math problem.\n\n{task.problem}\n\n"
                "Remember: put your final answer in the exact format ### <answer>."
            )
            row: dict[str, Any] = {
                "task_id": task.task_id,
                "problem": task.problem,
                "answer": task.answer,
                "source": task.source,
                "status": "started",
                "started_at": time.strftime("%F %T"),
            }
            last_error = ""
            for attempt in range(self.retries + 1):
                try:
                    output = client.call(user_prompt, system=prompt, max_tokens=self.max_tokens)
                    pred = _extract_answer(output)
                    row.update(
                        {
                            "status": "completed",
                            "output": output,
                            "pred_answer": pred,
                            "correct": _correct(output, task.answer),
                            "attempt": attempt + 1,
                            "finished_at": time.strftime("%F %T"),
                        }
                    )
                    break
                except Exception as exc:  # provider errors are retried; deterministic errors are recorded.
                    last_error = repr(exc)
                    if attempt < self.retries and is_retryable_remote_error(exc):
                        time.sleep(min(120.0, 5.0 * (attempt + 1)))
                        continue
                    row.update(
                        {
                            "status": "provider_error" if is_retryable_remote_error(exc) else "runtime_error",
                            "error": last_error,
                            "output": "",
                            "pred_answer": "",
                            "correct": False,
                            "attempt": attempt + 1,
                            "finished_at": time.strftime("%F %T"),
                        }
                    )
                    break
            _append_jsonl(result_file, row)
            done = len(existing) + index
            print(
                f"[{time.strftime('%F %T')}] {experiment}: {done}/{len(tasks)} "
                f"{task.task_id} status={row['status']} correct={row.get('correct')}",
                flush=True,
            )

        metrics = GEPAAIMEMetricProvider(result_path_fn=lambda _exp: str(result_file)).aggregate(experiment)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(
            run_dir / "run_status.json",
            {
                "status": "complete",
                "experiment": experiment,
                "results": str(result_file),
                "metrics": metrics,
                "finished_at": time.strftime("%F %T"),
            },
        )
        return str(run_dir)


def sample_stage1_traces(results_file: str, n_failed: int = 12, n_success: int = 4) -> str:
    from terrabox.evolution.promptevo.contrastive_sampler import default_render

    traces = list(GEPAAIMETrajectorySource(result_path_fn=lambda _exp: results_file).traces("current"))
    failed = [trace for trace in traces if not trace.success]
    success = [trace for trace in traces if trace.success]
    rng = random.Random(42)
    rng.shuffle(failed)
    rng.shuffle(success)
    picked = failed[:n_failed] + success[:n_success]
    rng.shuffle(picked)
    return "\n\n".join(default_render(trace, 2200) for trace in picked)
