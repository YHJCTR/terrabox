"""Template promptevo adapter for an external agent project.

Use this file as a starting point for a new project adapter. The minimum
integration for prompt evolution is:

1. `TemplatePromptStore`: load the original static prompt and save versions.
2. `TemplateTrajectorySource`: convert project logs into generic `Trace`s.

Everything else is optional:

- `TemplateMetricProvider` is useful when you want A/B comparisons and
  second-stage updates driven by metrics.
- `TemplateRolloutRunner` is useful when this repository should launch the
  external project. If the external project already has an official runner,
  keep this class absent or tiny, and document the official command instead.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from ...interfaces import MetricSpec, Step, TaskMetric, Trace


DEFAULT_STATIC_PROMPT = """Replace this with the external project's original static instruction."""
TEMPLATE_ADAPTER_DIR = os.path.dirname(__file__)
DEFAULT_TEMPLATE_EXPERIMENTS_DIR = os.path.join(TEMPLATE_ADAPTER_DIR, "experiments")


class TemplatePromptStore:
    """Required: read/write static prompt versions.

    Keep this narrow: store only the static task/system instruction that
    promptevo is allowed to optimize. Do not save dynamic tool descriptions,
    retrieved context, user requests, memory, or conversation history here.
    """

    def __init__(
        self,
        versions_dir: str = "evolution_store/promptevo/template_project/versions",
        base_prompt_path: str = "",
        base_prompt: str = DEFAULT_STATIC_PROMPT,
    ):
        self.versions_dir = versions_dir
        self.base_prompt_path = base_prompt_path
        self.base_prompt = base_prompt

    def _path(self, version: str) -> str:
        return os.path.join(self.versions_dir, f"{version}.txt")

    def load(self, version: str) -> str:
        """Required by `PromptStore`.

        Versions `base`, `orig`, and `original` should return the unmodified
        project prompt. Other versions should be promptevo-generated files.
        """
        if version in {"base", "orig", "original"}:
            if self.base_prompt_path:
                with open(self.base_prompt_path, encoding="utf-8") as f:
                    return f.read().strip()
            return self.base_prompt.strip()
        with open(self._path(version), encoding="utf-8") as f:
            return f.read().strip()

    def save(self, version: str, prompt: str, meta: dict) -> str:
        """Required by `PromptStore`."""
        os.makedirs(self.versions_dir, exist_ok=True)
        path = self._path(version)
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        with open(path.replace(".txt", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return os.path.abspath(path)


class TemplateTrajectorySource:
    """Required: convert project logs into generic promptevo traces."""

    def __init__(self, results_path_fn=lambda experiment: experiment):
        self._results_path = results_path_fn

    def traces(self, experiment: str) -> Iterable[Trace]:
        """Required by `TrajectorySource`.

        Yield one `Trace` per task. A useful trace should include:

        - `task_id`: stable task identifier.
        - `query`: user request/task statement.
        - `steps`: ordered user/assistant/tool events.
        - `success`: whether the run succeeded.
        - `final_answer`: final model answer if present.
        - `raw`: original row/file metadata for debugging.
        """
        path = self._results_path(experiment)
        raise NotImplementedError(
            f"Load project logs from {path!r} and yield Trace objects here."
        )


class TemplateMetricProvider:
    """Optional: compute task-level and aggregate metrics from result files.

    Implement this when you want promptevo's second stage or reports to compare
    base/v1/v2 runs using project-specific metrics. If the project already
    emits a summary JSON, this class can be a thin reader around that file.
    """

    def __init__(self, results_path_fn=lambda experiment: experiment):
        self._results_path = results_path_fn

    def per_task(self, experiment: str) -> dict[str, TaskMetric]:
        """Optional but required when implementing `MetricProvider`.

        Return one `TaskMetric` per task. Put project-specific details in
        `TaskMetric.extra`; put boolean failure buckets in `failure_flags`.
        """
        raise NotImplementedError

    def aggregate(
        self,
        experiment: str,
        task_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Optional but required when implementing `MetricProvider`."""
        metrics = self.per_task(experiment)
        if task_ids is not None:
            keep = set(task_ids)
            metrics = {k: v for k, v in metrics.items() if k in keep}
        vals = list(metrics.values())
        n = len(vals) or 1
        return {
            "n": len(vals),
            "success_rate": sum(m.success for m in vals) / n,
            "tool_f1": sum(m.tool_f1 for m in vals) / n,
        }

    def metric_specs(self) -> list[MetricSpec]:
        """Optional but recommended: describe each aggregate metric."""
        return [
            MetricSpec("success_rate", "Fraction of tasks marked successful.", "higher_better"),
            MetricSpec("tool_f1", "Mean tool-call F1 when available.", "higher_better"),
        ]


class TemplateRolloutRunner:
    """Optional: run the external project with a temporary prompt override.

    Only implement this when this repository should launch the benchmark. It
    must not overwrite the external project's original static prompt. The usual
    pattern is to write an experiment-local prompt file or pass a prompt string
    through an environment variable/CLI flag.
    """

    def __init__(self, prompts: TemplatePromptStore, output_dir: str = DEFAULT_TEMPLATE_EXPERIMENTS_DIR):
        self.prompts = prompts
        self.output_dir = output_dir

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        """Optional `RolloutRunner` method.

        Save `prompt` under a temporary version and launch the project runner.
        Return the experiment name or output path consumed by
        `TemplateTrajectorySource` and `TemplateMetricProvider`.
        """
        version = f"{experiment}_prompt"
        self.prompts.save(version, prompt, {"source": "TemplateRolloutRunner.run"})
        raise NotImplementedError(
            "Launch the external project with this prompt version without "
            "modifying the original source prompt."
        )


def make_template_components(
    results_path: str,
    versions_dir: str = "evolution_store/promptevo/template_project/versions",
):
    """Optional convenience factory mirroring concrete adapters."""
    prompts = TemplatePromptStore(versions_dir=versions_dir)
    traces = TemplateTrajectorySource(results_path_fn=lambda exp: results_path)
    metrics = TemplateMetricProvider(results_path_fn=lambda exp: results_path)
    runner = TemplateRolloutRunner(prompts)
    return prompts, traces, metrics, runner
