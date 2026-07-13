# ToolBench adapter

Static instruction slot:

- ToolBench prompts are usually assembled by its inference pipeline and method
  configuration rather than a single constant. Treat the selected system/task
  instruction as the prompt version and keep dynamic API documentation,
  retrieved tools, user query, and rollout history out of the saved prompt.

Local data:

- Source checkout: `/data1/yuhongjie2/ToolBench`
- Typical result files are JSON outputs from ToolBench inference/evaluation
  runs, for example files produced by `toolbench/inference/qa_pipeline.py`.

What this adapter currently does:

- Reads result JSON files without importing ToolBench.
- Converts CoT/DFS/DFSDT traces into generic promptevo `Trace` objects.
- Reports task success, pass/fail flags, tool-call counts, error-code buckets,
  transient-error rates, and final-answer availability from saved result files.

Minimal usage:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.toolbench import (
    ToolBenchMetricProvider,
    ToolBenchTrajectorySource,
)

results = "/path/to/toolbench/results"
traces = ToolBenchTrajectorySource(results_dir_fn=lambda exp: results)
metrics = ToolBenchMetricProvider(results_dir_fn=lambda exp: results)

print(metrics.aggregate("existing_run"))
print(next(iter(traces.traces("existing_run")), None))
PY
```

This adapter is deliberately read-only. Run ToolBench with its official
pipeline, then point promptevo at the saved results. If later we need a
first-class rollout runner, keep it in this project directory rather than in
the core promptevo optimizer.
