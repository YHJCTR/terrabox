# promptevo adapter template

Copy this directory when adding a new external project adapter.

## Required pieces

`TemplatePromptStore`

- Loads the original static prompt for `base` / `orig` / `original`.
- Saves promptevo-generated prompt versions under `evolution_store/`.
- Must only store the static prompt slot being optimized.
- Must not include dynamic tool descriptions, retrieved context, task input,
  conversation history, memory, or API responses.

`TemplateTrajectorySource`

- Reads existing project logs/result files.
- Converts each task into `Trace(task_id, query, steps, success, final_answer, raw)`.
- Should preserve original project metadata in `raw` for debugging.
- Should expose stable `task_id` values that match `MetricProvider.per_task()`
  when metrics are implemented.
- Should convert logs first, then let promptevo render `Trace` objects. Do not
  feed project-specific raw JSON directly into generic trace samplers unless
  the sampler explicitly supports that raw format.

These two are enough for first-stage promptevo prompt proposal.

## Optional pieces

`TemplateMetricProvider`

- Needed for rich reports, base/v1 comparisons, and metric-aware second-stage
  prompt updates.
- Can be a thin wrapper around the project's own evaluator output.
- Should describe metric meanings/directions through `metric_specs()`.
- Should include enough aggregate metrics to reveal tradeoffs, not just one
  final score. Useful examples are success, parse/format validity, action/tool
  selection, argument/value correctness, answer quality, latency, and error
  buckets when the project exposes them.

`TemplateRolloutRunner`

- Needed only if Terrabox should launch the external benchmark itself.
- Must apply prompt overrides only for the current experiment.
- Must not edit or replace the external project's original static prompt.
- Should write experiment-local outputs under this adapter's `experiments/`
  directory and should record prompt version, model endpoint, data path, shard
  info, and output paths in metadata.

## Suggested project directory

```text
adapters/<project>/
  __init__.py
  prompts.py      # required PromptStore
  traces.py       # required TrajectorySource
  metrics.py      # optional MetricProvider
  runner.py       # optional RolloutRunner
  files.py        # optional file helpers
  experiments/    # ignored experiment outputs for this adapter
  README.md
```

Small adapters may keep everything in `components.py`; larger adapters should
split responsibilities like the API-Bank adapter. If a larger adapter needs a
compatibility import surface, add a tiny `core.py` that only re-exports the
split modules.

## Static prompt slot checklist

Before implementing a new adapter, identify the exact static text slot being
optimized:

- It is written in code/config and reused across tasks.
- It describes general behavior, output contract, or role instructions.
- It does not contain the current user request, retrieved documents, tool/API
  schemas, tool observations, conversation history, or benchmark gold labels.
- It can be temporarily overridden for one experiment without changing the
  original project source file.

If a project builds a prompt from several static fragments, prefer exposing the
smallest complete fragment that can be safely replaced. If replacement is not
possible, document the fragment boundaries and implement a runner that injects
the evolved text only into that slot for the current experiment.
