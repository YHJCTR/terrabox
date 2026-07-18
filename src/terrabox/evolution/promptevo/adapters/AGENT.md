# promptevo adapters agent instructions

This directory contains optional adapters that let promptevo optimize static
prompt slots in external agent or benchmark projects.

## Adapter boundaries

- Keep adapters thin and file-based when possible.
- Do not put project-specific logic in `promptevo` core modules.
- Do not hard-code local checkout paths as the only option; accept paths through
  constructors or factory functions.
- Do not import third-party project code unless reading files is insufficient.
- Do not overwrite an external project's original prompt source file during an
  experiment.

## Static prompt slot

Promptevo optimizes only static behavior instructions. A valid adapter prompt
slot:

- is reused across tasks;
- describes role, behavior, constraints, output contract, or task policy;
- can be temporarily overridden for one experiment;
- does not include user input, dialogue history, retrieved documents, tool/API
  schemas, tool observations, memory, API responses, or gold labels.

If a project builds prompts from several static fragments, expose the smallest
complete static fragment that can be safely replaced. If the project cannot
replace that fragment directly, document the boundaries and implement an
experiment-local injection in the runner.

## Required adapter surface

- `PromptStore`: `load("base")` returns the unmodified static instruction;
  `save(version, prompt, meta)` writes a new version under
  `evolution_store/promptevo/<project>/versions/`.
- `TrajectorySource`: converts project logs/results into generic `Trace`
  objects. Stable `task_id` values must match `MetricProvider.per_task()` when
  metrics exist.

These two are enough for first-stage prompt proposal.

## Optional adapter surface

- `MetricProvider`: required for rich comparison and second-stage optimization.
  Expose task-level metrics, aggregate metrics, and `metric_specs()` with clear
  directions such as `higher_better`, `lower_better`, or `neutral`.
- `RolloutRunner`: required only when this repo should launch the benchmark.
  Apply prompt overrides only for the current experiment and write outputs under
  the adapter's `experiments/` directory.
- `tau2_bench` and `agentdojo` both expose optional runners. Keep their
  experiment-local prompt overrides and dependency preflight checks intact:
  missing external-project dependencies should produce `run_status.json`
  explaining the blocker, not fake metrics or source-tree edits.
- `toolbench` exposes a StableToolBench runner for the local
  `/data1/yuhongjie2/StepTool/stabletoolbench` checkout. It must only override
  `Prompts.ReAct_prompts.FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION` in the current
  process before importing the official pipeline; do not edit or copy the
  external source tree. The formal pipeline may explicitly start the cached API
  server and per-GPU vLLM lanes, must record those resources in experiment
  metadata/logs, and must stop resources it started at the end of each stage.

## Trace rendering

- Convert project logs to `Trace` first, then render with promptevo renderers.
- Do not feed project-private raw JSON directly into generic trace samplers
  unless the sampler explicitly supports that format.
- Preserve useful raw metadata in `Trace.raw` for debugging, but keep optimizer
  input compact.

## Experiment outputs

- Adapter-owned experiments should live under
  `src/terrabox/evolution/promptevo/adapters/<project>/experiments/`.
- Generated prompt versions should live under
  `evolution_store/promptevo/<project>/versions/`.
- Each runnable experiment should record prompt version, model endpoint, data
  path, shard info, prediction path, rollout path, and metric summary.
- Do not delete or overwrite previous experiment directories when searching meta
  prompts; create a new timestamped directory.
- New promptevo runs should use one shared stem:
  `promptevo_<dataset>_<stage>_<variant>_<YYYYMMDD_HHMMSS>`. Use the same stem
  for the prompt `.txt`, adapter experiment directory, and rollout experiment
  directory when possible. Do not rename historical `promptevo_v*` or other old
  outputs; this convention applies to new runs only.

## Terrabox/OEA adapter

- Use `adapters/terrabox/` for native OEA promptevo work instead of ad hoc
  `tmp/` conversion scripts.
- `results/<task_id>.json` under `tmp/trajectories/<experiment>/standard/results`
  is the source of truth. Rebuild `trajectories_full.jsonl` through
  `terrabox.files.rebuild_trajectory_files()` when needed.
- The optimized static prompt slot is `_REACT_SYSTEM_PROMPT`; apply versions
  with `TERRABOX_REACT_SYSTEM_PROMPT_FILE`, not `--evolution-method`.
- If a tool implementation changed, do not trust a full task count alone. For
  SAM2 refreshes, backup and rerun the union of tasks whose gold tools include
  `geo_perception.sam2_segment` and tasks whose old result actually called it.
  Move stale JSON files into a timestamped `results/bak_*` directory, then rerun
  the same experiment with `--resume`.

## Documentation sync

When changing an adapter interface, runner behavior, metric meaning, experiment
layout, GPU/service policy, or static prompt slot, update:

- this file and `CLAUDE.md` with the same operational rule when the change affects
  future coding agents;
- `adapters/README.md` for cross-adapter behavior;
- the project adapter's own `README.md` or `usage.md` for project-specific
  commands and outputs.
