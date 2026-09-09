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

## 类型化协议补丁（可选）

- adapter 不得向 PromptEvo core 注入项目专属的 patch 类型、工具名、任务 ID、实体、路径或
  固定 workflow。元提示词禁止这些内容，编译器会确定性拒绝任务 ID、路径和已知 benchmark
  标识。可用类型仅为 `tool_selection`、`argument_validation`、`error_recovery`、
  `termination_and_repetition`，其语义必须跨该 adapter 的未见任务通用。
- patch 模式仍只改静态 prompt slot；不得把动态 schema、对话历史、工具 observation、gold
  label 或 evaluator 输出编译进 prompt。
- adapter 若提供 `RolloutRunner`，必须支持固定 `dev_task_ids` 的真实 rollout。仅当该 rollout
  的指标通过接受门时，候选才能标记为 accepted；没有 runner 时只允许生成提案，不能声称验证
  成功。

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
- `gepa_aime` exposes a cached-HuggingFace AIME prompt-only runner aligned with GEPA's public example. It optimizes only the static math system prompt, keeps AIME solutions out of agent inputs, writes `results.jsonl` + `metrics.json` under `gepa_aime/experiments/<group>/<stage>/`, and should use `HF_HUB_OFFLINE=1` with the local cache unless the user explicitly wants network downloads.
- `toolbench` exposes a StableToolBench runner for the local
  `/data1/yuhongjie2/StepTool/stabletoolbench` checkout. It must only override
  `Prompts.ReAct_prompts.FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION` in the current
  process before importing the official pipeline; do not edit or copy the
  external source tree. The formal pipeline may explicitly start the cached API
  server and per-GPU vLLM lanes, must record those resources in experiment
  metadata/logs, and must stop resources it started at the end of each stage.

## External API pacing

- Adapters that call LongCat/DeepSeek through `make_llm_client()` inherit the
  shared `RemoteChatClient` pacing and retry behavior. Keep it configurable with
  `TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS`, provider-specific overrides such
  as `TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS`, and `TERRABOX_REMOTE_LLM_RATE_LOCK_DIR`;
  do not hard-code permanent serial execution. LongCat defaults to conservative
  pacing because multi-turn agent tasks can burst requests even at low job
  concurrency; set the interval to `0` only for a deliberate high-concurrency
  rerun.
- LongCat-heavy adapters must share the same remote-provider lock and JSON
  fairness state. Set `TERRABOX_REMOTE_LLM_WORKLOAD` to a stable workload name
  such as `tau2` or `experienceevo`; when both workloads are waiting, the shared
  pacer rotates request grants by workload. If only one workload is waiting, it
  may continue using the configured provider interval. Do not create a separate
  adapter lock to bypass the service-level LongCat limit, and restart the
  watcher/rollout parent after changing pacing or workload env vars.
- Adapters that bypass `RemoteChatClient` and call an upstream OpenAI-compatible
  SDK directly must implement equivalent cross-process pacing and finite queue
  retries. A single worker can still issue many rapid API calls inside one
  agent sample, so do not assume `API_WORKERS=1` is enough to avoid 429s. Route
  direct SDK calls through the same shared provider pacer; adapter-specific env
  names may remain only as compatibility aliases for interval settings, not as
  an independent rate-limit lock.

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
- API-Bank rollout uses task-level durable JSONL writes. On `--resume`, a task
  is complete only when both `predictions.jsonl` and `rollout.jsonl` contain
  the same `(file, id)` key; an interrupted half-pair is rerun. Do not replace
  these files with a deferred end-of-stage bulk write.
- For low-resource overnight chains, API-Bank's `pipeline.py` may run before
  `gepa_aime`, because both use only an external provider and no Terrabox GPU
  tool service. The watcher must start the next experiment only after the
  preceding pipeline reports `status=complete` *and* every Base/Stage1/Stage2
  API-Bank stage has full prediction coverage. A stopped or failed pipeline is
  terminal for that chain: do not treat partial JSONL output as completion and
  do not automatically start OEA, tau2-bench, AgentDojo, or ToolBench.

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

- this file and `AGENT.md` with the same operational rule when the change affects
  future coding agents;
- `adapters/README.md` for cross-adapter behavior;
- the project adapter's own `README.md` or `usage.md` for project-specific
  commands and outputs.
