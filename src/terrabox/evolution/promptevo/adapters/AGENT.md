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
  `query_abstraction`、`termination_and_repetition`、`answer_contract`、`tool_output_security`、`candidate_metric_guard`、
  `state_transition`，其语义必须跨该 adapter 的未见任务通用。
- API-Bank 可通过 `--task-profile tool_search` 提供任务类型级指导，限定优化器检查工具发现、
  下游调用、可观察依赖状态、精确参数键和语义搜索词；该 profile 不得注入具体 API、task id、gold 调用或固定
  认证 workflow，结果必须标为 `PromptEvo-ToolSearch-adapted`，并与 vanilla PromptEvo 分开报告。
- patch 模式仍只改静态 prompt slot；不得把动态 schema、对话历史、工具 observation、gold
  label 或 evaluator 输出编译进 prompt。
- API-Bank 可用 `--promptevo-protocol-mode conditional` 启用条件协议编译：保留 typed patch
  的 `trigger`、`rule`、`scope`，并用逐任务配对效果保护 baseline 成功、要求至少一个 exact-call
  gain；`query_abstraction` 会被强制限制为 `discovery/query fields only`，不得把查询规则扩大
  成全局实体值改写；该模式不写入具体 API、样本、认证流程或轨迹示例，必须在 manifest/provenance
  中单独标注。
- API-Bank 的 `--promptevo-patch-composition atomic_pairwise` 是 PromptEvo 的结构化组合搜索：
  保留 LLM 产出的 typed patches，逐条验证单 patch/两两组合并记录 lineage，整组候选只作高阶交互回退；
  该模式属于 PromptEvo 的方法变体，必须在 manifest/provenance 和结果表中与默认 whole 模式分开。
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
- `gepa_aime` exposes a cached-HuggingFace AIME prompt-only runner aligned with GEPA's public example. It optimizes only the static math system prompt, keeps AIME solutions out of agent inputs, writes `results.jsonl` + `metrics.json` under `tmp/promptevo_gepa_aime_experiments/<group>/<stage>/`, and should use `HF_HUB_OFFLINE=1` with the local cache unless the user explicitly wants network downloads.
- `toolbench` exposes a StableToolBench runner for the local
  `/data1/yuhongjie2/StepTool/stabletoolbench` checkout. It must only override
  `Prompts.ReAct_prompts.FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION` in the current
  process before importing the official pipeline; do not edit or copy the
  external source tree. The formal pipeline may explicitly start the cached API
  server and per-GPU vLLM lanes, must record those resources in experiment
  metadata/logs, and must stop resources it started at the end of each stage.
  Paper-style ToolBench chains must use the adapter's deterministic
  evolution/dev/test split manifest: both Stage1 and Stage2 require held-out
  dev rollouts for candidate acceptance, and the test split cannot be used to
  choose a prompt.

## External API pacing

- API-Bank 按用户新要求在 prompt 冻结后追加全部389条评测，保留79条独立test；前者包含优化数据，不可标独立泛化结果。旧进程由 `compare-methods --follow-full-evaluations` 补评，复用公共限流；`full_set_summary.json` 才是全量完成标志。

- API-Bank 外部对比入口 `api_bank.pipeline compare-methods` 使用对话级固定划分、公共 LongCat 限流和逐任务缓存。test 不参与优化，旧全量 389 条结果不能代替新 test 的 Base。方法适配差异记录于 `provenance.json` 和 API-Bank README，不得把导入检查算实验或把 adapted 算完整官方复现。账户错误停止，修改节流环境须重启父进程。

- Adapters that call LongCat/DeepSeek through `make_llm_client()` inherit the
  shared `RemoteChatClient` pacing and retry behavior. Keep it configurable with
  `TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS`, provider-specific overrides such
  as `TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS`, and `TERRABOX_REMOTE_LLM_RATE_LOCK_DIR`;
  do not hard-code permanent serial execution. LongCat defaults to conservative
  pacing because multi-turn agent tasks can burst requests even at low job
  concurrency; set the interval to `0` only for a deliberate high-concurrency
  rerun.
- LongCat-heavy adapters must share the same remote-provider lock. Leave
  `TERRABOX_REMOTE_LLM_WORKLOAD` unset for a single PromptEvo LongCat run, so the
  provider interval is the only pacing policy. Set a stable workload name only
  when multiple workload families must fairly share LongCat; then the shared
  pacer rotates request grants by workload. Do not create a separate adapter
  lock to bypass the service-level LongCat limit, and restart the watcher/rollout
  parent after changing pacing or workload env vars.
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
  `tmp/promptevo_<project>_experiments/`.
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

- this file and `CLAUDE.md` with the same operational rule when the change affects
  future coding agents;
- `adapters/README.md` for cross-adapter behavior;
- the project adapter's own `README.md` or `usage.md` for project-specific
  commands and outputs.
