# tau2-bench adapter

This adapter lets promptevo optimize the static instruction slot of
tau2-bench customer-service agents while keeping dynamic domain policy, tools,
task state, and dialogue history outside the optimized prompt.

## Local project

- Source: `/data1/yuhongjie2/tau2-bench`
- Adapter: `src/terrabox/evolution/promptevo/adapters/tau2_bench/`
- Adapter experiments:
  `src/terrabox/evolution/promptevo/adapters/tau2_bench/experiments/`
- Prompt versions:
  `evolution_store/promptevo/tau2_bench/versions/`

## Static prompt boundary

Optimized slot:

- `/data1/yuhongjie2/tau2-bench/src/tau2/agent/llm_agent.py::AGENT_INSTRUCTION`

Do not optimize or replace:

- `domain_policy`
- user scenario or dialogue history
- tool schemas
- tool observations
- task gold actions or reward labels

tau2 builds the runtime system prompt as:

```text
<instructions>
{AGENT_INSTRUCTION}
</instructions>
<policy>
{domain_policy}
</policy>
```

## Data scale

Local checkout size is about 844M. Existing official/example results under
`data/tau2/results` are about 577M.

Text task counts:

- `mock`: 10
- `airline`: 50
- `retail`: 114
- `telecom`: `small` 20, `base` 114, `full` 2285
- `banking_knowledge`: 97

The normal text/base non-mock scale is roughly 375 tasks:
`airline 50 + retail 114 + telecom base 114 + banking_knowledge 97`.

## Adapter modules

- `prompts.py`: required `PromptStore`; reads the original static instruction
  and writes versioned static prompts.
- `traces.py`: required `TrajectorySource`; converts tau2 Results JSON files
  into generic promptevo traces.
- `metrics.py`: optional `MetricProvider`; computes per-task and aggregate
  reward metrics.
- `runner.py`: optional rollout runner; launches tau2 text runs and mirrors
  results into this adapter's `experiments/` directory.
- `core.py`: backward-compatible re-export.

## No-model smoke test

```bash
cd /data1/yuhongjie2/terrabox
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.tau2_bench import (
    Tau2MetricProvider,
    Tau2PromptStore,
    Tau2TrajectorySource,
)

root = "/data1/yuhongjie2/tau2-bench"
result = root + "/data/tau2/results/final/gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json"
prompts = Tau2PromptStore(tau2_root=root)
metrics = Tau2MetricProvider(results_path_fn=lambda exp: result)
traces = Tau2TrajectorySource(results_path_fn=lambda exp: result)
print(prompts.load("base")[:120])
print(metrics.aggregate("existing"))
print(next(iter(traces.traces("existing"))).task_id)
PY
```

## Metrics

Primary metrics include:

- `success_rate`: full reward rate
- `avg_reward`: mean final tau2 reward
- `avg_db_reward`: final DB/end-state reward
- `avg_communicate_reward`: required communication reward
- `avg_action_reward`: action reward when the task uses ACTION in reward basis
- `db_failure_rate`, `communicate_failure_rate`, `action_failure_rate`
- `max_steps_rate`, `error_termination_rate`
- `avg_messages`, `avg_tool_calls`, `avg_duration_s`
- `macro_domain_reward`, `worst_domain_reward`

Do not treat the reference `evaluation_criteria.actions` as a required action
path unless tau2's `RewardType.ACTION` is in `reward_basis`. For airline,
retail, and normal telecom tasks, scoring mainly depends on final DB state and
required communication.

## Native run

tau2 requires its own core dependencies. The upstream recommended setup is:

```bash
cd /data1/yuhongjie2/tau2-bench
uv sync
```

If `uv` is unavailable, install it first or create an equivalent Python
environment with the dependencies in `pyproject.toml`.

A small local OpenAI-compatible run should use LiteLLM's OpenAI provider with a
local `api_base`:

```bash
cd /data1/yuhongjie2/terrabox
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.tau2_bench import (
    Tau2PromptStore,
    Tau2RolloutRunner,
    Tau2RunConfig,
)

root = "/data1/yuhongjie2/tau2-bench"
prompt = Tau2PromptStore(tau2_root=root).load("base")
runner = Tau2RolloutRunner(tau2_root=root)
cfg = Tau2RunConfig(
    domain="mock",
    num_tasks=1,
    num_trials=1,
    max_steps=40,
    max_concurrency=1,
    agent_llm="openai/local-qwen3-8b",
    user_llm="openai/local-qwen3-8b",
    agent_llm_args={"temperature": 0.0, "api_base": "http://localhost:9100/v1", "api_key": "EMPTY"},
    user_llm_args={"temperature": 0.0, "api_base": "http://localhost:9100/v1", "api_key": "EMPTY"},
)
print(runner.run(prompt, experiment="native_mock_1task_base", run_config=cfg))
PY
```

The runner does not edit tau2 source files. It writes the active static prompt
to the adapter experiment directory and patches `AGENT_INSTRUCTION` only inside
the launched process.

Native runs default to tau2's `--auto-resume`. Reusing the same experiment name
keeps `tmp/tau2_runtime/<experiment>/simulations/promptevo_<experiment>/` and
continues missing simulations. The adapter refuses to resume if the saved
static instruction differs from the requested prompt. Set
`Tau2RunConfig(auto_resume=False)` only when intentionally replacing that
experiment from scratch.

The adapter does not write into the tau2 checkout. It sets `TAU2_DATA_DIR` to
`tmp/tau2_runtime/<experiment>/`, links that runtime directory's read-only
`tau2/` input to the upstream dataset, and writes resumable simulation
checkpoints under its local `simulations/`. Completed results are mirrored to
the adapter experiment's `tau2_results/` directory.

## Three-stage pipeline and optional Stage3

`pipeline.py chain-after-base` continues from an existing Base group, while
`pipeline.py full-chain` is the formal Base -> Stage1 -> Stage2 entry. Both
commands require an explicit experiment group for every stage and accept a
named rollout profile.

历史正式口径里的“三阶段”指 Base / Stage1 / Stage2。新增的 Stage3 是可选的
post-Stage2 refinement，不替代 Stage2：它以 Stage2 prompt 为当前基线，比较
Stage1 与 Stage2 的同任务轨迹，只提出小幅 protocol patch 来修复 Stage2 引入的回归，
并继续使用同一固定真实 dev slice 验证；候选没有通过接受门时保留 Stage2。
可通过 `pipeline.py stage3-after-stage2` 单独接在已完成的 Stage2 后，也可在
`chain-after-base` / `full-chain` 中显式传入 `--stage3-group` 与
`--stage3-version` 自动接上。Stage3 的可恢复检查点为 `stage3_protocol_patch.json`。

- `legacy4` preserves the first four-domain experiment: local Qwen3 8B agent
  and user, one trial, `max_steps=80`, `max_tokens=512`, and offline BM25 for
  banking knowledge.
- `paper3` is the paper-comparable three-domain setup: airline/retail/telecom,
  local Qwen3 8B no-think agent, fixed LongCat2 no-think user simulator, four
  trials, `max_steps=100`, and `max_tokens=2048`. It follows the original
  domain/trial structure but is not a reproduction of the paper's frontier
  agent or GPT user-simulator models.
- `stable4` is the recommended reliable full-run profile: all four domains on
  GPU0-3, local Qwen3 8B for both agent and user simulator, four trials,
  `max_steps=100`, and `max_tokens=2048`. The external-user `paper3` profile
  remains experimental until every submitted domain has created `run_meta.json`
  and received inference requests, not just passed `/health`.
- `longcat_agent4` is the explicit external-agent profile: LongCat2 runs both
  the agent and user simulator through the OpenAI-compatible API, uses the same
  four-domain/four-trial/100-step setup, dynamic chunks, and no local vLLM
  containers. Credentials stay in the child environment and out of metadata.
  `stable4` uses dynamic chunk scheduling: GPU/port pairs are reusable lanes,
  tasks are split into small `task_ids` chunks, and early-finished lanes pull
  more work from the remaining queue. Chunk outputs are merged back into the
  canonical domain `tau2_results/results.json` with `(task_id, trial, seed)`
  de-duplication, so do not run multiple processes against the same domain
  result file directly.
- External-provider 429, rate-limit, timeout, and provider-overload failures in
  dynamic chunks are retryable queue events. The failed chunk waits briefly and
  is put at the back of the queue; `TERRABOX_TAU2_QUEUE_RETRIES` (default 12)
  caps this so real code/data bugs still surface.
- tau2 external-agent runs call LongCat/DeepSeek through LiteLLM inside tau2
  subprocesses, so they bypass Terrabox `RemoteChatClient`. The bootstrap must
  route those calls through `pace_remote_llm_request()` with the shared
  `TERRABOX_REMOTE_LLM_RATE_LOCK` and JSON fairness state. The pipeline injects
  `TERRABOX_REMOTE_LLM_WORKLOAD=tau2`; concurrent ExperienceEvo runs should use
  `TERRABOX_REMOTE_LLM_WORKLOAD=experienceevo` so LongCat request grants rotate
  by workload when both are waiting. `TERRABOX_TAU2_API_MIN_INTERVAL_SECONDS`
  remains a compatibility alias for the provider interval, but tau2 must not use
  an independent lock to bypass the service-level limit. `longcat_agent4`
  defaults to conservative `api_workers=1` / `run_concurrency=1`; raise
  `TERRABOX_TAU2_API_WORKERS`, `TERRABOX_TAU2_RUN_CONCURRENCY`, and/or set the
  interval to `0` only for an intentional high-concurrency rerun. Changing these
  env vars requires restarting the watcher or rollout parent process.

The external user key is loaded from gitignored provider configuration and is
passed only through the child environment. It must not appear in command-line
arguments, `run_meta.json`, or copied tau2 result metadata.

LongCat is forced to no-think for judging and prompt optimization. Rejudging
never overwrites native tau2 output: caches and materialized results live in
each experiment group's `rejudged_longcat/`. Stage1 sampling and Stage2 pairing
consume those copies. Multi-domain task keys include the domain
(`domain::task_id::trialN`) so equal numeric IDs cannot collide.

The external rejudge contract is compact and index-based. It does not ask the
model to repeat long assertion strings inside JSON, validates boolean verdicts
strictly, retries malformed responses, and preserves successful per-item
caches when a chain resumes. Only tasks whose `reward_basis` contains
`NL_ASSERTION` are sent to the provider; assertions on DB/COMMUNICATE-only tasks
cannot change reward and are reported as `skipped_non_scoring`. Common wrapper
keys and multiple top-level row objects are accepted. Exhausted parse retries
are written to `judge_failures/`, preserve the original reward info, and remain
visible in `rejudge_summary.json` instead of stopping the complete stage chain.

The selected rollout profile remains fixed across all three stages. Stage
containers are always released in a `finally` block.

Example paper-comparable full chain:

```bash
PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python -m \
  terrabox.evolution.promptevo.adapters.tau2_bench.pipeline full-chain \
  --profile paper3 \
  --base-group tau2_qwen3_8b_longcat_user_paper3_base_YYYYMMDD \
  --stage1-group tau2_qwen3_8b_longcat_user_paper3_stage1_YYYYMMDD \
  --stage2-group tau2_qwen3_8b_longcat_user_paper3_stage2_YYYYMMDD \
  --stage1-version tau2_qwen3_8b_longcat_user_paper3_stage1_YYYYMMDD \
  --stage2-version tau2_qwen3_8b_longcat_user_paper3_stage2_YYYYMMDD \
  --provider longcat
```

For thinking models such as Qwen3, the runner also strips `<think>...</think>`
from generated messages before they are appended to tau2 agent/user dialogue
state. The raw provider payload remains in `raw_data`, but the visible
conversation history and promptevo traces should not include private thinking
text.

## Local judge and context notes

tau2 has evaluator-side LLM calls that are separate from the agent/user LLM
settings. In particular, natural-language assertions import
`DEFAULT_LLM_NL_ASSERTIONS` from tau2 config, whose upstream default is an
external model. The adapter runner patches these evaluator defaults inside the
experiment process so they use the same local OpenAI-compatible vLLM endpoint
as the agent. Always pass `api_base`/`api_key` in `Tau2RunConfig.*_llm_args`;
the runner exports that endpoint to `TAU2_PROMPTEVO_LOCAL_API_BASE` and sets
`OPENAI_API_KEY=EMPTY` for LiteLLM compatibility.

`banking_knowledge` can exceed a 24k context window with BM25 retrieval
(for example `task_004` reached about 28.7k input tokens). Qwen3-8B on one
3090 has been smoke-tested with vLLM `--max-model-len 32768` and
`--gpu-memory-utilization 0.95`. Some formal requests reach about 34.6k; by
design they remain isolated context failures rather than increasing the server
to 40k and risking stage-wide OOM. The runner continues subsequent tasks.

The experiment bootstrap decodes JSON-looking tool arguments that Qwen/vLLM
occasionally double encodes as strings. Unknown tools and semantically wrong
arguments are not repaired and remain model failures.
