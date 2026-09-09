# tau2-bench adapter instructions

This directory adapts `/data1/yuhongjie2/tau2-bench` to promptevo.

## Prompt boundary

Promptevo may optimize only `src/tau2/agent/llm_agent.py::AGENT_INSTRUCTION`.
Do not optimize or replace `domain_policy`, tool schemas, dialogue history,
task data, gold actions, reward labels, retrieved documents, or tool
observations.

The runtime system prompt is static instruction plus dynamic domain policy.
Keep that boundary intact for every experiment.

## Experiment outputs

Put adapter-owned experiments under:

`src/terrabox/evolution/promptevo/adapters/tau2_bench/experiments/`

Each run should record:

- `run_meta.json`
- `run_status.json`
- `active_static_instruction.txt`
- `stdout.log` and `stderr.log`
- `tau2_results/` copied from tau2 `data/simulations/<save_to>/`
- `metrics_summary.json` when results exist

Do not overwrite tau2 source files to test a prompt. Use the adapter runner's
process-local prompt override.

For thinking models, do not let `<think>...</think>` enter tau2 visible
dialogue history or promptevo trace text. The adapter runner patches tau2
generation inside the experiment process so only the final visible message is
stored in agent/user state.

## Metrics

Use final reward and reward components as the acceptance signal:

- full reward / `success_rate`
- average final reward
- DB/end-state reward
- communication reward
- action reward only when ACTION is part of `reward_basis`

Do not require the agent to match `evaluation_criteria.actions` unless tau2
explicitly scores ACTION for that task.

## Running

tau2 needs its own dependencies. The upstream setup is `uv sync` in the tau2
repo. If dependencies are missing, the adapter runner should fail early and
write `run_status.json` instead of silently fabricating results.

For local vLLM, LiteLLM should receive `api_base`, `api_key`, and a model name
such as `openai/local-qwen3-8b` through `Tau2RunConfig.*_llm_args`.

`Tau2RunConfig.auto_resume` defaults to `True`. Keep it enabled for full runs:
the runner preserves tau2's upstream simulation directory and passes
`--auto-resume`. It rejects a resume when the experiment name exists with a
different static instruction, preventing cross-prompt result contamination.

Runtime writes must stay in the Terrabox workspace. The runner sets
`TAU2_DATA_DIR=tmp/tau2_runtime/<experiment>/`, links its `tau2/` input to the
upstream read-only dataset, and keeps resumable checkpoints in that runtime
directory before mirroring completed output to `tau2_results/`.

Use `pipeline.py chain-after-base` for the formal Base -> Stage1 -> Stage2
workflow. It performs resumable LongCat no-think NL rejudging without
overwriting native results, uses the rejudged Base for Stage1 sampling, and
uses paired rejudged Base/Stage1 task IDs for Stage2. Multi-domain task IDs must
remain `domain::task_id::trialN`; basename-only keys collide because every
domain output is named `results.json`.

历史正式口径里的“三阶段”指 Base / Stage1 / Stage2。可选 Stage3 是
post-Stage2 refinement，不替代 Stage2：它以 Stage2 prompt 为当前基线，比较
Stage1 与 Stage2 的同任务轨迹，只提出小幅 protocol patch 来修复 Stage2 引入的回归，
并继续使用同一固定真实 dev slice 验证；候选没有通过接受门时保留 Stage2。
可通过 `pipeline.py stage3-after-stage2` 单独接在已完成的 Stage2 后，也可在
`chain-after-base` / `full-chain` 中显式传入 `--stage3-group` 与
`--stage3-version` 自动接上。Stage3 的可恢复检查点为 `stage3_protocol_patch.json`。

Use `pipeline.py full-chain --profile paper3` for the paper-comparable rerun:
airline/retail/telecom only, Qwen3 8B no-think agent, fixed LongCat2 no-think
user simulator, four trials, `max_steps=100`, and `max_tokens=2048` on three
GPU lanes. This matches the original domain/trial structure but is not a model
reproduction. The older `legacy4` profile preserves the four-domain, one-trial,
80-step, 512-output-token experiment. External user credentials must stay in
the child environment and out of argv, run metadata, and copied results.

`stable4` is the recommended reliable full-run profile after the failed
external-user `paper3` attempt: all four domains run in parallel on GPU0-3,
both agent and user simulator use the local Qwen3 vLLM lane for that domain,
four trials, `max_steps=100`, and `max_tokens=2048`. A service that only passes
`/health` is not a running domain; every submitted domain must create
`run_meta.json` and receive inference requests before the run is considered
started.

Use `pipeline.py full-chain --profile longcat_agent4` only when intentionally
testing LongCat2 as the executable agent and user simulator. It keeps the same
four-domain, four-trial, `max_steps=100` profile, uses dynamic chunk scheduling,
does not start local vLLM containers, and passes LongCat credentials only
through the child environment. The provider config lookup must be cwd-independent: tau2 subprocesses run from `/data1/yuhongjie2/tau2-bench`, so `agent_config.yaml` must resolve via `AGENT_CONFIG_PATH`, `TERRABOX_LLM_API_*`, or the Terrabox repo-root fallback, never only via the current working directory.
For new LongCat agent experiments, use the generic PromptEvo v2 meta prompts by
passing `--optimizer-version v2`. When another LongCat API-heavy run is active,
start tau2 conservatively with `TERRABOX_TAU2_API_WORKERS=1-2`,
`TERRABOX_TAU2_CHUNK_SIZE=1-2`, and `TERRABOX_TAU2_RUN_CONCURRENCY=1`; these
change scheduling pressure only, not the benchmark prompt or task semantics.
Even with one worker, a single tau2 simulation can issue rapid agent/user/eval
LLM calls. External-provider runs therefore pace LiteLLM calls in the bootstrap
through `terrabox.agent.llm_provider.pace_remote_llm_request()`, using the same
provider lock and JSON fairness state as Terrabox `RemoteChatClient` and
LangChain remote rollouts. The tau2 pipeline injects
`TERRABOX_REMOTE_LLM_WORKLOAD=tau2`; ExperienceEvo or another concurrent run
should use its own workload name, for example `experienceevo`. When both are
waiting, LongCat grants rotate by workload; when tau2 is alone, it continues at
the configured provider interval. `TERRABOX_TAU2_API_MIN_INTERVAL_SECONDS`
remains a compatibility alias for the interval, but do not point tau2 at a
separate independent lock to bypass `TERRABOX_REMOTE_LLM_RATE_LOCK`. Restart the
watcher/rollout parent after changing pacing or workload env vars.
Keep `TERRABOX_TAU2_OPENAI_TIMEOUT_SECONDS` finite (default 300s) so LiteLLM
HTTP calls fail and requeue instead of hanging a chunk forever.

`stable4` uses dynamic chunk scheduling, not fixed domain-to-GPU scheduling.
Each GPU/port is a reusable lane; tau2 tasks are split into small `task_ids`
chunks, and a lane that finishes early immediately pulls another chunk from the
remaining queue. This avoids wasting a GPU when a short domain finishes before
longer domains. Chunk outputs are written under `<group>/_chunks/...` and then
merged back into the canonical `<group>/<domain>_base/tau2_results/results.json`
with `(task_id, trial, seed)` de-duplication. Do not run multiple processes
against the same domain `results.json` directly; use the chunk merger.

External API provider/rate-limit errors are retryable queue events, not final
rollout failures. Dynamic chunk workers inspect stderr/stdout for 429/rate-limit
/timeout/provider-overload markers, wait briefly, and put the same chunk at the
back of the queue. Keep `TERRABOX_TAU2_QUEUE_RETRIES` finite (default 12) so real
code/data bugs still surface instead of burning API tokens overnight.

Restarting the same chain must reuse completed artifacts: a valid
`rejudged_<provider>/rejudge_summary.json`, `stage1_protocol_patch.json`,
`stage2_protocol_patch.json`, or `stage3_protocol_patch.json` is a checkpoint.
Do not repeat paid rejudging or prompt optimization merely because the
orchestration process exited between stages.

NL rejudge responses use a compact index-based JSON contract rather than
echoing full assertion text. Parse the first valid JSON object, validate
boolean verdicts strictly, retry malformed external responses, and preserve
successful per-item caches. A single malformed provider response must not force
already judged tasks to consume API tokens again.

Only send assertions to the external judge when `NL_ASSERTION` participates in
the task's `reward_basis`. Assertions attached to DB/COMMUNICATE-only tasks do
not affect the metric and must be counted as `skipped_non_scoring`, not charged
to the provider. Accept common provider wrappers (`evaluations`, `checks`,
`outcomes`, `judgments`) and multiple top-level row objects. After retries, a
remaining malformed response is saved under `judge_failures/` and preserves the
original reward info instead of blocking every later stage; the summary must
surface `judge_failures` explicitly.

Qwen rollout stages load one 32k vLLM service per selected domain/GPU lane.
Keep `--safetensors-load-strategy eager` on this server: lazy
mmap loading stalled on the nearly full `/data1` disk. Stop every stage-owned
container in `finally`, including failed runs.

Some Codex execution sandboxes hide `/dev/nvidia*`, so direct `nvidia-smi` may
fail even when the host GPUs are healthy. Before declaring the host unavailable,
verify through the Docker daemon, for example with a short `docker run --rm
--gpus device=0 ... nvidia-smi` smoke using an existing Terrabox GPU image.

tau2 evaluator calls are not automatically tied to the agent/user LLM. Natural
language assertions use tau2 config defaults and may otherwise call an external
OpenAI/Anthropic model. Keep the adapter runner bootstrap patch that redirects
NL assertion/auth/review evaluator calls to `TAU2_PROMPTEVO_LOCAL_API_BASE`.

For full text/base with `banking_knowledge`, use vLLM context 32768 rather
than 24576. A local smoke test showed `banking_knowledge task_004` needs about
28.7k input tokens with BM25 retrieval. Qwen3-8B fits at 32768 on one 3090 with
`--gpu-memory-utilization 0.95`, but leaves very little spare VRAM. Some formal
banking requests reach about 34.6k and are intentionally recorded as isolated
infrastructure/context failures instead of raising the server to 40k. They must
not abort the remaining domain or stage.

The experiment bootstrap normalizes tool-call arguments that are valid JSON
but double encoded as strings before constructing tau2 `ToolCall`. This is a
transport repair only; unknown tool names, oversized context, and semantic
argument mistakes remain benchmark failures.

## Resource cleanup

This adapter should not leave vLLM or tau2 processes behind. After any native
run, check process/container ownership before cleaning:

- `ps -eo pid,ppid,user,stat,etime,cmd | rg 'tau2|vllm|api_server'`
- `docker ps --format '{{.Names}} {{.Ports}}'`
- `nvidia-smi`
