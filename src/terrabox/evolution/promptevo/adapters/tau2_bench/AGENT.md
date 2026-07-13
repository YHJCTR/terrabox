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

Restarting the same chain must reuse completed artifacts: a valid
`rejudged_<provider>/rejudge_summary.json`, `stage1_proposal.json`, or
`stage2_contrastive.json` is a checkpoint. Do not repeat paid rejudging or prompt
optimization merely because the orchestration process exited between stages.

NL rejudge responses use a compact index-based JSON contract rather than
echoing full assertion text. Parse the first valid JSON object, validate
boolean verdicts strictly, retry malformed external responses, and preserve
successful per-item caches. A single malformed provider response must not force
already judged tasks to consume API tokens again.

Qwen rollout stages load four 32k vLLM services sequentially and then run one
domain per GPU. Keep `--safetensors-load-strategy eager` on this server: lazy
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
