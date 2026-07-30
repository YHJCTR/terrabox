# AgentDojo PromptEvo adapter

This adapter optimizes AgentDojo's static `default` system message while
leaving suites, user tasks, tool schemas, tool outputs, attacks, and evaluators
unchanged.

## Local layout

- Upstream checkout: `/data1/yuhongjie2/agentdojo`
- Upstream version: package `0.1.35`, git `089ed468`, benchmark `v1.2.2`
- Static prompt source: `src/agentdojo/data/system_messages.yaml`, key `default`
- Prompt versions: `evolution_store/promptevo/agentdojo/versions/`
- Adapter experiments: `src/terrabox/evolution/promptevo/adapters/agentdojo/experiments/`

The upstream prompt file is never overwritten. Each run passes the selected
prompt through AgentDojo's official `--system-message` override and saves a copy
as `active_system_message.txt` in the experiment group.

## Benchmark scope

The formal local-Qwen configuration starts four equivalent Qwen3 worker lanes:

| Worker | GPU | vLLM port |
|---|---:|---:|
| lane0 | 0 | 9200 |
| lane1 | 1 | 9201 |
| lane2 | 2 | 9202 |
| lane3 | 3 | 9203 |

Jobs are dynamically pulled by the first free lane. They are not permanently
pinned by suite: workspace is much larger than the other suites, so fixed suite
pinning would leave three GPUs idle near the end.

For each suite and prompt version it runs two resumable phases:

1. `clean`: all user tasks without injection, measuring clean utility.
2. `attack`: `important_instructions` over all user-task/injection-task pairs;
   AgentDojo also evaluates injection tasks as normal user tasks.

The current v1.2.2 scope is 1,081 result JSON files per stage:

| Suite | Clean user | Injection-as-user | Attacked pairs | Total |
|---|---:|---:|---:|---:|
| workspace | 40 | 14 | 560 | 614 |
| travel | 20 | 7 | 140 | 167 |
| banking | 16 | 9 | 144 | 169 |
| slack | 21 | 5 | 105 | 131 |

The preflight recomputes these counts from the installed upstream source and
writes `expected_results`, so a future AgentDojo update cannot silently retain
the old total.

For balanced execution, the attack phase is split by injection task: 4 clean
jobs plus 35 disjoint attack jobs. Each attack shard contains one injection
task and all user tasks in that suite, so every user/injection pair and every
injection-as-user diagnostic is produced exactly once.

The agent is Qwen3 8B served by vLLM with native OpenAI tool calls:
`--model VLLM_PARSED` (its result directory/pipeline name remains
`vllm_parsed`) plus vLLM's Hermes parser. Do not switch to AgentDojo's
`local` provider unless deliberately testing its text-tag/regex tool parser.
The official CLI is also passed
`--module-to-load terrabox.evolution.promptevo.adapters.agentdojo.qwen_no_think`.
That adapter-local module changes only the upstream OpenAI request helper by
adding `chat_template_kwargs.enable_thinking=false` on every LLM turn. It is
needed because this vLLM image has no server-level `--chat-template-kwargs`
flag; without it the run would not match the Qwen3 no-think condition.
The agent loop itself remains AgentDojo's upstream `ToolsExecutionLoop` with
`max_iters=15`; PromptEvo does not alter that tool-call limit.

The same module rebuilds AgentDojo's `TaskResults` forward references for
Pydantic 2.13 before loading existing JSON. This is required for real resume:
fresh tasks can run without it, but an interrupted shard otherwise fails while
reading its already completed results. It also normalizes vLLM generic 400
context-window errors to `context_length_exceeded`, letting AgentDojo's upstream
per-sample context-limit fallback record the task as failed and continue.

Primary metrics are clean user-task utility, injection-task utility when those
objectives are presented legitimately, attacked utility, attacked security,
attack success rate, conjunction success, and a three-axis `balanced_score`.
AgentDojo does not expose a gold tool-call F1 in these result files, so the
adapter does not fabricate one.

## Dependency preflight

Run from the Terrabox repository:

```bash
PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.promptevo.adapters.agentdojo.pipeline preflight
```

The local checkout and Qwen3 model are present. The packages that were absent
from the read-only `unsloth` environment are:

- `cohere`
- `deepdiff`
- `google-genai` (imported as `google.genai`)

AgentDojo imports all provider implementations at module load time, so these
packages are required even for a local Qwen run. Because this server's conda
site-packages may be read-only, the adapter also supports an isolated dependency
layer at `tmp/agentdojo_site_packages/` (override with
`TERRABOX_AGENTDOJO_SITE_PACKAGES`). Preflight, inventory probes, and rollout
subprocesses add that path automatically. These packages are now installed in
that isolated layer and formal preflight passes. The adapter writes a structured
`preflight.json`/blocked `pipeline_status.json` instead of starting GPUs when
dependencies are missing.

## Commands prepared for the formal chain

Use timestamped names following the shared adapter convention. These commands
are examples only; no AgentDojo benchmark was started while preparing them.

Base rollout:

```bash
PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.promptevo.adapters.agentdojo.pipeline rollout \
  --group promptevo_agentdojo_base_qwen3_8b_YYYYMMDD_HHMMSS \
  --prompt-version base --stage base
```

Before the first Base, run the real local smoke after the GPUs are free:

```bash
PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.promptevo.adapters.agentdojo.pipeline smoke \
  --group qwen3_8b_v1
```

This starts only GPU0, runs one clean user task plus one attacked
user/injection pair (three result JSON files including injection-as-user), and
writes to `tmp/agentdojo_smoke/`. Full Base must not start unless this command
passes the same strict result-count checks.

After Base is running, the watcher/chain entry waits for all clean and attack
phases, creates Stage1 with LongCat no-think, runs Stage1, performs Base-vs-
Stage1 contrastive attribution, creates Stage2, and runs Stage2:

```bash
PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.promptevo.adapters.agentdojo.pipeline chain-after-base \
  --base-group promptevo_agentdojo_base_qwen3_8b_YYYYMMDD_HHMMSS \
  --stage1-group promptevo_agentdojo_stage1_qwen3_8b_YYYYMMDD_HHMMSS \
  --stage2-group promptevo_agentdojo_stage2_qwen3_8b_YYYYMMDD_HHMMSS \
  --stage1-version promptevo_agentdojo_stage1_qwen3_8b_YYYYMMDD_HHMMSS \
  --stage2-version promptevo_agentdojo_stage2_qwen3_8b_YYYYMMDD_HHMMSS \
  --provider longcat
```

For new generic meta-prompt experiments, append `--optimizer-version v2` and
include `metav2` in the group/version names. The default remains v1 for
backward compatibility; v2 is additive and does not overwrite historical prompt
versions or experiment directories.
Stage1 v2 and Stage2 v2 are separate meta prompts: Stage1 v2 diagnoses one
version's traces and proposes a conservative first edit, while Stage2 v2 uses
the exact Base-to-Stage1 prompt diff plus paired same-task traces to keep gains
and narrowly repair regressions.

The pipeline uses `force_rerun=False`. Existing upstream JSON files are the
checkpoint; rerunning the same group skips completed tasks and phases. Every
rollout invocation owns its four vLLM containers and removes them in `finally`.
It also verifies every job's valid result count and the stage-wide total before
writing `pipeline_status=complete`; a successful CLI exit with missing JSON is
recorded as `incomplete_results` and retried. If a single sample deterministically
fails because the local Qwen3 context window is exceeded (`maximum context
length` / `input tokens`), the adapter writes one failed synthetic result JSON
for that task (`terrabox_synthetic=true`,
`terrabox_skip_reason=context_length_exceeded`) and resumes the shard. This keeps
the formal result honest without truncating inputs or trapping the watcher on one
overlength task. If upstream AgentDojo's evaluator itself raises while checking
one already-written sample, the adapter marks that sample
`terrabox_skip_reason=evaluator_error` and resumes; this includes the case where
Rich-wrapped stdout no longer exposes a parseable trace marker and the adapter
must recover from the newest partial result JSON without utility/security
labels. `stdout.log` and `stderr.log` are written live,
timeout/launcher failures write `run_status.json`, and a file lock prevents two
AgentDojo pipelines from sharing the four lanes. Stage1/Stage2 proposal JSON
files are resumable checkpoints, so a watchdog restart does not repeat paid
prompt optimization after a prompt has already been saved.

## Experiment layout

```text
experiments/<group>/
  active_system_message.txt
  experiment_meta.json
  preflight.json
  pipeline_status.json
  metrics_summary.json
  workspace_clean/runs/vllm_parsed/...
  workspace_attack_injection_task_0/runs/vllm_parsed/...
  workspace_attack_injection_task_1/...
  travel_clean/...
  travel_attack_injection_task_0/...
  banking_clean/...
  banking_attack_injection_task_0/...
  slack_clean/...
  slack_attack_injection_task_0/...
```

Stage1 additionally saves `stage1_proposal.json`; Stage2 saves
`stage2_contrastive.json`. `metricViewer` discovers the top-level group and
recursively reads its authoritative AgentDojo JSON traces. Legacy upstream
`/data1/yuhongjie2/agentdojo/runs/` directories remain visible under the
`upstream/` prefix when they exist.

## Stage semantics

- Stage1 samples security failures, utility failures, and successful traces
  separately so a large utility bucket cannot hide injection regressions.
- Stage2 pairs Base and Stage1 by
  `suite/user_task/attack_type/injection_task`, then asks for changes that
  improve utility without accepting a material security regression.
- Controlled ablation experiments should create an explicit prompt version
  (for example a minimal prompt that keeps only role and identity) and pass it
  both to Base rollout as `--prompt-version <version>` and to the chained
  optimizer as `--base-prompt-version <version>`. This keeps Stage1/Stage2
  recovery anchored to the ablated prompt instead of accidentally optimizing
  from the official AgentDojo default prompt.
- By default, prompt optimization uses LongCat/DeepSeek only for proposing the
  static prompt, while rollout remains local Qwen3 8B. For explicit LongCat2
  agent experiments, pass `--agent-provider longcat`; the adapter then uses
  AgentDojo's upstream `OPENAI_COMPATIBLE` provider and keeps suites, attacks,
  evaluators, and result-count checks unchanged. LongCat job-level parallelism
  is controlled by `TERRABOX_AGENTDOJO_API_WORKERS` (default 1, configurable) and is separate
  from the four local-GPU lanes used by Qwen.
- LongCat/other external-provider 429, rate-limit, timeout, and provider-overload
  errors are treated as retryable queue events. The failed job waits briefly and
  is put at the back of the queue; `TERRABOX_AGENTDOJO_QUEUE_RETRIES` (default
  12) caps this so real benchmark or adapter bugs still surface. Even with one
  API worker, a single AgentDojo sample can issue rapid multi-turn model calls;
  external API calls are therefore paced with a cross-process file lock. Tune
  `TERRABOX_AGENTDOJO_API_MIN_INTERVAL_SECONDS` (LongCat default 8.0s) and
  `TERRABOX_AGENTDOJO_API_RATE_LOCK`; the old `TERRABOX_AGENTDOJO_LONGCAT_*`
  names remain aliases. Restart the watcher or rollout parent after changing
  pacing env vars, because running Python parents keep their old environment.
- Meta-prompt v2 is still domain-neutral: it uses repeated behavior patterns,
  metric directions, and paired trace evidence, but must not inject suite names,
  tool names, task IDs, or fixed workflows into the optimized static prompt.

## Validation completed

The adapter has synthetic tests for clean/attack classification, stable task
IDs, tool-call reconstruction, resumable CLI generation, single-GPU vLLM
pinning, Hermes parser flags, and metricViewer discovery. These tests do not
call paid APIs, start Docker, or execute benchmark tasks.
