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
`--model vllm_parsed` plus vLLM's Hermes parser. Do not switch to AgentDojo's
`local` provider unless deliberately testing its text-tag/regex tool parser.
The official CLI is also passed
`--module-to-load terrabox.evolution.promptevo.adapters.agentdojo.qwen_no_think`.
That adapter-local module changes only the upstream OpenAI request helper by
adding `chat_template_kwargs.enable_thinking=false` on every LLM turn. It is
needed because this vLLM image has no server-level `--chat-template-kwargs`
flag; without it the run would not match the Qwen3 no-think condition.
The agent loop itself remains AgentDojo's upstream `ToolsExecutionLoop` with
`max_iters=15`; PromptEvo does not alter that tool-call limit.

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

The pipeline uses `force_rerun=False`. Existing upstream JSON files are the
checkpoint; rerunning the same group skips completed tasks and phases. Every
rollout invocation owns its four vLLM containers and removes them in `finally`.
It also verifies every job's valid result count and the stage-wide total before
writing `pipeline_status=complete`; a successful CLI exit with missing JSON is
recorded as `incomplete_results` and retried. `stdout.log` and `stderr.log` are
written live, timeout/launcher failures write `run_status.json`, and a file lock
prevents two AgentDojo pipelines from sharing the four lanes. Stage1/Stage2
proposal JSON files are resumable checkpoints, so a watchdog restart does not
repeat paid prompt optimization after a prompt has already been saved.

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
- Prompt optimization uses LongCat/DeepSeek only for proposing the static
  prompt. Agent rollout remains local Qwen3 8B for all three stages.

## Validation completed

The adapter has synthetic tests for clean/attack classification, stable task
IDs, tool-call reconstruction, resumable CLI generation, single-GPU vLLM
pinning, Hermes parser flags, and metricViewer discovery. These tests do not
call paid APIs, start Docker, or execute benchmark tasks.
