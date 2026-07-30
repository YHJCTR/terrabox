# promptevo adapters

This package contains optional adapters for external agent projects.

Guidelines:

- Keep adapters thin and file-based when possible.
- Do not import third-party project code unless it is necessary.
- Do not hard-code local checkout paths; accept paths from constructors.
- Convert project outputs into `PromptStore`, `TrajectorySource`, and
  `MetricProvider` from `promptevo.interfaces`.
- Put project-specific rollout hooks or patches in documentation instead of
  mixing them into the core optimizer.
- Keep the optimized prompt slot narrow. Save only the static task/system
  instruction that promptevo may rewrite; leave tool descriptions, retrieved
  context, user input, dialogue history, memory, and API/tool observations as
  runtime context.
- Convert project logs into generic `Trace` objects before rendering them for
  promptevo. Do not pass project-private raw JSON directly to a generic sampler
  unless that sampler explicitly supports the format.
- Treat metrics as adapter-owned. The adapter should expose stable aggregate
  metrics and `metric_specs()` with directions so the two-stage optimizer can
  reason about real gains and regressions.
- Experiment outputs produced by an adapter should live under that adapter's
  `experiments/` directory, with generated prompt versions saved under
  `evolution_store/promptevo/<project>/versions/`.
- New promptevo experiments should use the shared naming stem
  `promptevo_<dataset>_<stage>_<variant>_<YYYYMMDD_HHMMSS>` for prompt
  versions, adapter experiment directories, and rollout output directories
  whenever possible. Do not rename historical experiments such as
  `promptevo_v1` or `promptevo_v2e`; apply the convention only to new runs.

Directory layout:

- `api_bank/` reads API-Bank lv1/lv2 JSONL conversations and model
  prediction JSONL files in the format used by API-Bank's evaluator.
- `toolbench/` reads ToolBench rollout JSON files and includes a
  StableToolBench runner that applies only an experiment-local ReAct static
  prompt override before calling the official `qa_pipeline_multithread.py`.
- `gepa_aime/` runs the prompt-only AIME scenario from GEPA's public example
  using cached HuggingFace AIME datasets; it optimizes only the static math
  system prompt and scores exact final integer answers in `### <answer>` form.
- `agentdojo/` reads AgentDojo system messages and `runs/**.json` traces;
  it also has an optional official-CLI runner that writes adapter-owned runs.
- `terrabox/` is the native OEA adapter. It reads Terrabox ReAct rollout
  `results/<task_id>.json`, rebuilds `trajectories_full.jsonl` from the
  authoritative results directory, and optimizes `_REACT_SYSTEM_PROMPT` through
  `TERRABOX_REACT_SYSTEM_PROMPT_FILE`.
- `tau2_bench/` reads tau2-bench agent instructions and Results JSON files;
  it also has an optional text-runner that mirrors tau2 outputs into the
  adapter experiment directory.
- `template/` documents the minimal required adapter surface and optional
  experiment-runner pieces for future projects.

Each project directory keeps documentation in `README.md` and an `__init__.py`
that re-exports the public adapter classes. Small adapters may keep code in a
single `components.py`; larger adapters should split by responsibility
(`prompts.py`, `traces.py`, `metrics.py`, `runner.py`, `files.py`) and may use a
thin `core.py` re-export for compatibility. This keeps project-specific code
out of one flat directory while preserving imports such as
`terrabox.evolution.promptevo.adapters.api_bank`.

Naming example:

```text
evolution_store/promptevo/terrabox/versions/promptevo_oea_stage1_sam2refresh_20260630_075048.txt
src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_stage1_sam2refresh_20260630_075048/
tmp/trajectories/promptevo_oea_stage1_sam2refresh_20260630_075048/standard/results/
```

Detailed quick starts:

- [API-Bank](api_bank/README.md)
- [API-Bank usage](api_bank/usage.md)
- [AgentDojo](agentdojo/README.md)
- [Terrabox/OEA](terrabox/README.md)
- [tau2-bench](tau2_bench/README.md)
- [GEPA AIME](gepa_aime/README.md)
- [Adapter template](template/README.md)

## API-Bank Summary

API-Bank is a good first external target because lv1/lv2 evaluation can be
reduced to static prompt -> next API-call prediction, without starting a tool
server or GPU service.

Expected prediction JSONL:

```json
{"file": "RegisterUser-level-1-2.jsonl", "id": 0, "pred": "[RegisterUser(username='johndoe')]"}
```

The adapter provides:

- `APIBankPromptStore`: versioned static prompt files under
  `evolution_store/promptevo/api_bank/versions/`.
- `APIBankTrajectorySource`: API-Bank gold conversations plus optional model
  predictions converted to generic `Trace` objects.
- `APIBankMetricProvider`: API-call accuracy and error buckets over
  prediction JSONL files. Default metrics are stable exact/API-name/argument
  metrics; `execute_api_calls=True` adds API-Bank-style execution diagnostics,
  and `include_responses=True` adds response Rouge-L metrics.
- `APIBankRolloutRunner`: runs one static instruction version and writes both
  `predictions.jsonl` and information-rich `rollout.jsonl`.
- `make_api_bank_components`: builds the prompt store, trajectory source,
  metric provider, and runner with consistent paths.
- `write_oracle_predictions`: deterministic smoke-test helper only; it writes
  gold API calls in API-Bank evaluator format and should score 1.0.

Lightweight validation, no model required:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import (
    APIBankMetricProvider,
    write_oracle_predictions,
)

data = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-1-given-desc"
pred = "tmp/api_bank_oracle_predictions.jsonl"
write_oracle_predictions(data, pred)
metrics = APIBankMetricProvider(
    data_dir_fn=lambda exp: data,
    prediction_path_fn=lambda exp: pred,
)
print(metrics.aggregate("oracle"))
PY
```

Dry-run prompt/message construction, no model required:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import make_api_bank_components

root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
prompts, traces, metrics, runner = make_api_bank_components(
    data_dir=data,
    api_bank_root=root,
    output_dir="tmp/promptevo/api_bank",
)
runner.run_version("base", "dry_base", limit=2, dry_run=True)
print(metrics.aggregate("dry_base"))
PY
```

Experiment loop:

1. `runner.run_version("base", "base_exp", limit=N)` runs the original static
   task instruction. Only the instruction slot is optimized; API descriptions
   and chat history remain dynamic context.
2. `metrics.aggregate("base_exp")` and `traces.traces("base_exp")` read the
   saved prediction/rollout files.
3. Use `PromptOptimizer` for a one-stage proposal from base traces, or
   `ContrastiveUpdater` for two-stage and later A/B updates.
4. Save the proposed static instruction with `APIBankPromptStore.save(...)`.
5. Rerun with `runner.run_version("<new-version>", "v1_exp", limit=N)`.
6. Continue with `ContrastiveUpdater.update("base", "v1", "base_exp",
   "v1_exp", "v2", ...)` for the next prompt version.

Adapter checklist:

- `PromptStore.load("base")` returns the unmodified static instruction.
- `PromptStore.save(...)` writes a new version file and metadata; it must not
  edit the source benchmark code.
- `TrajectorySource.traces(exp)` yields `Trace` objects with stable `task_id`s
  that match `MetricProvider.per_task(exp)`.
- `MetricProvider.aggregate(exp)` includes enough detail to diagnose tradeoffs,
  and `metric_specs()` marks directions such as `higher_better` or
  `lower_better`.
- `RolloutRunner`, if present, applies a prompt override only for the current
  experiment and writes rich `rollout.jsonl` plus evaluator-style predictions.

For API-Bank, keep the optimized prompt slot narrow: optimize the static
API-call task instruction, not the dynamic API descriptions or chat history.
The response task has its own static instruction (`API_BANK_RESPONSE_PROMPT`) and
should be treated as a separate experiment if needed.

## tau2-bench Notes

The tau2-bench adapter optimizes only
`src/tau2/agent/llm_agent.py::AGENT_INSTRUCTION`. Do not optimize dynamic
`domain_policy`, tool schemas, dialogue history, task state, gold actions, or
reward labels. Adapter-owned tau2 runs should write outputs under
`adapters/tau2_bench/experiments/<experiment>/` and use tau2 final reward plus
reward components as metrics. Reference actions are only required when tau2
includes ACTION in a task's `reward_basis`.

## Other External Adapters

AgentDojo follows the same adapter shape as API-Bank and tau2-bench: prompt
store, trajectory source, metric provider, official-CLI runner, and a formal
Base -> Stage1 -> Stage2 pipeline. The formal local-Qwen configuration runs the
four v1.2.2 suites on four single-GPU vLLM lanes, using AgentDojo's
`vllm_parsed` provider and Hermes native tool calls. Every stage includes both a
clean utility phase and an `important_instructions` attack phase. Results are
resumable and live under `agentdojo/experiments/<group>/`; metricViewer
discovers those top-level groups and legacy upstream runs separately.

AgentDojo metrics must stay faithful to its labels: clean user-task utility,
injection-task utility, utility under attack, security under attack, attack
success rate, and their balanced summary. Its JSON results do not define a
gold tool-call F1, so the adapter must not substitute success as a fake tool
F1. Before rollout, use the adapter
preflight because upstream imports all provider SDKs even for local Qwen.
