# API-Bank adapter

Static instruction slot:

- `API_BANK_API_CALL_PROMPT` in `api_bank/constants.py`
- Mirrors API-Bank's original `evaluator.py::api_call_prompt`
- Dynamic API descriptions and dialogue history are appended at rollout time
  and are not optimized by promptevo.
- `API_BANK_RESPONSE_PROMPT` is also exposed for API-Bank's response task, but
  it is a separate static prompt slot. Do not optimize API-call and response
  instructions as one merged prompt unless the experiment intentionally changes
  both tasks.

Local data:

- Source checkout: `/data1/yuhongjie2/DAMO-ConvAI-api-bank`
- Default lv1/lv2 data:
  `/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-1-given-desc`
- Default experiment output:
  `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/`

No-model smoke test:

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

Expected core smoke-test values for the current local checkout:

- `n = 389`
- `prediction_coverage_rate = 1.0`
- `success_rate = exact_match_accuracy = 1.0`
- `api_name_accuracy = argument_key_f1 = 1.0`

Rollout dry run, no model:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import make_api_bank_components

root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
prompts, traces, metrics, runner = make_api_bank_components(data, root)
runner.run_version("base", "dry_base", limit=2, dry_run=True)
print(metrics.aggregate("dry_base"))
PY
```

Main metrics include exact API-call accuracy, parse/call rates, API-name
accuracy, argument key precision/recall/F1, argument value accuracy, error
buckets, history/argument-count bucket accuracy, latency, and runtime-error
rate.

Official-style execution diagnostics:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import APIBankMetricProvider

root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
pred = "tmp/api_bank_oracle_predictions.jsonl"
metrics = APIBankMetricProvider(
    data_dir_fn=lambda exp: data,
    prediction_path_fn=lambda exp: pred,
    api_bank_root=root,
    execute_api_calls=True,
)
print(metrics.aggregate("oracle"))
PY
```

Keep this as a diagnostic, not the default optimization gate. API-Bank APIs
include external network calls, random tokens, and state-changing calls; some
gold API calls do not reproduce their saved gold result in a clean local
process. The default promptevo loop should optimize stable prompt-following
metrics first: exact API name, argument keys, argument values, parse success,
and error buckets.

## Internal layout

- `prompts.py`: required `PromptStore`; reads/writes static prompt versions.
- `traces.py`: required `TrajectorySource`; converts API-Bank logs into
  promptevo `Trace` objects.
- `metrics.py`: optional `MetricProvider`; computes exact-call, argument,
  response, and error-bucket metrics for experiment comparison.
- `runner.py`: optional rollout runner; temporarily applies a prompt version
  for one experiment without modifying API-Bank's original prompt.
- `api_utils.py` and `files.py`: API-Bank-specific parsing and file helpers.

Response metrics:

- Set `include_responses=True` on `APIBankMetricProvider` to include post-API
  dialogue samples.
- The adapter reports `response_rouge_l`, `response_success_rate`, and
  `low_response_rouge_rate`.
- A prediction JSONL can contain both API-call and response rows because both
  use API-Bank's `(file, id, pred)` format.

Tool-search data:

- For data directories whose basename does not end with `given-desc`, the
  rollout runner follows API-Bank's original behavior and gives the model only
  the `ToolSearcher` description.
- For `*-given-desc`, it gives descriptions for the APIs present in the
  conversation history.
