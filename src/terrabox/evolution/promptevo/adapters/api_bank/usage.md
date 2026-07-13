# API-Bank promptevo usage

This adapter runs API-Bank's static API-call instruction through the promptevo
interface. It optimizes only the written task instruction. API descriptions,
dialogue history, and API responses are dynamic inputs and should not be saved
as prompt versions.

## Paths

- API-Bank checkout: `/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank`
- Default data: `/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-1-given-desc`
- Default output root:
  `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/`
- For the original full API-Bank rerun in this document, use:
  `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_full/`

Each experiment writes:

- `predictions.jsonl`: API-Bank evaluator-style rows, one per target API call.
- `rollout.jsonl`: richer rows with messages, prompt context, latency, gold,
  prediction, parse analysis, and error flags.
- `meta.json`: prompt version, data path, shard info, model settings, and output
  paths.

## Start local vLLM services

Example for three RTX 3090 cards. These ports intentionally avoid the usual
Terrabox agent ports so an existing `9101` service can keep running.
These services are temporary for the run. Stop them after the experiment unless
you explicitly plan to reuse them immediately.

```bash
docker run -d --rm --gpus all --name apibank-qwen3-8b-gpu0-9200 \
  -p 9200:8000 -e CUDA_VISIBLE_DEVICES=0 \
  -v /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/:/model:ro \
  terrabox/agent-llm:latest \
  --model /model --trust-remote-code --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 24576 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes

docker run -d --rm --gpus all --name apibank-qwen3-8b-gpu1-9201 \
  -p 9201:8000 -e CUDA_VISIBLE_DEVICES=1 \
  -v /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/:/model:ro \
  terrabox/agent-llm:latest \
  --model /model --trust-remote-code --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 24576 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes

docker run -d --rm --gpus all --name apibank-qwen3-8b-gpu2-9202 \
  -p 9202:8000 -e CUDA_VISIBLE_DEVICES=2 \
  -v /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/:/model:ro \
  terrabox/agent-llm:latest \
  --model /model --trust-remote-code --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 24576 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Health check:

```bash
for p in 9200 9201 9202; do
  no_proxy=localhost,127.0.0.1 curl -s http://localhost:$p/v1/models >/dev/null && echo "$p ok"
done
```

## Run the original static instruction

Run one shard per service:

```bash
EVOLUTION_LLM_URL=http://localhost:9200 no_proxy=localhost,127.0.0.1 PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import make_api_bank_components
root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
_, _, _, runner = make_api_bank_components(data, root, output_dir="src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_level1_3gpu", max_tokens=128)
runner.run_version("base", "base_original_gpu0", shard_index=0, num_shards=3, resume=True)
PY

EVOLUTION_LLM_URL=http://localhost:9201 no_proxy=localhost,127.0.0.1 PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import make_api_bank_components
root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
_, _, _, runner = make_api_bank_components(data, root, output_dir="src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_level1_3gpu", max_tokens=128)
runner.run_version("base", "base_original_gpu1", shard_index=1, num_shards=3, resume=True)
PY

EVOLUTION_LLM_URL=http://localhost:9202 no_proxy=localhost,127.0.0.1 PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import make_api_bank_components
root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
_, _, _, runner = make_api_bank_components(data, root, output_dir="src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_level1_3gpu", max_tokens=128)
runner.run_version("base", "base_original_gpu2", shard_index=2, num_shards=3, resume=True)
PY
```

On three RTX 3090 cards with Qwen3-8B, the current 389 API-call samples finished
in about 2 minutes 20 seconds wall time. A single service smoke test usually
takes about 0.8-1.4 seconds per sample, so a conservative full-run estimate is
2-8 minutes depending on concurrent load and output length.

## Stop temporary services

Default policy: release any vLLM services started for this adapter run. Do not
leave them occupying GPU memory unless the next run will reuse them immediately.

```bash
docker stop apibank-qwen3-8b-gpu0-9200 \
  apibank-qwen3-8b-gpu1-9201 \
  apibank-qwen3-8b-gpu2-9202
```

Verify cleanup:

```bash
docker ps --format '{{.Names}} {{.Ports}}' | rg 'apibank|terrabox-agent-llm'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
for p in 9200 9201 9202; do
  no_proxy=localhost,127.0.0.1 curl -s --max-time 1 http://localhost:$p/v1/models >/dev/null && echo "$p up" || echo "$p down"
done
```

## Merge shards and score

The metric provider can score each shard directly. For a single combined result
file, merge rows by `(file, id)`:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from terrabox.evolution.promptevo.adapters.api_bank import APIBankMetricProvider

root = Path("src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_level1_3gpu")
out = root / "base_original_merged"
out.mkdir(parents=True, exist_ok=True)

seen = set()
for name in ["base_original_gpu0", "base_original_gpu1", "base_original_gpu2"]:
    for line in (root / name / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["file"], int(row["id"]))
        if key not in seen:
            seen.add(key)
            with (out / "predictions.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

for name in ["base_original_gpu0", "base_original_gpu1", "base_original_gpu2"]:
    for line in (root / name / "rollout.jsonl").read_text(encoding="utf-8").splitlines():
        with (out / "rollout.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

data = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-1-given-desc"
metrics = APIBankMetricProvider(
    data_dir_fn=lambda exp: data,
    prediction_path_fn=lambda exp: str(out / "predictions.jsonl"),
    rollout_path_fn=lambda exp: str(out / "rollout.jsonl"),
)
print(metrics.aggregate("base_original_merged"))
PY
```

For prompt evolution, use the merged experiment as the base run, then save a new
static prompt version with `APIBankPromptStore.save(...)` and rerun the same
three-shard command with a new experiment name.

## Metric notes

Default acceptance metrics are deterministic and stable:

- exact API-call success
- API-name accuracy
- argument key precision/recall/F1
- argument value accuracy
- parse/call rates
- error buckets
- prediction coverage

`execute_api_calls=True` adds API-Bank evaluator-style execution diagnostics,
but should be treated as debugging only. Some API-Bank tools depend on external
network calls, random tokens, mutable databases, or optional packages, so gold
calls may not reproduce their stored gold result in a clean local process.

`include_responses=True` adds post-API response Rouge-L metrics. That task uses
`API_BANK_RESPONSE_PROMPT`, a separate static instruction slot from API-call
prediction.

## Full original-task coverage

API-Bank's original evaluator has two switches:

- API-call mode: predict the next `[ApiName(...)]`.
- Dialog/response mode: predict the AI response after an API call and report
  Rouge-L.

The local checkout has two lv1/lv2 directories:

- `level-1-given-desc`: API descriptions are given directly.
- `level-2-toolsearcher`: API-call mode only gives the `ToolSearcher`
  description, following the original evaluator behavior.

The full local coverage therefore consists of four runs:

- `level1_api`
- `level1_response`
- `level2_toolsearch_api`
- `level2_toolsearch_response`

The latest full run is saved under:

```text
src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_full/
```

Score it again:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from terrabox.evolution.promptevo.adapters.api_bank import APIBankMetricProvider

root = Path("src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_full")
data_root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples"
configs = {
    "level1_api": (data_root + "/level-1-given-desc", False),
    "level1_response": (data_root + "/level-1-given-desc", True),
    "level2_toolsearch_api": (data_root + "/level-2-toolsearcher", False),
    "level2_toolsearch_response": (data_root + "/level-2-toolsearcher", True),
}
summary = {}
for exp, (data, include_responses) in configs.items():
    metrics = APIBankMetricProvider(
        data_dir_fn=lambda e, data=data: data,
        prediction_path_fn=lambda e, exp=exp: str(root / exp / "predictions.jsonl"),
        rollout_path_fn=lambda e, exp=exp: str(root / exp / "rollout.jsonl"),
        include_responses=include_responses,
    )
    agg = metrics.aggregate(exp)
    summary[exp] = agg
    print("\n[" + exp + "]")
    for key in [
        "n",
        "n_expected_api_calls",
        "n_responses",
        "n_expected_responses",
        "prediction_coverage_rate",
        "response_prediction_coverage_rate",
        "success_rate",
        "exact_match_accuracy",
        "parse_success_rate",
        "api_name_accuracy",
        "argument_key_f1",
        "argument_value_accuracy",
        "response_rouge_l",
        "response_success_rate",
    ]:
        if key in agg:
            print(key, agg[key])
with (root / "metrics_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
PY
```

The latest observed Qwen3-8B scores:

```text
level1_api exact_match_accuracy: 0.8098
level1_response response_rouge_l: 0.3512
level2_toolsearch_api exact_match_accuracy: 0.4286
level2_toolsearch_response response_rouge_l: 0.2845
```

The full run was executed on one RTX 3090 because GPU0/GPU2 were occupied at
the time; wall time was about 21 minutes excluding service startup. Temporary
vLLM services were stopped after completion.

## Meta-prompt search note

The current API-call prompt-evolution search output is saved under:

```text
src/terrabox/evolution/promptevo/adapters/api_bank/experiments/meta_v2c_stage1_20260629_172816/
```

Important files:

- `meta_prompt_search_summary.json`: compact comparison across base, stage1,
  and several stage2 candidates.
- `stage1_proposal.json`: first-stage optimizer diagnosis and revised static
  prompt.
- `stage2_*_attributions.json`: contrastive second-stage attributions.
- `stage2_*_candidates.json`: generated second-stage candidates.
- `stage*_prompt_diff*.txt`: prompt diffs against base or stage1.
- `metrics_summary*.json`: full metric summaries for each tested version.

Latest Qwen3-8B API-call macro success, averaged over `level1_api` and
`level2_toolsearch_api`:

```text
base:                  0.6192
stage1_v2c:            0.6515  (+0.0323 vs base)  current best
stage2_v2c_cand1:      0.6343  (+0.0152 vs base)
stage2_v2d_cand1:      0.6411  (+0.0219 vs base)
stage2_v2e_cand1:      0.6453  (+0.0261 vs base)
stage2_v2e_cand2:      0.6208  (+0.0016 vs base)
stage2_v2e_cand3:      0.6479  (+0.0287 vs base)
```

The best tested static API-call prompt version is:

```text
api_bank_api_call_meta_v2c_stage1_20260629_173024
```

Path:

```text
evolution_store/promptevo/api_bank/versions/api_bank_api_call_meta_v2c_stage1_20260629_173024.txt
```

The latest generic meta-prompt backups from this search are in:

```text
src/terrabox/evolution/promptevo/adapters/api_bank/experiments/meta_prompt_backups_20260629/
```

The second-stage meta prompt improved after adding two general constraints:
preserve structural prompt scaffold, and when the current version already has
net gains, inherit it conservatively and patch only the harmful subclause. These
constraints are intentionally domain-neutral and should carry to other agent
settings.

## Quality-fix experiment note

The latest quality-fix run is saved under:

```text
src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_fix_twostage_20260629_200956/
```

Important files:

- `stage1_candidates.json`, `stage1_static_candidate_scores.json`,
  `stage1_prompt_diff.txt`: first-stage candidates and the selected prompt.
- `stage2_attributions.json`, `stage2_paired_cases.json`: second-stage
  comparison evidence.
- `stage2_candidates.json`: initial second-stage candidate generation after
  switching to quote-safe candidate parsing.
- `stage2b_candidates.json`, `stage2c_*`: tightened second-stage candidate
  selection after penalizing value-relaxing phrases.
- `metrics_compare_base_stage1_stage2_stage2c.json`: base/stage1/stage2
  comparison.

Observed API-call macro success over `level1_api` and
`level2_toolsearch_api`:

```text
base:             0.6192
stage1:           0.6221  (+0.0029 vs base)
stage2_bad_relax: 0.6166  (-0.0026 vs base, -0.0055 vs stage1)
stage2c:          0.6192  (+0.0000 vs base, -0.0029 vs stage1)
```

This run fixed two process issues:

- Second-stage candidate output now has a quote-safe text format fallback,
  because full prompts often contain unescaped quotes that break JSON parsing.
- Candidate selection now penalizes phrases that loosen value fidelity, such as
  allowing the model to adjust, correct, infer, or rewrite provided values.

It also exposed one remaining process issue: when a static prompt ends with a
dynamic context anchor, new static rules must be inserted before that anchor.
Appending rules after the anchor can put static instructions inside the dynamic
context slot and should be rejected before rollout.

## Generalized v2c-style experiment

The latest successful v2c-style generalization run is saved under:

```text
src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_v2c_manual_general_20260629_210641/
```

This experiment tested a domain-neutral version of the original v2c insight:
when the user asks for an operation, lookup, computation, status, update, or
other result and a matching interface is available, the agent should use that
interface to produce the required structured output; it should still use only
context-supported information and should not invent fields or rewrite provided
values.

Observed API-call macro success over `level1_api` and
`level2_toolsearch_api`:

```text
base:       0.6192
stage1:     0.6398  (+0.0207 vs base)
stage2:     0.6398  (+0.0207 vs base, +0.0000 vs stage1)
stage2copy: 0.6470  (+0.0278 vs base, +0.0071 vs stage1)
```

Per split exact-call success:

```text
base       level1_api: 0.8098   level2_toolsearch_api: 0.4286
stage1     level1_api: 0.8175   level2_toolsearch_api: 0.4622
stage2     level1_api: 0.8175   level2_toolsearch_api: 0.4622
stage2copy level1_api: 0.8149   level2_toolsearch_api: 0.4790
```

The stage1 prompt diff is in `stage1_prompt_diff.txt`. The stage2 prompt kept
the interface-use requirement and only added "without rewriting provided
values"; this improved level1 argument-value accuracy slightly but did not move
exact success.

The stronger `stage2copy` candidate is saved as
`api_bank_api_call_quality_manual_general_stage2_copy_20260629_210641`. Its
diff is in `stage2copy_prompt_diff_stage1_to_stage2.txt`. It replaces the weak
"do not rewrite provided values" wording with a more concrete copy-fidelity
guardrail: copy provided values exactly unless the required output format
explicitly says to transform them; do not paraphrase, change capitalization,
add/remove words, replace placeholder-like values, or normalize values only for
style. This raised `level2_toolsearch_api` exact success from 0.4622 to 0.4790,
while `level1_api` dropped slightly from 0.8175 to 0.8149.

This suggests the useful general meta-prompt target is not merely "preserve
values"; it is "use the matching interface when the task clearly asks for a
result, while preserving context-supported fields and values." The old v2c
prompt remains stronger on this dataset because it used API-Bank-specific result
categories, but the generalized rule still produced a meaningful improvement
without naming API-Bank-specific domains.

For second-stage meta-prompt design, this run shows that a weak value rule such
as "without rewriting provided values" may not be enough for Qwen3-8B. When
paired traces show surface-form value mismatches, the second-stage meta prompt
should prefer a concrete, domain-neutral exact-copy rule over vague fidelity
wording.
