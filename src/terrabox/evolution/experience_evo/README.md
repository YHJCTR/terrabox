# ExperienceEvo

Offline artifact-transition experience evolution for Terrabox rollouts.

This module is intentionally separate from `promptevo/`: PromptEvo evolves a
static system prompt, while ExperienceEvo builds an external experience bank.
The first MVP is offline-only:

```text
historical rollout results
  -> artifact transition extraction
  -> LongCat/DeepSeek/local distillation
  -> JSONL + SQLite experience store
  -> top-k retrieval and prompt injection during eval
```

## Experience Unit

An entry describes a reusable transition:

```text
current artifact state -> next artifact/result state
```

Two levels are stored:

- `signature`: `task_type + input_signature + output_signature`
- `tool`: `task_type + input_signature + output_signature + tool`

Stored fields include `q`, `n`, `risk`, `status`, input constraints, output
checks, downstream use, recovery notes, and parameter notes.

The injected prompt never includes `expected_tools`, final answers, exact file
paths, or task ids. Historical F1/reward is used only to rank and label
experiences. Distillation examples are sanitized before they are sent to the
LLM: concrete task questions, place names, file paths, layer names, and
free-form text prompts are replaced with placeholders such as `<named_area>`,
`<artifact_reference>`, and `<task_specific_text_prompt>`.

## 2026-07-26 MVP Behavior

Current code is still an offline MVP, but it now follows the 4.2 product
transition direction more closely:

- Infra attribution: timeout, 429/rate-limit, network/proxy failures, provider
  overload, CUDA/OOM, context-length failures, Docker/container/service-health
  failures are marked `infra_error` and filtered out of experience distillation.
  They do not increase `risk`.
- Experience risk: only observable LLM/tool-use mistakes such as missing
  required parameters, invalid arguments, nonexistent file/layer/artifact
  references, invalid geometry/bbox/expression, etc. set `risk=1.0`.
- Retrieval: eval does not load all experiences into context. It uses a small
  lexical retriever with stopword removal, underscore token splitting, and OEA
  domain aliases (for example `fire/police/station -> poi/add_pois_layer`,
  `closest/nearest -> distance/compute_route_dist`). This is a cheap MVP
  retriever, not an embedding retriever.
- Two-stage prompt block: `ExperienceEvoPromptInjector` first retrieves
  signature-level product transitions, then finds tool-level children under the
  same `input_signature -> output_signature` and computes:

```text
lambda = Ntool / (Ntool + k)
Quse = lambda * Qtool + (1 - lambda) * Qsig
```

  The prompt shows Stage A product guidance and Stage B `Quse` tool
  recommendations. This is still task-start static injection because
  `scripts/run_trajectory_experiment.py` currently calls `augment(question)`
  once before the agent loop. True step-level retrieval/checker needs a later
  graph/tool-loop hook.
- Group selection: when `--max-groups` is set, distillation now uses a balanced
  group cut before filling by support/quality, so the store does not collapse to
  only high-frequency OSM/common-tool buckets.
- Distillation guardrails: LongCat/DeepSeek distillation prints bucket progress
  (`--progress-every`, default 10). `--allow-template-fallback` is capped by
  `--max-template-fallback-ratio` (default 0.25) and
  `--max-consecutive-template-fallbacks` (default 8), so provider/API problems do
  not silently degrade an entire store into template-only experiences.

## 2026-07-31 v2 Product-Transition Mode

旧模式没有被覆盖:

- `experience_evo` / `build` / `preview` 仍走 v1 bucket schema:
  `transitions.jsonl` -> `experiences.jsonl` -> `experience_evo.sqlite`.
- `experience_evo_v2` / `build-v2` / `preview-v2` 走新 schema:
  `events_v2.jsonl` -> `families_v2.jsonl` -> `experience_evo_v2.sqlite`.

v2 的核心变化是把一条经验定义成 atomic product-state transition family:

```text
Train rollout event
  -> typed input_product_state
  -> typed target_product_state
  -> local evidence score r / risk_observed
  -> product-level Qsig/Nsig/Rsig
  -> tool-level Qtool/Ntool/Rtool policies
  -> prompt-time Quse ranking
```

v2 不保存完整轨迹文本给 eval agent；它只保存当前产物状态、下一产物目标、输入绑定规则、
输出检查、下游消费规则、工具参数约束和恢复建议。`infra_error` 仍只过滤，不提升 risk；
risk 只来自可归因的工具使用错误，例如 missing required parameter、invalid argument、
不存在的文件/图层/产物引用等。

Build a v2 store without provider calls:

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner build-v2 \
  --source tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --source-name oea_train2000_longcat \
  --store-dir evolution_store/experience_evo/oea_train2000_v2 \
  --template-only \
  --min-support 1
```

Build a v2 store with LongCat distillation:

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner build-v2 \
  --source tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --source-name oea_train2000_longcat \
  --store-dir evolution_store/experience_evo/oea_train2000_v2_longcat \
  --provider longcat \
  --allow-template-fallback \
  --min-support 1 \
  --max-families 500
```

Preview v2 injection:

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner preview-v2 \
  --store-dir evolution_store/experience_evo/oea_train2000_v2 \
  --query "Calculate the percentage of high temperature pixels in the scene."
```

Run eval with v2:

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_v2_oea_eval \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_v2_oea_eval/standard \
  --llm-provider longcat \
  --evolution-method experience_evo_v2 \
  --evolution-store evolution_store/experience_evo/oea_train2000_v2 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos
```

Switching modes is only the method name and store:

```text
Old: --evolution-method experience_evo    --evolution-store <v1 store>
New: --evolution-method experience_evo_v2 --evolution-store <v2 store>
```

## Build From LongCat OEA Base

The default source is the existing LongCat OEA base rollout:

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner build \
  --source tmp/trajectories/promptevo_oea_base_longcat2_20260704_031626_rollout/standard/results \
  --source-name oea_longcat_base \
  --store-dir evolution_store/experience_evo/oea_longcat_base_offline \
  --provider longcat \
  --min-reward 0.8 \
  --max-groups 50
```

For a cheap smoke test without API calls:

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner build \
  --max-tasks 20 \
  --max-groups 10 \
  --template-only
```

Preview retrieval:

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner preview \
  --store-dir evolution_store/experience_evo/oea_longcat_base_offline \
  --query "Which fire station and police station are closest to each other in Banff National Park?"
```

Run an eval rollout with the store:

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_oea_longcat_smoke \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_oea_longcat_smoke/standard \
  --llm-provider longcat \
  --evolution-method experience_evo \
  --evolution-store evolution_store/experience_evo/oea_longcat_base_offline \
  --limit 5 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos
```

## Current Smoke

Commands already verified on 2026-07-24:

```bash
# LongCat distill smoke from the existing LongCat base, high-quality transitions only.
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.experience_evo.runner distill \
  --store-dir tmp/experience_evo/smoke_longcat_store \
  --provider longcat \
  --max-groups 3 \
  --max-transitions 232 \
  --min-support 1 \
  --allow-template-fallback \
  --progress-every 1

# One-task compute-only rollout smoke.
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  scripts/run_trajectory_experiment.py rollout \
  --task-file tmp/experience_evo/compute_smoke_tasks.json \
  --experiment experience_evo_compute_smoke \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_compute_smoke/standard \
  --llm-provider longcat \
  --evolution-method experience_evo \
  --evolution-store tmp/experience_evo/smoke_longcat_store \
  --limit 1 \
  --resume
```

Observed smoke result:

- ExperienceEvo: `oea_test_50`, completed, set-F1 `1.00`, tools `compute.solver`.
- Reflection with existing `oe_full_reflection_div1k` memory: completed, set-F1 `1.00`.
- MemRL source lite store built from 60 LongCat base trajectories: completed, set-F1 `0.67` due to one extra OSM tool call.

These smoke results only validate wiring and prompt injection. They are not
quality conclusions because the task set has one compute-focused example and
does not exercise OSM/perception services.

For external LongCat full OEA runs, follow the module-level provider guidance:
use three streams for full-scale OEA (`GPU0/1`, `GPU2/3`, plus OSM/nogpu) and
`TERRABOX_TOOL_SERVICE_SCOPE=call TERRABOX_KEEP_VLM_WARM=1`. Do not use
`session` scope for external API three-stream experiments.

## Comparison Plan

First smoke on the existing test-only LongCat base:

1. Treat `promptevo_oea_base_longcat2_20260704_031626_rollout` as the LongCat
   ReAct/base reference.
2. Build ExperienceEvo from that same historical base only as a transductive
   smoke/case-library test.
3. Run a few eval tasks with `--evolution-method experience_evo` and compare via
   `rollout_report compare`.

For a cleaner train->test comparison:

1. Shuffle OEA train tasks and select about 2,000 tasks covering diverse tool
   sequences.
2. Run train rollouts with LongCat for Reflection, MemRL source, and
   ExperienceEvo using the same train subset.
3. Build each offline memory/experience store from that train subset.
4. Evaluate all methods on the 1,162 OEA test tasks with the same LongCat
   rollout settings.

The train subset command used for the first planned comparison is:

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner select-train \
  --output tmp/experience_evo/oea_train_coverage_2000_seed42_tasks.json \
  --limit 2000 \
  --seed 42
```

This selected 2,000 of 14,538 OEA train rows and covered 87 distinct
`expected_tools` sequences.

The current after-train watcher is:

```bash
tmux attach -t expevo_after_train_20260726
tail -f tmp/experience_evo/nightly_20260724/logs_after_train/after_train_eval.log
```

It waits for `longcat_oea_train2000_seed42_base_nightly_20260724`, builds
`evolution_store/experience_evo/oea_train2000_longcat_balanced_20260726`, then
runs OEA test with:

Before building stores, the watcher runs a train repair pass. It re-launches the
same three train streams with `--resume`; `run_trajectory_experiment.py` now
re-runs existing transient failed result files that still have retry budget, but
does not retry deterministic LLM/tool-use errors such as context overflow,
missing required parameters, invalid arguments, or nonexistent files/layers.
The watcher also pins lane-specific non-VLM service ports for P1
(`9012`-`9017`) so P0 and P1 do not rebuild the same SAM2/RemoteSAM/InstructSAM/
Strip-RCNN/RemoteCLIP/ChangeOS container while running in parallel.

```text
experience_evo_oea_train2000_balanced_eval_20260726
reflection_oea_train2000_longcat_eval_20260726
memrl_source_oea_train2000_longcat_eval_20260726
```
