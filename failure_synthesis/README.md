# Failure Synthesis

This folder is an isolated test package for generating failure trajectories from
Terrabox SFT-style data. It intentionally stays outside `scripts/` and does not
write evolution-internal trajectory adapters.

When changing this module, update this document if the experiment command,
input contract, output contract, environment requirement, or failure definition
changes. Future experiment runs should start from the commands here and then
adjust limits, skips, ports, and output names as needed.

## Methods

- `agent_rollout_failures.py`: asks the deployed Docker/vLLM agent LLM to choose
  tools step by step, executes the selected tools through the Terrabox registry,
  feeds observations back to the LLM, and saves failed and optionally successful
  rollouts in the source dataset format.
- `replay_failures.py`: replays source gold tool calls against the current
  Terrabox registry. For Disaster v2, replay resolves common image placeholders
  such as `rgb.tif` / `pre.tif` / `post.tif` and step references such as
  `$step1.bboxes`; replay failures are marked `source_gold_invalid` so they are
  not confused with agent planning failures.
- `perturb_failures.py`: creates deterministic counterfactual failures by
  dropping steps, swapping adjacent actions, replacing tools, or mutating
  numeric/path arguments. It does not call an LLM.

## Main Experiment

The current real-agent experiment is `agent_rollout_failures.py`.

For each selected source sample, the script:

1. Reads the task prompt and source-native metadata.
2. Builds a dataset-specific allowed tool/action list from tools used by that
   dataset.
3. Sends the prompt, image paths, allowed tools, and previous observations to
   the deployed OpenAI-compatible Docker/vLLM agent LLM.
4. Requires the LLM to return one JSON tool decision at a time.
5. Executes the selected Terrabox tool through the registry.
6. Records tool arguments, observations, errors, final answer, and tool F1
   against the source trajectory.
7. Saves the rollout as a failure if a tool/runtime error occurs, the LLM emits
   invalid JSON, an unregistered tool is selected, max steps are reached, or
   tool F1 is below `--max-success-f1`.

The output is intentionally still source-format data, not `Trajectory JSON` or
`agentevolver.jsonl`. Feed-to-evolution conversion should be handled later by a
separate adapter.

## Environment

Use the `unsloth` conda environment for real replay/rollout because the default
system Python may be too old for the Terrabox runtime.

Check that the local Docker/vLLM agent LLM is available:

```bash
no_proxy=localhost,127.0.0.1 curl http://localhost:9100/v1/models
```

The rollout script defaults to `http://localhost:9100`. Override it with
`--llm-url`, `AGENT_LLM_URL`, or `AGENT_LLM_PORT`.

Use `no_proxy=localhost,127.0.0.1` when calling local LLM services from this
server environment.

## Inputs

### Disaster v2

Use the v2 SFT file for the current experiment because the original Disaster SFT
tool chains contain redundant steps.

- Main source: `data/disaster_sft_dataset_v2.json`
- Image mapping: `data/sft_augmented_image_mapping.json`
- Current experiment subset: first 90 samples, controlled by `--limit 90`

The mapping file is used to attach the corresponding pre/post image paths to
Disaster samples before the LLM sees the task.

### OpenEarth

- Main source: `data/openearth/train.json`
- Default ChangeOS skip: `ChangeDetection` and `geo_perception.change_os_detect`

OpenEarth records that use search actions need external API access. In the
current source file, `1329 / 14538` records use `GoogleSearch`/`BingSearch`,
which is about `9.14%`. These map to `bing_search.search` in Terrabox and need
`BRIGHTDATA_API_KEY` unless they are skipped.

ChangeOS records are `292 / 14538`, about `2.01%`, and should stay skipped until
the ChangeOS service is implemented instead of mocked.

## Outputs

Default outputs are written under `failure_synthesis/outputs/`.

- Disaster failure outputs keep the `disaster_sft_dataset` style with a
  top-level `samples` list.
- OpenEarth failure outputs keep the `train.json` conversation-list style.
- Successful real-agent rollouts are written only when `--success-output` is
  provided.
- Real-agent rollout writes checkpoint output every source sample by default
  through `--checkpoint-every 1`, so interrupted runs keep completed
  trajectories.
- Interrupted real-agent rollout can continue from existing output files with
  `--resume`; completed `source_id` / `source_idx` records are skipped.
- Each source sample has a timeout through `--sample-timeout` to prevent one
  hanging tool or LLM call from blocking the whole run.
- `outputs/summary.json` accumulates run statistics by dataset, method,
  failure type, tool, task type, skipped count, success count, and failure count.
- Tool artifacts produced during rollout are redirected under
  `failure_synthesis/outputs/artifacts/` so source data directories are not
  modified.
- Disaster replay can optionally write validated gold trajectories with
  `--success-output`; these are the only Disaster v2 source demonstrations that
  should be treated as execution-validated successes.

Important real-agent output fields:

- Disaster sample fields: `id`, `source_id`, `images`, `tool_calls`,
  `agent_final_answer`, `failure_type`, `failure_meta`.
- OpenEarth record fields: original `idx/images/type/question/label`,
  source-style `conversation`, `failure_id`, `source_idx`, `failure_meta`.
- `failure_meta` stores `method`, `failure_type`, `tool_f1`, expected tools,
  called tools, and captured error content.

## Real-Agent Rollout Commands

### Disaster v2, first 90 samples

```bash
conda run -n unsloth env \
  PYTHONPATH=src \
  no_proxy=localhost,127.0.0.1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python failure_synthesis/agent_rollout_failures.py \
  --dataset disaster \
  --input data/disaster_sft_dataset_v2.json \
  --mapping data/sft_augmented_image_mapping.json \
  --limit 90 \
  --max-steps 6 \
  --checkpoint-every 1 \
  --sample-timeout 600 \
  --output failure_synthesis/outputs/disaster_v2_agent_rollout_failures_90.json \
  --success-output failure_synthesis/outputs/disaster_v2_agent_rollout_success_90.json
```

### OpenEarth, skip ChangeOS only

Use this when you want to keep real search-service failures in the data. If
`BRIGHTDATA_API_KEY` is missing, search tasks will often become `tool_error`
failures.

```bash
conda run -n unsloth env \
  PYTHONPATH=src \
  no_proxy=localhost,127.0.0.1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python failure_synthesis/agent_rollout_failures.py \
  --dataset openearth \
  --input data/openearth/train.json \
  --skip-tools geo_perception.change_os_detect \
  --limit 200 \
  --max-steps 6 \
  --checkpoint-every 1 \
  --sample-timeout 600 \
  --output failure_synthesis/outputs/openearth_agent_rollout_failures.json \
  --success-output failure_synthesis/outputs/openearth_agent_rollout_success.json
```

### OpenEarth, skip ChangeOS and search API tasks

Use this as the cleaner main setting when you want fewer environment-induced
failures and do not have `BRIGHTDATA_API_KEY`.

```bash
conda run -n unsloth env \
  PYTHONPATH=src \
  no_proxy=localhost,127.0.0.1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python failure_synthesis/agent_rollout_failures.py \
  --dataset openearth \
  --input data/openearth/train.json \
  --skip-tools geo_perception.change_os_detect bing_search.search GoogleSearch \
  --limit 200 \
  --max-steps 6 \
  --checkpoint-every 1 \
  --sample-timeout 600 \
  --output failure_synthesis/outputs/openearth_agent_rollout_failures_no_search.json \
  --success-output failure_synthesis/outputs/openearth_agent_rollout_success_no_search.json
```

Resume an interrupted Disaster v2 run:

```bash
conda run -n unsloth env \
  PYTHONPATH=src \
  no_proxy=localhost,127.0.0.1 \
  PYTHONDONTWRITEBYTECODE=1 \
  python failure_synthesis/agent_rollout_failures.py \
  --dataset disaster \
  --input data/disaster_sft_dataset_v2.json \
  --mapping data/sft_augmented_image_mapping.json \
  --limit 90 \
  --max-steps 6 \
  --checkpoint-every 1 \
  --sample-timeout 300 \
  --resume \
  --output failure_synthesis/outputs/disaster_v2_agent_rollout_failures_90.json \
  --success-output failure_synthesis/outputs/disaster_v2_agent_rollout_success_90.json
```

### Optional search API setup

```bash
export BRIGHTDATA_API_KEY=your_key_here
```

## Current Limitations

- `--limit` selects the first N records. There is no `--offset` or sharding flag
  yet, so full OpenEarth runs should add that before parallel batching.
- Checkpoint files are overwritten with the latest completed sample state during
  a run. They are source-format partial outputs until the command finishes.
- `--sample-timeout` records timeout samples as `failure_type=sample_timeout`;
  partial tool artifacts may still exist under `outputs/artifacts/`.
- The LLM may still select invalid tool names even though the prompt lists
  allowed tools. These are saved as `unregistered_tool` failures.
- Search failures without `BRIGHTDATA_API_KEY` are real environment failures,
  but they can dominate OpenEarth failure statistics if search tasks are not
  skipped.
- OpenEarth ChangeOS samples are skipped because the registered tool is not a
  real implemented service for this experiment.
- Disaster image attachment depends on `data/sft_augmented_image_mapping.json`;
  v2 alternate IDs are handled conservatively, but new naming patterns may need
  extra mapping rules.
- The current success/failure split is tool-centric. Answer correctness is not
  judged semantically beyond tool errors, max steps, and tool F1.

## Replay And Perturb Examples

```bash
PYTHONPATH=src python failure_synthesis/replay_failures.py \
  --dataset disaster \
  --input data/disaster_sft_augmented.json \
  --mapping data/sft_augmented_image_mapping.json \
  --limit 20 \
  --output failure_synthesis/outputs/disaster_replay_source_invalid.json \
  --success-output failure_synthesis/outputs/disaster_replay_validated_gold.json
```

```bash
PYTHONPATH=src python failure_synthesis/replay_failures.py \
  --dataset openearth \
  --input data/openearth/train.json \
  --skip-tools geo_perception.change_os_detect \
  --limit 20
```

```bash
python failure_synthesis/perturb_failures.py \
  --dataset disaster \
  --input data/disaster_sft_augmented.json \
  --variants-per-sample 2 \
  --limit 50
```

```bash
python failure_synthesis/perturb_failures.py \
  --dataset openearth \
  --input data/openearth/train.json \
  --skip-tools geo_perception.change_os_detect \
  --variants-per-sample 2 \
  --limit 50
```
