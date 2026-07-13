# Terrabox/OEA adapter

This adapter is the native Terrabox/OpenEarthAgent path for promptevo. It
optimizes the static ReAct system prompt while leaving task input, tool
schemas, tool observations, and conversation history dynamic.

## Static Prompt Slot

Optimized prompt:

- `src/terrabox/agent/session.py::_REACT_SYSTEM_PROMPT`

Runtime override:

- `TERRABOX_REACT_SYSTEM_PROMPT_FILE=/path/to/version.txt`

The standard eval path reads the override in
`src/terrabox/agent/eval_modes/standard.py`; if the variable is unset, it uses
the original hard-coded prompt.

## Base OEA Experiment

Current full OEA base rollout:

- experiment: `oe_full_react_offline`
- results: `tmp/trajectories/oe_full_react_offline/standard/results/`
- task file: `data/oea_full_sft/openearth_test_tasks.json`
- total tasks: 1162

`results/<task_id>.json` is the authority. Derived files such as
`trajectories_full.jsonl` can be rebuilt with:

```bash
cd /data1/yuhongjie2/terrabox
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.terrabox import rebuild_trajectory_files
print(rebuild_trajectory_files())
PY
```

## Refreshing Stale Tool Results

If a tool implementation changes, a full-looking experiment may still contain
stale per-task results. Do not judge freshness only from `1162/1162`.

For SAM2 changes, refresh the union of:

- tasks whose gold `expected_tools` include `geo_perception.sam2_segment`;
- tasks whose existing `results/<task_id>.json` actually called
  `geo_perception.sam2_segment`.

Move those JSON files into a timestamped backup directory under the same
`results/` directory, then rerun the same experiment with `--resume`. This keeps
old evidence recoverable while forcing only the affected tasks to rerun.

The current temporary pipeline does this before stage-1 optimization:

```bash
bash tmp/run_oea_stage1_after_sam2_refresh_20260630.sh
```

It writes adapter-local artifacts under:

```text
src/terrabox/evolution/promptevo/adapters/terrabox/experiments/<run_id>/
```

## Stage-1 Prompt Optimization

```bash
cd /data1/yuhongjie2/terrabox
no_proxy=localhost,127.0.0.1 PYTHONPATH=src python -m terrabox.evolution.promptevo.run propose \
  --trajectories tmp/trajectories/oe_full_react_offline/standard/trajectories_full.jsonl \
  --out src/terrabox/evolution/promptevo/adapters/terrabox/experiments/<run>/stage1_proposal.json
```

Then accept it as a Terrabox-scoped version:

```bash
PYTHONPATH=src python -m terrabox.evolution.promptevo.run accept \
  --proposal src/terrabox/evolution/promptevo/adapters/terrabox/experiments/<run>/stage1_proposal.json \
  --name <version> \
  --versions-dir evolution_store/promptevo/terrabox/versions
```

## Full Rollout

Use the ReAct/OEA runner documented in `src/terrabox/evolution/ReAct/AGENTS.md`.
For promptevo, export the prompt version before running:

```bash
export TERRABOX_REACT_SYSTEM_PROMPT_FILE=$(pwd)/evolution_store/promptevo/terrabox/versions/<version>.txt
```

Use the shared promptevo naming stem for new experiments:

```text
promptevo_oea_<stage>_<variant>_<YYYYMMDD_HHMMSS>
```

For example, after a SAM2 refresh:

```text
promptevo_oea_stage1_sam2refresh_20260630_075048
```

Use that same stem for:

```text
evolution_store/promptevo/terrabox/versions/<stem>.txt
src/terrabox/evolution/promptevo/adapters/terrabox/experiments/<stem>/
tmp/trajectories/<stem>/standard/results/
```

Historical directories such as `promptevo_v1`, `promptevo_v2e`, and the current
`oea_stage1_sam2refresh_20260630_015032_rollout` do not need to be renamed.
Apply the convention only to future runs.

For full OEA promptevo experiments, choose the rollout layout by agent LLM
provider:

- **Local agent LLM**: use the ReAct baseline **two-flow / 3+1 GPU layout**.
  Flow A uses GPU0 for the local agent LLM, GPU1 for non-VLM perception, and
  GPU2 for VLM; it runs offline tasks first, then online `--gpu-class gpu`.
  Flow B uses GPU3 for the second local agent LLM and runs online
  `--gpu-class nogpu` in parallel.
- **External API agent LLM** (`--llm-provider longcat|deepseek`): use the
  **three-flow layout**. Perception flow P0 uses GPU0/1, perception flow P1
  uses GPU2/3, and the OSM/nogpu flow uses only the external API plus CPU/network
  tools.
- All flows write the same experiment name and rely on `--resume` to merge
  results. Apply the prompt only through `TERRABOX_REACT_SYSTEM_PROMPT_FILE`.

Do not replace the full rollout with a GPU0/1/2/3 multi-shard `nogpu` run unless
the perception flow is intentionally not running. A pure `nogpu` shard command
is only a special-case backfill mode.

## Metrics

Use the shared rollout report commands:

```bash
PYTHONPATH=src python -m terrabox.evolution.shared.rollout_report status <experiment> --scope all --total 1162
PYTHONPATH=src python -m terrabox.evolution.shared.rollout_report compare <experiment> oe_full_react_offline
```
