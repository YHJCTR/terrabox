# GEPA AIME PromptEvo Adapter

This adapter runs a prompt-only AIME scenario aligned with GEPA's public
`gepa.examples.aime` example:

- training/validation split: `AI-MO/aimo-validation-aime`, shuffled with seed 0
  and split in half;
- test split: `MathArena/aime_2025`;
- optimized prompt slot: only the static math system prompt;
- metric: exact final integer answer extracted from the required `### <answer>`
  format.

The adapter reads local HuggingFace cache by default:

```bash
HF_HOME=/data1/yuhongjie2/hf_cache
HF_DATASETS_CACHE=/data1/yuhongjie2/hf_datasets_cache
HF_HUB_OFFLINE=1
```

Run a LongCat no-think three-stage experiment:

```bash
cd /data1/yuhongjie2/terrabox
AGENT_CONFIG_PATH=/data1/yuhongjie2/terrabox/agent_config.yaml \
PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.promptevo.adapters.gepa_aime.pipeline \
  run-three-stage \
  --group promptevo_gepa_aime_longcat_base_stage1_stage2_$(date +%Y%m%d_%H%M%S) \
  --provider longcat \
  --split train \
  --limit 45 \
  --candidate-validation-tasks 8 \
  --min-interval 2
```

`--candidate-validation-tasks N` enables the GEPA-like selection loop for both
PromptEvo stages: Stage1 generates several prompt candidates and runs each on
the same small real validation slice before choosing one; Stage2 does the same
for contrastive candidates. Set it to `0` to use the older static-only selector.
The validation rollouts are saved under `<group>/validation/` and are not mixed
with the final base/stage1/stage2 metrics.

Check progress:

```bash
cd /data1/yuhongjie2/terrabox && PYTHONPATH=src /data/yhj/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.promptevo.adapters.gepa_aime.pipeline status --group <group>
```

Outputs live under:

```text
src/terrabox/evolution/promptevo/adapters/gepa_aime/experiments/<group>/
evolution_store/promptevo/gepa_aime/versions/
```
