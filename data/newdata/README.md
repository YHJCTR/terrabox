# Terrabox OpenEarth + EarthBench SFT Data

Generated from `data/merged/sft_full.json` and aligned to the current Terrabox tool interface.

## Files

- `sft_train_full.jsonl`: all 8878 merged training samples used by the tokens experiment.
- `sft_train_strict.jsonl`: 8646 samples whose adapted gold tool calls are executable under the current Terrabox schemas.
- `tools_catalog.json`: the 47-tool catalog injected into the SFT system prompt.
- `manifest.json`: counts, source provenance, and format notes.

## Recommended Use

Use `sft_train_strict.jsonl` for cold-start SFT. Use `sft_train_full.jsonl` for auditing, repair, and experiments that can handle lossy or schema-mismatched arguments.

Each JSONL row contains `messages` in chat format. Assistant turns use OpenEarthAgent-style JSON actions, but tool names and arguments are converted to current Terrabox slugs and schemas.
