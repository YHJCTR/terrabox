# ToolBench adapter

Static instruction slot:

- ToolBench prompts are usually assembled by its inference pipeline and method
  configuration rather than a single constant. Treat the selected system/task
  instruction as the prompt version and keep dynamic API documentation,
  retrieved tools, user query, and rollout history out of the saved prompt.

Local data:

- ToolBench source checkout: `/data1/yuhongjie2/ToolBench`
- StableToolBench / StepTool checkout: `/data1/yuhongjie2/StepTool/stabletoolbench`
- Typical result files are JSON outputs from ToolBench inference/evaluation
  runs, for example files produced by `toolbench/inference/qa_pipeline.py`.

What this adapter currently does:

- Provides `make_toolbench_components(...)` for the standard
  prompt/trace/metric adapter trio.
- Provides `StableToolBenchRolloutRunner` for running StableToolBench's
  official `qa_pipeline_multithread.py` with an experiment-local static prompt
  override.
- Provides `pipeline.py` for formal Base -> Stage1 -> Stage2 runs on all six
  StableToolBench solvable query groups.
- Reads result JSON files without importing ToolBench.
- Converts CoT/DFS/DFSDT traces into generic promptevo `Trace` objects.
- Reports task success, pass/fail flags, tool-call counts, error-code buckets,
  transient-error rates, and final-answer availability from saved result files.

Static prompt override:

- The optimized slot is only `Prompts/ReAct_prompts.py`'s
  `FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION`.
- The runner does **not** rewrite ToolBench/StableToolBench source. It starts a
  Python wrapper that preloads `Prompts.ReAct_prompts`, replaces that constant
  for the current process, and then imports StableToolBench's official pipeline.
- Dynamic tool descriptions, retrieved tools, user query, rollout history, API
  server behavior, and ToolEval judging remain outside promptevo's prompt slot.

Minimal usage:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.toolbench import (
    make_toolbench_components,
)

results = "/path/to/toolbench/results"
prompts, traces, metrics = make_toolbench_components(
    results_dir=results,
    # Optional: ToolEval pass labels from JSON/CSV/TSV.
    tool_eval_labels="",
    # Optional: plain-text static prompt extracted from a prior result.
    base_prompt_path="",
)

print(metrics.aggregate("existing_run"))
print(next(iter(traces.traces("existing_run")), None))
PY
```

StableToolBench dry-run command construction:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.toolbench import (
    StableToolBenchRolloutRunner,
    StableToolBenchRunConfig,
)

runner = StableToolBenchRolloutRunner(
    run_config=StableToolBenchRunConfig(
        group="G1_instruction",
        backbone_model="qwen2",
        model_path="qwen2",              # served model name in vLLM
        vllm_api_base="http://127.0.0.1:8084/v1/",
        service_url="http://localhost:8081/virtual",
        method="DFS_woFilter_w2",
        num_thread=4,
    )
)
exp_dir = runner.run_version("base", "promptevo_stabletoolbench_base_smoke", dry_run=True)
print(exp_dir)
PY
```

Preflight before real runs:

```bash
conda run -n unsloth bash -lc 'cd /data1/yuhongjie2/terrabox && PYTHONPATH=src python - <<"PY"
from terrabox.evolution.promptevo.adapters.toolbench import stable_toolbench_has_core_dependencies
print(stable_toolbench_has_core_dependencies())
PY'
```

On this machine, the adapter preflight passes in `unsloth` after installing the
small missing official dependency `termcolor`. The local StableToolBench source
also imports several optional LLM/retrieval wrappers unconditionally; the runner
adds process-local compatibility aliases/stubs for wrappers that are not used by
the qwen2/llama3/ToolLLaMA_vllm path. Selecting one of those unavailable
backbones still fails explicitly rather than silently falling back.

Actual StableToolBench rollout prerequisites:

1. Either start services manually, or use `pipeline.py` below. The formal
   pipeline starts the StableToolBench cached tool server and per-GPU vLLM
   lanes explicitly, records logs/metadata, and stops resources it started.
2. If using `StableToolBenchRolloutRunner` directly, start a vLLM
   OpenAI-compatible server whose served model name matches
   `StableToolBenchRunConfig.model_path`; for Qwen-style runs the official
   wrapper uses `--backbone_model qwen2` and `VLLM_API_BASE`.
3. Run `StableToolBenchRolloutRunner.run_version(...)`. Outputs are saved under
   `src/terrabox/evolution/promptevo/adapters/toolbench/experiments/<experiment>/answers/<group>/`.
4. For official pass-rate/preference metrics, use StableToolBench's own
   `toolbench/tooleval` conversion and evaluation scripts on the generated
   answer directory. The adapter's `ToolBenchMetricProvider` can read raw JSON
   immediately, but raw structural success is not a replacement for ToolEval
   semantic pass-rate.

For upstream ToolBench, the adapter can still be used read-only: run ToolBench
with its official pipeline, then point promptevo at the saved results. The
included runner is specifically for the local StableToolBench/StepTool checkout
and mirrors that project's official evaluation command.

Local notes:

- `/data1/yuhongjie2/ToolBench` is present but appears to contain the upstream
  source checkout plus demo data, not the full benchmark data dump.
- `/data1/yuhongjie2/StepTool/stabletoolbench` is also present and includes a
  StableToolBench-style cached tool-response server.

Formal three-stage pipeline:

```bash
conda run -n unsloth bash -lc 'cd /data1/yuhongjie2/terrabox && PYTHONPATH=src \
python -m terrabox.evolution.promptevo.adapters.toolbench.pipeline preflight'
```

```bash
conda run -n unsloth bash -lc 'cd /data1/yuhongjie2/terrabox && PYTHONPATH=src \
python -m terrabox.evolution.promptevo.adapters.toolbench.pipeline full-chain \
  --base-experiment promptevo_stabletoolbench_base_qwen3_YYYYMMDD_HHMMSS \
  --stage1-experiment promptevo_stabletoolbench_stage1_qwen3_YYYYMMDD_HHMMSS \
  --stage2-experiment promptevo_stabletoolbench_stage2_qwen3_YYYYMMDD_HHMMSS \
  --stage1-version promptevo_stabletoolbench_stage1_qwen3_YYYYMMDD_HHMMSS \
  --stage2-version promptevo_stabletoolbench_stage2_qwen3_YYYYMMDD_HHMMSS \
  --provider longcat --optimizer-version v2'
```

The default profile:

- uses all six stable groups:
  `G1_instruction,G1_category,G1_tool,G2_instruction,G2_category,G3_instruction`;
- schedules those groups dynamically across GPU lanes `0,1,2,3` on ports
  `9300..9303`;
- serves the local Qwen3 8B checkpoint as model name `qwen2` so the upstream
  `Qwen2Model` wrapper can keep using its official completion prompt path;
- mirrors StepTool's official `qwen2` script: `DFS_woFilter_w2`,
  `max_observation_length=1024`, `max_query_count=30`, `num_thread=4`.

Progress/status:

```bash
conda run -n unsloth bash -lc 'cd /data1/yuhongjie2/terrabox && PYTHONPATH=src \
python -m terrabox.evolution.promptevo.adapters.toolbench.pipeline status \
  --experiment promptevo_stabletoolbench_base_qwen3_YYYYMMDD_HHMMSS'
```

Outputs:

- answer JSONs:
  `src/terrabox/evolution/promptevo/adapters/toolbench/experiments/<experiment>/answers/<stable_group>/`
- group statuses:
  `src/terrabox/evolution/promptevo/adapters/toolbench/experiments/<experiment>/status/<stable_group>.json`
- stage metadata and aggregate structural metrics:
  `pipeline_status.json`, `metrics_summary.json`, `run_meta_<stable_group>.json`
- generated prompts:
  `evolution_store/promptevo/toolbench/versions/<version>.txt`

Important metric note:

- `metrics_summary.json` is an immediate structural metric over rollout JSONs
  (`valid_data` / `Finish(give_answer)` and error buckets). It is useful for
  PromptEvo sampling and quick diagnosis.
- Paper-style semantic pass-rate should still be produced with StableToolBench's
  official `toolbench/tooleval` conversion and pass-rate scripts when reporting
  final results.
