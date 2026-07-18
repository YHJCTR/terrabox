# ToolBench / StableToolBench adapter instructions

This adapter targets the local StableToolBench checkout at
`/data1/yuhongjie2/StepTool/stabletoolbench` and the upstream ToolBench source
checkout at `/data1/yuhongjie2/ToolBench`.

## Scope

- Optimize only the static ReAct system prompt slot:
  `toolbench/inference/Prompts/ReAct_prompts.py::FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION`.
- Do not edit StableToolBench source files for experiments. Apply prompt
  versions by process-local override before importing the official pipeline.
- Do not optimize dynamic API documentation, retrieved tools, user query,
  rollout history, observations, cached API responses, or ToolEval labels.

## Official rollout口径

- Use StableToolBench's official `qa_pipeline_multithread.py` path through
  `runner.py` / `pipeline.py`.
- Default local profile mirrors StepTool's qwen2 script:
  `backbone_model=qwen2`, `method=DFS_woFilter_w2`,
  `max_observation_length=1024`, `max_query_count=30`, `num_thread=4`.
- The local pipeline serves `/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B` as
  model name `qwen2` so the upstream qwen2 completion wrapper remains unchanged.
- Experiment outputs live under
  `src/terrabox/evolution/promptevo/adapters/toolbench/experiments/<experiment>/`.
- Generated prompt versions live under
  `evolution_store/promptevo/toolbench/versions/`.

## Services

- `pipeline.py` may explicitly start the StableToolBench cached tool server
  (`server/main.py`, port 8081) and per-GPU vLLM lanes.
- Record service logs and per-group metadata. Stop only resources started by the
  current pipeline run.
- Do not silently fall back to another benchmark implementation, another prompt
  slot, or synthetic metrics if the official pipeline fails.

## Metrics

- Adapter `metrics_summary.json` is structural: it reflects saved rollout JSONs,
  `valid_data` / `Finish(give_answer)`, and error buckets.
- For paper-style final reporting, use StableToolBench's official
  `toolbench/tooleval` conversion and pass-rate scripts on generated answers.
