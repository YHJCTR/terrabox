# AgentDojo adapter instructions

This adapter targets the upstream checkout at `/data1/yuhongjie2/agentdojo`
without modifying that checkout.

- Optimize only the static `default` system message. Pass prompt versions with
  AgentDojo's `--system-message`; never overwrite `system_messages.yaml`.
- Formal experiments use benchmark `v1.2.2`, all four suites, a clean phase,
  and an `important_instructions` attack phase. Four Qwen lanes dynamically
  consume 4 clean jobs plus one disjoint job per injection task; do not regress
  to fixed suite-to-GPU pinning because workspace dominates the workload.
- Local Qwen3 8B uses AgentDojo provider `vllm_parsed` with vLLM Hermes native
  tool parsing. The `local` provider is a different regex/text-tag parser and
  must not be substituted silently. Keep the adapter-local `qwen_no_think`
  module in every formal run so each request explicitly sends
  `chat_template_kwargs.enable_thinking=false`.
- Keep result task IDs stable as
  `suite/user_task/attack_type/injection_task`. Treat `None`, empty strings,
  and the string `none` as non-attack sentinels.
- Do not invent tool F1: AgentDojo result JSON contains utility/security labels,
  not a gold tool sequence suitable for that metric.
- Experiment groups live under `agentdojo/experiments/<group>/`; prompt versions
  live under `evolution_store/promptevo/agentdojo/versions/`.
- Runs must be resumable (`force_rerun=False`) and must stop their own vLLM
  containers in `finally`. A job is complete only when its valid result count
  exactly matches `expected_results`; CLI return code 0 alone is insufficient.
  A group is complete only when all 39 job statuses are complete and the stage
  total matches the preflight total (currently 1081).
- Long jobs stream stdout/stderr directly to experiment-local log files. Timeout
  and launcher failures must write `run_status.json`, so watchdog retries never
  depend on an empty terminal or an in-memory subprocess buffer.
- Only one AgentDojo rollout may own the four GPU lanes. Keep the cross-process
  `tmp/agentdojo_rollout.lock`, clean only stale `agentdojo-qwen3-*` containers
  after acquiring it, and never stop unrelated Tau2 or user containers.
- Before any smoke/full run, execute the pipeline `preflight`. Missing upstream
  dependencies are a blocker and must be reported, not bypassed with mock data.
- If the conda environment is read-only, install missing AgentDojo-only packages
  into `tmp/agentdojo_site_packages/`; core preflight and runner add it to
  `PYTHONPATH` automatically. Do not fall back to `/home/*/.local`.
- Stage1 must sample both security and utility failures. Stage2 must compare the
  same Base/Stage1 task IDs and explicitly protect security while improving
  utility. Existing `stage1_proposal.json` and `stage2_contrastive.json` are
  paid-API checkpoints and must be reused on watcher restart.
- Keep `AGENTS.md`, `CLAUDE.md`, this adapter README, the shared adapter README,
  and `EXPERIMENT_TODO.md` synchronized when changing scope, metrics, provider,
  GPU layout, or output paths.
