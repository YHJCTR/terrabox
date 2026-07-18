# AgentDojo adapter instructions

This adapter targets the upstream checkout at `/data1/yuhongjie2/agentdojo`
without modifying that checkout.

- Optimize only the static `default` system message. Pass prompt versions with
  AgentDojo's `--system-message`; never overwrite `system_messages.yaml`.
- Formal experiments use benchmark `v1.2.2`, all four suites, a clean phase,
  and an `important_instructions` attack phase. Four Qwen lanes dynamically
  consume 4 clean jobs plus one disjoint job per injection task; do not regress
  to fixed suite-to-GPU pinning because workspace dominates the workload.
- Local Qwen3 8B uses AgentDojo CLI provider `VLLM_PARSED` (result pipeline name
  `vllm_parsed`) with vLLM Hermes native
  tool parsing. The `local` provider is a different regex/text-tag parser and
  must not be substituted silently. Keep the adapter-local `qwen_no_think`
  module in every formal run so each request explicitly sends
  `chat_template_kwargs.enable_thinking=false`. Keep a finite local output cap
  (`TERRABOX_AGENTDOJO_QWEN_MAX_TOKENS`, default 2048) so one AgentDojo turn
  cannot decode indefinitely and starve a GPU lane.
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
- Local Qwen3 8B has a finite context window. If one AgentDojo sample fails
  deterministically with `maximum context length` / `input tokens` overflow,
  record that sample as a failed `terrabox_synthetic` result with
  `terrabox_skip_reason=context_length_exceeded`, then resume the shard so later
  tasks continue. Do not let a single overlength sample trap the watcher in an
  infinite job retry loop; also do not silently truncate the prompt/history for
  this formal local-model condition. The adapter-local `qwen_no_think` module
  must normalize vLLM generic 400 context-window errors to
  `context_length_exceeded`, because upstream AgentDojo only continues after
  seeing that OpenAI-style error code.
- If upstream AgentDojo evaluator code crashes on one already-written sample
  (for example a traceback while checking utility/security), first check whether
  it is a reproducible adapter/runtime compatibility bug and patch it in the
  adapter-local module instead of scoring it as a model failure. Only annotate
  the sample as failed with `terrabox_skip_reason=evaluator_error` when the
  exception cannot be safely repaired without changing benchmark semantics. Rich
  stdout can wrap trace markers, so the fallback must also inspect the newest
  partially written result JSON that lacks utility/security labels.
- The adapter-local module must rebuild `TaskResults` forward references under
  Pydantic 2.13 before resume loads existing JSON, and must keep the
  `CalendarEvent` hash compatibility patch used by AgentDojo workspace
  evaluator/DeepDiff under Pydantic 2.x. Do not remove these as no-think-only
  cleanup.
- Long jobs stream stdout/stderr directly to experiment-local log files. Timeout
  and launcher failures must write `run_status.json`, so watchdog retries never
  depend on an empty terminal or an in-memory subprocess buffer.
- Only one AgentDojo rollout may own the four GPU lanes. Keep the cross-process
  `tmp/agentdojo_rollout.lock`, clean only stale `agentdojo-qwen3-*` containers
  after acquiring it, and never stop unrelated Tau2 or user containers.
- Before any smoke/full run, execute the pipeline `preflight`. Missing upstream
  dependencies are a blocker and must be reported, not bypassed with mock data.
- Before the first full Base for a configuration, run pipeline `smoke`. It uses
  GPU0 for one clean user task and one attacked user/injection pair (3 results),
  validates the same vLLM/Hermes/evaluator/result-count path, writes only under
  `tmp/agentdojo_smoke/`, and must finish before the 1081-result Base starts.
- If the conda environment is read-only, install missing AgentDojo-only packages
  into `tmp/agentdojo_site_packages/`; core preflight and runner add it to
  `PYTHONPATH` automatically. Do not fall back to `/home/*/.local`.
- Stage1 must sample both security and utility failures. Stage2 must compare the
  same Base/Stage1 task IDs and explicitly protect security while improving
  utility. Existing `stage1_proposal.json` and `stage2_contrastive.json` are
  paid-API checkpoints and must be reused on watcher restart.
- PromptEvo meta-prompt v2 is additive and opt-in: use `chain-after-base
  --optimizer-version v2` for new generic optimization experiments. The default
  remains v1 so historical prompt versions and old chains are not overwritten.
  Stage1 v2 and Stage2 v2 are intentionally different: Stage1 reads one
  version's traces to propose a conservative first edit; Stage2 reads the exact
  Base->Stage1 prompt diff plus paired task traces to keep gains and narrowly
  repair regressions. Do not collapse them into one shared meta prompt.
- Keep `AGENTS.md`, `CLAUDE.md`, this adapter README, the shared adapter README,
  and `EXPERIMENT_TODO.md` synchronized when changing scope, metrics, provider,
  GPU layout, or output paths.
