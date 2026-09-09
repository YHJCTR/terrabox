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
- Do not treat copied `tmp/agentdojo_site_packages/` as portable across servers:
  binary wheels such as `cryptography` may require a newer glibc than the target
  machine provides. After `sshpass`/rsync migration, run `pipeline preflight`;
  if it reports glibc/import errors from this layer, move the directory aside and
  rebuild only the missing AgentDojo dependencies in-place before any formal run.
- Stage1 must sample both security and utility failures. Stage2 must compare the
  same Base/Stage1 task IDs and explicitly protect security while improving
  utility. ProtocolPatch v2 uses a fixed real dev slice, three candidates per
  stage, and resumable `optimization/stage1_protocol_patch.json` and
  `optimization/stage2_protocol_patch.json` checkpoints.
- 历史正式口径里的“三阶段”指 Base / Stage1 / Stage2。可选 Stage3 是
  post-Stage2 refinement，不替代 Stage2：它以 Stage2 prompt 为当前基线，比较
  Stage1 与 Stage2 的同任务轨迹，只提出小幅 protocol patch 来修复 Stage2 引入的回归，
  并继续使用同一固定真实 dev slice 验证；候选没有通过接受门时保留 Stage2。
  可通过 `pipeline.py stage3-after-stage2` 单独接在已完成的 Stage2 后，也可在
  `chain-after-base` 中显式传入 `--stage3-group` 与 `--stage3-version` 自动接上。
  Stage3 的可恢复检查点为 `optimization/stage3_protocol_patch.json`，且必须继续保护
  AgentDojo security gate，不允许用明显安全回归换取 utility 改善。
- To intentionally test LongCat2 as the executable AgentDojo agent, pass
  `--agent-provider longcat`. This uses upstream `OPENAI_COMPATIBLE` provider,
  the same suites/attacks/evaluators/result-count checks, and does not start
  local vLLM containers. The default remains `qwen`. External API rollouts use
  `TERRABOX_AGENTDOJO_API_WORKERS` concurrent job lanes (default 1, raise only for deliberate high-concurrency reruns); this knob
  must not affect local Qwen runs, which remain bound to the four GPU lanes.
  If LongCat/other API providers enter repeated rate-limit/timeout retries,
  first reduce this to 4 or lower, then increase the API pacing interval and
  resume the same group instead of continuing a high-concurrency retry storm.

- External API provider/rate-limit errors are retryable queue events, not final
  benchmark failures. LongCat job lanes inspect job stderr/stdout and run JSON
  for 429/rate-limit/timeout/provider-overload markers, wait briefly, and put
  the same job at the back of the queue. Keep
  `TERRABOX_AGENTDOJO_QUEUE_RETRIES` finite (default 12), and do not synthesize
  these as evaluator failures. Even with `TERRABOX_AGENTDOJO_API_WORKERS=1`,
  one AgentDojo sample can issue many rapid model calls, so external API
  requests are cross-process paced by the shared Terrabox
  `pace_remote_llm_request` limiter, using workload
  `TERRABOX_REMOTE_LLM_WORKLOAD=agentdojo` by default and the shared
  `tmp/service_locks/remote_llm_longcat.lock` by default. This must be shared
  with any concurrent LongCat rollout, including ExperienceEvo and tau2. Legacy
  `TERRABOX_AGENTDOJO_LONGCAT_MIN_INTERVAL_SECONDS` / `_RATE_LOCK` remain aliases,
  but new watcher scripts should use the generic API names. LongCat
  OpenAI-compatible calls must disable `thinking` explicitly and must not forward
  OpenAI `reasoning_effort`; that parameter is provider-specific and can create
  false 400 failures. After changing pacing env vars, restart the watcher or
  rollout parent process; already-running Python parents keep their old env.
  Keep `TERRABOX_AGENTDOJO_OPENAI_TIMEOUT_SECONDS` finite (default 300s) so one
  wedged HTTPS request cannot stall the whole watcher overnight.
- New formal PromptEvo chains default to meta-prompt v2; select v1 explicitly
  only to reproduce a historical chain. Stage1 v2 and Stage2 v2 are intentionally
  different: Stage1 reads one version's traces to propose a conservative first edit; Stage2 reads the exact
  Base->Stage1 prompt diff plus paired task traces to keep gains and narrowly
  repair regressions. Do not collapse them into one shared meta prompt.
- For controlled prompt ablations, save the ablated prompt as a normal
  PromptStore version and pass the same version to Base rollout
  `--prompt-version` and to `chain-after-base --base-prompt-version`. Without
  this, Stage1/Stage2 optimization will intentionally keep historical official
  `base` behavior and the ablation recovery result will be invalid.
- Keep `AGENTS.md`, `CLAUDE.md`, this adapter README, the shared adapter README,
  and the current Chinese experiment record synchronized when changing scope,
  metrics, provider, GPU layout, or output paths.
