# API-Bank adapter

## 外部方法固定划分对比（2026-09-19）

用户更新要求：所有冻结方法最终都评测该设置全部 **389 条**，保存
`full_set_final_results.json` / `full_set_summary.json`。这包括 train/dev，明确标为全量覆盖评测，
不能宣称独立泛化测试。原 79 条独立 test 的 `test_results.json` / `summary.json` 保留。
PromptEvo 额外保存 Stage1 全量结果，EvoTool 额外保存模块化 Base 全量结果；同 prompt/样本复用已有调用缓存。
当前旧 GEPA 进程无需重启；新增父进程使用同一命令加 `--follow-full-evaluations`，
监测已冻结且完成 test 的方法并补齐全量评测（最长等待 72 小时）。与原队列使用相同 LongCat 节流，
每方法全量评测有独立防重复锁。汇总见 `full_set_comparison.json`，未完成不能写为全量完成。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS=1 TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS=1 \
python -m terrabox.evolution.promptevo.adapters.api_bank.pipeline compare-methods \
  --group api_bank_external_longcat_grouped_20260919 \
  --methods gepa base scope aho promptevo evotool --workers 4
```

- 使用 `unsloth`，全部调用 LongCat，不启动 GPU。389 条样本按对话文件分组，seed=20260919：232 train / 78 dev / 79 test。指标为 API 名与参数精确匹配，不是工具执行成功率。
- GEPA：官方搜索，2000 metric calls；反思读取可见轨迹而不发送 gold answer，搜索使用 train/dev 评测分数。
- SCOPE：官方规则生成/分类/战略记忆，train 学习后冻结；每任务重置战术计数。没有同任务多步重试，标为 `frozen next-call adapted`。
- AHO：官方 BetterHarness 5 轮历史/接受循环；未安装 DeepAgents，提案改为工作区证据 JSON（最多 12 个失败、4 个通过案例），标为 `prompt-only adapted`，不是完整 harness 复现。
- EvoTool：官方归因/定向修改/多样性搜索，3 epoch；API-Bank 下一次调用适配，planner/selector/caller 后由 synthesizer 输出 bracketed-call，不执行工具。需独立模块化 Base，多次 actor 调用与单次调用成本不可混淆。
- PromptEvo：既有 Stage1 补丁和 Stage2 对比优化，新增显式 dev ID/dev baseline 参数，旧命令默认不变；两个阶段冻结后才测试。
- PromptEvo 可选 `--task-profile tool_search`：只向优化器说明“工具发现/下游调用/可观察依赖状态/精确参数键/语义搜索词”这些待分析维度，具体规则仍必须从 train/dev 轨迹生成；该结果命名为 `PromptEvo-ToolSearch-adapted`，不能与 vanilla PromptEvo 混报。
- `tool_search` profile 不应预设 credential/authentication、具体 prerequisite、API 名或固定工作流；若实验显式加入这些结构，只能另标为更强的场景适配，不能与结构性 profile 结果合并。
- PromptEvo 的 Stage1/Stage2 候选数可通过 `--promptevo-stage1-candidates` / `--promptevo-stage2-candidates` 调整；这属于搜索预算变化，必须记录在 `provenance.json`，不能与默认 3+3 结果混报。
- PromptEvo Stage2 可用 `--promptevo-patch-composition atomic_pairwise` 启用 typed patch 组合搜索：对每个候选的单 patch、两两组合和必要的整组回退分别做固定 dev rollout，优先 exact-call、API/schema/value 指标，并在分数相同时选择更小组合。该模式记录 `candidate_evaluations` 和 `selected_patch_ids`，命名为结构化组合搜索变体，不能与 `whole` 模式混报。
- PromptEvo 可用 `--promptevo-protocol-mode conditional` 启用条件协议演化：`trigger`、`rule`、`scope` 会以 `IF ... THEN ... [Scope: ...]` 编译进静态 prompt，`evidence` 只留在审计元数据；支持 `state_transition` 和 `query_abstraction` 类型。后者只用于动态描述约束下的搜索词抽象，并在编译器中强制限定为 `discovery/query fields only`；规则可以要求保留 schema-significant 的实体或标识值，但不能把查询规则扩大成全局实体改写，也不写入具体查询词。Stage1/Stage2 记录逐任务 gain/regression 和按对话文件的 lineage，baseline 已成功的任务不得回归，且候选至少要产生一个 exact-call gain。该模式是 PromptEvo 的结构化条件编译变体，不是 GEPA 的轨迹 few-shot 或全局重写，必须与 `legacy` 分开报告。
- 调用异常传播，不写成模型答错；逐条保存结果与响应。GEPA 原生恢复；SCOPE 按已提交任务规则恢复；AHO/EvoTool 固定种子重放控制流程并复用缓存，不是上游原生 resume。
- 所有方法共享 LongCat 总节流；`queue.lock` 仅用于防重复启动。修改环境后须重启父进程。
- Watcher：`PROMPTEVO_EXTERNAL_APIBANK_ONLY=1 bash tmp/promptevo_watchers/watch_after_agentdojo_20260917.sh`。每方法最长 8 小时、最多两次尝试，账户错误停止；方法代码失败可继续独立方法。`summary.json` 仅表示 79 条 test 完成；全量还必须有 `full_set_summary.json`，不能只看旧队列的 `queue_summary.json`。
- 这些方法搜索预算不同，不得宣称等 token 对比。首次 GEPA 缓存未记录 usage 时记缺失；AHO 上游 proposer token 占位零值不能当作零成本。


Static instruction slot:

- `API_BANK_API_CALL_PROMPT` in `api_bank/constants.py`
- Mirrors API-Bank's original `evaluator.py::api_call_prompt`
- Dynamic API descriptions and dialogue history are appended at rollout time
  and are not optimized by promptevo.
- `API_BANK_RESPONSE_PROMPT` is also exposed for API-Bank's response task, but
  it is a separate static prompt slot. Do not optimize API-call and response
  instructions as one merged prompt unless the experiment intentionally changes
  both tasks.

Local data:

- Source checkout: `/data1/yuhongjie2/DAMO-ConvAI-api-bank`
- Default lv1/lv2 data:
  `/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-1-given-desc`
- Default experiment output:
  `tmp/promptevo_api_bank_experiments/`

No-model smoke test:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import (
    APIBankMetricProvider,
    write_oracle_predictions,
)

data = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-1-given-desc"
pred = "tmp/api_bank_oracle_predictions.jsonl"
write_oracle_predictions(data, pred)
metrics = APIBankMetricProvider(
    data_dir_fn=lambda exp: data,
    prediction_path_fn=lambda exp: pred,
)
print(metrics.aggregate("oracle"))
PY
```

Expected core smoke-test values for the current local checkout:

- `n = 389`
- `prediction_coverage_rate = 1.0`
- `success_rate = exact_match_accuracy = 1.0`
- `api_name_accuracy = argument_key_f1 = 1.0`

Rollout dry run, no model:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import make_api_bank_components

root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
prompts, traces, metrics, runner = make_api_bank_components(data, root)
runner.run_version("base", "dry_base", limit=2, dry_run=True)
print(metrics.aggregate("dry_base"))
PY
```

Main metrics include exact API-call accuracy, parse/call rates, API-name
accuracy, argument key precision/recall/F1, argument value accuracy, error
buckets, history/argument-count bucket accuracy, latency, and runtime-error
rate.

## 断点续跑

`APIBankRolloutRunner` 每完成一条样本就立即追加写入 `predictions.jsonl` 与
`rollout.jsonl`。传入 `resume=True` 时，只有两个文件中都存在同一 `(file, id)`
记录的样本才会被跳过；进程在两次写入之间中断时，该样本会在下次运行时重做。这样不会把
未完成的 provider 调用误判为已完成。

## 三阶段 ProtocolPatch 实验

正式三阶段入口为 `pipeline.py`。它只优化 API-call 的静态指令，动态 API 描述与对话历史
不会写入提示词。Stage1 从 Base 的真实轨迹生成 3 个类型化协议补丁候选，Stage2 从 Base/Stage1
同任务配对轨迹生成候选；两个阶段都使用固定 24 条真实 API-call rollout 验证后才接受候选。若
所有候选在验证集退步，阶段会明确记录为保留上一版本，而不会静默挑一个静态上看似更好的提示词。
长时间运行被中断后应直接复用同一 `--group` 续跑：Stage1 候选会先写入
`optimization/stage1_candidates.json`，Stage2 候选写入
`optimization/stage2_candidates.json`；validation rollout 使用 `resume=True`，只补齐缺失
的 `(file, id)` 样本，不会重新消耗已经完成的 dev 调用，也不会重新生成不同候选。

```bash
PYTHONPATH=src python -m terrabox.evolution.promptevo.adapters.api_bank.pipeline run-three-stage \
  --group promptevo_apibank_protocol_patch_longcat_20260814 --provider longcat \
  --stage1-candidates 3 --stage2-candidates 3 --validation-tasks 24 --min-interval 10
```

Tool Search 场景适配版使用独立实验组，不写入具体 API、task id、gold 调用或固定认证流程：

```bash
PYTHONPATH=src python -m terrabox.evolution.promptevo.adapters.api_bank.pipeline run-three-stage \
  --group promptevo_apibank_toolsearch_adapted_longcat_20260920 \
  --data-dir /data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-2-toolsearcher \
  --provider longcat --task-profile tool_search \
  --stage1-candidates 3 --stage2-candidates 3 --validation-tasks 24
```

该 profile 是场景适配，不是原始 PromptEvo 的无修改复现；报告时需同时保留 vanilla PromptEvo 结果。

条件协议变体示例：

```bash
PYTHONPATH=src python -m terrabox.evolution.promptevo.adapters.api_bank.pipeline compare-methods \
  --group api_bank_toolsearch_prompt_evo_conditional_transition_longcat_grouped_20260920 \
  --data-dir /data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank/lv1-lv2-samples/level-2-toolsearcher \
  --methods promptevo --workers 4 --promptevo-task-profile tool_search \
  --promptevo-patch-composition atomic_pairwise \
  --promptevo-protocol-mode conditional
```

该变体仍只从 train/dev 证据生成短 typed patch；`trigger` 只描述可见条件，`rule` 描述动作，
`scope` 描述适用边界，不允许出现具体 API、凭据、task id、gold 或固定 workflow。候选测试的
配对 lineage 保存在 `candidate_evaluations`，全量结果仍不能替代 held-out 泛化结果。

查看阶段与全量指标：

```bash
PYTHONPATH=src python -m terrabox.evolution.promptevo.adapters.api_bank.pipeline status \
  --group promptevo_apibank_protocol_patch_longcat_20260814
```

结果保存于 `tmp/promptevo_api_bank_experiments/<group>/{base,stage1,stage2}/`，其中每阶段都有
`predictions.jsonl`、`rollout.jsonl`、`metrics.json`；优化提案、固定验证集和接受决策保存于
`tmp/promptevo_api_bank_experiments/<group>/optimization/`。外部 API 的可重试网络/限流错误在 `resume=True` 时会重新排队，
确定性格式或参数错误仍保留为真实结果。

Official-style execution diagnostics:

```bash
PYTHONPATH=src python - <<'PY'
from terrabox.evolution.promptevo.adapters.api_bank import APIBankMetricProvider

root = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
data = root + "/lv1-lv2-samples/level-1-given-desc"
pred = "tmp/api_bank_oracle_predictions.jsonl"
metrics = APIBankMetricProvider(
    data_dir_fn=lambda exp: data,
    prediction_path_fn=lambda exp: pred,
    api_bank_root=root,
    execute_api_calls=True,
)
print(metrics.aggregate("oracle"))
PY
```

Keep this as a diagnostic, not the default optimization gate. API-Bank APIs
include external network calls, random tokens, and state-changing calls; some
gold API calls do not reproduce their saved gold result in a clean local
process. The default promptevo loop should optimize stable prompt-following
metrics first: exact API name, argument keys, argument values, parse success,
and error buckets.

## Internal layout

- `prompts.py`: required `PromptStore`; reads/writes static prompt versions.
- `traces.py`: required `TrajectorySource`; converts API-Bank logs into
  promptevo `Trace` objects.
- `metrics.py`: optional `MetricProvider`; computes exact-call, argument,
  response, and error-bucket metrics for experiment comparison.
- `runner.py`: optional rollout runner; temporarily applies a prompt version
  for one experiment without modifying API-Bank's original prompt.
- `api_utils.py` and `files.py`: API-Bank-specific parsing and file helpers.

Response metrics:

- Set `include_responses=True` on `APIBankMetricProvider` to include post-API
  dialogue samples.
- The adapter reports `response_rouge_l`, `response_success_rate`, and
  `low_response_rouge_rate`.
- A prediction JSONL can contain both API-call and response rows because both
  use API-Bank's `(file, id, pred)` format.

Tool-search data:

- For data directories whose basename does not end with `given-desc`, the
  rollout runner follows API-Bank's original behavior and gives the model only
  the `ToolSearcher` description.
- For `*-given-desc`, it gives descriptions for the APIs present in the
  conversation history.
