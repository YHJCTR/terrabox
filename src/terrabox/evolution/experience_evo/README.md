# ExperienceEvo

Terrabox rollout 的离线产物状态转移经验自进化模块。

本模块刻意放在 `promptevo/` 同级而不是内部：PromptEvo 优化静态系统 prompt，
ExperienceEvo 构建外部经验库。当前 MVP 是离线优先：

```text
历史 rollout results
  -> 产物状态转移抽取
  -> LongCat/DeepSeek/local 蒸馏
  -> JSONL + SQLite 经验库
  -> eval 时 top-k 检索并注入 prompt
```

## 经验单元

一条经验描述可复用的产物状态转移：

```text
当前产物状态 -> 下一产物 / 结果状态
```

当前存两层：

- `signature`: `task_type + input_signature + output_signature`
- `tool`: `task_type + input_signature + output_signature + tool`

字段包含 `q`、`n`、`risk`、`status`、输入约束、输出检查、下游消费规则、
恢复建议和参数注意事项。

注入 prompt 时不能包含 `expected_tools`、最终答案、精确历史路径或 task id。
历史 F1/reward 只用于经验排序和标签。发给 LLM 蒸馏前会先脱敏：
具体任务文本、地点名、文件路径、layer 名和自由文本参数会被替换为
`<named_area>`、`<artifact_reference>`、`<task_specific_text_prompt>` 等占位符。

## 2026-07-26 MVP 行为

当前代码仍是离线 MVP，但已经更接近 4.2 的产物状态转移方向：

- 基建归因：timeout、429/rate-limit、网络/代理失败、provider overload、
  CUDA/OOM、上下文长度、Docker/container/service-health 问题会标记为
  `infra_error` 并从经验蒸馏中过滤，不提升 `risk`。
- 经验风险：只有可观测的 LLM 工具使用错误才设 `risk=1.0`，例如 missing
  required parameter、invalid argument、不存在的文件/图层/产物引用、
  invalid geometry/bbox/expression 等。
- 检索：eval 不把所有经验塞进上下文，而是用停用词过滤、下划线拆词和 OEA
  domain alias 的轻量 lexical retriever，例如
  `fire/police/station -> poi/add_pois_layer`、
  `closest/nearest -> distance/compute_route_dist`。这只是低成本 MVP 检索，
  还不是 embedding retriever。
- 两阶段 prompt block：`ExperienceEvoPromptInjector` 先检索签名级产物转移，
  再在同一 `input_signature -> output_signature` 下找工具级子经验，并计算：

```text
lambda = Ntool / (Ntool + k)
Quse = lambda * Qtool + (1 - lambda) * Qsig
```

  prompt 展示 Stage A 产物级指导和 Stage B `Quse` 工具推荐。当前仍是任务开始前
  的静态注入，因为 `scripts/run_trajectory_experiment.py` 只在 agent loop 前调用
  一次 `augment(question)`；真正 step-level retrieval/checker 需要后续在
  graph/tool-loop 附近加 hook。
- 分组选择：设置 `--max-groups` 时，蒸馏会先做 balanced group cut，再按
  support/quality 填充，避免经验库只剩高频 OSM/common-tool bucket。
- 蒸馏护栏：LongCat/DeepSeek 蒸馏会打印 bucket 进度（`--progress-every`，
  默认 10）。`--allow-template-fallback` 受到 `--max-template-fallback-ratio`
  （默认 0.25）和 `--max-consecutive-template-fallbacks`（默认 8）限制，
  避免 provider/API 问题把整库静默降级成 template-only 经验。

## 2026-07-31 v2 产物转移模式

旧模式没有被覆盖:

- `experience_evo` / `build` / `preview` 仍走 v1 bucket schema:
  `transitions.jsonl` -> `experiences.jsonl` -> `experience_evo.sqlite`.
- `experience_evo_v2` / `build-v2` / `preview-v2` 走新 schema:
  `events_v2.jsonl` -> `families_v2.jsonl` -> `experience_evo_v2.sqlite`.

v2 的核心变化是把一条经验定义成原子级产物状态转移 family：

```text
Train rollout event
  -> typed input_product_state
  -> typed target_product_state
  -> local evidence score r / risk_observed
  -> product-level Qsig/Nsig/Rsig
  -> tool-level Qtool/Ntool/Rtool policies
  -> prompt-time Quse ranking
```

v2 不保存完整轨迹文本给 eval agent；它只保存当前产物状态、下一产物目标、输入绑定规则、
输出检查、下游消费规则、工具参数约束和恢复建议。`infra_error` 仍只过滤，不提升 risk；
risk 只来自可归因的工具使用错误，例如 missing required parameter、invalid argument、
不存在的文件/图层/产物引用等。

### Gold 与 Rollout 来源

当前 v2 主实验是 rollout-derived offline self-evolution：

```text
train task
  -> LongCat base/react rollout
  -> actual conversation_history tool calls / args / observations
  -> events_v2.jsonl
  -> families_v2.jsonl + experience_evo_v2.sqlite
```

`expected_tools` / gold sequence 不进入 LongCat 经验蒸馏 prompt。它们当前只用于:

- `select-train` 时优先覆盖不同 gold tool sequence；
- rollout 后计算 tool F1 / AnyOrder / SameOrder / Unique；
- 后续做 gold replay audit 时作为 teacher-forced replay 的来源。

这样做的原因是 OEA gold trajectory 不一定等于当前 Terrabox 真实工具栈下可执行的正确轨迹。
gold 可能存在参数/schema 版本差异、文件/layer 契约变化，或需要 observation 动态绑定。
因此未经 replay 验证的 gold 不应直接进入主 ExperienceEvo store。

推荐后续补一个单独的 gold replay audit：

```text
train gold_tool_calls
  -> teacher-forced real tool replay
  -> replay success / tool-schema error / missing artifact / infra failure buckets
  -> optional gold_replay_store as upper-bound diagnostic
```

正式主结果仍应使用 train rollout-derived store；gold-derived store 如果构建，只作为
upper-bound 或诊断表，不和主 self-evolution 结果混用。

### Gold 真实回放入口

`gold-audit` 只是静态审计：检查 `gold_tool_calls` 是否存在、工具名是否在当前 Terrabox
live registry 中、参数 schema 是否大体对齐。默认不用 `data/oea_full_sft/tools_catalog.json`
旧快照，避免把已修复的 live 工具参数（例如 `add_text.color`、`count_given_object.bbox`）
误报成 gold 数据问题；需要复现旧 catalog 口径时可显式传 `--catalog <path>`。它不会启动工具，
也不能证明 gold 按当前 Terrabox 工具栈能得到正确结论。

`gold-replay` 是 teacher-forced 真实执行：按每条样本的 `gold_tool_calls` 顺序直接调用
Terrabox tools，不调用 LongCat 作为 actor，也不把 gold 参数暴露给 eval agent。执行前会把
OEA symbolic artifact alias（`gpkg_N` / `tif_N` / `img_N`，以及少量命名 GeoPackage alias 如 `marienplatz_gpkg_1` / `gpkg_jeronimos_1`）绑定到样本输入或前序工具产物；
也会把 OEA 原始 observation 中的固定产物名（例如 `out.tif` / `out.png` /
`dummy_generated_image.jpg`）绑定到最近一次真实生成的图像或栅格产物，但不会改写
`out_file` / `output_path` 这类输出参数本身。
每条任务会使用独立 artifact 目录，实际重跑某条任务前会先清空该任务自己的 artifact 子目录，
避免旧 `artifact_index.json` 污染新的 alias 绑定。输出是标准
`results/<task_id>.json` schema，可直接用 `rollout_report` / `rollout_metrics` 看工具链执行
指标；其中 `final_answer_full` 保存 replay observation evidence，需要再跑 answer judge 才能回答
“工具链跑通后是否支持 ground_truth 结论”。

直接用 CLI 跑 `gold-replay` 时，模块会默认写入单卡 VLM 安全配置：
`VLM_TENSOR_PARALLEL_SIZE=1`、`VLM_MAX_MODEL_LEN=8192`、`VLM_MIN_IMAGE_MODEL_LEN=8192`、
`VLM_GPU_MEMORY_UTILIZATION=0.95`、`VLM_MAX_NUM_SEQS=1`、`TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS=4096`。这些默认值会覆盖
`agent_config.yaml` 中更激进的双卡 VLM 配置，避免单 lane gold replay 启动 VLM 时因为
`tensor_parallel_size > 可见 GPU 数` 直接失败；手写三流脚本仍可以显式设置自己的 lane 配置。
注意上下文长度和输出 token 预算不同：8192 是服务上下文，默认输出预算先设为 4096，
避免每次都用 8192 输出预算触发 context retry。
InstructSAM Docker 默认传 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，缓解
`CountGivenObject` / `instructsam` 在 24GB 卡上的 PyTorch 显存碎片 OOM；如需改动，用
`INSTRUCTSAM_PYTORCH_CUDA_ALLOC_CONF` 覆盖，不要在实验脚本里直接改容器入口。
OEA `CountGivenObject` 的 `bbox` / `region` 参数按历史 gold 约定做兼容：全图 bbox 宽高互换会归一化为
真实图像全图，轻微越界但仍与图像相交的 bbox 会 clamp；完全无效的 bbox 仍返回工具错误。这个口径用于
避免把历史 bbox 坐标约定差异误判成 gold 数据错误。

断点续跑时，`gold-replay --resume` 只保留干净的 `completed` 与 OOM 终态；普通 failed 会重跑，
旧版误写成 `completed` 但 observation 中仍含 `Error in calculator` / 工具 `status=error` 的脏结果
也会重跑，避免修复 replay 适配后被历史结果跳过。`--max-transient-retries` 默认 5，只重试
`infra_or_provider` / `timeout` 等瞬时 replay 失败；schema、artifact binding 和工具参数语义错误不会重试。
需要精确复测某些旧失败样本时，用可重复传入的 `--task-id <task_id>`，不要临时改数据文件。

小规模验证：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner gold-replay \
  --data data/oea_full_sft/openearth/test.jsonl \
  --out-dir tmp/experience_evo/gold_replay_smoke/test \
  --task-id oea_test_187 \
  --task-id oea_test_556 \
  --use-docker \
  --max-transient-retries 5 \
  --progress-every 1
```

全量 train / eval：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner gold-replay \
  --data data/oea_full_sft/openearth/train.jsonl \
  --subset-file tmp/experience_evo/oea_train_coverage_2000_seed42_tasks.json \
  --out-dir tmp/experience_evo/gold_replay_train2000_20260731/train2000 \
  --use-docker \
  --max-transient-retries 5 \
  --progress-every 20

PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner gold-replay \
  --data data/oea_full_sft/openearth/train.jsonl \
  --out-dir tmp/experience_evo/gold_replay_full_20260731/train \
  --use-docker \
  --max-transient-retries 5 \
  --progress-every 20

PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner gold-replay \
  --data data/oea_full_sft/openearth/test.jsonl \
  --out-dir tmp/experience_evo/gold_replay_full_20260731/eval \
  --use-docker \
  --max-transient-retries 5 \
  --progress-every 20
```

answer judge：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/judge_answers.py \
  --results tmp/experience_evo/gold_replay_full_20260731/eval/results \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --provider longcat \
  --judge-non-completed
```

`distill-v2` 支持断点续跑：如果 `families_v2.jsonl` 已存在部分 family，CLI 会先读取
已有 family id，跳过已完成项，并按 `--progress-every` 定期写回 JSONL + SQLite。LongCat
额度耗尽/API 402 属于确定性 provider failure，保存到 rollout `results/*.json` 后不会被
`--resume` 自动重跑；需要先把 failed + 0 tool_calls 的 402/额度/空响应结果备份移出
`results/`，再执行同一 rollout 命令续跑。

无 provider 调用构建 v2 store：

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner build-v2 \
  --source tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --source-name oea_train2000_longcat \
  --store-dir evolution_store/experience_evo/oea_train2000_v2 \
  --template-only \
  --min-support 1
```

使用 LongCat 蒸馏构建 v2 store：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner build-v2 \
  --source tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --source-name oea_train2000_longcat \
  --store-dir evolution_store/experience_evo/oea_train2000_v2_longcat \
  --provider longcat \
  --allow-template-fallback \
  --min-support 1 \
  --max-families 500
```

预览 v2 注入：

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner preview-v2 \
  --store-dir evolution_store/experience_evo/oea_train2000_v2 \
  --query "Calculate the percentage of high temperature pixels in the scene."
```

使用 v2 跑 eval：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_v2_oea_eval \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_v2_oea_eval/standard \
  --llm-provider longcat \
  --evolution-method experience_evo_v2 \
  --evolution-store evolution_store/experience_evo/oea_train2000_v2 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos
```

切换新旧模式只需要换 method 名称和 store：

```text
Old: --evolution-method experience_evo    --evolution-store <v1 store>
New: --evolution-method experience_evo_v2 --evolution-store <v2 store>
```

## 2026-07-31 v2 nightly 初步结果

当前 v2 实验已不在运行：

```text
tmux: expevo_v2_nightly_20260731（已停止，无活跃 rollout 进程）
store: evolution_store/experience_evo/oea_train2000_v2_longcat_20260731
eval:  tmp/trajectories/experience_evo_v2_oea_train2000_longcat_eval_20260731/standard/results
```

构建口径：

- 来源：`longcat_oea_train2000_seed42_base_nightly_20260724` 的 train2000 LongCat base rollout。
- 抽取：`13243` 条 `events_v2`。
- 蒸馏：LongCat 生成 `500` 个 product-transition family。
- eval：OEA test `1162` 条；截至 2026-07-31 14:30 CST，目录停在 `1011/1162`
  （`completed=933`、`failed=78`），当前无活跃 LongCat rollout/GPU/container 进程。
  后续先确认 gold replay 口径，再决定是否补完剩余样本。

与 LongCat base test rollout 配对对比（当前已完成的共同 `999` 条）：

| 指标 | LongCat base | ExperienceEvo v2 | Delta |
|---|---:|---:|---:|
| success_rate | 86.29% | 87.19% | +0.90 pp |
| set-F1 | 0.676 | 0.688 | +0.012 |
| multiset-F1 | 0.608 | 0.623 | +0.015 |
| exact_match | 15.22% | 17.22% | +2.00 pp |
| ordered_exact | 11.51% | 12.71% | +1.20 pp |
| AnyOrder / SameOrder / Unique | 55.76 / 54.65 / 60.06 | 56.06 / 54.45 / 62.86 | +0.30 / -0.20 / +2.80 pp |
| perception F1 | 34.07 | 33.99 | -0.08 |
| operation F1 | 35.67 | 41.07 | +5.40 |
| logic F1 | 30.76 | 38.68 | +7.92 |
| gis F1 | 83.12 | 83.75 | +0.62 |
| same-tool>=4 tasks | 161 | 141 | -20 |
| errors/task | 0.40 | 0.47 | +0.07 |
| tokens/task | 54,807 | 68,233 | +13,426 |
| time/task | 62.1s | 74.4s | +12.3s |

阶段性判断：v2 比 v1 train-store eval 更健康，也相对 LongCat base 有正向信号，
尤其 logic/operation 工具类别和重复调用下降；但整体提升还不是“好看”的大幅提升。
主要问题是 prompt 注入仍偏静态、tokens/task 增加明显、perception 几乎没改善，
后续优先做 step-level 检索、压缩经验块、Quse 命中诊断和高风险 checker。

## 从 LongCat OEA Base 构建

默认来源是已有 LongCat OEA base rollout：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner build \
  --source tmp/trajectories/promptevo_oea_base_longcat2_20260704_031626_rollout/standard/results \
  --source-name oea_longcat_base \
  --store-dir evolution_store/experience_evo/oea_longcat_base_offline \
  --provider longcat \
  --min-reward 0.8 \
  --max-groups 50
```

低成本、无 API 调用的 smoke test：

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner build \
  --max-tasks 20 \
  --max-groups 10 \
  --template-only
```

预览检索注入：

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner preview \
  --store-dir evolution_store/experience_evo/oea_longcat_base_offline \
  --query "Which fire station and police station are closest to each other in Banff National Park?"
```

使用经验库跑 eval rollout：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_oea_longcat_smoke \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_oea_longcat_smoke/standard \
  --llm-provider longcat \
  --evolution-method experience_evo \
  --evolution-store evolution_store/experience_evo/oea_longcat_base_offline \
  --limit 5 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos
```

## 当前 Smoke

2026-07-24 已验证的命令：

```bash
# 从已有 LongCat base 做小规模 LongCat 蒸馏 smoke，只取高质量 transition。
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.experience_evo.runner distill \
  --store-dir tmp/experience_evo/smoke_longcat_store \
  --provider longcat \
  --max-groups 3 \
  --max-transitions 232 \
  --min-support 1 \
  --allow-template-fallback \
  --progress-every 1

# 单条 compute-only rollout smoke。
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  scripts/run_trajectory_experiment.py rollout \
  --task-file tmp/experience_evo/compute_smoke_tasks.json \
  --experiment experience_evo_compute_smoke \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_compute_smoke/standard \
  --llm-provider longcat \
  --evolution-method experience_evo \
  --evolution-store tmp/experience_evo/smoke_longcat_store \
  --limit 1 \
  --resume
```

已观察到的 smoke 结果：

- ExperienceEvo：`oea_test_50`，completed，set-F1 `1.00`，工具为 `compute.solver`。
- Reflection：使用已有 `oe_full_reflection_div1k` memory，completed，set-F1 `1.00`。
- MemRL source lite：用 60 条 LongCat base 轨迹构建，completed，set-F1 `0.67`，原因是多调用了一个 OSM 工具。

这些 smoke 结果只验证链路和 prompt 注入能跑通，不代表质量结论；任务只有一条
compute-focused 样本，没有覆盖 OSM/perception 服务。

外部 LongCat 全量 OEA 实验遵循模块级 provider 说明：全量 OEA 用三流
（`GPU0/1`、`GPU2/3`，另加 OSM/nogpu），并设置
`TERRABOX_TOOL_SERVICE_SCOPE=call TERRABOX_KEEP_VLM_WARM=1`。外部 API 三流实验
不要用 `session` scope。

## 对比计划

先在已有 test-only LongCat base 上 smoke：

1. 把 `promptevo_oea_base_longcat2_20260704_031626_rollout` 视为 LongCat ReAct/base 参考。
2. 只把同一个历史 base 构建成 ExperienceEvo，作为 transductive smoke/case-library 测试。
3. 用 `--evolution-method experience_evo` 跑少量 eval，并通过 `rollout_report compare` 对比。

更干净的 train->test 对比：

1. shuffle OEA train，并选约 2,000 条覆盖尽量多的工具序列。
2. 用同一个 train subset 跑 LongCat base rollout，作为 Reflection、MemRL source 和 ExperienceEvo 的共同经验来源。
3. 分别从该 train subset 构建 offline memory / experience store。
4. 用相同 LongCat rollout 设置在 `1162` 条 OEA test 上评估所有方法。

后续 GRPO/RL 实验中，经过 replay 审计的 gold 轨迹主要用于 SFT/warm-start 和 reward
校准；GRPO 主样本应使用当前策略的 on-policy rollout。reward/verifier 信号来自
answer judge、tool execution status、product-transition validity、downstream
consumption 和 schema validity。network、OOM、context、provider 等基建失败只做重试、
跳过或单独记录，不作为负经验风险。

第一轮正式对比使用的 train subset 命令：

```bash
PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner select-train \
  --output tmp/experience_evo/oea_train_coverage_2000_seed42_tasks.json \
  --limit 2000 \
  --seed 42
```

该命令从 `14538` 条 OEA train 中选择 `2000` 条，覆盖 `87` 个不同
`expected_tools` sequence。

当前 after-train watcher：

```bash
tmux attach -t expevo_after_train_20260726
tail -f tmp/experience_evo/nightly_20260724/logs_after_train/after_train_eval.log
```

它等待 `longcat_oea_train2000_seed42_base_nightly_20260724`，然后构建
`evolution_store/experience_evo/oea_train2000_longcat_balanced_20260726`，并在 OEA test 上跑：

建库前 watcher 会先做 train repair pass：用同一三流 `--resume` 重启。
`run_trajectory_experiment.py` 会重跑仍有 retry budget 的 transient failed result，
但不会重跑 context overflow、missing required、invalid argument、文件/图层不存在等确定性
LLM/tool-use 错误。watcher 还会为 P1 钉住非 VLM 服务端口（`9012`-`9017`），
避免 P0/P1 并行时重建同一组 SAM2/RemoteSAM/InstructSAM/Strip-RCNN/RemoteCLIP/ChangeOS 容器。

```text
experience_evo_oea_train2000_balanced_eval_20260726
reflection_oea_train2000_longcat_eval_20260726
memrl_source_oea_train2000_longcat_eval_20260726
```
