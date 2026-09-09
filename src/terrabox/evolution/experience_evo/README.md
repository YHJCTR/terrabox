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

## 2026-08-14 v4-clean 严格口径

历史 v1--v4 store 曾把 OEA 的 `task_type` 写进 family key，因此保留为可复现的
历史结果，但不再作为严格无标签自进化主表。`experience_evo_v4_clean` 是独立实现，
不会覆盖旧 v4 的代码、store 或结果：

```text
train base rollout（仅任务、输入、实际工具调用/参数/observation/可观测错误）
  -> 去除 task_id/task_type/expected_tools/gold answer/gold calls/F1 等标注
  -> query-derived intent + input product state + target product state family
  -> LongCat 蒸馏 product/tool experience 与 Qsig/Qtool/N/R
  -> JSONL + SQLite strict store
  -> eval: 前置产物状态硬过滤
         + lexical BM25 与结构化状态/意图排序的 RRF 融合
         + Quse/risk 重排
         + v4 soft verifier 与 step hint
```

`task_type` 在 runtime 中仅为统一 agent 接口而接收，随后立即丢弃；它不参与建库、
family id、候选过滤、排序、路由或 prompt。严格 store 的事件字段中会留下固定的
`"rollout"` / `"general"` 占位值以兼容 JSONL/SQLite schema，但不保留任何样本标签。
Q 与 risk 只使用当前 rollout 可观测的输入绑定、输出有效、目标产物形成、下游消费、
终态可用性与可归因参数/产物错误；OOM、网络、timeout、配额、上下文与服务问题只过滤。

正式建库示例：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.experience_evo.runner build-v4-clean \
  --source tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --source-name oea_train2000_longcat_rollout \
  --store-dir evolution_store/experience_evo/oea_train2000_v4_clean \
  --provider longcat --min-support 2 --progress-every 25
```

其中 2000 条 train 子集按 oracle `expected_tools` 序列覆盖挑选，因此要在论文中标为
`oracle-sequence-coverage subset`；这只影响 subset 选择，不会写入 store 或暴露给 agent。

注入 prompt 时不能包含 `expected_tools`、最终答案、精确历史路径或 task id。
历史 F1/reward 只用于经验排序和标签。发给 LLM 蒸馏前会先脱敏：
具体任务文本、地点名、文件路径、layer 名和自由文本参数会被替换为
`<named_area>`、`<artifact_reference>`、`<task_specific_text_prompt>` 等占位符。

## 2026-08-22 v5：最终答案证据校验增强

`experience_evo_v5` 是 v4-clean 的独立增强版，不覆盖 v4-clean 代码、store 或结果。它复用
v4-clean 的 rollout-only transition store、混合检索、`Quse` 排序、risk 约束和 soft verifier，
额外在主 agent 准备输出最终答案时触发一次 verifier 子 agent：

```text
当前任务 + 当前 artifact state + 成功/失败工具调用 + observation + draft final answer
  -> artifact-state evidence summary
  -> final-answer verifier 检查数值、单位、阈值、最近/最远、计数、实体选择和产物是否被证据支持
  -> accept：直接输出；revise：回注修正提示，让主 agent 再回答或补一次必要工具调用
```

v5 verifier 只看当前运行证据，不看 train 轨迹、eval gold、`task_type`、`expected_tools`、task id
或 answer judge。它的目的不是通用压缩整段对话，而是把工具产物压成面向最终答案的结构化证据表，
专门修正“工具链基本正确但最终数值/单位/阈值/计数/实体解释错误”的问题。

运行入口：

```bash
PYTHONPATH=src python scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_v5_full_dockerfixed_20260822 \
  --mode standard \
  --llm-provider longcat \
  --evolution-method experience_evo_v5 \
  --evolution-store evolution_store/experience_evo/oea_train2000_v4_clean_longcat_20260814 \
  --use-docker --resume
```

可用环境变量：

- `TERRABOX_EXPEVO_V5_DISABLE_FINAL_VERIFIER=1`：关闭最终答案 verifier，应退化为近似 v4-clean，用于 sanity/消融。
- `TERRABOX_EXPEVO_V5_VERIFIER_PROVIDER=longcat`：指定 verifier 子 agent provider。
- `TERRABOX_EXPEVO_V5_VERIFIER_MAX_TOKENS=520`：控制 verifier JSON 输出预算。

## 2026-08-24 v5-hybrid-qwen：Qwen embedding 混合检索

`experience_evo_v5_hybrid_qwen` 是 v5 的独立检索增强版，不覆盖 `experience_evo_v5`、v4-clean
store 或已有结果。它保留 v5 final-answer verifier，只把 transition family 的候选排序改成：

```text
当前任务 + 当前 product state
  -> v4-clean 前置硬过滤：Q/Risk、precondition、已满足 target、可用工具、query profile hard mismatch
  -> 三路排序：BM25 lexical rank + structured artifact-state rank + Qwen embedding semantic rank
  -> RRF 融合 + Q/N/Risk rerank
  -> prompt 中仍展示 product transition、Quse 工具排序和 v5 final verifier
```

embedding 文档只由 rollout-derived family 构成，包括 `intent_signature`、`input_product_state`、
`target_product_state`、product/tool experience、输入绑定规则、输出检查、恢复建议和公开工具名；不包含
`task_type`、task id、`expected_tools`、gold answer、gold tool calls 或 gold 指标。embedding 只参与候选排序，
不替代 product-state hard filter，避免语义相似但当前状态接不上的经验被塞入 prompt。

使用前必须先启动 OpenAI-compatible Qwen embedding 服务，并构建 family index：

```bash
PYTHONPATH=src TERRABOX_EXPEVO_EMBEDDING_URL=http://127.0.0.1:9101/v1/embeddings \
TERRABOX_EXPEVO_EMBEDDING_MODEL=/model \
python -m terrabox.evolution.experience_evo.runner build-v5-hybrid-index \
  --store-dir evolution_store/experience_evo/oea_train2000_v4_clean_longcat_20260814 \
  --batch-size 24
```

运行入口：

```bash
PYTHONPATH=src TERRABOX_EXPEVO_EMBEDDING_URL=http://127.0.0.1:9101/v1/embeddings \
TERRABOX_EXPEVO_EMBEDDING_MODEL=/model \
python scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_v5_hybrid_qwen_oea_train2000_longcat_eval_20260824 \
  --mode standard --llm-provider longcat \
  --evolution-method experience_evo_v5_hybrid_qwen \
  --evolution-store evolution_store/experience_evo/oea_train2000_v4_clean_longcat_20260814 \
  --use-docker --resume
```

该模式显式依赖 `qwen_family_embedding_index.json`。index 缺失、模型名不一致或 family 覆盖不完整时，
runtime 会报错退出；不得静默退化为纯 BM25/v5，否则实验口径会被污染。调试时可临时设置
`TERRABOX_EXPEVO_HYBRID_DISABLE_SEMANTIC_RANK=1` 或
`TERRABOX_EXPEVO_HYBRID_ALLOW_PARTIAL_INDEX=1`，正式实验不能使用这两个开关。

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

GeoPackage 产物契约：`osm_gis.add_index_layer`、`osm_gis.compute_index_change`、
`osm_gis.add_pois_layer` 和 `osm_gis.compute_route_dist` 的 append-style 写入都会检查
SQLite/GPKG catalog，并在 GDAL 写入前对同一 GeoPackage 加进程锁、在锁内再次检查。完整的
raster/vector layer 会幂等复用；不完整 layer、类型冲突、catalog 检查失败或 GDAL 写入失败会
返回明确错误，不会删除或覆盖 artifact，也不会把统计结果伪造成成功。STAC
raster 读取会把窗口裁剪到源影像范围；这些修复不改变工具 timeout。
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

## 2026-08-04 v3 运行时修正版

v3 不覆盖 v1/v2，也不要求重新构建 store。它直接复用 `build-v2` 产出的
`families_v2.jsonl` / `experience_evo_v2.sqlite`，但修正 eval-time 检索和注入：

- `experience_evo_v2`：按 query lexical overlap 取 family，容易在任务开局塞入后置产物转移，
  例如 `gpkg + vector_layer -> display_on_map`；当 test task 没有 `task_type` 时，还会被
  `type30/type31/gis_type*` 这类训练集 task type token 干扰。
- `experience_evo_v3`：先从当前任务输入推断初始产物状态（`task_request`、`input:image`、
  `input:raster`、`input:gpkg` 等），只检索 preconditions 已满足的第一步产物转移；
  再用 query intent、输入形态和可见工具全集过滤明显错配的 family。
- v3 注入更短，默认 top-k 为 3；统一 rollout 会把 `images`、`data_files` 和
  `available_tools` 传给 augmenter，但不会传当前 eval 的 `expected_tools`、gold answer
  或 task id。
- v3 现在是“初始静态检索 + 工具返回后的动态 step hint”：任务开始前只注入当前输入
  满足的第一步经验；standard eval 的 sequential tool loop 每次工具 observation 返回后，
  用 `agent.artifacts` 更新当前产物状态，再调用 `step_hint()` 检索下一步 family。
  结果文件会写入 `evolution_trace`，用于检查每步命中了哪些经验、推荐了哪些工具、
  agent 实际选择了什么工具，以及选择是否落在经验推荐中。
- v3 动态阶段会降低已经完成 target 的 family 和泛化 `task_request` 起步 family 权重；
  对“已有感知产物 + 明确计算需求”的场景，会优先把经验推进到 `compute.*`，避免只重复
  调用其它感知工具。
- v3 动态阶段补了极窄的 runtime fallback：当离线 store 缺少某个显然必要的下游 family，
  但当前产物状态和工具契约已经唯一指向下一步时，临时生成一条可审计的产物转移提示。
  当前只覆盖已实测缺口：index change 在已有 1 个 `add_index_layer` 产物时继续推荐
  第二个 `add_index_layer`，已有 2 个 index layer 后才推荐
  `raster_layer:from:osm_gis.compute_index_change`；multi-target 的图像测量任务在已有 1 个
  `geo_perception.instructsam` 结果时继续推荐第二个不同目标的 InstructSAM，在已有一个
  `compute.calculator` 结果但仍缺目标计算时继续推荐 calculator；以及任务要求属性/健康/
  状态判断但属性描述次数不足时，继续推荐
  `geo_perception.region_attribute_description`。fallback 不读取 gold / expected tools /
  answer / task id，也不写回经验库；它只用于避免 store 覆盖缺口让 step hint 推荐错工具。
- v3 动态阶段新增 answer-ready 防护：如果当前产物状态已经包含可直接回答的结果
  （例如 `compute.calculator`、`compute_route_dist`、`compute_index_change`、OCR/属性描述结果等），
  step hint 会优先要求 final answer；如果模型仍继续发起工具调用，sequential loop 会拦截
  该额外工具并要求直接输出最终答案。对 GSD/像素面积/距离测量任务，`vlm_analyze` 的描述文本
  不视为可回答结果，必须等到可计算的像素/掩码证据和 calculator 等结果。对 `both/two/each`
  等 multi-target 任务，answer-ready 会保留重复产物计数；只完成一个对象的属性描述或一次计算，
  不会被当成整题已完成。
- v3 还新增了窄口径执行 guard：对 GSD/像素距离/面积这类精确测量任务，如果模型首步想调用
  `vlm_analyze` 或 `strip_rcnn_detect`，sequential loop 会在真实执行前拦截；单目标/局部对象
  测量要求先用 `geo_perception.instructsam` 取得可测量像素/掩码证据。显式 `segment all` /
  `sum pixel areas` / `combined ground area` 这类 bulk all-object segmentation 任务例外，会允许
  并优先提示 `geo_perception.sam2_segment`。该 guard 会写入 `evolution_trace.guard_preview` /
  `blocked_tool_calls`，便于后续审计经验是否真的约束了工具选择。
- v3 guard 还会检查两类常见可观测工具误用：第一，图片参数必须来自当前 task image 或当前运行
  产物，发现不存在/非当前任务的历史路径会在执行前拦截；第二，重复 evidence 工具时不再按
  set token 粗暴拦截，而是记录 `successful_call_records` 中的工具参数，只拦截“同工具+同语义
  target”的重复调用。multi-target 任务如果换了不同 `text`/target（例如先 `garbage pile`，
  再 `big pond`），会允许继续调用同一 evidence 工具。对 damage/symmetry/health 这类已完成
  定位、下一步应做属性描述的任务，若模型继续重复 `geo_perception.instructsam`，guard 会明确
  要求改用 `geo_perception.region_attribute_description`，避免只“拦截重复”但模型不知道下一步。
  非 visual 请求还会过滤 `geo_perception.add_text` 这类标注产物，避免把可视化动作混进计算/属性链路。
- 工具状态更新现在不会把明确失败观察（例如 `Error in calculator:`）记录为成功产物状态；
  失败调用只进入 `failed_calls`，避免 answer-ready 或下一步检索被错误结果污染。
- OSM POI 参数归一也做了一个小修复：`marketplace(s)` 以及 `{"shop": "marketplace"}` 会归一为
  `{"amenity": "marketplace"}`，避免 LongCat 把 marketplace 误当 shop tag 导致 no matching
  features 后再重试。

v3 额外过滤的典型错配：

```text
image-only detection task     -> 不推荐 OSM/display_on_map 后置经验
no raster input task          -> 不推荐 get_bbox_from_raster
single image non-change task  -> 不推荐 change_os_detect
non-search geospatial task    -> 不推荐 bing_search.search
OSM/感知任务开局             -> 不推荐 compute.* 直接计算，除非已有工具证据
已有 target product          -> 不重复推荐同一产物转移
已有感知结果且问题要求计算    -> 优先推荐 compute.* 下游计算
已有 1 个 index layer 且问题要求变化 -> store 缺口时 runtime fallback 继续推荐 add_index_layer
已有 2 个 index layer 且问题要求变化 -> store 缺口时 runtime fallback 推荐 compute_index_change
已有感知+计算且还需属性判断    -> store 缺口时 runtime fallback 推荐 region_attribute_description
multi-target 已有 1 个定位结果    -> store 缺口时 runtime fallback 推荐第二个不同目标的 InstructSAM
multi-target 已有 1 个计算/属性结果 -> 未达到所需次数前不触发 answer-ready，继续推荐下游工具
非 visual 请求                -> 不推荐 draw/display/plot 等可视化产物转移
GSD/像素精确测量首步          -> 执行前拦截 VLM/SAM2/Strip-RCNN，改用 InstructSAM
非当前任务图片路径             -> 执行前拦截，要求使用当前 task image / 当前运行产物
已有可回答结果               -> 优先 final answer，不继续扩展工具链
失败工具 observation          -> 不进入成功 product state
```

预览 v3 注入：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner preview-v3 \
  --store-dir evolution_store/experience_evo/oea_train2000_v2_longcat_20260731 \
  --query "Which fire station and police station are closest in Banff National Park?"

PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner preview-v3 \
  --store-dir evolution_store/experience_evo/oea_train2000_v2_longcat_20260731 \
  --query "Detect all domestic garbage regions and calculate their combined area." \
  --images /data1/yuhongjie2/OpenEarthAgent/data/test/TG_70028.jpg

# 模拟工具返回后的动态下一步检索
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner preview-v3 \
  --store-dir evolution_store/experience_evo/oea_train2000_v2_longcat_20260731 \
  --query "Detect all domestic garbage regions and calculate their combined area." \
  --images /data1/yuhongjie2/OpenEarthAgent/data/test/TG_70028.jpg \
  --current-state "task_request,input:image,result:from:geo_perception.instructsam"

# 模拟已有计算结果后的停止提示
PYTHONPATH=src $PY -m terrabox.evolution.experience_evo.runner preview-v3 \
  --store-dir evolution_store/experience_evo/oea_train2000_v2_longcat_20260731 \
  --query "Detect all domestic garbage regions and calculate their combined area." \
  --images /data1/yuhongjie2/OpenEarthAgent/data/test/TG_70028.jpg \
  --current-state "task_request,input:image,result:from:geo_perception.instructsam,result:from:compute.calculator"
```

使用 v3 跑 eval：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_v3_oea_train2000_longcat_eval_20260804 \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_v3_oea_train2000_longcat_eval_20260804/standard \
  --llm-provider longcat \
  --evolution-method experience_evo_v3 \
  --evolution-store evolution_store/experience_evo/oea_train2000_v2_longcat_20260731 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos \
  --resume
```

当前 v3 版本不需要额外模型；它先修正“经验是否被正确检索和注入”。如果这一步仍然提升不够，
下一阶段再考虑两个模型增强：一是用 LongCat 对候选 family 做 query relevance rerank，
二是加本地 embedding/reranker（例如 BGE/e5 系列）替代纯 lexical 检索。

切换新旧模式只需要换 method 名称和 store：

```text
v1: --evolution-method experience_evo    --evolution-store <v1 store>
v2: --evolution-method experience_evo_v2 --evolution-store <v2 store>
v3: --evolution-method experience_evo_v3 --evolution-store <v2 store>
v4: --evolution-method experience_evo_v4 --evolution-store <v2 store>
```

## 2026-08-07 v4 软约束运行时

v4 是独立于 v3 的新运行模式，复用同一个 rollout-derived v2 store，不覆盖
`experience_evo`、`experience_evo_v2` 或 `experience_evo_v3`。v4 针对 v3 的主要问题做了
运行时修正：v3 的 answer-ready 判断过早、hard guard 拦截过多，并在连续拦截后合成答案，
这些行为会让工具序列更短，却损伤最终答案和生成类任务。

v4 的核心变化：

- 保留产物状态转移检索、`Qsig/Nsig/Rsig`、`Qtool/Ntool/Rtool` 和 `Quse` 排序；不读取
  当前 eval 的 gold、expected tools、task id 或答案。
- 默认关闭 v3 的 answer-ready hard guard、premature-final guard 和 synthetic final；模型
  仍然可以根据当前观测自行决定是否继续调用工具或输出答案。
- 只保留两类可直接验证的硬约束：图片必须来自当前任务/当前运行产物，
  `compute.calculator` 必须有合法表达式。产物链推断只作为软提示，不因为启发式判断而阻断
  分析工具或最终绘图、标注、地图展示工具。
- 每次提示中增加通用的 verifier checkpoint，根据任务文本和当前产物状态标记缺失的证据槽位，
  例如 aggregate、visual output、localized attribute、count/plot；这不是针对某个任务写死的
  gold 规则。
- 保留多 agent 设计接口：可选用 LongCat 调 checker/verifier，对当前任务、产物状态、最近
  observation 和候选转移做独立审查。默认关闭，避免正式全量评测把 LongCat 请求量直接翻倍。

可选 LongCat checker：

```bash
export TERRABOX_EXPEVO_V4_LLM_CHECKER=1
export TERRABOX_EXPEVO_V4_CHECKER_PROVIDER=longcat
export TERRABOX_EXPEVO_V4_LLM_CHECKER_MAX_CALLS=1
```

v4 消融开关默认都关闭，只用于固定子集/正式消融，不改变主方法默认行为：

| 开关 | 作用 | 用途 |
|---|---|---|
| `TERRABOX_EXPEVO_V4_DISABLE_STEP_HINT=1` | 关闭每次工具 observation 后的 `step_hint()`，只保留任务开始前注入 | 验证逐步产物状态检索是否贡献收益 |
| `TERRABOX_EXPEVO_V4_DISABLE_QUSE=1` | 保留产物转移和工具候选文本，但不展示 `Quse` 排序 | 验证 `Qsig/Qtool/N/R` 组合排序是否影响工具选择 |
| `TERRABOX_EXPEVO_V4_DISABLE_TOOL_RANKING=1` | `DISABLE_QUSE` 的别名 | 便于实验脚本语义化命名 |
| `TERRABOX_EXPEVO_V4_DISABLE_VERIFIER=1` | 关闭 deterministic verifier checkpoint 和可选 LLM checker | 验证 verifier checklist 是否降低过早回答/漏产物 |
| `TERRABOX_EXPEVO_V4_DISABLE_VERIFICATION=1` | `DISABLE_VERIFIER` 的别名 | 便于实验脚本语义化命名 |

推荐消融顺序：先跑同一固定子集的 v4 default，再跑 `checker_on`、`quse_off`、`step_hint_off`；
每组都要同时保存 rollout report 和 LongCat answer judge，不能只用 success rate 判断。

### v4-clean 因果对照模式

为避免把通用执行修复误认为经验收益，当前代码还提供两个不覆盖主方法的显式入口：

| 入口 | 保留内容 | 移除内容 |
|---|---|---|
| `experience_evo_v4_no_store_soft_only` | 当前运行产物状态和通用 evidence-state checklist | 历史 transition、经验文本、工具推荐、Quse 和 store |
| `experience_evo_v4_generic_guard` | 当前任务图片路径检查、`compute.calculator` expression schema 检查 | 所有经验检索、产物转移提示和历史统计 |

这两个模式只用于固定子集的因果消融，不能替代 `experience_evo_v4_clean`，也不能读取
`task_type`、`expected_tools` 或 gold。`v4-clean checker_on` 则通过
`TERRABOX_EXPEVO_V4_LLM_CHECKER=1` 开启独立 LongCat checker，用于量化多 agent 校验的
收益与额外请求成本。正式结果必须同时保存 rollout report、完整结果 JSONL 和 answer judge。

正式 OEA eval 示例：

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
PYTHONPATH=src $PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment experience_evo_v4_oea_train2000_longcat_eval \
  --mode standard \
  --output-dir tmp/trajectories/experience_evo_v4_oea_train2000_longcat_eval/standard \
  --llm-provider longcat \
  --evolution-method experience_evo_v4 \
  --evolution-store evolution_store/experience_evo/oea_train2000_v2_longcat_20260731 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos \
  --resume
```

外部 LongCat 全量 watcher 必须使用 `TERRABOX_TOOL_SERVICE_SCOPE=call` 和
`TERRABOX_KEEP_VLM_WARM=1`，并按 lane 钉定各感知服务 GPU/端口；不要使用
`TERRABOX_TOOL_SERVICE_SCOPE=session`。v4 的实验结果必须同时报告工具链指标和 LongCat
`answer_acc`/`answer_acc_w_gen`，不能只看 success rate。

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
- eval：OEA test `1162` 条，已跑完；实际结果为 `completed=1070`、`failed=90`、
  `empty_final=2`。`rollout_report status` 旧展示只统计 completed+failed，所以曾显示
  `1160/1162`，手工按 results 文件确认共有 1162 条。

与 LongCat base test rollout 配对对比（全量 1162 条）：

| 指标 | LongCat base | ExperienceEvo v2 | Delta |
|---|---:|---:|---:|
| success_rate | 87.26% | 87.78% | +0.52 pp |
| set-F1 | 0.696 | 0.707 | +0.011 |
| multiset-F1 | 0.634 | 0.643 | +0.009 |
| exact | 15.49% | 17.13% | +1.64 pp |
| ordered exact | 11.96% | 12.48% | +0.52 pp |
| AnyOrder | 59.72% | 59.90% | +0.17 pp |
| SameOrder | 58.69% | 58.43% | -0.26 pp |
| Unique | 63.51% | 65.83% | +2.32 pp |
| perception F1 | 33.75 | 33.79 | +0.04 |
| operation F1 | 33.53 | 39.62 | +6.09 |
| logic F1 | 29.12 | 36.50 | +7.38 |
| gis F1 | 83.09 | 82.03 | -1.06 |
| same-tool>=4 tasks | 179 | 155 | -24 |
| errors/task | 0.43 | 0.57 | +0.14 |
| tokens/task | 54,145 | 68,948 | +14,804 |
| time/task | 67.5s | 95.2s | +27.7s |

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
