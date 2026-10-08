# Qwen2.5-3B RL 研究与实验仓库治理总计划

更新时间：2026-10-07

> 本文是 Qwen2.5-3B SFT、Swift RL、ExperienceEvo 和产物来源感知信用分配工作的跨阶段执行计划，也记录与这些实验直接相关的数据和代码治理任务。各算法的实现细节继续以对应 README/计划为准；本文负责排序、依赖、验收和当前状态。

## 1. 研究目标与证据边界

主线是在同一个 Terrabox/OEA 真实工具环境中，逐步回答四个问题：

1. 当前 harness 格式 SFT 与真实工具 replay SFT，分别能否让 Qwen2.5-3B 学会稳定调用工具？
2. 在同一数据、rollout budget、reward 和测试集上，GRPO、RLOO、REINFORCE++ 等 Swift 机制是否有可靠差异？
3. ExperienceEvo 的经验提示或 reward prior 能否提高任务完成率、降低工具错误，并能否与参数更新方法公平结合？
4. 产物来源感知的 action-level credit assignment 是否能比序列级 GRPO 更准确地把收益分配给正确的工具动作？

当前不得把“运行时加入 ExperienceEvo 经验”描述为已完成的新 RL 算法；产物来源感知 GRPO 仍是设计阶段。论文或简历中的方法主张必须对应可复现实现、独立对照和消融结果。

## 2. 当前状态快照

状态会随实验推进而变化；正式汇报前应重新读取对应 manifest、`metrics_summary.json`、watcher 状态和 checkpoint 路径。

### 2.1 数据与 SFT

| 项目 | 当前证据 | 状态与后续 |
|---|---|---|
| Current-harness SFT 数据 | `data/oea_current_harness_sft/`；14,538 train / 1,162 eval | 已用于 Swift SFT；数据只代表原 harness 监督，不是当前工具的真实执行 observation |
| Current-harness SFT 训练 | `tmp/agent_rl_runs/sft/qwen25_3b_current_harness_sft_swift_coldstart_20260916/`；checkpoint-1817 已保存并 merge | 训练完成；OEA standard eval 已完成，工具调用能力较弱，不能当作成功冷启动结论 |
| Current-harness SFT OEA eval | `tmp/agent_rl_runs/sft/qwen25_3b_sft_coldstart_ckpt1817_oea_standard_eval_tok24576_20260917/` | 1162 条；success 24.61%、Set-F1 0.1688、tools/task 0.79。后续保留为负结果和数据流程参照 |
| Real-replayed SFT 准备 | `tmp/agent_rl_runs/sft/qwen25_3b_real_replayed_sft_20260929/`；静态审计曾记录 2,000 个来源样本、9,915 个 gold tool calls、23 个工具，schema 和序列静态问题为 0 | 当前重点数据线；真实 teacher-forced replay 完成后才可定可用样本数 |
| Real gold replay | 输出 `tmp/agent_rl_runs/sft/qwen25_3b_real_replayed_sft_20260929/gold_replay/results/` | 2026-10-04 已完成 offline scope 全部 1,310 条（GPU 1,306 + nogpu 4），全部 completed；剩余 690/2,000 条属于 online scope，未纳入本次 SFT 数据 |
| Real-replayed SFT 数据 | `tmp/agent_rl_runs/sft/qwen25_3b_real_replayed_sft_20261004/replayed_sft_offline_full_20261004/` | 1,179 train / 131 val；结构、assistant 字段和 split 去重通过；Qwen tokenizer 下长度中位数约 6.1k，100% 超过旧配置 max_length=4096，正式训练前必须解决长度截断问题 |
| Real-replayed SFT 全量 OEA rollout | `tmp/agent_rl_runs/sft/qwen25_3b_replayed_sft_oea_full_safe_unsloth_20261005/` | 2026-10-07 完成 1,162/1,162；`sft-json`、24576 context、4096 completion、15 turns；runner success 37.95%、Set-F1 0.282、MultiSet-F1 0.215、AnyOrder 11.96%；local answer accuracy 15.37%、含生成类任务口径 35.21%（judge 1,020 条、生成类 142 条、0 error）。309 条（26.6%）撞 max-turn，594 条有同工具连续调用≥4；OOM/context/infra 桶均为 0。失败审计发现 228 条调用了不存在的图片路径，其中 220 条测试任务没有图片输入；另有 6 条调用了无效工具名。local judge 仅作诊断，不能直接与历史 LongCat 主表比较。该结果说明 rollout 稳定、模型工具行为仍弱，不能据此声称 SFT 有效或直接开始 artifact-credit 主实验。状态文件已按 1,162 个 manifest task id 补齐完成审计字段。

### 2.2 已有 RL / ExperienceEvo 结果

以下是各自 `metrics_summary.json` 中的 1162 条 OEA test 结果快照。success 和 Set-F1 的计分口径仍须结合 manifest 统一审计；结果目录不能仅凭指标接近就视作完全公平对照。

| 方法 / checkpoint | success | Set-F1 | 有工具错误任务 | 当前解释 |
|---|---:|---:|---:|---|
| Pure GRPO ckpt-1900 | 70.65% | 0.5102 | 49.74% | 相较已知 Base 快照没有明显增益 |
| Swift GRPO-RLOO ckpt-1900 | 72.72% | 0.5191 | 48.19% | 训练与全量 OEA rollout 已完成 |
| Swift GRPO-REINFORCE++ ckpt-1900 | 72.03% | 0.5197 | 48.11% | 已完成；比较时需报告 KL/reward 与 rollout 配置 |
| Pure GRPO + ExperienceEvo v4-clean | 73.75% | 0.5137 | 34.17% | success 和工具错误任务较好；这属于运行时经验增强评测，不是新的 RL trainer |
| Base + ExperienceEvo v4-clean（独立评测目录） | 74.44% | 0.5156 | 37.69% | 命名含 `after_sft`，须先查 model path 和 eval manifest，不能直接作为 Base 纯模型结果 |

Pure GRPO、RLOO、REINFORCE++ 的路径：

- `tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_ckpt1900_oea_standard_eval_tok24576_stable3_20260914/`
- `tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_rloo_ckpt1900_oea_standard_eval_tok24576_20260923/`
- `tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_reinforcepp_ckpt1900_oea_standard_eval_tok24576_20260928/`
- ExperienceEvo eval：`tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_ckpt1900_experience_evo_v4clean_eval_tok24576_20260915/`

当前尚未确认所有实验都完成答案 judge；工具选择指标不能替代自然语言答案正确性。主表前应生成同 provider、同 prompt、同 judge cache policy 的 answer accuracy，并报告 infrastructure/system-limited 桶。

## 3. 执行总顺序与验收门

### 阶段 A：收尾并审计 gold replay（已完成）

**动作**

1. 监控既有 tmux、进程、日志与 result 文件数；以 result JSON 为任务粒度证据，日志进度行为批量汇总。
2. 原 GPU lane 的 1,306 条已完成；随后按 CLI 默认 resume 语义补跑 offline nogpu lane，新增 4 条。注意 runner 没有 `--resume` 参数，默认续跑，`--no-resume` 才会关闭续跑。
3. replay 完成后统计 completed/failed/OOM/system limitation、工具错误、精确工具序列匹配、每任务 observation 数、重试与遗漏结果。对可用轨迹做抽样人工核查，但不要打印或复制大段 gold 内容。

**验收**：offline scope 1,310 条均有 completed 结果且无工具错误 observation；online scope 的 690 条未跑，本次 SFT 明确排除。整理派生报告时注意它按结果目录重建全量指标，但 `expected_total_tasks` 取当前调用筛选数，不能据此判断历史批次总量。

**输出**：保留原始 replay；后续 SFT 数据另写新目录，生成含输入路径、数据哈希、split seed、usable/rejected 数量和原因的 manifest。

### 阶段 B：建立可复现的 real-replayed SFT 版本（数据已生成，长度审计未通过）

1. 用 `terrabox.evolution.sft.runner prepare-replayed-data` 生成新的、唯一的输出目录。当前代码默认拒绝覆盖已有文件；除非明确要重建同一版本，否则不要使用 `--overwrite`。
2. 审核输出仅含 `messages`，训练侧不含 `ground_truth`、`expected_tools`、gold tool calls 或任务标签；gold answer 只可按当前 adapter 设计作为 final-answer supervision，不能进入 rollout prompt/test 输入。
3. 已核验 train/validation 内容无交集、JSON schema 无额外 gold 字段、系统 prompt 哈希与旧版本一致；采用 Qwen2.5-3B tokenizer 统计 train 中位长度 6,136 / val 6,164，p95 分别 7,134 / 7,035，最大分别 9,478 / 9,463，全部超过旧 `max_length=4096`。下一步必须做 Swift tokenizer/template 与 assistant loss mask 的实际预处理审计，并在 8k/10k 长度策略间评估显存和截断位置；不得直接沿用 4096 训练。
4. 对数据文件生成 SHA-256 和行数记录，冻结版本后再训练；改任何过滤或 observation 规则都创建新数据版本。

**验收**：预检无 schema/tokenization 错误；Swift 实际模板下的长度截断和 assistant loss mask audit 通过；超长样本的保留/过滤策略有记录；manifest 能从原始 source、scope 和 replay 目录追溯到最终行数；测试集与训练来源隔离。

### 阶段 C：训练并评估 real-replayed SFT

1. 以已记录的 Qwen2.5-3B base checkpoint、Swift 训练配置和固定 1 epoch 作为首个正式版本；先 preflight，再启动训练。禁止与 gold replay 或其他感知任务争用相同 GPU lane。
2. 保存 checkpoint、训练配置、tokenizer、数据哈希、训练日志、合并模型 manifest；确认完整性后释放本次启动的训练服务/进程。
3. 用 `data/oea_full_sft/openearth_test_tasks.json` 的 1162 条 prompt-only test 做 OEA `standard` 真实工具 rollout。模型、prompt、工具目录、max iterations、上下文长度、GPU lane、timeout、缓存策略和 judge 设置应与主对照固定。首轮 rollout 已完成，详见状态表；当前应先定位重复工具/不终止问题并完成 answer judge，再决定是否修改协议后补一组隔离 smoke。
4. 复用标准 rollout 的 gpu/nogpu 分 lane 策略；检查 watcher、结果数量、遗漏/失败桶、answer judge 和完整资源清理。

**验收**：SFT 至少与 Base、Pure GRPO 处于同一有效 eval manifest；报告 success、Set/Multiset F1、exact/ordered、tool error、tools/task、answer accuracy、system-limited 和成本/时延。若 real-replayed SFT 没有改善，不以训练 loss 或 token accuracy 宣称成功。2026-10-07 修复前，公共 `rollout_report` 未能从 `sft-json` 的顶层规范化 `tool_calls` 还原序列，曾输出虚假的全零工具指标；修复后重新生成的 `metrics_summary.json/md` 才是本次 SFT 工具匹配结果权威报告。

### 阶段 C0：冻结 Qwen2.5-3B rollout 合同（2026-10-05 起执行）

旧 Base/RL 结果使用 vLLM Hermes 原生 `tool_calls`；real-replayed SFT 的训练目标则是
OpenEarthAgent 风格的 assistant JSON `actions` + `OBSERVATION` 用户消息。两者不能共用
parser 或 system prompt。统一入口 `scripts/run_trajectory_experiment.py` 已加入显式
`--tool-protocol native|sft-json`、训练 system prompt 校验、每次请求的 completion 上限和
`rollout_manifest.json`。固定约定为：

1. Base、GRPO、RLOO、REINFORCE++ 及其它原生 tool-call checkpoint 使用 `native`；Swift
   文本 ReAct SFT（包括 real-replayed SFT）使用 `sft-json`，并传同一数据版本产生的
   `system_prompt.txt`。
2. 本地 Qwen2.5-3B rollout 使用 `AGENT_LLM_MAX_MODEL_LEN=24576`、每次请求最多
   `TERRABOX_AGENT_LLM_MAX_TOKENS=4096`、`max_iterations=15`。输入历史不主动截断；如果
   vLLM 仍报告 context overflow，对应结果必须保留为 `status=context_overflow`、
   `has_context_overflow=true` 的基础设施失败，并跳过瞬时错误重试，不能把不完整上下文计作完成。
   runner 会把任务 `images` 与 `data_files` 合并到当前任务输入解析范围，修复模型复述训练集绝对
   图片路径时的 basename 解析；无当前任务输入的幻觉路径仍按工具错误统计。
   工具服务容器启动失败或中途退出时，结果记录为 `status=infra_error`、
   `has_infra_error=true`，按有限瞬时重试处理；后续 calculator/solver 不能覆盖这类基础设施失败。
3. 感知任务使用单独 GPU lane、`workers=1`、服务锁和 `TERRABOX_TOOL_SERVICE_SCOPE=call`；
   agent LLM 与 VLM/感知服务不得共卡。先跑 3–10 条 smoke，核对 `tool_calls`、每次 token
   使用、finish/context 错误和 manifest，再以同一目录 `--resume` 做全量。
4. 结果目录必须包含模型路径、协议、system prompt 哈希、context、completion budget、
   工具目录、端口/GPU 和 provider；主表只比较 contract 一致的结果。

详细命令与排障见 `src/terrabox/evolution/sft/README.md` 的“Qwen2.5-3B rollout 固定合同”。

### 阶段 D：核实并冻结 Swift RL 公平对照表

1. 审核 GRPO、RLOO、REINFORCE++ 的 canonical train manifest、训练 step、采样数量、reward、KL、generation budget、base model、merge checkpoint 和 OEA eval manifest。
2. 确认三者是否同一 train subset、相同 rollout 数量和 token budget；如有差异，在结果表标成不同预算/诊断对照，不重跑已经完成的训练来追求形式整齐。
3. 读取 Pure GRPO 其他稳定性评测，报告均值/方差或 task-level paired bootstrap；处理 eval manifest 的 worker 数差异和 system-limited 样本，确定主分析口径。
4. 核实 Base+ExperienceEvo 目录的模型身份；分别保留 Base-only、ExperienceEvo-only、RL-only、ExperienceEvo+RL 的清晰命名。
5. 完成所有待处理的答案 judge，并锁定统一缓存与判分配置。

**决策门**：如果已有方法训练配置或评测条件无法公平对照，先补 manifest/统计与小规模审计。只有问题会实质影响主结论时才补跑；不重跑纯粹重复的 RLOO/REINFORCE++ 全量实验。

### 阶段 E：选择 Swift 额外 RL 对照（条件执行）

1. 首选候选为 DAPO loss：研究问题是 OEA 长短轨迹差异是否让 token-level loss weighting 更稳；保持同一 Swift GRPO trainer、数据、reward、rollout 数和训练预算，只改变 loss normalization。
2. GDPO 只有在 reward 保持为分量（成功、工具契约、格式、证据等）并记录分量日志时才有意义；先实现/审计 reward trace，再决定是否训练。
3. GSPO-style sequence importance sampling 是 loss/importance sampling 消融；不能写成已使用独立 GSPO trainer。
4. DPO/ORPO/KTO 属离线 preference baseline，只有完成 chosen/rejected 轨迹构造、任务配对和数据污染检查后再立项；PPO 暂缓。

**验收**：冻结配置与 command manifest；记录训练异常、KL、clip fraction、reward components、有效 token、吞吐和 checkpoint；结果与同预算 GRPO 配对分析。

### 阶段 F：ExperienceEvo 引导 RL 与消融

1. 固定经验库来源、版本、可见性和 train/test 边界；actor/store 不得读取测试 gold、`expected_tools`、`ground_truth`、task id 或数据集标签。
2. 至少区分：纯 RL、ExperienceEvo prompt-only、reward-only/ECPR、prompt+reward；每种都用同一 base、rollout budget、reward 基础项和 eval。
3. 报告经验命中/使用、工具错误桶、成功率、answer accuracy、token/time/cost，并作 task-level 配对统计。
4. 先审计已有 v4-clean 和 QNR/ECPR 代码及结果，再决定是补消融还是推进新的 reward 版本。

**验收**：ExperienceEvo 增益不能仅由不同评测配置、筛选或模型来源解释；负结果、经验未命中和基础设施错误如实保留。

### 阶段 G：实现产物来源感知 action-level credit assignment

这是研究创新阶段，依赖 A–F 的稳定基线。实现保持在独立 `artifact_credit_rl/`（或现有明确隔离的方法目录）；不得为方便而改公共 `agent_rl/rewards.py`、Pure GRPO reward 或旧 baseline。

1. 定义 tool action、artifact、consumer edge、tool contract、来源/时空/对象绑定和局部结果的结构化 trace schema。
2. 先做离线 trace parser 与确定性审计：验证 artifact parent-child 链、相同前缀动作对照、错误绑定/无效消费、跨 episode 隔离；不把数据集 gold 序列当作训练期 credit 信号。
3. 构造不依赖 gold 的 action-level advantage：基础信号来自实际工具返回、schema/contract 校验、artifact 依赖图和后续实际消费；gold 仅用于离线评估诊断。
4. 明确 token-span 到 action 的映射、缺失 observation、并行/重复工具调用、无 artifact 工具、工具报错、截断轨迹和 credit normalization 的行为。
5. 用小样本 CPU/模拟 trace 测试公式与边界，再跑少量真实工具 pilot；检查梯度、KL、NaN、方差、动作覆盖和额外开销。
6. 最小消融：序列级 GRPO；仅工具成功局部 credit；加入 artifact provenance；去除来源约束；随机/打乱 provenance 对照。各组用同一 rollout budget 和 paired task set。

**验收**：实现、trace schema、公式、日志和消融结果均可复现；相对 GRPO 的增益不能仅来自额外 reward 总量或更大采样预算；正式结果后再更新方法说明和简历表述。

### 阶段 H：论文/汇报结果冻结

1. 用一个机器可读索引汇总每个 run 的 model/data/config/eval manifest、commit、指标 JSON 和 answer judge。
2. 主表与附表标明官方复现、adapted/reimplemented、数据/训练/评测预算差异；不能把诊断结果与正式主结果混在一起。
3. 完成 success、tool matching、answer correctness、工具错误、资源和配对统计的统一口径；报告置信区间和 system-limited 处理方式。
4. 对外写作只陈述已完成证据；“设计中”“已实现未验证”“完成对照”分开描述。

## 4. 数据与仓库整理计划

### 4.1 数据目录分类

| 类别 | 例子 | 处理策略 |
|---|---|---|
| 当前主力原始/派生输入 | `data/oea_full_sft/`、`data/oea_current_harness_sft/` | 保留；固定 manifest、生成命令、split 和 SHA-256；绝不因 mtime 老就删 |
| OpenEarth 兼容数据与影像 | `data/openearth/`、`data/openearth/images/`、`data/openearth/geo/` | 多个 runner/方法仍引用；保留，逐步在目录索引中标明主要消费者 |
| Disaster/旧方法训练数据 | `data/disaster_*`、mapping、`drone_video_sft_dataset.json` | 当前旧 runner、转换器仍有默认引用；确认方法是否退役及代码默认路径迁移后再评估归档 |
| 可再生成的报告/缓存 | `batch_test_report.json` 等 | 只在无消费者、已知生成命令且有归档/哈希时移动到 archive；默认不永久删除 |
| 已归档本地文件 | `data/archive/legacy_20261003/` | 保留来源路径、理由、大小、哈希；不参与训练或评测 |
| 大型模型/rollout/checkpoint | `tmp/agent_rl_runs/`、`tmp/trajectories/` | 不纳入源码树；按实验 manifest 和资源状态管理，运行中不可清理 |

“几个月没更新”只能触发引用审计，不能作为删除条件。任何候选删除必须同时满足：代码/文档/配置零引用、不是唯一原始来源、结果已有备份或可再生成、哈希和变更清单已留存。

### 4.2 代码与文档审计优先级

| 优先级 | 检查/改进 | 验收 |
|---|---|---|
| P0 | 数据生成输出不静默覆盖；replayed SFT 已加拒绝覆盖及显式 `--overwrite` | CLI 与隔离目录 smoke 通过 |
| P0 | 训练/测试数据隔离、system prompt 与 JSON-actions loss mask、来源哈希 | manifest + audit 输出能够复核 |
| P1 | `sft.runner` 的旧 strict/TRL、Swift、veRL 入口和默认值容易混淆 | README 明确当前默认和 legacy/debug，不改下游接口 |
| P1 | 数据目录目录索引过时或被 `.gitignore` 忽略 | 在受版本管理的 evolution 文档中维护当前数据目录索引；说明被忽略的 `data/DATA_README.md` 不是权威入口 |
| P1 | OEA rollout manifest 的 worker、端口、模型、token 长度和 provider 口径差异 | 汇总表明确可比/不可比项；不得用不同口径数字做直接优劣结论 |
| P2 | 各旧方法转换器、batch script、历史 runner 的消费者清点 | 仅对确认退役入口加 deprecation/迁移说明；不批量删除脚本 |
| P2 | 实验结果与代码树边界 | 模型、轨迹、judge cache、临时产物只放 `tmp/` 或明确 archive |

新增脚本仍须先向用户说明用途、命名、参数和输出位置并取得确认；优先在现有 runner 增强检查。修改数据契约、rollout flag、GPU/service 配置后同步对应 README、实验 manifest 与就近 `AGENTS.md`。

## 5. 当前已完成的治理改动

- `sft.data_adapter.build_replayed_sft_dataset()` 增加输出覆盖保护；已有四个受管文件时默认报错，API 和 CLI 显式 `overwrite` 才允许重建。
- `sft/README.md` 已把当前 Qwen2.5-3B Swift 主线放到开头，并标注旧 Qwen3-8B / strict 数据内容为历史记录。
- `data/batch_test_report.json` 与两份字节相同的 mapping 备份移入 `data/archive/legacy_20261003/`；文件未删除，`archive_manifest.json` 保存来源、大小和 SHA-256。`scripts/run_sft_batch.py` 仍可在需要时重新生成默认报告。
- 验证已完成：目标 Python 文件 `py_compile`、`prepare-replayed-data --help`、防覆盖临时目录 smoke、`git diff --check`。

## 6. 执行约束与停止条件

- 当前 gold replay 已完成；GPU0 上仍有来源属于既有环境的常驻服务，本次未停止。新的训练启动前继续核实 GPU/CPU 和服务资源，并只清理本次启动的资源。
- 使用 `unsloth` 环境；RL/SFT 正式训练前同时检查 GPU、CPU 内存、数据版本和 manifest。长训练默认周期保存 checkpoint、支持 resume；实验结束核实并释放本次启动的资源。
- 不隐藏工具目录、不基于测试 gold 改 prompt、经验、reward 或训练数据；测试 gold 只用于隔离的离线评估与诊断。
- watcher/rollout 共享输出时用 `--resume` 和 task id 独立结果；外部 API 按仓库 AGENTS 的 provider 节流与多 lane 规则运行。
- 任何比较若出现数据集交叉、base model 错配、服务失败、answer judge 口径不同或 checkpoint 不能复现，先标为诊断并暂停主结论，不通过删样本或改口径“修好”指标。
- 当前工作区已有大量用户未提交修改/删除；不 reset、不 checkout、不批量格式化，也不覆盖已有计划/实验记录。仅提交与本总计划明确相关的窄范围变更。

## 7. 下一步操作清单

1. 将 real-replayed SFT 的 local judge 结果留作诊断；若要和历史主表比较，使用同一 LongCat judge 配置补齐答案判分，并记录 provider、prompt 和 cache 口径。
2. 审计 228 条错误图片路径的轨迹与输入分布，区分“测试任务无图却调用感知工具”和“有图但模型复用了训练路径”；通过独立小规模 prompt/数据诊断评估通用修正，不改写本次正式结果。
3. 按统一 OEA manifest、checkpoint、tool protocol 和 judge provider 核对 Base、current-harness SFT、real-replayed SFT、GRPO、RLOO、REINFORCE++ 与 ExperienceEvo 的可比性，并生成 paired 结果表。
4. 另行决定是否生成 online scope teacher-forced replay 数据；它包含网络/在线工具任务，不得把本次 offline SFT 误称为完整 2,000 条覆盖。
5. 依据公平对照结果决定是否跑 DAPO/GDPO 消融；随后再进入独立的 artifact-credit pilot。
6. 完成当前 RL 主线后，按引用图逐类清理或归档旧数据/脚本，所有归档可追溯且不破坏兼容入口。

### 2026-10-07 状态更新与紧接着的动作

1. Real-replayed SFT 全量 rollout 已完成且结果目录与 1,162 个 task id 完整一致；run_status 中补写了旧 runner 未记录的 `completed_selected`、`completed_manifest`、`invocation_complete`、`manifest_complete` 和 `partial` 字段。GPU1 的本实验 vLLM、rollout tmux 已退出，GPU1–3 空闲；保留既有 GPU0 常驻 `terrabox-vllm-9000`。
2. 公共 `rollout_metrics.reconstruct()` 现在在 native conversation history 没有调用记录时回退到 SFT/RL 结果顶层 `tool_calls`；对应单测覆盖 `sft-json`。不能再使用修复前 `metrics_summary` 的全零工具指标。
3. SFT rollout 指标与 answer judge 均已完成：runner success 37.95%、Set-F1 0.282、MultiSet-F1 0.215、local answer_acc 15.37%、answer_acc_w_gen 35.21%；judge 1,020 条、生成类 142 条、0 error，已并入 `report.json`。本地 judge 和历史 LongCat judge 口径不同，暂不并入跨方法主表。
4. 失败审计中 309 条均到 max-turn；其中 228 条出现 `Image not found`（220 条对应测试任务 `images=[]`，8 条任务有图但模型路径未匹配），6 条有无效工具名，78 条没有这两类错误但仍到轮数上限。stop-loop fake LLM smoke 已通过，说明 action/observation 配对与停止逻辑可工作；正式轨迹暴露的是模型重复和输入选择问题，不支持通过降低轮数或加 gold 特判修复。
5. watcher 生成器 `agent_rl.runner write-oea-eval-command` 现在支持显式 `--tool-protocol native|sft-json`；SFT 必须同时传同一训练版本的 `system_prompt.txt`，生成器会把协议、prompt 路径、SHA-256、24576 context/4096 completion 写入 manifest 并传给两条 lane。`tmp/watcher_validation2/configs/run_oea_eval.sh` 仅做过静态 `bash -n` 验证，且使用 SFT checkpoint 却走 native protocol，**不得启动**；已生成并验证的修正版只位于 `tmp/watcher_validation2_fixed/`，本轮也没有启动它。正式 RL/SFT watcher 都应从生成器重新生成；生成器已有 lane 后状态门（要求 `status=complete` 且 `invocation_complete=true`），两 lane 显式 `workers=1`、全程 `--resume`。
6. 本地 Evolution LLM/judge 已支持未固定地址时探测 `EVOLUTION_LLM_URLS` 或 `9100`–`9103`、`9000`；正式 watcher 仍应在 manifest/日志记录实际端口，显式 `--llm-url` 可关闭自动探测。
7. 2026-10-08 诊断 smoke 发现 SFT 分片的工具目录契约问题：任务文件只含 5 个 OSM gold 工具时，旧 runner 会把 runtime allow-list 错缩成 5 个，而 SFT system prompt 仍展示完整 23 工具目录，导致 `geo_perception`/`compute` 被判为不存在并触发 max-turn。已修复 `scripts/run_trajectory_experiment.py`：`sft-json` 从训练版本 prompt 的 `Tool catalog` 解析并校验 registry，native 仍按任务文件并集；README、模块 AGENTS 和测试已同步。旧正式 1,162 条结果不覆盖，需用修复后的隔离 smoke 验证后再决定是否补跑正式 SFT rollout。

## 8. 相关详细文档

- 真实 SFT 流程：`sft/README.md`、`sft/Qwen3B_CurrentHarness_SFT_冷启动流程_20260916.md`
- Swift/veRL 公共适配：`agent_rl/README.md`、`agent_rl/Swift_veRL_双框架适配计划_20260911.md`
- Swift 算法候选与公平对照：`agent_rl/Swift_其他RL算法实验规划_20260923.md`
- ExperienceEvo Qwen2.5-3B RL 矩阵：`ExperienceEvo_3B_RL大范围实验计划_20260902.md`
- artifact-provenance 方法设计：`ExperienceEvo_产物来源感知GRPO_小白说明与简历写法_20260920.md`
- 仓库运行与资源规则：`AGENTS.md`、仓库根目录 `AGENTS.md`
