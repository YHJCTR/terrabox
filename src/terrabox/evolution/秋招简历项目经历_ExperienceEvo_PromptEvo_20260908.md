# 秋招简历项目经历：地理空间多工具 Agent 自进化系统

> 项目名称可写作：**地理空间多工具 Agent 自进化系统｜ExperienceEvo × PromptEvo**

## 版本 A：算法/Agent 岗位（可直接使用）

**地理空间多工具 Agent 自进化系统｜ExperienceEvo × PromptEvo**
**技术栈：** Python、LangGraph、LangChain、vLLM、Docker、Milvus Lite、OpenEarthAgent（OEA）

- **ExperienceEvo 经验复用算法：**针对多工具 Agent 在长尾工具选择和中间产物依赖下的规划不稳定，提出“产物状态转移”经验表示，统一建模前置条件、参数绑定、输出契约、后验检查和失败恢复；进一步设计 Q/N/R（质量、支持度、风险）统计与 `Quse` 风险感知排序，在每轮工具 observation 后按当前产物状态动态检索并注入候选策略，同时通过证据审核模块约束产物、数值和最终答案的可验证性。在 Base、Reflection、MemRL、ExpeL、ACE 等方法对比中取得当前最高工具链指标，Success **86.90%→92.51%**、Set/Multiset F1 **.703/.637→.774/.722**、工具错误率 **15.4%→9.6%**、平均调用 **6.48→4.80**。
- **PromptEvo 提示词自进化：**针对固定 system prompt 难覆盖复杂多步工具决策的问题，构建 rollout 失败归因与协议补丁生成流程；通过 Stage1 生成候选补丁、Stage2 成对 rollout 对比优化静态提示词，OEA Qwen3 Set F1 **.628→.661**、Exact/Ordered **22.85%/16.11%→34.55%/23.18%**。
- **实验与评测闭环：**统一真实工具 rollout、执行环境和指标口径，沉淀任务轨迹、产物状态、协议版本与资源日志，支持断点续跑、失败定位、指标重算和结果复核。

## 版本 B：工程/大模型应用岗位（更短）

**地理空间多工具 Agent 自进化系统｜ExperienceEvo × PromptEvo**

- **Agent 基础设施：**基于 LangGraph + LangChain 构建可恢复的地理空间 ReAct Agent，统一工具 schema、调用编排、产物状态管理、RAG、代码执行沙箱和多轮会话。
- **经验复用优化：**设计 ExperienceEvo，将产物状态转化为可检索的工具策略，结合前置条件进行结构化注入，使 success **86.90%→92.51%**、Set F1 **.703→.774**、平均工具调用 **6.48→4.80**。
- **静态提示词自进化与实验治理：**设计 PromptEvo，针对 system prompt 的跨任务行为缺陷生成通用协议补丁；Stage1 从失败轨迹提出首版候选，Stage2 通过 base/候选成对 rollout 做对比式优化并检查副作用，同时实现提示词版本化、断点恢复、任务级日志和指标复盘。

## 版本 C：突出 PromptEvo（算法岗可选）

**PromptEvo：面向工具调用 Agent 静态 system prompt 的失败驱动自进化**

- **失败归因：**将工具选择、参数、产物衔接、错误恢复和提前结束等行为结构化，聚合跨任务重复失败模式。
- **补丁生成与两阶段优化：**由 LLM 生成任务无关的 protocol patch；Stage1 根据基础提示词的失败轨迹完成问题自发现并生成首版候选，Stage2 通过 base/候选成对 rollout 的差异进行对比式优化，同时检查过调用、工具错误和效率副作用，并保留 shadow 结果与提示词版本。
- **效果验证：**在同配置对照中，API-Bank tool-search 的 exact API-call/success **54.62%→56.30%（+1.68 个百分点）**，protocol patch **.8226→.8380（+0.0154）**；OEA Qwen3 对照中 Set F1 **.628→.661（+0.033）**、Exact/Ordered **22.85%/16.11%→34.55%/23.18%（+11.70/+7.07 个百分点）**，对存在 trade-off 的指标不做全面提升表述。

## 关键模块与技术细节（面试备查）

### ExperienceEvo

1. **经验生成：**解析历史任务中的用户目标、工具调用、参数、工具返回和产物变化；将完整对话压缩成可迁移的状态转移，而不是保存任务答案。
2. **质量治理：**检查工具 schema、必填参数、执行成功、产物落盘、后续产物使用和重复调用；按执行质量、跨任务支持度和相似度去重，形成冻结经验库。
3. **经验注入：**以“任务目标 + 当前文件/图层 + 已有产物 + 已执行工具 + 最近观察”检索 Top-K；过滤前置条件不满足或 schema 不兼容的经验，仅注入结构化候选策略，模型仍自主决定最终调用。
4. **防泄漏：**经验只保留可观察的状态、工具约束和执行结果，不保留任务答案、标签或固定任务标识；测试时使用冻结的经验库。

可解释的经验排序可概括为：

```text
Q_use = λ Q_tool + (1 - λ) Q_sig
λ = N_tool / (N_tool + k)
```

其中 `Q_tool` 表示工具执行/产物状态质量，`Q_sig` 表示模式稳定性与支持度，`N_tool` 为有效支持次数。

### PromptEvo

1. 收集固定 system prompt 的真实 rollout，并按工具误选、参数错误、产物断链、无效重复调用、恢复失败和过早结束分类。
2. 将重复失败归纳为与具体任务无关的静态提示词/行为协议缺陷，调用 LLM 生成 protocol/prompt patch。
3. Stage1 基于基础提示词和失败轨迹做开放式问题自发现，编译出首版候选静态提示词；Stage2 在同一任务与工具环境下对比 base/候选的成对 rollout，根据行为差异继续优化候选，并检查工具错误、调用数、token、超时和新失败类型。

### 工程化与可观测性

- 根据产物状态动态披露可用工具，减少无关 schema 和上下文；平台实验中复杂地理任务平均 token 消耗 **21.5K→14.2K**。
- 采用结构化摘要压缩历史观察，保留最近原文和关键产物证据，降低长轨迹上下文超限及 OOM 风险。
- 每次运行保存 prompt/protocol 版本、数据 manifest、任务级 episode trace、原始工具观察、压缩统计、GPU/CPU 记录、checkpoint 和 judge 缓存，支持后续画 reward、工具分布、失败类型和效率曲线。

## 指标使用边界

- ExperienceEvo 的主结论应写成**工具链完成度、产物状态衔接和调用效率提升**；普通文本 answer accuracy 基本持平（49.67%→49.32%），不能写成答案准确率全面提升。
- OEA 主表指标来自完整真实工具 rollout；离线 action 诊断和 Qwen 3B GRPO 训练 reward 不替代 OEA test 指标。
- Qwen 3B online GRPO 曾因 CUDA OOM 在 step 0 中止，只有部分轨迹，没有最终 checkpoint 和完整 OEA test 结果，不写入已完成成果。
- “13 个末级风险点、Judge 驳回率/成功率、77.7%→92.2%”属于另一个商品审核项目，不属于本项目。

## 一分钟项目介绍

我做的是一个多工具地理空间 Agent 自进化平台。底层用 LangGraph 统一管理 ReAct 状态、工具 schema 和中间产物；ExperienceEvo 将工具操作抽象为产物状态转移，在测试时按当前状态检索可迁移的工具策略；PromptEvo 面向固定 system prompt，从真实失败轨迹中自发现跨任务的行为协议缺陷，由 LLM 生成静态提示词候选，再通过 Stage1 首版提案和 Stage2 base/候选成对 rollout 对比进行优化。在统一 OEA 测试中，ExperienceEvo 将 success 从 86.90% 提升到 92.51%，Set F1 从 .703 提升到 .774，平均工具调用从 6.48 降到 4.80，同时保留了完整的轨迹、资源和版本化实验记录。

## ExperienceEvo 与 PromptEvo：完整技术说明

### 1. ExperienceEvo 解决什么问题

多工具 Agent 的失败不一定是模型不会推理，很多时候是“当前已经有什么产物、下一步缺什么产物、哪个工具能完成转换”没有被稳定地组织起来。ExperienceEvo 不把历史对话当作普通文本知识库，而是把可迁移的工具操作经验表示为产物状态转移：

```text
当前输入产物/状态 → 目标产物/状态 → 候选工具与参数策略 → 新增产物/状态
```

例如，已有一幅栅格影像、目标是得到裁剪后的 AOI 统计结果，经验中保留的是输入类型、必要前置条件、候选工具、关键参数约束和预期输出产物，而不是某个任务的文件名、答案或完整对话。

### 2. ExperienceEvo 的经验生成流程

1. **经验抽取：**从历史真实 rollout 中提取可复用的状态、工具和产物信息；经验库建立后冻结，测试任务不反向写入 store。
2. **轨迹解析：**从真实轨迹中提取任务目标、工具调用、参数、工具返回、错误/超时、产物变化和后续是否使用该产物。
3. **状态抽象：**将连续多轮调用压缩成输入状态、目标状态、工具/参数策略、输出状态四元组，去除任务特定文本和无关历史。
4. **质量过滤：**检查工具是否存在、参数是否符合 schema、调用是否成功、输出是否落盘、产物是否被后续步骤消费，以及是否存在无效重复调用或过早结束。
5. **经验治理：**对相似经验做去重和合并，按照执行质量、跨任务支持度、稳定性和风险进行排序与容量淘汰，并保留命中次数、相似度和使用结果用于审计。
6. **防泄漏：**显式删除 task id、task type、expected tools、gold answer、gold tool calls、F1 等标签字段；经验只描述可观察状态和工具策略。

### 3. ExperienceEvo 的推理时注入

每个决策点使用“用户目标 + 输入文件/图层 + 已有 artifact + 已执行工具 + 最近观察”构造检索状态。候选经验需要同时满足：

- 当前输入产物和前置条件存在；
- 目标状态与当前任务相容；
- 工具在当前目录中可用，参数 schema 可满足；
- 历史执行质量和支持度达到阈值。

通过过滤后只注入结构化的候选策略块，顺序为“系统约束 → 工具 schema → 用户任务 → 当前 artifact state → 经验候选 → 模型决策”。经验不是强制动作，模型仍可拒绝、改写或回退到通用 ReAct；调用结果会继续更新 artifact state 并记录到 episode trace。

#### 一个脱敏的实际经验条目示例

ExperienceEvo 经验库中的一条经验 family 可抽象为如下结构。具体任务名、文件路径和事件 ID 已泛化：

```json
{
  "family_id": "<hash>",
  "intent_signature": "distance",
  "input_product_state": ["task_request"],
  "target_product_state": ["result:from:geo_perception.instructsam"],
  "product_experience": {
    "goal": "产生 InstructSAM 的可验证结果产物",
    "preconditions": ["运行时已有任务请求和匹配影像"],
    "output_checks": ["返回中必须包含该工具的有效结果", "拒绝空输出、工具错误和虚构路径"],
    "downstream_rule": "后续调用只能使用工具真实返回的路径、图层名或数值",
    "recovery": ["缺少前置产物时先生成/恢复", "输出无效时修正参数绑定后再重试"],
    "q": 0.607,
    "n": 171,
    "risk": 0.0
  },
  "tool_policies": [
    {
      "tool": "geo_perception.instructsam",
      "required_input_roles": ["task_request"],
      "parameter_binding_rules": ["image 必须绑定当前 artifact 中的匹配影像", "text_prompt 从当前任务推导，不复制历史样例字面量"],
      "output_contract": ["result:from:geo_perception.instructsam"],
      "post_checks": ["工具无 error", "返回的路径或图层名只能原样传递"],
      "q": 0.607,
      "n": 171,
      "risk": 0.0
    }
  ],
  "provenance_summary": {
    "events": 195,
    "non_infra_events": 171,
    "locally_attributable_failures": 0,
    "strict_rollout_only": true
  }
}
```

这里的 `n=171` 表示该状态—工具模式的有效支持次数，不表示模型看过 gold demonstration；`q` 是运行质量/稳定性信号，`risk` 用于降低高风险候选的优先级。

#### 从经验到一次真实决策的例子

假设用户要求“在给定卫星影像中分割某个文本描述的对象，并将结果用于后续面积或距离分析”。运行时执行以下闭环：

1. 当前 state 有用户任务和输入影像，检索到上面的 `instructsam` family；其 `required_input_roles` 和当前 artifact 能对齐，因此通过 guard。
2. 注入给模型的不是旧任务答案，而是类似下面的短策略块：

```text
[Retrieved state-transition policy]
Goal: create a validated result from geo_perception.instructsam.
Precondition: bind image to a current artifact; derive text_prompt from this task.
Post-check: continue only if the observation has no error and returns a real artifact/value.
Downstream: reuse returned artifact references exactly; do not invent paths.
Recovery: if binding/output validation fails, correct it before the next step.
```

3. 模型依据当前任务自主生成 `geo_perception.instructsam(image=<当前影像>, text_prompt=<当前对象描述>)`；经验不提供固定文件名或固定参数值。
4. 工具返回 mask/像素统计或 artifact 路径后，executor 将它写入当前 `artifact state`。若返回错误、空结果或不存在的路径，post-check 拦截后续工具链，模型转入修正/恢复；若通过，则下一步经验可基于新状态推荐面积统计、距离计算或展示工具。

这个例子体现了 ExperienceEvo 的核心：检索的对象是“什么时候可以做什么、如何绑定参数、成功后什么能作为下一步输入”，而不是“这道题应该调用什么 gold 工具序列”。

经验排序可用下式向面试官解释：

```text
Q_use = λ Q_tool + (1 - λ) Q_sig
λ = N_tool / (N_tool + k)
```

`Q_tool` 表示工具执行和产物状态质量，`Q_sig` 表示模式稳定性/支持度，`N_tool` 表示有效支持次数，`k` 为平滑项。这样可以避免一次偶然成功的经验被过度复用。

### 4. PromptEvo 解决什么问题

ExperienceEvo 主要解决“局部状态下如何复用操作经验”；PromptEvo 解决“固定 system prompt 在不同任务中反复出现的全局行为缺陷”。它不直接更新模型参数，而是从真实失败轨迹归纳静态提示词/协议层缺陷，例如：工具选择偏差、必填参数遗漏、产物没有传给下一步、重复调用、错误后不会恢复、展示工具漏调用或过早输出 final。

### 5. PromptEvo 的进化流程

1. **失败采集：**运行基础静态 system prompt，在固定任务切分上保存完整 rollout、工具错误、参数校验、artifact 变化和结束原因。
2. **结构化归因：**将失败映射到工具选择、参数、状态衔接、恢复、效率和终止协议等维度，聚合跨任务重复模式。
3. **补丁生成：**把重复失败及其证据交给 LLM，要求生成任务无关的静态 protocol/prompt patch，而不是针对某个任务硬编码答案。
4. **Stage1 首版候选：**让 LLM 从基础提示词和失败轨迹中开放式发现问题、完成结构化归因，并生成首版候选静态提示词。
5. **Stage2 对比式优化：**在同一任务与工具环境下成对运行 base 和 Stage1 候选，基于工具选择、参数、产物衔接等行为差异继续生成或筛选候选，同时检查总体 success、Set/Multiset F1、Exact/Ordered、工具错误、调用数、token、超时和新失败类型。
6. **版本与结果留存：**保存 patch、prompt 版本、任务 manifest、shadow 结果和接受/拒绝原因，支持版本间复盘和多轮对比。

其中，**shadow 对比**指让候选协议在同一批任务、同一工具环境下旁路运行或成对重放，只记录候选的工具选择、错误、成功、调用数和 token 等结果，不改变当前正式版本的执行结果；它用于判断补丁是否真的改善了目标行为，以及是否引入新的副作用。

### 6. 两个方法的关系与区别

| 维度 | ExperienceEvo | PromptEvo |
|---|---|---|
| 优化对象 | 运行时可复用的局部操作经验 | 固定 system prompt / 全局行为协议 |
| 输入 | 成功/失败 rollout 中的 artifact 状态转移 | 静态提示词与失败 rollout 的结构化归因 |
| 输出 | 冻结经验库与 Top-K 候选策略 | 版本化静态提示词与 protocol patch |
| 是否更新模型参数 | 否 | 否 |
| 主要收益 | 产物衔接、工具规划、调用效率 | 工具选择、参数遵循和协议一致性 |
| 主要风险 | 错误经验传播、检索误匹配 | prompt 过拟合、局部修复导致全局回归 |
| 控制机制 | schema/artifact guard、质量排序、回退 ReAct | Stage1/Stage2 验证、shadow、版本管理 |

## RL 框架评估与当前实验状态

### 1. 当前是否使用 veRL

是。当前 Qwen 3B 的在线实验入口使用 `/data1/yuhongjie2/verl` 中的 veRL，训练器为 `verl.trainer.main_ppo`，算法配置为 `algorithm.adv_estimator=grpo`，接入了多轮工具 agent loop、真实工具执行、LoRA、FSDP、vLLM rollout 和自定义 reward function。也就是说，当前不是静态 action 打分，而是“模型生成 → 工具真实执行 → 观察返回 → 计算 reward → GRPO 更新”的在线 RL 路径。

### 2. 为什么当前场景优先选 veRL

veRL 适合本项目的原因是：

- 原生支持 GRPO、PPO 等 on-policy 训练和 rollout/ref/actor 分工；
- 支持 vLLM rollout、FSDP、LoRA/参数高效训练和多 GPU；
- 可以通过 agent loop 接入多轮工具调用，而不是只能处理一次性文本 completion；
- 自定义 reward function 可以组合任务完成、工具执行和轨迹质量信号；
- 训练 batch、old log-prob、优势估计、checkpoint 和断点恢复链路较完整。

对当前“真实工具执行、多轮轨迹、3B LoRA、四卡资源受限”的目标，veRL 是目前最匹配的主框架，不建议为了换框架而重写已经接通的工具 agent loop。

### 3. 其它框架是否更合适

| 框架 | 适合程度 | 适用情况 | 当前不优先原因 |
|---|---|---|---|
| **veRL** | **最高** | 多轮 agentic RL、GRPO/PPO、vLLM/FSDP、多 GPU | 当前主要问题是显存和长序列配置，不是框架选型错误 |
| TRL | 中等 | SFT、DPO/KTO、离线 GRPO、小规模单轮实验 | 多轮真实工具 rollout、分布式 actor/rollout 和长轨迹需要较多自定义代码 |
| OpenRLHF | 中等 | 大规模 RLHF、成熟 reward/reference/actor 集群 | 更偏通用 RLHF，接入本地异构工具环境和 artifact state 需要额外适配 |
| verl-agent / Agent Lightning 类方案 | 可作为后续研究 | 希望把 agent loop、trajectory 和 RL trainer 更彻底解耦 | 需要评估现有工具协议、reward 和 checkpoint 是否能迁移，当前不应中途切换 |

因此建议保持 veRL 作为训练底座；TRL 可用于离线 SFT/DPO/KTO 基线，不能替代当前真实工具在线 GRPO 主实验。

### 4. 当前实验真实状态

- **已完成：**Qwen 2.5 3B 离线 QLoRA/GRPO 训练，使用 OEA 2000 条数据（1900 train / 100 val），保存 reward trace、训练日志和 adapter checkpoint。但它不是完整真实工具在线 RL，也没有独立 OEA test 全量指标。
- **未完成：**Qwen 3B veRL 真实工具 online GRPO。最新安全重跑目录为 `tmp/experience_evo_rl/qwen25_3b_pure_online_grpo_compressed_tok10240_b1_valsafe_keepvllm_train2000_20260908`，在第一个训练 step 的 old-log-prob/entropy 阶段因 CUDA OOM 中止，未生成最终 checkpoint，不能报告完整 OEA test 结果。
- **已保存：**失败运行的真实 episode trace、raw tool observation、reward trace、配置、数据 manifest 和日志，可用于诊断，但不能当作训练完成结果。
- **已完成的可比较主结果：**ExperienceEvo 的 OEA 完整评测；这属于非参数更新的经验增强结果，不是 RL 训练结果。

### 5. 对当前路线的判断

当前应把问题定义为“veRL + 真实工具环境的显存/长序列工程稳定性”，而不是“是否选错 RL 框架”。在现有硬件上，优先稳定一条可复现的 3B online GRPO 配置，再做算法或 reward 创新。建议顺序是：

1. 保持 veRL、LoRA 和真实工具 loop 不变，先解决 old-log-prob/entropy 阶段的峰值显存；
2. 固定 `n=2` rollout、batch/micro-batch 及 token 上限，记录每 step 的最大序列长度和显存峰值；
3. 训练完成后，用固定 OEA test manifest 分别评测 Base、Pure GRPO adapter、ExperienceEvo+GRPO adapter 和最终 online checkpoint；
4. 只有在线闭环稳定后，再比较 reward 设计（任务完成、工具成功、序列匹配、效率惩罚、KL）或 PPO/GRPO 变体。

在此之前，不应把离线 reward 曲线、50 条 action 诊断或中止的在线轨迹写成 RL 已取得 OEA 提升。
