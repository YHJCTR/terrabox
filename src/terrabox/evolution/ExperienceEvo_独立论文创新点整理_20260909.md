# ExperienceEvo 独立论文创新点整理

更新时间：2026-09-09

## 结论先行

ExperienceEvo 可以单独整理成一篇“小论文 / workshop paper / technical report”级别的创新点，但需要把贡献边界讲清楚：它不是单纯的 RAG，也不是保存完整历史轨迹的 case memory，而是面向多工具 Agent 的**产物状态转移经验复用方法**。核心价值在于把历史 rollout 中可迁移的工具使用经验抽象成“当前有哪些产物、下一步要生成什么产物、应该如何绑定工具参数、成功后如何继续”的结构化策略，并用 Q/N/R 质量统计、`Quse` 排序、逐步状态检索和证据审核机制控制经验注入。

如果写论文，建议主张放在：**non-parametric agent experience reuse for artifact-centric tool planning**。中文可写成：**面向产物状态的非参数化工具经验复用**。

## 适合单独成文的原因

### 1. 问题是清晰的

复杂地理空间任务中的 Agent 失败，很多不是最终语义理解错误，而是工具链执行中的中间状态管理问题：

- 工具集合大且长尾，模型知道工具存在，但不知道当前状态下哪个工具最合适；
- 多步任务依赖文件、图层、mask、bbox、GeoPackage、统计值等中间产物；
- 工具返回经常包含路径、图层名、数值或错误信息，模型容易虚构路径、漏传参数或重复调用；
- 单条 system prompt 很难覆盖所有“前置条件—参数绑定—后验检查—失败恢复”的组合。

ExperienceEvo 解决的不是“记住答案”，而是“复用工具链中可迁移的状态转换经验”。这个问题定义比普通 prompt engineering 更像算法问题，也更容易写成论文贡献。

### 2. 方法有可解释的结构

ExperienceEvo 的经验单元不是自然语言示例，而是结构化的产物状态转移：

```text
输入产物状态 → 目标产物状态 → 工具/参数策略 → 输出产物状态
```

对应到实现中，可以拆成：

- **状态抽象：**从真实 rollout 中抽取任务目标、已有 artifact、工具调用、参数绑定、工具返回和新增 artifact；
- **Q/N/R 经验统计：**将相似状态转移合并为经验 family，同时维护 product-level 的 `Qsig/Nsig/Rsig` 和 tool-level 的 `Qtool/Ntool/Rtool`；
- **Quse 风险感知排序：**用 `Quse = λ Qtool + (1-λ) Qsig` 融合模式可靠性与具体工具质量，并结合 risk 降低不稳定经验优先级；
- **逐步状态检索：**不只在任务开头注入经验，而是在每次工具 observation 后更新产物状态，再通过 `step_hint()` 检索下一步候选；
- **证据审核：**用 soft verifier / final-answer verifier 检查当前证据是否足以继续或作答，减少虚构路径、漏产物和过早结束；
- **安全过滤：**根据工具 schema、必填参数、当前 artifact 是否存在、工具是否可用来过滤候选；
- **推理注入：**只注入候选策略和约束，不注入 task id、gold answer、gold tool sequence 或固定文件名；
- **回退机制：**如果没有合适经验，回到普通 ReAct；如果经验与当前状态不匹配，不强制执行。

这种结构可以与 Memento、Reflection、ExpeL、MemRL 等方法区分开：它不是保存完整案例文本，也不是事后反思文本，而是把工具执行轨迹压缩成 artifact-centric transition policy。

### 3. 已有完整同口径实验支撑

目前最适合作为主结果的是 LongCat/OEA 完整真实工具 rollout 对比。可以谨慎表述为：

- 在 Base、Reflection、MemRL、ExpeL、ACE 等对比方法中，ExperienceEvo 取得当前最高工具链指标；
- 相比 LongCat Base，Success 从 **86.90%** 提升到 **92.51%**；
- Set/Multiset F1 从 **.703/.637** 提升到 **.774/.722**；
- Exact/Ordered 从 **19.6%/15.7%** 提升到 **42.3%/34.2%**；
- 工具错误率从 **15.4%** 降到 **9.6%**；
- 平均工具调用从 **6.48** 降到 **4.80**。

注意：普通文本 answer accuracy 基本持平，不能写成答案准确率全面提升。更稳妥的主张是：ExperienceEvo 显著提升工具链规划、产物衔接和调用效率，并提升 generated-artifact-inclusive 完成质量。

## 推荐论文定位

### 可选题目

1. **ExperienceEvo: Artifact-Centric Experience Reuse for Multi-Tool Geospatial Agents**
2. **Non-Parametric Experience Reuse for Tool-Augmented Agents via Artifact State Transitions**
3. **Learning to Reuse Tool Experience without Parameter Updates for Geospatial ReAct Agents**

中文内部标题可写：

> ExperienceEvo：面向地理空间多工具 Agent 的产物状态经验复用方法

### 核心贡献表述

建议写成四点方法贡献，另把严格评测闭环放到实验可信度部分，而不是作为方法贡献：

1. **提出产物状态转移经验表示。** 将工具 rollout 抽象为输入产物、目标产物、工具策略、参数绑定、输出契约和失败恢复，而不是保存完整对话或任务答案。
2. **提出 Q/N/R 与 Quse 的经验可靠性建模。** 分别在 product-level 和 tool-level 统计质量、支持度和风险，用 `Quse` 融合“状态转移模式是否可靠”和“具体工具是否稳定可用”，避免只按语义相似度检索经验。
3. **设计逐步状态感知的经验注入机制。** 基于当前任务目标、已有 artifact、已执行工具和最近 observation 检索候选经验，并在每次工具返回后通过 `step_hint()` 重新检索下一跳策略。
4. **引入证据审核式经验控制。** 通过 soft verifier / final-answer verifier 检查证据槽位、产物存在性、数值/单位和最终答案支撑关系，降低过早回答、虚构路径和错误经验传播风险。

严格无标签真实工具评测闭环仍然很重要，但它更适合作为**实验设计与可信度保障**：证明方法没有使用 test 侧 gold answer、expected tools、task id 或历史完整轨迹，并能在同口径真实工具环境中比较。

## 方法细节

### 1. 经验数据结构

一条经验 family 可以抽象为：

```json
{
  "family_id": "hash_of_transition_family",
  "intent_signature": "distance_or_segmentation_or_osm_query",
  "input_product_state": ["task_request", "image", "aoi_boundary"],
  "target_product_state": ["validated_mask", "distance_result", "rendered_map"],
  "product_experience": {
    "goal": "生成可被后续工具消费的中间产物",
    "preconditions": [
      "当前状态中必须存在输入影像或 GeoPackage",
      "目标对象或 AOI 必须能从任务描述中解析"
    ],
    "output_checks": [
      "工具返回不能包含 error",
      "路径、图层名或数值必须来自真实 observation",
      "输出产物需要能被后续工具引用"
    ],
    "downstream_rule": "后续工具只能使用真实返回的 artifact 引用，不虚构路径或图层名",
    "recovery": [
      "缺少前置产物时先调用边界/感知/转换工具补齐",
      "参数绑定失败时使用当前 artifact 重新绑定，而不是复用历史字面量"
    ],
    "Qsig": 0.61,
    "Nsig": 171,
    "Rsig": 0.0
  },
  "tool_policies": [
    {
      "tool": "geo_perception.instructsam",
      "required_input_roles": ["image", "text_prompt"],
      "parameter_binding_rules": [
        "image 绑定当前任务中的真实影像 artifact",
        "text_prompt 从当前任务目标生成，不复制历史样例文本"
      ],
      "output_contract": ["mask_or_segmentation_result"],
      "post_checks": ["无 error", "返回 artifact 可被后续统计/展示工具消费"],
      "Qtool": 0.64,
      "Ntool": 58,
      "Rtool": 0.03,
      "Quse": 0.63
    }
  ]
}
```

这里的 `Q/N/R` 可以理解为质量、支持度和风险：`Qsig/Nsig/Rsig` 描述某类产物状态转移本身是否稳定，`Qtool/Ntool/Rtool` 描述具体工具策略是否可靠。代码和文档中不一定把它整体命名为 `QNR`，但实现上确实是这组三元统计；`Quse` 是最终展示给模型的工具候选排序分。

### 2. 经验生成流程

```text
真实 rollout
  → 解析工具调用、参数、observation、artifact 变化
  → 抽象输入/目标/输出产物状态
  → 过滤失败调用、schema 不合法调用、无效 artifact 和泄漏字段
  → 合并相似状态转移为经验 family
  → 计算质量分、支持度、风险分
  → 冻结经验库
```

关键点：经验库保存的是“可迁移的操作策略”，不是具体任务答案。对论文来说，这一点必须反复强调，否则容易被审稿人误解为 retrieval over demonstrations。

### 3. 经验注入流程

```text
当前任务 + 当前 artifact state + 已执行工具 + 最近 observation
  → 构造检索 query
  → 检索 Top-K 经验 family
  → schema/artifact guard 过滤不可执行候选
  → Q/N/R + Quse 对候选工具排序并标记风险
  → 注入结构化策略块
  → Agent 自主选择工具或回退 ReAct
  → 工具返回后更新 artifact state
  → step_hint() 基于新状态检索下一步经验
```

注入内容应控制为短策略块，例如：

```text
候选经验：先生成可验证的分割结果，再用于后续统计。
前置条件：必须存在当前任务影像；文本提示从当前问题生成。
参数绑定：image 使用当前 artifact 中的影像路径；text_prompt 使用当前目标对象。
后验检查：仅当工具返回无 error 且包含真实 artifact 路径时继续。
恢复策略：若返回空结果或路径无效，修正参数后重试或改用可替代感知工具。
```

### 4. Q/N/R、Quse 与逐步状态检索

ExperienceEvo 不是只做“相似任务检索”。经验被拆成两层可靠性：

- `Qsig/Nsig/Rsig`：产物状态转移 family 的质量、支持度和风险；
- `Qtool/Ntool/Rtool`：family 内具体工具策略的质量、支持度和风险。

最终工具排序可以使用如下解释式：

```text
Q_use = λ Q_tool + (1 - λ) Q_sig
λ = N_tool / (N_tool + k)
```

- `Q_tool`：工具执行质量，包括成功率、参数合法性、产物是否落盘、后续是否被消费；
- `Q_sig`：经验模式稳定性，包括跨任务支持度、语义一致性、相似经验聚合质量；
- `N_tool`：有效支持次数；
- `k`：平滑项，避免单次偶然成功经验被过度放大。

这个设计的意义是：当某个具体工具的证据很多时，排序更相信 tool-level 的实际执行质量；当具体工具样本少时，排序更多回退到 product-level 的状态转移可靠性。同时，risk 分数会提示或降权易误用经验，避免高相似但不稳定的候选被优先注入。

逐步状态检索是另一点关键创新：经验不是一次性塞进 prompt，而是在工具 observation 返回后更新当前产物状态，再重新检索下一步候选。这样可以把“工具执行 → 产物状态更新 → 下一步经验检索”变成闭环，避免初始检索无法覆盖后续状态变化。

### 5. 证据审核与安全控制

Verifier / 审核 agent 应作为 ExperienceEvo 的方法组成，而不是另一个商品审核项目。这里的审核含两层：

1. **Soft verifier checkpoint：**根据任务文本和当前产物状态标记缺失证据槽位，例如是否缺少统计值、展示产物、区域属性、计数结果或分割证据；它只给 checklist/risk note/recommended focus，不直接替 Agent 解题。
2. **Final-answer verifier 子 agent：**在 Agent 准备输出最终答案时，检查数值、单位、阈值、最近/最远、计数、实体选择和产物存在性是否被当前运行证据支持；它只读取当前 artifact state、工具调用和 observation，不读取 gold 或 expected tools。

安全控制建议写成四类 guard：

1. **Schema guard：**工具存在、参数类型合法、必填参数可绑定；
2. **Artifact guard：**候选经验要求的输入产物必须在当前状态真实存在；
3. **Leakage guard：**不注入 task id、expected tools、gold answer、gold trajectory、F1 等标签；
4. **Fallback guard：**经验不匹配时回退普通 ReAct，不强制执行检索结果。

## 近期开题相关工作（重点 2025-12 之后）

本节用于支撑后续“自进化训练 / self-evolving agentic RL / 自评分 reward / 状态转移 reward”的论文定位。检索口径主要覆盖：`self-evolving LLM agents`、`agentic RL`、`GRPO`、`LLM-as-a-judge reward`、`process reward`、`tool-use agent`、`skill library`、`self-certainty reward`。优先记录 2025-12 之后论文；少数 2025 年早期工作作为必要背景保留。

### 1. 最相关论文清单

| 论文 / 方法 | 时间 | 核心思想 | 实验数据 / 场景 | 与我们的关系 |
|---|---:|---|---|---|
| **Reinforcement Learning for Self-Improving Agent with Skill Library / SAGE** | 2025-12 | 将 skill library 纳入 GRPO；Sequential Rollout 中前序任务生成 skill，后续任务复用；reward 同时奖励任务完成和 skill 生成/使用质量。 | AppWorld | 与 ExperienceEvo 最接近：都是“历史交互 → 可复用策略/技能 → 后续 rollout 注入”。区别是我们把经验显式建模为地理工具的产物状态转移、参数绑定和输出契约。 |
| **Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization** | 2025-12 | 针对 agentic RL 稀疏奖励和 GRPO 组内同分导致零 advantage 的问题，提出 progressive reward shaping 和 value-based sampling。 | 需读正文确认具体 benchmark | 支撑我们做 dense / step-level reward，而不是只用整条 trajectory 的终局分数。 |
| **SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards** | 2026-02 | 将 entropy、least-confidence、margin 等 token-level uncertainty 合成 dense 信号，并做 failure-aware reward reshaping。 | ALFWorld、WebShop | 对应用户提出的“奖励模型肯定性/置信度 token”方向；可作为 confidence auxiliary reward 的直接相关工作。 |
| **AgentFlow / Flow-GRPO** | 2026 ICLR | 在真实工具系统中收集 states/actions/tool events 的 in-the-flow rollout，用 Flow-GRPO 优化 planner，使多轮工具任务转化为更稳定的训练信号。 | 多领域工具集成 agent 系统 | 支撑“真实工具在线 rollout + GRPO”路线；与我们当前 Swift/veRL 工具环境接入方向一致。 |
| **SOAR: Supervision from Observation for Agentic Reinforcement Learning** | 2026 ACL | 不只奖励 action/outcome，而是把环境 observation 纳入监督；对 observation token 分配与前序 action 负熵相关的正 advantage，使模型学习工具反馈和环境动态。 | 三个领域、14 个 benchmark | 与我们的状态转移 reward 高度相关：我们也应该奖励“工具观察导致的产物状态推进”，而不是只奖励最终答案。 |
| **Q-Evolve: Self-evolving LLM Agents with In-distribution Optimization** | 2026 ICML | 用 expert + agent-generated trajectories 构造 hybrid off-policy 数据，训练 in-distribution critic，再用 advantage 生成 step-wise process reward 并优化 policy。 | ALFWorld、WebShop、ScienceWorld | 支撑“过程奖励 + 策略共同进化”；但工程复杂度高，适合作为后续方法对标，不建议当前第一阶段直接复刻。 |
| **SkillMaster: Toward Autonomous Skill Mastery in LLM Agents** | 2026-05 | 提出 DualAdv-GRPO，分别估计 task-solving actions 和 skill-editing decisions 的 advantage，联合训练解题和技能管理。 | 需读正文确认具体 benchmark | 与 ExperienceEvo 的“经验库/技能库是否可训练化”相关，可作为后续把经验选择、经验编辑纳入 RL 的参考。 |
| **OpenSkill: Open-World Self-Evolution for LLM Agents** | 2026-06 | 研究 open-world self-evolution：agent 在没有目标任务监督的情况下构建技能和验证信号。 | open-world agent 场景 | 支撑“自建验证信号/自进化技能库”方向，但离我们当前 OEA 固定任务和真实工具评测更远。 |
| **Progress Advantage for LLM Agents** | 2026-06 | 从 RL post-training 的 policy/reference log-prob ratio 中恢复隐式 step-level advantage，作为无额外 reward model 的过程信号。 | 五个 benchmark、四类模型族 | 对我们有两点启发：可以把训练后的 log-prob 差异用于 failure attribution；也可探索不训练额外 PRM 的 step-level 轨迹评分。 |
| **LLM-as-a-Verifier: A General-Purpose Verification Framework** | 2026-07 | 不要求 judge 生成离散分数，而是用评分 token 的 logits / probability distribution 得到连续 verifier score；可作为 dense reward。 | LIBERO robotics、数学 GRPO 等 | 直接对应“用初始/冻结模型通过肯定性 token 或 score token 打分”的设想；建议用 frozen verifier，而不是当前 policy 自评。 |
| **More Convincing, Not More Correct** | 2026-07 | 指出 reference-free self-judge / self-play reward 容易让模型变得更会说服 judge，而不一定更正确；提出 hidden-anchor / de-anchored judge 等审计。 | GSM8K、Qwen3 policies 等 | 是我们做 self-judge reward 必须引用的风险论文。结论：自评 reward 可做消融，但主方法应使用真实工具状态或冻结/去锚定 judge。 |
| **APPO: Agentic Procedural Policy Optimization** | 2026-06 | 用 token uncertainty 和后续 continuation likelihood gain 选择关键 branching location，改善长程 agent 的过程级探索。 | 需读正文确认具体 benchmark | 可作为 ARPO/Flow-GRPO 一类 agentic RL 算法参考；说明多轮工具任务的关键难点是 step/turn-level credit assignment。 |

### 2. 必要背景工作（早于 2025-12，但与问题强相关）

| 论文 / 方法 | 时间 | 核心思想 | 与我们的关系 |
|---|---:|---|---|
| **Self-Rewarding Language Models** | 2024 | 同一个 LLM 既生成回答，也通过 LLM-as-a-Judge prompt 产生 reward，用于后续训练。 | 是“自己给自己打分”路线的起点；但对工具 Agent 应注意 reward hacking 和 judge drift。 |
| **Process-based Self-Rewarding Language Models** | 2025 | 将 self-rewarding 扩展到 step-wise LLM-as-a-Judge 与 step-wise preference optimization。 | 支撑过程级自评分，而不是只给最终答案打分。 |
| **Learning to Reason without External Rewards / Intuitor** | 2025 / ICLR 2026 | 用模型 self-certainty 作为唯一 reward，把 GRPO 的外部 reward 替换为内部置信度信号。 | 对应“奖励肯定性 token / confidence”的方向；可做辅助 reward 或消融，不建议在 OEA 中作为唯一主 reward。 |
| **Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning** | 2025-11 | 用 curriculum agent 和 executor agent 共进化，结合工具生成更复杂任务和解题能力。 | 虽早于 2025-12，但与“工具集成 + 自生成任务/课程 + GRPO”相关，可作为边界相关工作。 |
| **ARPO: Agentic Reinforced Policy Optimization** | 2025 / ICLR 2026 | 在工具调用后的高熵回合自适应分支采样，增强多轮 agent 的 step-level 探索。 | 支撑我们关注“工具 observation 后的下一步选择”而不是只看最终结果。 |

### 3. 对我们后续方法的直接启发

1. **状态转移 reward 是最稳的主线。** SOAR、Q-Evolve、Progress Advantage、ARPO/APPO 都在强调 agentic RL 的关键不是单个最终 reward，而是如何给中间 observation、state、tool event 分配 credit。我们的 OEA 场景天然有文件、图层、mask、bbox、统计值等产物状态，因此可以把 ExperienceEvo 的产物状态转移直接改造成 GRPO 的 dense reward。
2. **冻结 judge 比当前模型自评更稳。** Self-Rewarding、Process-based Self-Rewarding 和 LLM-as-a-Verifier 证明 LLM judge 可以给训练提供信号；但 More Convincing 说明 reference-free self-play judge 容易 reward hacking。因此如果做“自己打分”，建议至少固定一个 frozen judge / initial model judge，并保存 judge prompt、score token probability、rationale 和 disagreement case。
3. **置信度/肯定性 token 可以作为辅助项，不宜单独作为主 reward。** Intuitor 和 SELAUR 说明 confidence / uncertainty signal 能带来训练信号，但 OEA 是外部工具环境，模型自信不等于工具链正确。更合理做法是 `R_total = R_state_transition + α R_confidence`，其中 `α` 小一些，并监控是否出现自信错误和虚构路径。
4. **经验库可以从非参数增强扩展到可训练 skill policy。** SAGE、SkillMaster、OpenSkill 都在做 skill library 与 RL 的结合。ExperienceEvo 可以保持“经验库冻结 + 检索注入”的主结果，同时扩展一个 RL 版本：训练模型学会什么时候采纳、拒绝或修正某条产物状态经验。

### 4. 建议实验矩阵

| 实验 | Reward 设计 | 目的 | 优先级 |
|---|---|---|---:|
| Pure Online GRPO | 当前工具执行 reward：成功工具、错误工具、artifact、turn/token 惩罚 | 建立严格 no-label、真实工具 GRPO baseline | P0 |
| State-Transition GRPO | 当前 reward + 产物状态推进、参数绑定真实性、后续产物复用、失败恢复 | 验证 ExperienceEvo 的状态转移思想是否能变成训练信号 | P0 |
| Frozen-Judge Hybrid GRPO | `R_state_transition + α * R_frozen_judge` | 验证初始模型/外部模型 trajectory judge 是否补充语义判断 | P1 |
| Verifier-Token GRPO | `R_state_transition + α * E[score_token]` 或 `P(success_token)` | 验证 score-token/logit reward 是否比自然语言 judge 更稳定 | P1 |
| Self-Judge GRPO 消融 | 当前 policy 自己给 rollout 打分 | 只用于风险和 ablation，不作为主方法 | P2 |
| Confidence Auxiliary GRPO | `R_state_transition + α * self-certainty/uncertainty` | 验证肯定性/置信度信号是否改善探索 | P2 |

### 5. 对论文定位的修正

如果后续加入 RL 训练，论文主张可以从“非参数经验复用”扩展为两阶段：

```text
ExperienceEvo-Retrieve：用产物状态转移经验做非参数检索增强；
ExperienceEvo-RL：把产物状态转移进一步转化为 step-level reward，训练模型内化工具状态推进能力。
```

这样与近期工作的差异会更清楚：SAGE/SkillMaster 强调 skill library 与 skill editing，Q-Evolve/Progress Advantage 强调过程奖励估计，SOAR 强调 observation supervision，而我们的核心是**地理空间多工具任务中的 artifact-state transition：当前有什么产物、下一步需要什么产物、工具参数如何绑定、输出如何被后续消费**。

### 6. 参考链接备查

- SAGE / Skill Library GRPO: https://arxiv.org/abs/2512.17102
- Progressive Reward Shaping for Agentic RL: https://arxiv.org/abs/2512.07478
- SELAUR: https://arxiv.org/abs/2602.21158
- AgentFlow: https://agentflow.stanford.edu/
- SOAR: https://aclanthology.org/2026.acl-long.1624/
- Q-Evolve: https://arxiv.org/abs/2606.07367
- SkillMaster: https://arxiv.org/abs/2605.08693
- OpenSkill: https://arxiv.org/abs/2606.06741
- Progress Advantage: https://arxiv.org/abs/2606.26080
- LLM-as-a-Verifier: https://llm-as-a-verifier.com/
- More Convincing, Not More Correct: https://arxiv.org/abs/2607.05904
- APPO: https://arxiv.org/abs/2606.12384
- Intuitor / RLIF: https://proceedings.iclr.cc/paper_files/paper/2026/hash/052e2404709e8df9dc38fbc7bfec5b80-Abstract-Conference.html

## 与相关方法的区别

| 方法类型 | 常见做法 | ExperienceEvo 的区别 |
|---|---|---|
| 普通 RAG | 检索相似任务文本或知识片段 | 检索的是产物状态转移与工具策略，不是任务答案 |
| CaseBank / Memento | 保存历史案例或成功轨迹摘要 | 更强调 artifact 前置条件、参数绑定和输出契约 |
| Reflection / ExpeL | 从失败或成功中生成自然语言反思 | 将反思落到可执行的工具状态转换结构 |
| MemRL | 学习或更新 memory/Q-value | 当前主方法是非参数化经验复用，可冻结、审计、回滚 |
| Semantic retrieval | 按语义相似度找历史经验 | ExperienceEvo 先做产物状态和 schema hard filter，再用 Quse/risk 排序 |
| PromptEvo | 修改全局 system prompt/protocol | ExperienceEvo 是运行时局部状态经验注入，粒度更细 |

## 推荐实验设计

### 主实验

使用同一模型、同一工具目录、同一真实工具执行环境、同一 OEA test manifest，对比：

1. Base / ReAct；
2. Reflection；
3. MemRL；
4. ExpeL；
5. Memento / CaseBank；
6. ACE Playbook；
7. ExperienceEvo。

主表指标建议保留：Success、Set F1、Multiset F1、Exact/Ordered、OEA Any/Same/Unique、Category F1、Tool error、Tools/task、Tokens/task、Answer/Gen.Acc。

### 消融实验

建议至少做以下消融：

1. **No state transition：**只检索自然语言经验，不使用产物状态转移结构；
2. **No artifact guard：**不检查当前 artifact 是否满足前置条件；
3. **No schema guard：**不做工具 schema 和必填参数过滤；
4. **Random retrieval：**随机经验替代相似检索；
5. **No step hint / no verifier：**复用已有 late-August 诊断，说明 step-level state hint 和验证机制贡献；
6. **Top-K sensitivity：**比较 Top-1、Top-3、Top-5、Top-10 的效果和 token 成本；
7. **Risk score ablation：**去掉风险降权，看是否引入更多工具错误或重复调用。

### 诊断图

适合画的图包括：

- 工具调用数分布：Base vs ExperienceEvo；
- 工具错误类型堆叠图：参数错误、路径错误、timeout、unknown tool、empty output；
- 不同工具类别的 F1 提升；
- Top-K 经验命中率与成功率关系；
- 经验支持度 `support_count` 与被采纳率/成功率关系；
- 状态转移图示：输入 artifact → 中间 artifact → 最终 artifact。

## 可以写成论文的主线

### 摘要草稿

多工具 Agent 在地理空间任务中需要连续调用遥感感知、GIS 分析、地图渲染和数值计算工具。现有 ReAct 或案例检索方法通常把历史轨迹作为文本示例复用，难以显式约束中间产物、参数绑定和后续工具依赖，容易产生虚构路径、错误参数和重复调用。本文提出 ExperienceEvo，一种面向产物状态的非参数化经验复用方法。ExperienceEvo 从真实工具 rollout 中抽取输入产物、目标产物、工具策略、输出契约和失败恢复，构建可审计的经验库；推理时根据当前 artifact state 检索候选经验，并通过 schema/artifact guard 过滤不可执行策略。实验表明，ExperienceEvo 在完整真实工具 OEA 评测中提升工具链成功率、工具序列 F1 和调用效率，并降低工具错误率。该结果说明，面向中间产物状态的经验复用可以作为参数更新之外提升多工具 Agent 可靠性的有效路径。

### Introduction 逻辑

1. 多工具 Agent 的关键困难不是单步调用，而是长链路状态管理；
2. 地理空间任务尤其依赖文件、图层、mask、bbox、统计值等 artifact；
3. 现有 prompt、reflection、case memory 对 artifact state 的建模不足；
4. ExperienceEvo 把经验表示为产物状态转移，并用状态感知检索注入；
5. 实验显示该结构带来更高工具链指标和更低调用成本。

### Method 章节结构

1. Task and artifact-state formulation；
2. Experience extraction from rollout traces；
3. Transition family construction with Q/N/R statistics；
4. Quse ranking and risk-aware candidate selection；
5. Step-level state-aware retrieval and schema/artifact guarded injection；
6. Evidence verifier and leakage/fallback policy。

### 结果章节结构

1. Main comparison against Base and prior experience/reflection baselines；
2. Ablation on retrieval quality, step hint, verifier, schema/artifact guard；
3. Efficiency analysis: tool calls, token usage, tool errors；
4. Case studies showing corrected artifact chaining；
5. Failure analysis: when retrieved experience is insufficient or harmful。

## 当前还需要补强的证据

如果目标是正式投稿，而不是内部技术报告，建议补齐以下证据：

1. **严格 ablation 表。** 尤其是 No artifact guard、No schema guard、Random retrieval、Top-K sensitivity。
2. **检索命中审计。** 统计多少任务命中经验、命中经验是否被采纳、采纳后是否成功。
3. **Q/N/R 与 Quse 消融。** 区分只用语义相似度、只用 product-level Q、只用 tool-level Q、完整 Quse/risk 的效果差异。
4. **审核机制消融。** 分别关闭 soft verifier、final-answer verifier，观察过早回答、虚构路径和缺失产物的变化。
5. **泄漏审计。** 明确证明经验库不含 task id、expected tools、gold answer、gold trajectory。
6. **失败案例分析。** 展示 3–5 个 Base 失败而 ExperienceEvo 成功的工具链状态变化。
7. **跨模型验证。** LongCat 主结果已经强，但如果 Qwen 3B/8B 或另一个外部模型也有提升，论文说服力更强。

## 风险与边界

- 不应声称 ExperienceEvo 学到了通用地理知识；它主要学习/复用工具链操作经验。
- 不应声称 answer accuracy 全面提升；主结果更强的是工具链完成度、artifact 生成和调用效率。
- 不应把离线 action 诊断或在线 GRPO 训练 reward 当作 OEA 主评测指标。
- 不应把 MemRL 中使用 label-augmented memory 的旧结果作为严格 rollout-only 主对比，需要明确数据边界。

## 一句话定位

ExperienceEvo 的论文定位可以是：

> 我们提出一种面向中间产物状态的经验复用机制，将历史工具调用轨迹抽象为可检索、可过滤、可审核的产物状态转移策略，并通过 Q/N/R 可靠性统计、Quse 风险感知排序和逐步状态检索，在不更新模型参数、不泄漏测试标签的前提下提升多工具 Agent 的工具链规划可靠性与调用效率。

这个定位是可以独立成文的；关键是不要把它包装成“又一个 RAG”，而要突出产物状态转移表示、Q/N/R+Quse 排序、逐步状态检索、证据审核，以及严格真实工具 rollout 作为实验支撑。
