# PromptEvo 研究定位与跨 Agent 实验建议

更新日期：2026-07-29

## 1. 结论先行

当前 PromptEvo 已经不是一个简单的“让 LLM 根据失败轨迹重写 prompt”脚本。它已经包含：

- Stage1 单版本轨迹诊断；
- Stage2 同任务 base/Stage1 配对归因；
- 多候选生成；
- 可选的真实 dev rollout 验证；
- 跨 API-Bank、OEA、tau2-bench、AgentDojo、ToolBench、AIME 的 adapter 接口。

但是，仅将这些机制概括为“轨迹反思 + 两阶段静态提示词优化”，已经不足以形成强创新点。GEPA、Self-Harness、SePO、RHO 和 2026 年 7 月的 GSME 已经分别覆盖轨迹反思、多候选搜索、最小 harness patch、优化器自进化、历史轨迹自偏好和失败语义归档。

本项目最值得发展的方向是：

> **跨 Agent、消费者感知的协议补丁进化**：从异构 agent 轨迹中提取有类型、带触发条件的行为协议补丁；先验证补丁是否正确，再验证目标模型是否会激活并遵守补丁；最后将同一抽象补丁编译进不同场景、不同模型的静态 system prompt。

暂定英文名称可以使用：

> **ProtoPatch-Evo: Consumer-Aware Protocol Patch Evolution for Cross-Agent Self-Improvement**

该方向保留 PromptEvo 的核心约束：**部署时仍使用一份有界的静态行为提示词，不向上下文注入任务级经验，不修改模型权重**。

建议把论文的主要创新压缩为三点：

1. **协议级归因**：把轨迹失败定位到通用 agent 生命周期中的具体协议槽，而不是自由改写整个 prompt。
2. **消费者感知**：显式区分“补丁写得对”“目标 agent 会激活/遵守”“最终任务确实获益”三个阶段。
3. **跨场景迁移**：学习的是带适用条件的抽象协议补丁，同一补丁可以编译到不同 agent prompt；用 leave-one-scene-out 证明迁移，而不只是证明 adapter 代码可以复用。

Stage2 的“配对回归修复”应继续保留，但更适合作为可靠性机制，而不是论文唯一创新点。

## 2. 最新相关工作与本项目的边界

### 2.1 直接相关工作

| 工作 | 时间 | 主要机制 | 与当前 PromptEvo 的重叠 | 仍可区分的空间 |
|---|---|---|---|---|
| [PromptBreeder](https://arxiv.org/abs/2309.16797) | 2023-09 | 同时进化任务 prompt 和 mutation prompt | 多候选、元提示词 | 不面向 agent 轨迹中的协议级失败和跨 agent 激活 |
| [TextGrad](https://arxiv.org/abs/2406.07496) | 2024-06 | 用自然语言 textual gradient 优化复合系统中的文本组件 | 轨迹/反馈驱动改写 | 缺少本项目计划中的类型化协议 IR、消费者激活和跨场景迁移 |
| [GEPA](https://arxiv.org/abs/2507.19457) | 2025-07；ICLR 2026 Oral | 反思执行轨迹，生成、测试、组合 prompt，维护 Pareto archive | 轨迹反思、多候选、真实验证 | 需要把差异放在跨 agent 协议迁移和 updater-consumer 解耦，而不是“比 GEPA 多一个阶段” |
| [DelvePO](https://arxiv.org/abs/2510.18257) | 2025-10 | prompt 组件分解、working memory、方向引导进化 | 组件化改写、方向聚合 | 可进一步做 agent 生命周期协议槽、触发条件和消费者编译 |
| [ACE](https://arxiv.org/abs/2510.04618) | 2025-10；ICLR 2026 | Generator-Reflector-Curator 持续维护结构化 playbook/context | 从轨迹积累通用规则 | ACE 更接近持续上下文/记忆；本项目应坚持有界静态协议，不存任务经验 |
| [Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621) | 2026-05 | 区分 harness 更新能力和目标 agent 使用 harness 的能力 | 直接解释当前“prompt 看起来合理但指标不涨”的现象 | 本项目可进一步把该诊断变成 consumer-aware compiler 和 activation probe |
| [Continual Harness](https://arxiv.org/abs/2605.09998) | 2026-05 | 在连续运行中共同更新 prompt、sub-agent、skill、memory | 在线持续改进 | 范围远大于本项目；不建议现阶段扩到完整 harness |
| [SePO](https://arxiv.org/abs/2606.04465) | 2026-06 | 同时进化 task agent prompt 与 prompt optimizer 自身 prompt | 当前固定元提示词是其直接批评对象 | 可把 optimizer 自进化作为消融或后续增强，不应作为唯一主线 |
| [Adaptive Auto-Harness](https://arxiv.org/abs/2606.01770) | 2026-06 | harness tree、状态化 evolver、求解时路由，处理异质任务流 | 反对不断膨胀的单一全局 prompt | 本项目可做更受约束、可解释、最终仍编译为静态 prompt 的协议补丁体系 |
| [RHO](https://arxiv.org/abs/2606.05922) | 2026-06 | 历史轨迹 coreset、多次重放、自一致与 pairwise self-preference | 轨迹采样、候选验证 | 本项目可以使用真实指标和跨场景协议迁移，不必把无标签自偏好作为主贡献 |
| [Self-Harness](https://arxiv.org/abs/2606.09498) | 2026-06 | Weakness Mining -> Harness Proposal -> Proposal Validation；最小、模型特定补丁 | 与当前 Stage1/验证流程高度重叠 | 本项目应强调同一抽象补丁如何跨模型/跨场景编译与激活，而不是只做模型特定 patch |
| [GSME](https://arxiv.org/abs/2607.13683) | 2026-07 | LLM 诊断/提案与确定性统计门控分离；按 WHERE×WHY 建语义质量多样性档案 | 失败归因、多候选、确定性 gate、语义补丁库 | 最新且最接近；本项目必须进一步证明 protocol IR、消费者激活和零/少反馈跨场景迁移 |
| [Rethinking the Evaluation of Harness Evolution](https://arxiv.org/abs/2607.12227) | 2026-07 | 指出同一公开 benchmark 上搜索再报告会高估；要求预算匹配和真正 held-out 测试 | 对当前小 dev 选候选再看同分布全量构成直接警告 | 本项目应把 sealed split、场景级迁移和相同 rollout 预算写进方法本身 |

2026 年综述 [Self-Improvements in Modern Agentic Systems](https://arxiv.org/abs/2607.13104) 将 scaffold improvement 分为 prompt、memory、tool 和 full scaffolding，并把 prompt 优化归纳为标量反馈、定性改写、种群进化和 textual gradient 四类。PromptEvo 当前主要处于“定性轨迹反馈 + 小规模种群搜索 + 对比式 textual gradient”的交叉位置。

### 2.2 不能再单独声称的创新点

以下内容可以作为系统组成，但不能单独作为核心创新：

- 使用 LLM 阅读失败轨迹并改 system prompt；
- 生成多个候选并在 dev 集选择；
- 使用 Pareto 或候选 archive；
- 先挖弱点、再生成 patch、最后回归验证；
- 优化 prompt optimizer 的元提示词；
- 将失败按语义类别归档；
- 为多个 benchmark 写 adapter。

其中“多个 adapter 共用一套 core”证明的是**软件可移植性**，不是“学到的行为改进可以迁移”。论文需要证明的是后者。

## 3. 现有实验为什么经常不提升

### 3.1 已有结果呈现的规律

- API-Bank 是当前最清楚的正例：主实验 Macro Success 从约 0.6192 提高到 Stage1 的 0.6398，再到较强 Stage2 候选的 0.6470。
- tau2-bench 的总体成功率从 8.47% 到 9.20%，但不同 domain 有升有降，且对话轮数、工具调用和耗时增加。
- AgentDojo 的 Qwen3 8B 三阶段几乎持平，Stage2 还有轻微回退。
- OEA LongCat2 think 的若干工具指标有局部改善，但 success rate 从 Base 到 Stage1/Stage2 下降。
- AIME 小规模实验出现 dev 候选改善但正式任务没有泛化，说明小 dev 选择和随机 API 输出会形成明显选择偏差。

这些结果不说明“静态提示词优化完全无效”，而说明当前的一次性全局 prompt 改写缺少稳定的**可优化杠杆、归因粒度和泛化约束**。

### 3.2 五个主要原因

#### 原因一：官方 base prompt 已经较强

AgentDojo、tau2-bench 等官方 prompt 通常已包含工具使用、状态维护和安全规则。继续追加通用规则的边际收益很小，反而容易增加冲突、长度和执行负担。

这也是为什么系统性消融实验有意义：它用于测量“方法能否恢复缺失协议”，但必须同时报告原始完整 prompt 上的 ceiling 结果，不能只展示人为削弱后的提升。

#### 原因二：许多失败并非 prompt 可修复

AIME 的主要瓶颈通常是数学推理能力；OEA 的一部分失败来自服务、网络或模型能力；tau2 的失败还受 user simulator、长上下文和数据库状态影响。让优化器对所有失败都改 prompt，会把不可修复噪声写成多余规则。

当前需要显式预测 `prompt_fixability`，只对“改变静态行为协议可能改善”的失败生成 patch。

#### 原因三：整段重写缺少可靠 credit assignment

全局改写会同时改变工具选择、重试、停止、安全和回答格式。指标变化后很难知道是哪一条规则造成的，也无法只撤销有害部分。当前 Stage2 虽然能看到 base/Stage1 配对，但最终仍主要产出整份 prompt。

#### 原因四：补丁质量不等于目标模型能使用补丁

同一句抽象规则对 LongCat2 可能足够，对 Qwen3 8B 可能太隐晦；过长的 checklist 又可能让模型机械执行、增加轮次。当前没有单独测量：

1. 规则是否在正确条件下被注意到；
2. 模型是否按规则行动；
3. 按规则行动后是否真正改善任务。

这正是 Harness Updating 与 Harness Benefit 的区别。

#### 原因五：候选验证容易对小 dev 过拟合

候选越多，越容易偶然选到在 8-12 条 dev 任务上高分的 prompt。若任务随机性大、没有重复 seed、dev 与最终集高度同分布，最终结果很容易回归均值。AIME 实验已经出现该现象。

## 4. 建议的新方法：协议补丁而不是整份 prompt 搜索

### 4.1 通用协议中间表示

将不同 agent 的静态 prompt 统一映射到以下生命周期槽位：

1. `goal_and_completion`：任务目标、成功和完成条件；
2. `context_and_evidence`：可信上下文、证据和状态来源；
3. `planning`：分解、依赖关系和行动顺序；
4. `tool_selection`：何时选哪个工具；
5. `argument_grounding`：如何从观测中构造参数；
6. `observation_update`：如何解释工具返回并更新状态；
7. `recovery`：错误分类、重试、替代路径；
8. `verification`：结果校验和交叉检查；
9. `stopping_and_answer`：何时停止、如何给最终答案；
10. `trust_and_safety`：权限、注入、不可信内容和副作用；
11. `budget`：token、调用次数、延迟和费用约束。

每个候选不再首先表示为一整段自然语言，而表示为一个 `ProtocolPatch`：

```text
patch_id: recovery.no_identical_retry
slot: recovery
trigger: 上一次工具调用返回确定性参数错误
precondition: 工具、参数和输入观测均未变化
action: 解析错误字段，只修改有证据支持的参数；否则切换方案或停止
forbid: 原样重复同一调用
scope: 有结构化工具错误的多步 agent
evidence: 轨迹和 task_id 列表
prompt_fixability: high
risk: 可能把瞬时网络错误误判为确定性错误
expected_effect: 降低重复调用和 max-turn，保持任务成功率
```

该结构存的是**行为协议**，不是任务答案、具体地点、具体 API 名或历史工作流，因此与经验库有清晰边界。

### 4.2 新的两阶段定义

#### Stage1：Failure-Localized Patch Induction

输入 base prompt、base 轨迹和分维度指标，输出若干有类型补丁：

1. 区分基建错误、模型能力错误和 prompt 可修复错误；
2. 为可修复错误定位 `slot + trigger + cause`；
3. 只生成最小 patch，不重写无关槽；
4. 同类跨任务证据不足时不提案；
5. 用小规模真实 rollout 验证 patch，而不是只让 LLM 给文本打分。

#### Stage2：Counterfactual Scope Repair

输入 base/Stage1 的同任务配对轨迹，目标不是再写一次 Stage1，而是修正 patch 的适用范围：

1. 对改善样本提取“何时应该激活”；
2. 对回退样本提取“何时不应该激活”；
3. 收窄 trigger、增加 precondition 或拆分 patch；
4. 只撤销有害 patch，保留已验证收益；
5. 形成 patch-level ledger：每个 patch 对哪些指标、哪些场景、哪些模型有益或有害。

这比“Stage1 负责改，Stage2 负责再改”更容易形成可检验的因果叙述。

### 4.3 Consumer-Aware Compiler

同一抽象补丁不应直接复制到所有模型，而应编译成不同表达：

- 对较弱或较小模型：短、显式、可执行的条件-动作规则；
- 对较强模型：较简洁的原则，避免 checklist 过度约束；
- 对工具型 agent：保留工具参数和观测更新的操作语义；
- 对安全型 agent：突出信任边界、授权和不可信内容；
- 对已有强 prompt：在对应槽位做最小插入，不重排整份 prompt。

编译后先运行廉价的 `ActivationProbe`：构造不含 benchmark 答案的微型情境，检查模型是否在 trigger 出现时遵守规则、在 trigger 不出现时保持原行为。只有通过 activation probe 的候选才进入昂贵 rollout。

因此评估被拆成三项：

| 层次 | 问题 | 指标示例 |
|---|---|---|
| Patch validity | 补丁是否针对真实失败机制 | 诊断一致率、反事实合理性 |
| Activation/compliance | 目标模型是否正确激活和遵守 | trigger recall、false activation、rule compliance |
| Harness benefit | 遵守后是否改善正式任务 | held-out success、tool correctness、security、成本、回归率 |

### 4.4 协议补丁档案与路由

档案以 `slot + trigger + cause + scope` 为索引，不以 task_id 为索引。每个 patch 维护：

- 来源场景和证据；
- 适用/不适用条件；
- 对各模型的 activation 统计；
- 对成功率、工具正确性、安全、token、延迟的影响；
- 与其他 patch 的冲突和依赖；
- 置信区间和最后验证时间。

最终仍可将选中的 patch 编译为单一静态 prompt。档案和路由只存在于优化阶段，不需要在每个任务运行时检索历史经验。

## 5. 真正能证明跨 Agent 有效的场景

### 5.1 第一主线：工具协议迁移

优先选择 API-Bank、StableToolBench、OEA、tau2-bench。这四个场景虽然任务不同，但共享大量协议失败：

- 工具选择错误；
- 参数没有从观测中 grounding；
- 忽略工具错误信息；
- 相同参数重复调用；
- 没有更新状态；
- 过早结束或达到 max-turn；
- 最终回答没有引用已得到的证据。

推荐做 leave-one-scene-out：

1. 在三个场景的训练轨迹中发现协议补丁；
2. 不看第四个场景的标签和 rollout，将补丁编译进其 base prompt；
3. 在第四个场景 sealed test 上直接评估零反馈迁移；
4. 再给同等少量 target feedback，比较少样本适配速度。

这能证明“学到的改进可迁移”，而不仅是 adapter 可复用。

API-Bank 应作为第一源场景，因为已经观察到稳定正收益，任务便宜、错误结构清楚。StableToolBench 可作为第一个目标场景；OEA 成本和基建噪声较高，放在机制稳定之后。tau2 更适合测试长对话状态和恢复协议。

### 5.2 第二主线：消费者/模型迁移

固定同一组抽象 patch，比较：

- LongCat2 updater -> Qwen3 8B consumer；
- LongCat2 updater -> LongCat2 consumer；
- 可选 Qwen3 8B updater -> 两种 consumer。

需要比较：

1. 直接把相同自然语言 patch 塞给两个模型；
2. 使用 consumer-aware compiler；
3. compiler + activation probe。

这是对 Harness Updating Is Not Harness Benefit 的直接方法性回答，也是当前已有实验资源最容易支持的新贡献。

### 5.3 AgentDojo：作为安全-效用多目标验证

AgentDojo 不适合作为唯一的“追求成功率大幅上涨”场景，但非常适合证明 Stage2 的回归控制：

- utility 与 security 不能合并成一个模糊指标；
- 一个 safety patch 可能提高 security、降低 clean utility；
- Stage2 应学习更精确的信任边界和触发条件，而不是简单增加“忽略不可信内容”。

建议同时报告：

- 官方完整 base prompt；
- minimal prompt；
- 分槽消融：去掉 trust/safety、tool rule、verification；
- PromptEvo 恢复后的 prompt；
- held-out attack family 和正常任务结果。

系统性消融适合论文，但必须明确称为 **protocol recovery study**，并且不能隐藏官方 base 上提升有限的事实。

### 5.4 外部强验证：Terminal-Bench 或代码 agent

如果资源允许，后期增加一个代码/终端场景。代码 agent 的测试、错误恢复、验证和停止规则具有较高 prompt 可修复性，也便于与 Self-Harness、GSME 的结果对话。

但该场景成本高，且完整 harness 差异较大，不应在核心方法尚未通过 API-Bank/ToolBench 跨场景迁移前优先投入。

### 5.5 不建议作为主场景：AIME

AIME 可作为“能力瓶颈下 prompt evolution 的负结果”或与 GEPA 的接口测试，但不适合作为本方法主证明：

- 它不是典型多步工具 agent；
- 静态协议对基础数学能力的控制力有限；
- 任务量小，候选选择极易过拟合；
- 当前 45 条实验已经出现 dev 提升不泛化。

## 6. 建议的正式实验矩阵

### 6.1 Baseline

每个场景至少比较：

1. Official base prompt；
2. One-shot generic rewrite；
3. TextGrad 或同预算 textual-gradient baseline；
4. GEPA；
5. 当前 PromptEvo v2；
6. ProtoPatch-Evo；
7. ProtoPatch-Evo 去掉 Stage2；
8. ProtoPatch-Evo 去掉 consumer-aware compiler；
9. ProtoPatch-Evo 去掉 activation probe；
10. ProtoPatch-Evo 去掉跨场景 patch archive。

若完整 GEPA 成本过高，至少在 API-Bank 和一个外部场景上做同预算实现，不应只引用论文数字。

### 6.2 数据划分

禁止只做随机 item split。至少包含：

- task-level train/dev/sealed-test；
- tool/API family held-out；
- domain held-out；
- scene held-out；
- 可选 model held-out。

优化过程中不能读取 sealed-test 的轨迹、标签、judge 反馈或 aggregate 指标。测试集只在最终候选冻结后运行一次。

### 6.3 预算公平

统一统计并限制：

- optimizer LLM 调用次数和 token；
- solver rollout 数；
- 验证任务数；
- wall-clock 和 API 费用；
- 候选数量；
- 是否允许多 seed 或重试。

必须与简单 test-time scaling 比较。例如同样 100 次额外 solver 调用，可以用于搜索 prompt，也可以直接为测试任务采样多个答案。否则无法证明收益来自可复用协议，而不是额外计算。

### 6.4 核心指标

除各 benchmark 原始指标外，新增：

- `heldout_gain`：sealed test 相对 base 的提升；
- `regression_rate`：base 正确、候选错误的任务比例；
- `gain_retention`：测试提升 / dev 提升；
- `activation_recall`：该触发时是否激活；
- `false_activation_rate`：不该触发时是否误激活；
- `compliance_rate`：激活后是否按 patch 行动；
- `transfer_gain`：无目标反馈时在新 scene/model 上的提升；
- `adaptation_efficiency`：达到相同提升需要的目标场景 rollout 数；
- token、工具调用、延迟和费用变化；
- utility/security 等多目标 Pareto 变化。

统计上使用同任务 paired bootstrap 或随机化检验，并对 API 模型的随机性至少跑多个 seed。接受 patch 时应使用置信下界，而不是单次均值大于 base 即接受。

## 7. 与当前 PromptEvo v2 的关系

不需要丢弃现有实现。建议保留三个层级：

| 版本 | 定位 | 用途 |
|---|---|---|
| v1 | 单边轨迹、一次改写 | 最弱 baseline |
| v2 | base/Stage1 配对归因、多候选和 dev 验证 | 强工程 baseline、回归修复 |
| v3 ProtoPatch-Evo | 类型化协议 patch、消费者编译、激活验证、跨场景迁移 | 论文主方法 |

v3 可以复用现有的 `PromptStore`、`TrajectorySource`、`MetricProvider`、`RolloutRunner`、配对 sampler 和实验 adapters。主要新增的是中间表示、patch 账本、compiler、activation probe 和跨场景评估协议。

建议新增的核心数据结构如下，具体命名可以在实现时调整：

```python
@dataclass
class ProtocolPatch:
    patch_id: str
    slot: str
    trigger: str
    preconditions: list[str]
    action: str
    forbidden_actions: list[str]
    scope: list[str]
    evidence_ids: list[str]
    prompt_fixability: float
    risks: list[str]

@dataclass
class ConsumerProfile:
    model: str
    agent_scene: str
    prompt_style: str
    capability_notes: list[str]

@dataclass
class PatchEvaluation:
    patch_id: str
    consumer: ConsumerProfile
    activation_recall: float
    false_activation_rate: float
    compliance_rate: float
    metric_delta: dict[str, float]
    confidence_interval: tuple[float, float]
```

## 8. 建议的论文叙事

### 8.1 主问题

> 现有 agent prompt/harness evolution 能从轨迹产生越来越好的更新，但通常将“更新质量”与“目标模型能否正确使用更新”混为一谈，并且主要在同一 benchmark 内搜索。我们研究能否把失败归纳为带适用条件的行为协议补丁，使其可以被不同 agent/model 编译、激活并在 sealed held-out 场景中产生可迁移收益。

### 8.2 可主张的贡献

1. 提出一种 agent 生命周期协议 IR，将异构轨迹中的失败归因到有类型的行为槽位，并产生带 trigger、scope 和 risk 的最小 patch。
2. 提出 consumer-aware compiler 和 activation probe，首次在优化闭环中显式分离 patch validity、activation/compliance 与 task benefit。
3. 提出两阶段 patch induction / counterfactual scope repair，利用 base-vs-patched 配对轨迹收窄适用条件并控制回归。
4. 在多个工具 agent 上采用 scene-level sealed split 和预算匹配评估，证明抽象协议 patch 可以零样本或少样本迁移，而不仅是 prompt optimizer 代码可以复用。

“首次”表述需要在正式投稿前再次做完整相关工作检索；当前检索支持这是一个明显比“轨迹反思改 prompt”更空缺的组合方向，但不应现在就绝对化。

### 8.3 不应使用的叙事

- 不要把“比 GEPA 多 Stage2”作为核心差异；
- 不要只在 minimal/ablated prompt 上报告结果；
- 不要把同一测试集上的反复搜索称为泛化；
- 不要把 adapter 数量称为跨领域学习；
- 不要为了让 Stage2 好看而删除非基建失败；
- 不要宣称每个阶段必须单调提升。可靠方法应该允许拒绝候选并保持 no-op。

## 9. 最小可行研究路线

### M1：先验证新假设，不急着重写全部代码

在 API-Bank 现有正例上手工/半自动抽取 10-20 个 typed patch，选 3-5 个最通用的工具协议 patch，分别编译到 StableToolBench 和 tau2-bench prompt。目标是确认“跨场景协议迁移”确实有信号。

通过标准：在完全不看目标训练轨迹的条件下，至少一个目标场景的工具正确性或成功率显著提升，且回归率受控。

### M2：实现 Stage1 类型化归因和 patch-level 验证

把现有自由改写改为“先输出结构化 patch，再编译 prompt”。保留 current v2 作为对照，不替换旧版本。

### M3：实现 consumer-aware compiler 和 activation probe

先支持 Qwen3 8B 与 LongCat2 两个 consumer。证明同一 patch 的模型特定表达优于直接复制相同文本。

### M4：实现 Stage2 scope repair 和 patch archive

用 paired regression cases 自动收窄 trigger/scope；确定性程序负责 gate 和统计，LLM 只负责诊断、提案和文本编译。

### M5：正式跨场景实验

建议顺序：

1. API-Bank -> StableToolBench；
2. API-Bank + StableToolBench -> tau2 leave-one-domain-out；
3. 三个工具场景 -> OEA；
4. AgentDojo 做安全/效用多目标和协议恢复；
5. 资源允许后增加 Terminal-Bench/代码 agent 外部验证。

## 10. 最终判断

当前 PromptEvo 的问题不是“机制完全错误”，而是论文问题定义仍停留在已经拥挤的 prompt search 范式中。已有实验已经提供了两个重要信号：

- API-Bank 证明协议级静态提示词确实存在可优化空间；
- tau2、AgentDojo、OEA 和 AIME 的不稳定结果证明全局自由改写、同场景小 dev 选择和只看最终指标不足以稳定泛化。

因此不建议通过进一步破坏 base prompt 或增加更多候选来“找一个会涨的 benchmark”。更有价值的路线是把这些负结果转化为方法动机：**为什么一个看似正确的 prompt 更新没有被目标 agent 正确激活，为什么同一规则在一个场景有益、在另一个场景有害，以及怎样学习带适用条件、可跨场景编译的协议补丁。**

若该假设在 API-Bank -> StableToolBench/tau2 的零反馈迁移实验中成立，它会比“再实现一个 GEPA 风格搜索器”更有创新性，也更符合本项目已经投入的大量跨场景 adapter 和真实 agent 基建。
