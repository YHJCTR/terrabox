# ExperienceEvo 与 PromptEvo 中期答辩材料

> 用途：10 分钟中期答辩的内容底稿、指标依据与逐页 PPT 剧本。
>
> 更新时间：2026-09-21
>
> 口径标记：**已完成**表示已有完整、可追溯结果；**评测中**表示方法和训练侧构建已完成，但 OEA test 尚未产生最终结论；**研究计划**表示尚未实现或验证。三者不可混写。

## 1. 一页总览：研究问题、路线与当前状态

### 1.1 研究问题

地理空间 Agent 往往要经过多步工具调用并持续生成、消费中间产物。困难不只在于“选择某个工具”，而在于：工具可以执行成功，但输入文件、图层、时间阶段、对象、单位或 CRS 绑定仍可能错误；这种错误还会沿后续产物依赖链传播。

本工作围绕两类互补的自进化对象展开：

```text
历史真实 rollout
  ├─ 动态、状态相关经验  -> ExperienceEvo
  │    ├─ v4-clean：从产物状态转移中检索可复用经验             [已完成]
  │    └─ Boundary：学习经验何时适用、何时应拒绝或拆分        [评测中]
  └─ 静态、任务无关协议  -> PromptEvo
       从失败轨迹生成 typed protocol patch，再由 gate 审核    [已完成]

下一步：把产物来源检查转成工具调用级 credit assignment，训练模型 [研究计划]
```

### 1.2 答辩中必须守住的结论边界

| 模块 | 当前能作出的结论 | 不能作出的结论 |
|---|---|---|
| ExperienceEvo v4-clean | 在 matched OEA 全量真实工具评测中，工具链可靠性、序列匹配和效率显著改善 | 普通文本最终答案准确率全面提升 |
| ExperienceEvo Boundary | 已完成训练侧反例生成、双 Agent 审核、规则冻结，并在公平 OEA test 评测中 | 已提升 OEA Success/F1，或已证明降低时间/CRS 负迁移 |
| PromptEvo | typed patch 与 gate 能在部分严格划分上改善工具调用与错误恢复 | 每次 Stage2 都会单调优于 Stage1/Base，或对安全性无副作用 |
| 标准 GRPO | 已完成真实工具 online GRPO 基线和运行时经验增强对照 | Pure GRPO 已稳定优于 Base |
| 产物来源感知 GRPO | 有明确、可实现的后续算法设计 | 已实现或已有提升结果 |

## 2. ExperienceEvo v4-clean：已完成的主方法

## 2A. 答辩需要交代的背景与相关工作

### 2A.1 背景：为什么地理空间 Agent 需要“经验自进化”

地理空间 Agent 与普通问答 Agent 的差异在于：一次任务通常不是“输入文本—输出文本”，而是一个由工具调用和中间产物组成的执行图。Agent 需要持续处理影像、矢量图层、表格、mask、GeoPackage 等异构产物，并维护它们的来源、对象、时间、单位与坐标系关系。

答辩中建议用下面三层错误定义问题：

```text
工具是否执行成功？             API/程序层
输入产物是否绑定正确？          数据依赖层
输出是否满足任务语义？          任务目标层
```

现有工具成功信号通常只覆盖第一层；而第二、三层错误会沿后续产物依赖链传播。我们的研究问题因此不是简单地“让模型记住更多案例”，而是：如何让 Agent 在真实工具环境中积累可复用经验，并判断一条经验在当前产物状态下是否适用。

### 2A.1.1 从数据分布和真实 rollout 暴露出的两个结构性问题

下面两点不是抽象地假设出来的缺陷，而是可以在 OEA 全量任务和真实工具轨迹中直接观察到的现象。它们也决定了本项目为什么不能只把“完整历史对话”继续塞回 prompt。

#### 问题一：完整轨迹记忆太粗，无法告诉模型“下一步需要补什么证据”

很多经验/记忆方法保存的是一条完整案例、整段反思或较长的轨迹片段。这样的记录能够复述“以前做过什么”，但不一定能明确表示每一个局部决策的输入、参数、产物和输出契约。例如 OEA 中有如下任务：

> **任务：**“How many buildings are in between domestic and construction waste?” 输入是一张 `TG_40010.jpg` 遥感影像。正确的工具链需要先定位 domestic waste 和 construction waste，再根据两者空间关系统计中间的建筑，最后给出数量。

在 LongCat Base 的真实轨迹中，工具调用按实际顺序完整记录为：

```text
1  geo_perception.ocr_extract
2  geo_perception.vlm_analyze
3  geo_perception.vlm_analyze
4  geo_perception.vlm_analyze
5  geo_perception.vlm_analyze
6  geo_perception.vlm_analyze
7  geo_perception.vlm_analyze
8  geo_perception.vlm_analyze
9  geo_perception.vlm_analyze
10 geo_perception.vlm_analyze
11 geo_perception.vlm_analyze
12 geo_perception.vlm_analyze
13 geo_perception.vlm_analyze
14 geo_perception.count_given_object
15 geo_perception.vlm_analyze
```

这条轨迹中 OCR 没有得到文本，连续多次 VLM 分析都没有产生可用于定位两类垃圾区域的明确产物，最后才调用一次建筑计数，并再次回到 VLM。最终状态是 `failed`，答案为 `ERROR: max sequential tool turns reached before final answer`；工具序列相对任务要求的 F1 只有 `.333`，multiset F1 只有 `.105`。这不是单纯的“模型不知道答案”，而是它没有在每一步形成可检查的中间证据，也没有在重复分析无新增信息时停止。

同一任务在 ExperienceEvo v4-clean 中的实际轨迹则是：

```text
1  geo_perception.count_given_object(object="building")
```

工具返回了当前运行产生的建筑计数结果，Agent 随即结束，避免了 Base 中 15 次调用的重复分析；但这条轨迹仍然只覆盖了期望链路的一部分，序列 F1 为 `.5`、multiset F1 为 `.4`，不能把它描述成已经完整解决了任务。这个例子同时说明两件事：

1. **逐步提示确实能抑制无证据的重复调用和过早进入 VLM 反思循环；**
2. **仅有“下一步提示”仍不够，还需要记录每一步缺失的输入产物、输出契约和任务完成条件。**

如果把 Base 的整段对话原样存为案例，检索器最多得到“这类任务过去调用过 OCR/VLM/计数工具”；它很难直接回答：当前是否已经拿到两类垃圾的空间证据？下一步应该调用哪个工具？一个工具返回“成功”后，是否真的产生了后续步骤可消费的 artifact？因此我们需要把轨迹拆成可执行的**步骤级产物状态转移**，而不是只保留长文本。

#### 问题二：工具调用是长尾分布，高频经验会掩盖低频工具风险

在 OEA test 的 1162 个任务中，按任务提供的期望工具序列做**事后诊断**，23 个工具的调用次数差异很大：

| 工具 | 事后期望调用次数 | 工具 | 事后期望调用次数 |
|---|---:|---|---:|
| `osm_gis.add_pois_layer` | 1067 | `osm_gis.get_area_boundary` | 668 |
| `compute.calculator` | 569 | `osm_gis.compute_route_dist` | 496 |
| `geo_perception.instructsam` | 366 | `geo_perception.region_attribute_description` | 361 |
| `osm_gis.add_index_layer` | 246 | `compute.solver` | 191 |
| `geo_perception.count_given_object` | 158 | `osm_gis.compute_index_change` | 123 |
| `geo_perception.change_os_detect` | 98 | `geo_perception.draw_bboxes` | 71 |
| `geo_perception.sam2_segment` | 34 | `bing_search.search` | 32 |
| `geo_perception.add_text` | 29 | `osm_gis.show_index_layer` | 29 |
| `osm_gis.get_bbox_from_raster` | 25 | `compute.plot` | 22 |
| `geo_perception.strip_rcnn_detect` | 21 | `osm_gis.display_on_geotiff` | 16 |
| `geo_perception.ocr_extract` | 10 | `geo_perception.vlm_analyze` | 19 |

这个分布带来三个具体风险：

- 经验库容易被 `add_pois_layer`、`get_area_boundary`、`calculator` 等高频工具占据，检索结果看起来“很可靠”，但并不代表低频工具的参数绑定和异常恢复也可靠；
- 低频工具只有少量历史证据，单一成功或失败就可能造成过高或过低的全局质量估计；
- 高频工具的大量正常调用会稀释少数条件化错误，例如特定时间图层、对象类别、单位或 CRS 下的错误绑定。一个全局 Q 值很难回答“这个工具在当前产物条件下是否适用”。

这里的工具频次只用于评测后的分布诊断，未被写入 ExperienceEvo 的训练提示、经验索引或质量值；自进化阶段仍只使用任务文本、公开工具、真实调用和真实观察。

#### 一个完整的长尾/异常恢复案例：Kluane National Park 任务

另一个 OEA 任务要求：

> **任务：**“Assign each park to its nearest museum within a 3000m radius in Kluane National Park, Yukon, Canada.” 任务需要构造区域边界，分别获取 parks 和 museums 图层，计算候选配对距离，并筛选 3000m 内的最近博物馆。

在相同任务上，不同 adapted 方法的真实轨迹表现出相同的结构性困难：OSM 查询容易无结果或网络失败，Agent 随后反复改变关键词、重新创建边界或扩大缓冲区。

| 方法（均为本地 OEA adapted/reimplemented 口径） | 完整实际轨迹摘要 | 结果与可观察问题 |
|---|---|---|
| LongCat Base matched | 共 11 次工具调用：`get_area_boundary`、多次 `add_pois_layer`、重复 `get_area_boundary`，最后 `compute_route_dist` 和 `display_on_map`；耗时约 482.8s | 最终 `completed=True`，但 `has_tool_error=True`；日志中出现多次 “No matching features” 和一次 `buffer_m=50000 exceeds ... 10000`。它能够给出表格，却没有把失败的查询与后续候选集合充分区分开。 |
| ExperienceEvo v4-clean | 共 7 次工具调用：`get_area_boundary`、`add_pois_layer`、`compute_route_dist` 等；耗时约 379.4s | `completed=True`、`success=True`，调用次数更少，但仍出现 3 次 “No matching features”。这说明状态约束和已有恢复经验能减少重复，但不能把外部 OSM 的不可用结果伪装成确定事实。 |
| ExpeL adapted | 共 15 次调用，反复 `add_pois_layer`，中间穿插多次 `bing_search.search`；耗时约 906.2s | 最终 `failed`，错误为 `max sequential tool turns reached before final answer`；多次查询失败后仍继续尝试相近请求，没有形成稳定的失败/回退边界。 |
| Memento CaseBank adapted | 共 15 次调用，重复边界/POI 查询和搜索；耗时约 468.3s | 最终 `failed`；轨迹中同时出现 “No matching features”、网络不可达和过大 buffer 错误，案例检索没有阻止错误策略重复使用。 |
| ACE Playbook adapted | 共 12 次调用，多次创建边界、添加 POI，最后计算距离并绘图；耗时约 874.2s | `completed=True` 但 `has_tool_error=True`；能够结束，但调用冗余和错误恢复成本仍然较高。 |

这组结果不能被解释为对原论文官方实现的严格排行榜，因为它们是在本地 OEA 工具和 harness 上的适配版本；但它清楚暴露了一个适合本项目研究的问题：**当工具调用长尾、外部服务有噪声、同一工具存在多个参数条件时，完整案例/全局质量统计很容易鼓励“再试一次”，却缺少“当前证据已经不足，应换策略或安全回退”的局部边界。**

### 2A.1.2 从问题自然引出本项目的设计要求

上述两个案例把方法需求具体化为四点：

1. 经验必须细化到工具动作级，记录输入产物、关键参数、输出产物、失败模式和恢复条件；
2. 检索必须先检查当前产物状态和工具输入契约，再使用文本相似度或历史质量排序；
3. 质量统计要同时保留支持次数、风险和当前使用收益，不能让高频工具的总体成功率掩盖低频/条件化失败；
4. 当证据不足或条件未知时，系统需要 verifier 和通用 ReAct fallback，而不是强制复用一条表面相似的历史轨迹。

这四点分别引出 ExperienceEvo 的产物状态转移、前置状态过滤、Q/N/R 与 Quse、逐步提示和 evidence verifier；当经验本身的适用条件仍不确定时，再由 Boundary 通过反例测试进行补充、分裂、隔离或延迟决策。PromptEvo 则处理另一类问题：把跨任务反复出现的协议缺口编译为任务无关的 typed patch，而不是把某一次 OSM 或某一张影像的具体路径写进系统提示词。

#### 案例复核路径与解释边界

上面的两个完整案例可由以下结果文件复核：

- “domestic waste—construction waste—building”任务：`tmp/trajectories/longcat_base_matched_v4clean_dockerfixed_20260818/standard/results/oea_test_1000.json` 与 `tmp/trajectories/experience_evo_v4_clean_oea_train2000_longcat_eval_dockerfixed_20260816/standard/results/oea_test_1000.json`；
- Kluane National Park 任务：各方法对应结果目录中的 `oea_test_1001.json`，包括 Base matched、ExperienceEvo v4-clean、ExpeL live、Memento strict CaseBank 和 ACE Playbook adapted。

这些案例用于解释失败模式和方法动机，不替代 1162 条任务的总体统计。尤其是 Kluane 任务受到 OSM/网络服务波动影响，不能单条样本证明某个方法的因果优势；它的价值在于把“重复查询、错误恢复、过大 buffer、max-turn”这些在汇总工具错误率中容易被压扁的机制问题具体展示出来。

### 2A.2 与本项目最相似的方向

| 方向/论文 | 主要机制 | 尚未解决的问题 | 本地 OEA 适配中观察到的现象 | 我们的对应切入点 |
|---|---|---|---|---|
| **Reflexion**（Shinn et al., 2023） | 用语言反馈和反思记忆指导下一次尝试，不更新模型参数 | 记忆通常以轨迹/反思文本存在，缺少地理产物来源和工具契约约束 | Reflection adapted 的工具错误率为 13.0%、平均 6.17 次调用；对长工具链的局部参数绑定没有显式记录 | 将经验绑定到 artifact state transition，并在运行时检查可用产物 |
| **ExpeL**（Zhao et al., 2024） | 从成功/失败轨迹抽象跨任务自然语言经验 | 抽象经验的适用条件和负迁移边界不够显式 | ExpeL adapted 工具错误率为 52.2%，平均 6.36 次调用；Kluane 任务中出现 15 次调用后达到最大轮数 | 用输入产物、输出契约、恢复条件和 Q/N/R 表示可执行经验 |
| **ACE**（Zhang et al., ICLR 2026） | 通过生成、反思、整理持续更新 playbook/context | 静态上下文演化难以表达“当前状态下哪条经验可用” | ACE adapted 工具错误率为 15.6%、平均 6.27 次调用；错误工具结果并未自动变成条件化回退 | PromptEvo 采用 typed protocol patch；ExperienceEvo 采用动态状态检索 |
| **SkillRL**（Xia et al., 2026） | 将轨迹蒸馏为分层技能，并与 RL 策略递归共进化 | 技能复用仍需处理条件失配；不专门建模地理产物 provenance | 当前未将本地 SkillRL 结果作为严格主表；其“技能可复用”仍需回答何时不应复用 | Boundary 学习经验适用边界；后续再将边界反馈接入 RL |
| **Memento / CaseBank 类方法** | 通过检索历史案例帮助规划下一步 | 案例粒度较粗；相似案例可能携带错误参数或过期产物上下文 | Memento adapted 工具错误率为 37.2%、平均 6.95 次调用；Kluane 任务中重复查询并最终 max-turn 失败 | 先做状态/前置条件过滤，再做检索与风险排序 |

论文链接：

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)
- [Agentic Context Engineering](https://arxiv.org/abs/2510.04618)
- [SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234)

### 2A.3 一页讲清“现有方法不足—我们的改进”

| 真实现象 | 仅保存轨迹/案例或使用全局统计的直接后果 | 本项目的改进要求 |
|---|---|---|
| Base 在“domestic waste—construction waste—building”任务中连续 13 次 VLM 分析，最后 max-turn 失败 | 记忆告诉模型“以前调用过什么”，却没有指出当前缺失的产物、下一步工具和停止条件 | 将每一步表示为 `当前产物状态 → 工具动作/参数 → 输出产物/契约`，在每轮给出可消费的 step hint |
| `add_pois_layer` 事后期望调用 1067 次，而 `ocr_extract` 只有 10 次；高频工具占据经验库 | 全局高 Q 容易代表“常见工具总体成功”，无法代表低频工具或特定条件下的可靠性 | 使用支持次数 N、风险 R、使用收益 Quse 和工具签名共同排序，并保留低证据/未知状态 |
| Kluane 任务中 OSM 查询无结果、网络不可达或 buffer 超限，ExpeL/Memento 继续重复查询 | 相似案例会把失败策略再次带回当前上下文，错误调用和耗时不断累积 | 先做输入/产物前置条件过滤，记录 failure mode 与 recovery；验证失败时触发 fallback，而不是强制复用 |
| 工具返回 `success`，但文件、时间、对象、单位或 CRS 可能绑定错误 | API 层成功被误当成任务层成功，后续错误产物继续传播 | 用 provenance、输出契约和下游消费关系做 evidence verifier；未知字段不能直接当作正确 |

将这些现象串起来，研究缺口不是“历史经验太少”，而是**经验的粒度、适用条件和证据闭环没有被显式建模**：

```text
完整轨迹/技能记忆
        ↓  粒度过粗：不知道每一步缺什么
相似度或全局质量检索
        ↓  频率偏置：高频工具掩盖低频/条件化风险
工具 success 信号
        ↓  语义盲区：成功执行不等于来源绑定正确
错误经验被重复注入或持续重试

ExperienceEvo：产物状态转移 + 前置契约过滤 + Q/N/R/Quse + step hint + verifier/fallback
Boundary：通过可执行反例学习经验何时适用、何时补条件/分裂/隔离
PromptEvo：把跨任务共性协议缺口编译为受保护的 typed patch，并用 gate 接受、拒绝或回滚
后续 RL：将来源约束信用对齐到具体工具调用 token，而不是把整条轨迹共享一个局部错误信号
```

注意：答辩中不要说“首次提出经验记忆、状态转移或步骤级奖励”。更准确的贡献是把**产物来源、语义绑定、工具输出契约和经验适用边界**引入地理空间 Agent 的自进化闭环。

### 2A.4 适合答辩引用的较新工作（替代“只讲 Reflection”）

答辩不需要铺开所有论文，建议主讲下面四篇，Reflexion/ExpeL 只在备用页作为历史起点带过：

| 论文 | 年份 | 主要解决什么 | 仍存在的空缺 | 我们如何区别 |
|---|---:|---|---|---|
| **GiGPO: Group-in-Group Policy Optimization for LLM Agent Training** | 2025 | 在 episode-level GRPO 之外，按重复/相近状态组织 step-level 相对优势 | 状态锚点和粒度策略主要由训练框架预设，不显式检查产物来源和语义绑定 | 我们把“状态”具体化为 artifact provenance state，并检查输入来源、输出契约和下游消费 |
| **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL** | 2025 | 用 RL 学习 memory 的 ADD/UPDATE/DELETE/NOOP 管理动作 | 重点是记忆生命周期与问答利用，不是长工具链的动作级产物信用 | ExperienceEvo 关注可执行状态转移；Boundary 关注经验的条件化适用边界 |
| **MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory** | 2026 | 先按语义相关性筛选，再用 Q-value 学习经验的运行时效用 | 主要学习“检索哪条经验”，对经验内部条件缺失、规则冲突和技能分裂建模不足 | 我们将 Q/N/R 从全局质量扩展到状态约束、风险和边界治理，并保留 fallback |
| **GACA: Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent RL** | 2026 | 根据决策关键性动态混合 episode-level 与 step-level advantage | 解决的是信用粒度，未显式处理持久经验库、技能组合和产物 provenance | 方案 A 将局部信用绑定到具体工具调用 token，并把 provenance 检查作为信用来源 |

推荐引用链接：

- [GiGPO（2025）](https://arxiv.org/abs/2505.10978)
- [Memory-R1（2025）](https://arxiv.org/abs/2508.19828)
- [MemRL（2026）](https://arxiv.org/abs/2601.03192)
- [GACA（2026）](https://arxiv.org/abs/2609.12424)

**答辩中的一句话综述：**近期研究已经分别推进了“经验效用学习”“主动记忆管理”和“长轨迹信用分配”，但这些机制通常彼此分离；本项目面向地理空间工具链，把可执行产物状态、经验适用边界和协议补丁化自进化放入同一条真实工具 rollout 闭环，后续再将 provenance 信号接入 RL 的动作级信用分配。

**证据边界：**上述是相关工作定位，不代表我们的 Boundary 或方案 A 已经超过这些方法；本项目当前已验证的是 v4-clean 和 PromptEvo 的对应实验，Boundary 仍以冻结全量评测为准，方案 A 仍是后续研究计划。

### 2.1 核心想法

不把历史轨迹作为一段长文本案例直接塞入 prompt，而是将一次工具操作抽象为**产物状态转移经验**：在什么产物状态下，调用何工具、消费什么输入、应产生什么输出，以及失败/恢复条件是什么。运行时只检索满足当前已有产物状态的经验，并以软提示形式辅助 Agent 决策。

```text
train2000 的真实 LongCat rollout（严格不读 gold）
  -> 提取 input artifact / tool action / output artifact
  -> 聚类为 transition family，统计 Q/N/R
  -> 当前任务观察到的 artifact state
  -> 前置产物状态过滤 + BM25/结构化状态 RRF 检索
  -> Quse/risk 重排序
  -> initial guidance + step hint + 轻量 evidence verifier
  -> 当前 Agent 继续真实工具 rollout
```

严格性：建库与运行时不读取 `task_type`、`expected_tools`、gold answer、gold tool calls、task id 或 F1 reward；这些信息只用于评测和事后诊断。

### 2.2 方法模块

| 模块 | 作用 | 为什么不是普通 RAG |
|---|---|---|
| 产物状态转移 family | 表示输入产物、工具操作、输出产物与恢复策略 | 检索对象是可执行状态转移，不是问题-答案文本 |
| 前置状态过滤 | 只有当前已具备所需产物时才允许候选经验进入 | 先约束可用性，再做语义相似度排序 |
| 状态约束检索 | BM25 与结构化 artifact state 经 RRF 融合 | 避免仅按自然语言关键词复制案例 |
| Q/N/R 与 Quse | 分别追踪质量、支持次数、风险；再按工具 signature 融合排序 | 区分“相似”与“历史上可靠” |
| step hint | 在每轮给出当前可行的下一步转移提示 | 直接作用于长链路的局部决策 |
| evidence verifier | 基于当前 rollout 的真实工具输出检查证据是否闭合 | 不以数据集 gold 来强制纠正 Agent |

### 2.3 可直接讲的贡献表述

> 针对地理空间多工具任务中中间产物依赖导致的长链规划不稳定问题，提出基于产物状态转移的 ExperienceEvo：从严格无标签真实 rollout 抽取可复用转移经验，结合状态约束检索、Q/N/R 可靠性建模、逐步提示与运行时证据校验，降低错误调用和冗余决策。

## 3. v4-clean 正式主实验：OEA 全量结果（已完成）

### 3.1 设置与口径

- 模型：LongCat；matched Docker/工具配置。
- 数据：OEA test，共 1162 个任务；Base 覆盖 1161/1162，v4-clean 覆盖 1162/1162。
- 比较对象：LongCat Base matched 与 `ExperienceEvo v4-clean`。
- `answer_acc` 是普通文本回答口径；`answer_acc_w_gen` 对生成/展示任务采用扩展回答口径。工具序列、Success、错误率和效率为独立维度，不能互相替代。

### 3.2 完整主指标

| 指标 | LongCat Base matched | ExperienceEvo v4-clean | 变化 |
|---|---:|---:|---:|
| Answer accuracy | 49.67% | 49.32% | -0.35pp |
| Answer accuracy with generation | 66.20% | 76.06% | +9.86pp |
| Task success | 86.90% | 92.51% | +5.61pp |
| Tool error rate | 15.4% | 9.6% | -5.8pp |
| Set F1 | .703 | .774 | +.071 |
| Multiset F1 | .637 | .722 | +.085 |
| Exact tool sequence | 19.6% | 42.3% | +22.7pp |
| Ordered exact sequence | 15.7% | 34.2% | +18.5pp |
| Any / Same / Unique | 59.0 / 58.4 / 63.7% | 59.6 / 59.0 / 67.9% | +0.6 / +0.6 / +4.2pp |
| Perception F1 | 34.47 | 44.67 | +10.20 |
| OSM F1 | 35.68 | 58.50 | +22.82 |
| Local compute F1 | 30.00 | 40.29 | +10.29 |
| GIS F1 | 82.46 | 90.11 | +7.65 |
| Tools per task | 6.48 | 4.80 | -1.68 |
| LLM calls per task | 7.39 | 5.89 | -1.50 |
| Tokens per task | 55,810 | 53,835 | -1,975 |
| Total tokens | 64,821,232 | 62,555,906 | -2,265,326 |
| Time per task | 336.4s | 268.1s | -68.3s |
| Errors per task | .50 | .25 | -.25 |

### 3.3 行为与覆盖情况

| 指标 | Base | v4-clean |
|---|---:|---:|
| Completed / failed | 1057 / 104 | 1131 / 31 |
| Empty final | 1 | 0 |
| Cap rate | 9.0% | 2.7% |
| Repeated calls >= 4 | 192 | 90 |
| Set precision / recall | .645 / .818 | .750 / .833 |
| Multiset precision / recall | .582 / .785 | .714 / .785 |

**结论应这样讲：**v4-clean 的主要收益是工具链执行成功、调用序列匹配、错误控制与调用效率；普通文本 `answer_acc` 与 Base 基本持平且略降，因此不应泛化为“所有最终答案均提升”。

## 4. v4-clean 对照与机制证据

### 4.1 外部方法 adapted 对照（OEA）

> 这些是对方方法在本地 OEA/harness 上的 adapted 或 reimplemented 结果，不能等同于原论文官方 leaderboard；MemRL 旧版本含 label-augmented 历史设置，尤其不能作为严格公平比较。

| 方法 | Success | Set / Multi F1 | Tool error | Tools/task | Answer acc. with generation |
|---|---:|---:|---:|---:|---:|
| Reflection | 89.4% | .706 / .639 | 13.0% | 6.17 | 69.72% |
| MemRL（旧 label-augmented adapted） | 89.8% | .699 / .662 | 9.7% | 6.18 | 65.49% |
| ExpeL adapted | 88.6% | .723 / .651 | 52.2% | 6.36 | 71.13% |
| ACE Playbook adapted | 89.0% | .723 / .657 | 15.6% | 6.27 | 75.35% |
| Memento CaseBank adapted | 85.0% | .691 / .626 | 37.2% | 6.95 | 65.49% |
| **ExperienceEvo v4-clean** | **92.51%** | **.774 / .722** | **9.6%** | **4.80** | **76.06%** |

### 4.2 固定 200 条子集消融

> 这是固定 200 条子集，不能与第 3 节 1162 条全量主表直接做严格差值计算。

| 变体 | Answer acc. | Answer acc. w/ gen | Success | Tool error | Set / Multi F1 | Exact / Ordered | Tools/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic guard only | 45.30% | 71.15% | 84.5% | 16.0% | .657 / .589 | 23.5 / 16.0% | 6.40 |
| no store, soft guidance only | 47.39% | 75.00% | 91.5% | 13.5% | .716 / .644 | 35.0 / 26.5% | 5.47 |
| checker on | 50.36% | 65.38% | 89.5% | 7.5% | .748 / .688 | 40.0 / 33.0% | 4.84 |

可讲的机制观察：仅加通用 guard 不足；experience store 带来更强的序列指导；checker 有助于降低错误，但对生成/展示类最终回答存在收益权衡。

### 4.3 全量 rollout diagnostics（机制趋势，不替代主表）

| 变体 | Set / Multi F1 | Exact / Ordered | Success | Tool error | Tools/task |
|---|---:|---:|---:|---:|---:|
| v4-clean | .774 / .722 | 42.3 / 34.2% | 92.5% | 9.6% | 4.80 |
| -Quse/QNR | .770 / .717 | 42.0 / 34.6% | 92.5% | 9.2% | 4.83 |
| -step hint | .739 / .675 | 31.6 / 24.8% | 87.9% | 10.2% | 5.85 |
| -verifier | .753 / .701 | 35.0 / 28.5% | 92.1% | 8.9% | 5.24 |
| random retrieval | .745 / .694 | 38.5 / 31.2% | 89.9% | 9.0% | 5.06 |

稳妥解读：step hint、verifier 和检索质量具有明显作用；QNR 的单独绝对差异较小，需要更细粒度配对统计才能声称强因果贡献。

## 5. ExperienceEvo Boundary：从“可靠经验”到“经验适用边界”（评测中）

### 5.1 为什么需要 Boundary

v4-clean 的 Q/N/R 回答“经验过去总体是否可靠”，但无法完整回答“经验在当前对象、时间、单位、图层、分辨率或 CRS 条件下是否仍适用”。经验越抽象，跨任务复用越容易；同时也越可能遗漏造成负迁移的条件。

例如“影像 -> 分割 mask -> 面积统计”在一般情形是可靠经验，但若 mask 与当前影像不同时间、不同对象，或空间分辨率/CRS 不可用于面积换算，工具仍可能成功运行，却给出语义错误结果。

### 5.2 完整 Boundary 流程

```text
v4-clean parent store（只读）+ train2000 rollout
  -> 训练题按 hash 划分 discovery / validation，隔离同题证据
  -> 确定性条件学习 + 成对真实工具重放
  -> Boundary Proposer 提出语义/前置条件反例 probe
  -> Boundary Auditor 仅根据真实 control/transform 观察决策
  -> add_condition / split_family / quarantine / keep / defer_unknown
  -> 冻结 boundary_rules.jsonl（test 前不再更新）
  -> 1162 条 OEA test 的完整真实工具 rollout
```

反例类型：

- **semantic preserving**：文件名、路径别名等变化，策略应保持；
- **semantic changing**：对象、时间、单位、图层等变化，原经验应重新绑定或拒绝；
- **precondition breaking**：缺少必需产物或不兼容输出，经验应过滤或回退。

语义条件为软引导；信息未知时不能硬拒绝，必须保留通用 ReAct fallback。当前成对重放是单工具调用级验证，并不声称已经证明完整长轨迹后缀的因果效果。

### 5.3 已完成的训练侧构建证据

| 项目 | 数值 |
|---|---:|
| 经验 family 数 | 803 |
| active rules | 697 |
| conditioned family | 455 |
| accepted conditions | 11 |
| 总证据数 | 13,331 |
| 成功 / 失败 / 未知 evidence | 10,269 / 311 / 2,751 |
| 真实 counterfactual tool replay | 44 |
| metadata probe：validated / unknown | 1,316 / 6,580 |
| Proposer/Auditor 已执行 pairs | 662 |
| 审核：defer / add / quarantine / keep / split | 490 / 64 / 106 / 122 / 21 |
| 规则冻结 | 是，`eval_frozen=true` |

当前严格性：父 v4-clean store 保持只读；Boundary 使用独立 sidecar；规则在测试前被冻结；不从 OEA test 或 gold 标注学习条件。

### 5.4 当前评测状态与答辩表述

- OEA test 共 1162 条，LongCat + 完整真实工具环境；已进入 `full_eval` 阶段，结果持续独立落盘。
- 评测尚未结束，**本次答辩不得展示 Boundary 的最终 Success/F1 柱状提升或声称其已超过 v4-clean**。
- 可展示第 5.2 节流程图和第 5.3 节训练侧审计表，表述为“完整方法已构建并进入冻结测试”。

## 6. PromptEvo：规则补丁化的静态提示词自进化（已完成）

### 6.0 从真实结果引出的 PromptEvo 问题

ExperienceEvo 解决的是“当前已有产物状态下，应该复用哪条经验”；但还有一类跨任务、与具体文件无关的缺口会反复出现，例如工具参数校验、错误恢复、重复调用和终止条件。把这些缺口全部留给固定 system prompt，会出现两个问题：提示词过于冗长却仍覆盖不全；一次全局改写可能修复一种错误，同时改变原本稳定的行为。

已有结果已经显示出这种非单调性，而不是一个只存在于理论上的风险：

| 证据 | Base | Stage1/Stage2 | 暴露的问题 | PromptEvo 的应对 |
|---|---:|---:|---|---|
| ToolBench strict paper split（冻结 test=160） | Success 86.25%，Tool error 14.37%，平均 5.14 次工具调用 | Stage1：88.12%、7.50%、4.67；Stage2：84.38%、10.00%、5.59 | 第一版协议补丁有效，但后续修复候选引入冗余和不完成，演化不是单调上升 | Stage1/Stage2 分离；用 paired trace attribution 定位新增/消失行为；由 gate 接受、拒绝或回滚 |
| OEA Qwen3 matched n=906 | Set F1 .628，Exact 22.85%，Ordered 16.11%，Success 86.87% | Stage2：Set F1 .661，Exact 34.55%，Ordered 23.18%，但 Success 85.21% | 工具序列更接近目标，不等于端到端成功率同步提高 | 同时检查目标指标、工具错误、终止和新失败类型，不用单一 F1 自动接受补丁 |
| API-Bank 389（同集合诊断/门控） | Missing arg 4.63%，Extra arg 4.37% | Stage2：3.34% / 2.83%，Tool F1 .9572 | 参数格式更规范，但该集合存在优化/dev 重叠，不能当作独立泛化证明 | 将结果定位为执行诊断和回归门控，独立 test 结论另行报告 |

因此 PromptEvo 的研究问题不是“让 LLM 再写一版更长的 prompt”，而是：**如何把失败轨迹归因成最小、可审计、任务无关的协议补丁，并验证补丁没有引入新的失败。**这自然引出下面的 typed patch、Stage1/Stage2 对比归因和 dev gate，而不是直接覆盖原 system prompt。

### 6.1 与 ExperienceEvo 的区别

| 维度 | ExperienceEvo | PromptEvo |
|---|---|---|
| 优化对象 | 当前状态下动态检索的经验 | 固定 system prompt 中的通用行为协议 |
| 更新时机 | 运行时注入 | 离线从失败 rollout 演化后固定 |
| 表示 | 产物状态转移、前置条件、Q/N/R | typed protocol patch |
| 解决问题 | 当前产物状态如何继续 | 跨任务通用的工具选择、参数验证、恢复与终止规则 |

### 6.2 Typed protocol patch 流程

PromptEvo 不让 LLM 自由重写全部 system prompt，而是要求输出受类型约束的补丁，再由确定性 compiler 渲染为静态 prompt：

```text
Base rollout failure
  -> failure attribution
  -> typed patch candidate
  -> deterministic compiler
  -> Stage1 candidate prompt
  -> paired Base/Stage1 trace regression analysis
  -> Stage2 repair candidate
  -> dev gate：目标指标 + error/repetition/termination guard
  -> accept / reject / rollback
```

补丁类型包括：`tool_selection`、`query_abstraction`、`argument_validation`、`error_recovery`、`termination_and_repetition`、`answer_contract`、`tool_output_security`、`candidate_metric_guard`、`state_transition`。

硬约束：不得把 task id、固定路径、benchmark 名、gold 信息或固定工具工作流写入补丁。

### 6.3 PromptEvo 结果：三种证据需分开报告

#### API-Bank 389 条完整诊断实验

> Base/Stage1/Stage2 均完成；但优化轨迹与 dev 从同一全量集合抽取并在同集合报告，因此是完整执行诊断/回归门控，不是严格独立 test 泛化。

| 指标 | Base | Stage1 | Stage2 | Stage2 - Base |
|---|---:|---:|---:|---:|
| Success | 82.78% | 82.01% | 83.55% | +0.77pp |
| Tool F1 | .9469 | .9444 | .9572 | +.0103 |
| Parse rate | 97.94% | 96.66% | 97.69% | -0.25pp |
| API name accuracy | 95.63% | 95.63% | 96.92% | +1.29pp |
| Missing argument | 4.63% | 3.60% | 3.34% | -1.29pp |
| Extra argument | 4.37% | 3.08% | 2.83% | -1.54pp |
| Runtime error | 0 | 0 | 0 | 0 |

#### ToolBench strict paper split：最干净的 PromptEvo 正向证据

> evolution=455，dev=150，冻结 test=160；LongCat；Base/Stage1/Stage2 均为 160/160。

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| Success | 86.25% | **88.12%** | 84.38% |
| Avg tools | 5.14 | **4.67** | 5.59 |
| No-finish | 11.88% | 11.88% | 15.62% |
| Give-up | 4.38% | **0.62%** | 0 |
| Tool error | 14.37% | **7.50%** | 10.00% |
| Repeat | 10.62% | **9.38%** | 8.75% |

Stage1 相比 Base：Success +1.87pp、平均调用 -0.47、Tool error -6.87pp、Give-up -3.76pp。Stage2 出现回退，说明 contrastive repair 仍可能引入过度探索，正是 gate 与回退机制必要的原因。

#### OEA Qwen3 contrastive（matched n=906）

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| Set F1 | .628 | .653 | .661 |
| Exact | 22.85% | 32.23% | 34.55% |
| Ordered exact | 16.11% | 20.97% | 23.18% |
| Success | 86.87% | 未作为主要结论 | 85.21% |

这说明协议补丁改善工具调用集合与顺序，但不支持“任务成功率必然提升”的说法。

### 6.4 PromptEvo 的反例与局限

- AgentDojo LongCat（1081）：Stage3 balanced 62.28 vs Base 61.53，但 security、attacked security 下降，attack success 上升；只能说明效用-安全权衡。
- tau2 Qwen3 8B（1500）：success 8.47 -> 9.20，提升有限。

这两组不建议作为 10 分钟主结果，但可在答辩最后用一句话说明：PromptEvo 采用 gate 的原因是不同 benchmark、指标与安全目标之间存在非单调权衡。

### 6.5 外部提示词/Agent Harness 方法的问题：证据、案例与对应机制

这一节用于回答答辩专家可能提出的追问：“GEPA、SCOPE、AHO 或 EvoTool 也能改进 Agent，为什么还需要 PromptEvo？”这里不把某个方法简单说成“无效”，而是把问题限定为**在本项目的真实工具调用场景中，哪些机制仍然不够**。外部方法的本地结果是 adapted/reimplemented 口径，不能当作官方 leaderboard；389 条 API-Bank 结果包含优化 train/dev，79 条才是冻结 held-out。更不能为了让表格好看而伪造不存在的提升。

本节的案例分为两类：带有结果文件路径或完整工具序列的，标为“真实诊断案例”；由审计出的候选 patch 和工具契约构造的，标为“机制示意/待配对重跑”。后者用于解释为什么需要某个设计，不冒充已经完成的单条因果实验。

#### 6.5.1 GEPA/反思式 whole-prompt mutation：平均分提高不等于回归被保护

GEPA 类方法通过反思轨迹和指标反馈生成新的完整提示词。它适合快速寻找有效的文本策略，但若候选以一个聚合均值被接受，往往难以回答三个问题：

1. 改善究竟来自哪条失败轨迹，是否只是少数任务变好；
2. 哪些原本稳定的行为被候选破坏；
3. 某条规则只在什么触发条件和作用域内生效。

本地 API-Bank 结果给出了一个具体的边界信号：GEPA 与 Base 在 389 条 full coverage 诊断上都是 `322/389`；在冻结 79 条 held-out 上也都是 `64/79`。这不是“GEPA 没有价值”的证明，而是说明**仅凭该聚合口径不能证明独立泛化提升**。在 ToolBench strict paper split 中，Stage1 的 success 为 88.12%，但后续 Stage2 为 84.38%，平均工具调用从 4.67 增至 5.59，no-finish 从 11.88% 增至 15.62%。这说明一个候选可能修好一类错误，却同时引入冗余探索或不终止。

一个完整的**机制示意/待配对重跑案例**是：Stage1 发现“工具参数缺失时先检查必填字段”可以减少 malformed call，于是候选提示词继续把它泛化成“缺少任意参数时都向用户询问”。在 API-Bank 的 next-call 预测接口中，Agent 并不总能向用户发起真实追问；如果候选不区分 `required-and-uninferable` 与 `optional-or-context-inferable`，就可能出现：

```text
原任务：根据已有对话调用天气 API，城市已在上一轮出现，单位没有出现但 API 有默认值。
候选规则：只要发现参数缺失就停止并向用户询问。
结果：模型没有生成 next tool call，parse/finish 失败；原本可以由上下文或工具默认值完成的调用被过早中止。
```

PromptEvo 的对应设计不是继续重写全文，而是把修复写成有条件的 typed patch：`kind=argument_validation`，`trigger=required argument is missing and cannot be inferred`，`scope=tool-call construction`，并显式保留 `optional/defaultable/context-inferable` 例外。Stage1/Stage2 使用同一任务、同一 provider 做 paired trace attribution，gate 同时检查 success、parse/no-finish、tool error、重复调用和新增失败类型；任何保护指标退化就 reject 或 rollback。这样“参数检查”不会被编译成无条件的“向用户提问”。

#### 6.5.2 SCOPE/guideline append：不断追加规则会造成上下文膨胀和条件冲突

SCOPE 风格的方法把新的 guideline 追加到上下文，优点是实现简单、容易解释；但它通常缺少统一的 `trigger`、`scope`、优先级和冲突消解协议。长时间演化后，多个看似合理的规则可能同时命中：一条要求“遇到工具错误立即重试”，另一条要求“避免重复调用”，第三条要求“缺参先询问”。模型只能在自然语言规则之间自行仲裁，规则数量和上下文长度也会一起增长。

本地 API-Bank 389 条 full coverage 中 SCOPE 为 `325/389`，高于 Base 的 `322/389`；但在 79 条 held-out 上 SCOPE 与 Base 都是 `64/79`。因此更准确的结论是：追加 guideline 在当前执行诊断中有收益，但还没有证明稳定的独立泛化。答辩中应把这个现象作为**为什么需要作用域和冻结 gate**的动机，而不是把 full coverage 的 3 个样本差异包装成普适优势。

一个完整的**机制示意/待配对重跑案例**是：一个 OEA 工具超时后，旧式追加规则同时激活“Retry once”与“Do not repeat a failed call”。如果没有结构化优先级，模型可能先重复同一参数调用，再因为第二条规则停止；如果工具实际上是瞬时网络超时而非参数错误，这会浪费一次重试机会；如果是确定性 `buffer_m` 超限，重试只会再次触发相同错误。PromptEvo 将其拆成两个 typed patch：

```json
{
  "kind": "error_recovery",
  "trigger": "transient tool timeout or temporary network failure",
  "scope": "retryable external-service failures",
  "rule": "retry at most once with the same semantic request, then continue to fallback"
}
```

以及一个互斥的确定性错误 patch：

```json
{
  "kind": "error_recovery",
  "trigger": "invalid argument, unsupported layer, or deterministic no-match",
  "scope": "non-retryable tool failures",
  "rule": "do not repeat unchanged arguments; repair, ask for a missing value, or use fallback"
}
```

compiler 按 `trigger → scope → priority` 渲染条件，而不是把两条 `rule` 无条件拼在一起；paired replay 再检查重试次数、错误类型和任务终止。当前代码审计确实发现旧 `compile_patches()` 主要追加 `patch.rule`、可能丢失 `trigger/scope` 的风险，这里是**已审计的编译语义问题与修复设计**，不能表述成因果修复已经完成。

#### 6.5.3 AHO/BetterHarness 的 prompt-only 适配：协议文字不能替代运行时保护

AHO/BetterHarness 类工作覆盖的能力范围可能包含 prompt、工具编排或 middleware。若在本地只复刻 prompt-only 变体，就必须把它与完整 harness 的结果分开；否则会把运行时重试、缓存、工具选择器等收益错误归因给提示词。即使平均 utility 提高，也仍需单独监测 no-finish、错误工具、重复调用和安全回归。

当前 API-Bank 389 条 adapted 结果中 AHO 与 Base 都是 `322/389`；这说明在本项目的固定 provider、工具解析器和任务口径下，prompt-only 适配没有表现出独立提升。一个来自 OEA 的完整失败链路说明了原因：在 `How many buildings are in between domestic and construction waste?` 任务中，Base 连续 13 次 VLM 分析后仍调用 `count_given_object`，最后因 `max sequential tool turns` 失败。仅在 system prompt 中写“避免重复分析”并不能保证模型知道已经分析过哪些视觉产物，也不能保证它识别出当前该消费哪一个中间结果。

PromptEvo 的解决范围更窄也更可审计：用 `termination_and_repetition` patch 规定“同一语义输入、同一工具和等价参数的重复调用必须触发停止或 fallback”，用 `state_transition`/`answer_contract` patch 要求在工具循环结束前检查是否已有可消费 artifact；但它不声称替代 artifact runtime verifier。这样 PromptEvo 负责跨任务的协议约束，ExperienceEvo 负责当前产物状态，二者各自有明确边界。

#### 6.5.4 EvoTool/blame-aware mutation：模块化搜索很强，但适配边界与保护目标仍需显式化

EvoTool 将 Planner、Selector、Caller、Synthesizer 等模块分开，并利用 blame-aware mutation 和 diversity-aware selection 搜索改进方向；它的优势是搜索空间更丰富，但也带来两个答辩中必须说明的限制：模块接口、工具执行语义和评估器一旦变化，原论文机制并不能自动等价迁移；多模块改动同时发生时，单条指标变化也更难归因。

本地 API-Bank 389 条结果中 EvoTool 为 `301/389`（77.378%），modular-base 变体为 `281/389`（72.237%），79 条 held-out 分别为 `57/79`（72.152%）和未作为主表的对应变体。由于这是 adapted/reimplemented 口径，不能把数字直接解释为论文方法的普遍弱点，也不应拿它做未经控制的 headline ranking；它至少提醒我们：**方法的模块化能力与当前 harness 的适配正确性是两件事**。

一个完整的**机制示意/待配对重跑案例**是：如果 Planner 生成了“先查询边界、再添加 POI、最后计算路线”的合理计划，而 Caller 没有把上一工具返回的 GeoPackage layer id 传给下一工具，所有模块都可以“局部看起来正确”，但真实工具链仍会出现 `No matching features`。PromptEvo 不尝试重写整个 Planner/Caller，而是提出一个低侵入的 `argument_validation` patch：在调用依赖前序 artifact 的工具前，必须确认输入引用来自当前会话、layer 类型与 schema 匹配；无法确认时走 fallback。paired trace 会把“规划正确但参数绑定错误”单独归因出来，避免用一个总分把错误归给整个模块链。

#### 6.5.5 Free whole-prompt rewrite、TextGrad、PromptWizard 与 A-Evolve：作为待补对照，不提前编结果

自由全文改写、TextGrad/PromptWizard 的文本优化，以及 A-Evolve/AutoHarness 的 workspace/harness 演化，分别代表更宽的搜索空间或更强的系统改造能力。本项目当前没有在完全相同的 actor、provider、预算、冻结 manifest 和评估器下完成这些方法的正式对照，因此它们只能写入“待补 baseline/相关工作”，不能填入假数字。研究上仍可用它们提出一个共同问题：候选修改越自由，越需要明确变更类型、证据来源、作用域和回滚条件。

### 6.6 PromptEvo 的完整机制案例：从编译语义丢失到受保护 patch

下面案例来自对 PromptEvo 编译器的代码审计，目的是展示方法为什么需要 typed patch 和 condition-aware compiler；它不是一条已经完成的因果重跑结果。

**问题输入。**失败轨迹显示：工具请求偶发超时时，模型需要最多一次重试；但参数非法、图层不存在或 `buffer_m` 超限时，重复相同调用只会制造新的错误。LLM 提出的候选补丁包含触发条件、适用范围和规则：

```json
{
  "patch_id": "p17",
  "kind": "error_recovery",
  "trigger": "transient tool timeout",
  "scope": "retryable external-service failures",
  "priority": 20,
  "rule": "Retry the same semantic request once; if it fails again, use fallback."
}
```

**旧实现的风险。**审计发现 `compile_patches()` 主要把 `patch.rule` 追加进最终 system prompt；若 `trigger` 和 `scope` 没有被渲染为可执行条件，最终提示词只剩下“Retry the same semantic request once”。这会把本来只适用于瞬时超时的规则扩大到参数错误、无匹配结果和确定性 schema 错误，导致重复调用，甚至与“不要重复失败工具调用”的另一条规则冲突。

**PromptEvo 的修复路径。**typed schema 先校验 `kind/trigger/scope/priority/rule`，compiler 再生成带条件和优先级的协议段；paired Base/candidate trace 对比固定任务的错误类型、重复次数、是否完成和新失败；dev gate 采用 protected metrics：候选只有在目标行为改善且 no-finish、tool error、repeat、新回归均不恶化时才 accept，否则 reject/rollback。该流程把“LLM 写了一句看似合理的话”转化为“一个有证据、有作用域、可回滚的协议变更”。

### 6.7 PromptEvo 的预注册目标与不能伪造的指标

为避免答辩中把目标写成结果，后续 PromptEvo 对照实验应在运行前冻结以下判据：

| 目标类型 | 预注册判据 | 当前状态 |
|---|---|---|
| 主要目标 | held-out exact call accuracy 高于 Base，且置信区间/配对检验支持差异 | 待验证；当前 79 条 held-out 中 PromptEvo 与 Base 均为 64/79 |
| 参数安全 | missing/extra argument 不增加，parse/no-finish 不恶化 | 待验证；389 full coverage 中 Stage2 已观察到参数错误下降，但不是独立 test |
| 工具行为 | tool error、重复调用、无效重试不增加 | ToolBench Stage1 已改善；Stage2 回退说明需保留 gate/rollback |
| 泛化与安全 | 未见任务/安全子集不出现新的明显回归 | AgentDojo 当前只能说明效用—安全权衡，不能称全面提升 |

因此，本答辩可以展示真实的 Stage1 正向证据、Stage2 回退证据和机制设计，但不能“假装” PromptEvo 在所有 benchmark 上优于外部方法。若后续实验达到预注册判据，再把“待验证”替换为带 manifest、样本量和置信区间的正式结果。

## 7. 已完成标准 GRPO 与下一阶段 RL 设计

### 7.1 已完成：Qwen2.5-3B Pure GRPO

训练流程：同一任务采样多条真实工具 rollout，给每条轨迹计算不依赖 gold 的 episode reward，再作组内相对优势并用 LoRA/QLoRA 更新模型。

在线规则 reward：成功工具调用 +0.35，若有 artifact/path 再加 +0.15（单步最高 +0.60）；empty -0.20；error -0.60。episode reward 同时考虑成功比例、error/empty、artifact、过长工具链、过多轮数和过长输出：

\[
R=-0.15+0.95\frac{N_{success}}{N}-0.45N_{error}-0.20N_{empty}
+\min(0.20,0.05N_{artifact})-P_{long}-P_{turn}-P_{token},
\]

无工具调用时给 -1，并将 reward 截断到 [-1, 1]。组内优势为：

\[
A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}.
\]

| OEA 完整 eval | Success | Set F1 | Tool error |
|---|---:|---:|---:|
| Qwen2.5-3B Base | 71.86% | .5142 | 48.45% |
| Pure GRPO checkpoint-1900 | 70.65% | .5102 | 49.74% |
| Pure GRPO + ExperienceEvo v4-clean | 73.75% | .5137 | 34.17% |

结论：Pure GRPO 没有稳定超过 Base；引入 v4-clean 后工具错误明显下降。因此当前可将 RL 作为已完成的真实环境基线，而不是已验证的性能主贡献。

### 7.2 下一阶段：面向产物来源约束的工具调用信用分配（研究计划）

标准 GRPO 对一条 rollout 得到一个总优势，通常广播给整段 assistant token。它难以区分“前面一次错误绑定、后面重试修正”的不同责任。

拟议方法保留组内 GRPO，但额外：

1. 建立 artifact provenance graph，记录工具消费/产生的产物及对象、时间、单位、CRS、下游使用关系；
2. 为第 t 个工具调用计算局部信用：

\[
c_{i,t}=w_sS_{i,t}+w_oO_{i,t}+w_pP_{i,t}-w_nN_{i,t};
\]

3. 在相同产物前缀下比较候选动作，得到局部相对收益：

\[
D_{i,t}=Y_{i,t}-\operatorname{mean}(Y_{\text{other same-prefix}});
\]

4. 将局部优势与 episode advantage 混合并仅对齐到该工具调用的 assistant token：

\[
\widetilde A_{i,t}=(1-\lambda c_{i,t})A_i+
\lambda c_{i,t}\operatorname{clip}(D_{i,t}/s,-a_{max},a_{max}).
\]

真正的算法差别在于“动作级 advantage 与 token span 对齐”；若只把步骤分数加成总 reward 再训练，仍只是普通 GRPO 的 reward shaping。该方案需要 veRL 修改 action span 与 advantage 构造；Swift 只能做近似 shaping。

### 7.3 RL 内容在答辩中的定位与可协商指标

建议把 RL 讲成“已完成基线 + 明确的下一步算法创新”，不要把当前 Pure GRPO 结果包装成已经验证的新算法：

| 层次 | 当前状态 | 答辩中的说法 | 可展示指标 |
|---|---|---|---|
| Pure GRPO baseline | 已完成真实工具 online rollout、LoRA 训练和完整 OEA eval | 建立真实环境 RL 基线，并暴露 episode-level reward 的局限 | Success、Set F1、tool error、reward 曲线、KL、turns、训练稳定性 |
| Pure GRPO + ExperienceEvo | 已完成运行时经验增强对照 | 说明策略更新与外部经验检索可以组合，但收益主要体现在错误控制 | Success、tool error、经验命中率、tools/task |
| 方案 A：Provenance-aware credit assignment | 研究计划，尚未实现 | 下一步将产物来源与语义绑定信用对齐到工具调用 token | 产物绑定准确率、tool F1、错误率、长链任务成功率、未见组合泛化 |

Pure GRPO 的已完成指标可以放在备用页或第 11 页的左半部分：

| OEA eval | Success | Set F1 | Tool error |
|---|---:|---:|---:|
| Qwen2.5-3B Base | 71.86% | .5142 | 48.45% |
| Pure GRPO checkpoint-1900 | 70.65% | .5102 | 49.74% |
| Pure GRPO + ExperienceEvo v4-clean | 73.75% | .5137 | 34.17% |

**答辩口径建议：**这组结果适合用来提出问题，而不是宣称 RL 已经带来稳定性能提升。可以说：“当前纯 GRPO 尚未稳定超过 Base，但经验增强显著降低了工具错误；这说明仅使用 episode-level 工具执行奖励无法定位错误产物绑定，促使我们设计方案 A。”

**不要作为主结论的指标：**单次训练 reward 上升、KL 变化、turns 下降不能直接等价于任务能力提升；它们只能作为训练过程诊断。若老师追问，可展示 reward 曲线并说明最终判断仍以固定 OEA eval 和工具/产物指标为准。

**方案 A 的创新点要这样说：**不是重新提出 GRPO、步骤奖励或组内优势，而是把可验证的 artifact provenance 作为局部信用来源，并将该信用对齐到具体工具调用的 assistant token span；如果只把步骤分数求和后继续使用整轨迹 advantage，则仍属于 reward shaping。

## 8. 10 分钟 PPT 逐页剧本

### 第 1 页：封面与一句话目标（20 秒）

- 标题：**面向地理空间多工具 Agent 的经验自进化与协议自进化**。
- 一句话：让 Agent 不仅记住“做过什么”，还知道“当前产物状态下何时可以复用、何时必须重新决策”。
- 只放姓名、课题组、日期；不放指标。

### 第 2 页：问题与挑战（50 秒）

- 左侧画简短链路：灾后影像 -> 分割 -> mask -> 面积/变化分析。
- 右侧放对比：**工具成功 != 产物绑定正确 != 任务语义正确**；下方补两条数据证据：轨迹级记忆缺少步骤级参数/产物契约，工具调用呈长尾分布。
- 讲稿重点：用“domestic waste—construction waste—building”任务的完整失败链路说明 Base 连续 13 次 VLM 后仍到达 max-turn；再用 OEA 23 工具的高频/低频调用差异说明全局经验统计会偏向高频工具。
- 过渡句：因此要把经验拆成“当前产物状态—工具动作—输出契约”的步骤级转移，并在检索前检查其适用条件。

### 第 3 页：总体研究路线（45 秒）

- 放第 1.1 的两路图：ExperienceEvo（动态经验）与 PromptEvo（静态协议）。
- 用状态标签区分：v4-clean 已完成；Boundary 评测中；provenance-aware GRPO 为后续。
- 讲稿重点：两条路线分别回应两类问题：ExperienceEvo 处理当前产物状态下的经验复用和长尾工具风险，PromptEvo 处理跨任务重复出现的协议缺口；二者不能混为单一 prompt trick。

### 第 4 页：ExperienceEvo v4-clean 方法（70 秒）

- 放第 2.1 的六步流程图。
- 高亮四个词：`step-level artifact transition`、`state-constrained retrieval`、`Q/N/R + Quse`、`verifier + fallback`。
- 页脚写“严格 rollout-only：不读取 OEA gold/tool label”。
- 讲稿重点：经验不是答案案例；先检查当前是否已有可消费产物，再检索工具动作和参数。低证据或未知状态不强制套用历史经验。

### 第 5 页：v4-clean 核心结果（80 秒）

- 正文只做 5 组柱状/数字卡：Success 86.90 -> 92.51，Set F1 .703 -> .774，Tool error 15.4 -> 9.6，Tools/task 6.48 -> 4.80，Answer w/ gen 66.20 -> 76.06。
- 页脚小字说明：OEA 1162、LongCat、matched tools；普通 answer_acc 49.67 -> 49.32，不夸大。
- 讲稿重点：提升主要来自工具链可靠性和效率，不回避普通文本答案持平略降。

### 第 6 页：为什么有效，及外部对照（65 秒）

- 左侧放外部对照的精简表：Base / Reflection / ACE / Memento / v4-clean，只留 Success、Set F1、Tool error、Tools/task。
- 左下或讲稿中补 Kluane 任务案例：ExpeL/Memento 15 次调用后 max-turn 失败，Base 11 次调用且有 OSM/buffer 错误，v4-clean 7 次调用但仍保留外部服务不确定性。
- 右侧放 diagnostics 小表或三条结论：去 step hint、去 verifier、random retrieval 都退化；这证明改进来自步骤提示、证据闭环和检索质量的组合，而不是单纯增加上下文长度。
- 页脚注明“外部方法为 OEA adapted 复现；消融与主表采样口径不同”。

### 第 7 页：Boundary：经验也要学习适用边界（70 秒）

- 放第 5.2 的闭环图。
- 用一个例子：同样的 mask 面积统计，当前影像时间/CRS 改变时不能盲目复用。
- 高亮：Proposer 提出反例，Auditor 基于真实执行观察决定 `add/split/quarantine/defer`。
- 讲稿重点：这是独立模式，父 v4-clean store 不改写；Boundary 不是 RL。

### 第 8 页：Boundary 训练侧证据与当前状态（45 秒）

- 放数字：803 families、697 active rules、662 agent audit pairs、44 real tool replays、21 splits、106 quarantines。
- 右下角放醒目状态：**OEA 1162 条冻结全量评测进行中，暂不报告最终性能提升。**
- 讲稿重点：强调公平测试冻结与结论边界，比展示未完成曲线更可信。

### 第 9 页：PromptEvo 规则补丁化（65 秒）

- 放 `failure -> typed patch -> compiler -> paired trace -> dev gate -> accept/reject/rollback`。
- 左侧用三张小卡片概括外部方法的缺口：GEPA/whole-prompt mutation 的平均分不能保护回归；SCOPE append 容易上下文膨胀和规则冲突；AHO prompt-only 不能替代 artifact/runtime verifier。右侧用三类 patch 示意：参数校验、错误恢复、重复/终止控制。
- 用一个完整但不堆字的案例说明：`transient timeout -> retry once`，而 `invalid argument/no-match -> no unchanged retry`；compiler 必须保留 trigger/scope，gate 检查 no-finish、tool error、repeat。
- 对比 ExperienceEvo：PromptEvo 优化固定协议，不看当前 artifact state。

### 第 10 页：PromptEvo 结果与诚实的非单调性（60 秒）

- 主表使用 ToolBench strict split：Base vs Stage1 vs Stage2，突出 Stage1 的 Success、error、tools、give-up，并用醒目标记 Stage2 回退。
- 右侧加一个小表：API-Bank 389 full coverage 中 Base/GEPA/AHO/PromptEvo 的 `322/389`、SCOPE `325/389`，下方单列 79 held-out：Base/GEPA/SCOPE/AHO/PromptEvo 均 `64/79`；明确 full coverage 含优化 train/dev，不是独立泛化。
- 讲稿重点：Stage2 回退、GEPA 与 Base held-out 持平、OEA 工具序列与 Success 不一致，都说明补丁不是“越演化越好”；paired attribution、compiler、gate/rollback 是方法组成而非工程细节。不要把未完成的自由全文改写对照填成结果。

### 第 11 页：RL 基线、下一步与总结（55 秒）

- 上半：Pure GRPO 表（Base、GRPO、GRPO+ExperienceEvo），一句话说明当前纯 GRPO不足。
- 下半：provenance-aware credit assignment 示意图，标记“下一阶段，未验证”。
- 最后一句总结：
  > v4-clean 已证明产物状态经验能改善长工具链；Boundary 正在把经验从全局可靠性推进到条件化适用性；PromptEvo 验证失败轨迹可驱动受保护的协议更新；下一步通过产物来源感知 RL 将这些约束内化到策略中。

## 8A. AI 生图提示词：答辩中需要的 pipeline 图

以下提示词用于生成“视觉草图”，生成后应在 PowerPoint/Figma 中重新排版文字、箭头和数字。不要直接把生图模型生成的乱码文字用于正式答辩。统一视觉要求：白色或浅灰背景、深蓝/青绿/橙色三色体系、扁平矢量科研信息图、细线箭头、无人物、无装饰性渐变球、16:9 横向、留出标题和中文标注空间。

### 图 A：总体研究路线图（主图，必须制作）

```text
Create a clean 16:9 academic vector infographic for a mid-term defense presentation about a geospatial multi-tool AI agent. Show one input task flowing into a tool-use loop with raster imagery, vector layers, masks, tables and GeoPackages as intermediate artifacts. Split the loop into two parallel self-evolution branches: the upper branch is “ExperienceEvo” with artifact-state transition, state-constrained retrieval, Q/N/R reliability and runtime verifier; the lower branch is “PromptEvo” with failed rollout, typed protocol patch, compiler and dev gate. On the right, show a future research branch “provenance-aware GRPO” that assigns local credit to tool-call token spans. Use three restrained colors: navy for the base agent, teal for ExperienceEvo, orange for PromptEvo, purple only for future RL. Minimal labels, no dense paragraphs, no decorative illustration, no photorealism, no fake logos, no illegible text.
```

### 图 B：ExperienceEvo v4-clean pipeline（方法页）

```text
Create a publication-style 16:9 vector pipeline diagram. Left: real rollout trajectories from a geospatial tool-use agent, explicitly marked “rollout-only, no gold labels”. Middle: decompose each trajectory into step-level input artifacts, tool action and parameters, output artifacts, output contracts and recovery conditions; cluster them into artifact-state transition families and estimate quality, support and risk statistics. Next: observe the current artifact state, apply precondition and provenance filtering, combine structured state matching with lexical retrieval, then rerank by reliability and current-use benefit. Right: inject an initial guidance and step hint into the agent, execute real tools, and run an evidence verifier; when evidence is unknown or inconsistent, fall back to generic ReAct instead of forcing a historical case. Show a small example chain: post-disaster image -> segmentation mask -> area/change analysis. Flat scientific vector style, white background, navy-teal-orange palette, clear arrows, no people, no gradients, no fake text.
```

### 图 C：Boundary 反例驱动闭环（方法页）

```text
Create a clean 16:9 academic systems diagram for boundary-aware experience evolution in a geospatial agent. Start with a read-only parent experience store. Branch into three controlled probes: semantic-preserving transformation, semantic-changing transformation, and precondition-breaking transformation. Feed paired real tool observations to a Boundary Proposer and Boundary Auditor. Show five possible outcomes as compact nodes: keep, add condition, split family, quarantine, defer unknown. Then freeze boundary rules before the test set and send them into a full real-tool evaluation, with a safe fallback to generic ReAct when conditions are unknown. Emphasize “train-only discovery, frozen test, parent store unchanged”. Use teal for evidence, orange for counterexamples, red only for quarantine, navy for evaluation. Minimal text, precise arrows, no decorative elements.
```

### 图 D：PromptEvo typed protocol patch（方法页）

```text
Create a 16:9 publication-quality vector flowchart showing protected prompt evolution for a tool-using language agent. Left: base system prompt and failed rollout traces. Middle: failure attribution produces typed patch candidates such as tool selection, argument validation, error recovery, repetition termination and answer contract. Then a deterministic compiler renders a candidate protocol patch. Right: paired base-versus-candidate rollout regression, with a gate checking target success, tool error, repetition and new failure types; the result is accept, reject or rollback. Use orange for failure evidence, blue for compiler and gate, green for accepted patch, gray for rejected patch. Keep the diagram sparse and readable, no large paragraphs, no human figures, no fake logos.
```

### 图 E：普通 GRPO 与方案 A 对比（研究计划页）

```text
Create a split-screen 16:9 academic infographic comparing standard GRPO and provenance-aware tool credit assignment. Left panel: several complete rollouts receive one episode reward, one group-relative advantage is broadcast to all assistant tokens. Right panel: a provenance graph records tool, input artifact, output artifact, semantic fields and downstream consumption; local credit is computed for each tool action and aligned only to its corresponding assistant tool-call token span, then mixed with the episode advantage. Include a small example where a wrong pre/post-disaster image binding gets negative local credit and the corrective action gets positive local credit. Clearly label the right panel as “future research, not yet evaluated”. Navy and gray for standard GRPO, purple and teal for the proposed extension, clean vector style, no decorative gradients.
```

### 图 F：主结果图的生图建议

结果柱状图、折线图和表格不要用生成式图像模型绘制，应用 Python/matplotlib 或 PowerPoint 原生图表生成，以保证数值、坐标轴和误差口径正确。AI 生图只用于方法 pipeline、问题示意和概念图。

## 8B. 最终 10 分钟 PPT 制作规划（按页执行）

| 页码 | 标题 | 页面必须放什么 | 讲述目标 | 时间 |
|---:|---|---|---|---:|
| 1 | 研究目标与一句话贡献 | 项目标题、地理空间工具链示意、两条路线名称 | 让听众先知道“研究对象不是普通问答，而是多步真实工具执行” | 20s |
| 2 | 背景：工具成功不等于语义正确 | 灾后影像→mask→分析的三层错误图；突出来源/时间/单位/CRS | 说明为什么普通成功率和工具 API 返回值不够 | 55s |
| 3 | 相关工作与研究缺口 | GiGPO、Memory-R1、MemRL、GACA 四个小卡片；底部一句 gap | 说明已有工作分别解决信用、记忆管理或经验效用，但没有统一处理 provenance 与边界 | 60s |
| 4 | 总体路线 | 图 A；标注 v4-clean 已完成、Boundary 评测中、RL 计划 | 建立两条自进化路线和后续 RL 的层次关系 | 45s |
| 5 | ExperienceEvo v4-clean | 图 B；只突出 state transition、constrained retrieval、Q/N/R、verifier | 讲清经验不是文本 RAG，而是可执行产物状态转移 | 75s |
| 6 | v4-clean 主结果 | 5 个数字卡：Success、Set F1、tool error、tools/task、answer w/ gen；小字写 answer_acc 持平略降 | 用证据证明主要收益是工具链可靠性、序列匹配与效率 | 85s |
| 7 | v4-clean 为什么有效 | step hint/verifier/random retrieval 消融；外部 adapted 对照只放精简四列 | 回答“是不是只因为 prompt 变长”，展示机制证据与公平口径 | 65s |
| 8 | Boundary：经验适用边界自进化 | 图 C；803 family、662 audit pairs、44 replay、规则冻结 | 说明从“检索什么经验”推进到“什么时候经验适用” | 70s |
| 9 | PromptEvo：固定协议补丁化 | 图 D；GEPA/SCOPE/AHO 的缺口卡片；ToolBench strict 三项结果：success、tool error、avg tools | 说明静态通用规则与动态产物经验是互补关系，并说明 typed patch/compiler 的必要性 | 65s |
| 10 | 非单调演化与当前状态 | Stage1/Stage2 回退；API-Bank full/held-out 口径对照；Boundary 当前 full eval 状态；明确“尚无最终 Boundary 性能结论” | 展示研究诚实性：演化需要 gate/rollback，评测中结果不提前下结论 | 60s |
| 11 | RL 基线、算法创新与总结 | 左侧放 Pure GRPO 三行指标表，右侧放图 E；底部列出三句 takeaways | 说明 Pure GRPO 是已完成基线，方案 A 是尚未验证但机制清晰的下一步创新 | 55s |

总时长约 9 分 55 秒，剩余 5 秒用于翻页和停顿。若老师要求压缩到 8 分钟，删除第 7 页外部对照，只保留一张消融小图；若要求展开到 12 分钟，增加备用页 A 的完整指标和备用页 B 的数据泄漏审计。

### PPT 制作顺序

1. 先用真实数据制作第 6、7、9 页的图表，不要先做装饰图。
2. 用图 A–E 生成方法草图，再在 PPT 中统一重画文字、箭头和公式。
3. 每页最多一个主结论、一个主图、三组数字；详细表格全部移到备用页。
4. 所有“评测中”“研究计划”标签使用统一灰色角标，避免听众误解为已验证结果。
5. 最后按 9 分 55 秒完整试讲一次，删掉超时页中的背景解释，不删结果边界和限制说明。

## 9. 备用页与答辩问答准备

### 备用页 A：全部 v4-clean 指标

直接使用第 3.2、3.3 节两张表；回答“最终回答为何未全面提升”时强调答案评测与工具链正确性并非同一目标，并展示 expanded generation answer 的改善。

### 备用页 B：Boundary 是否泄漏测试集

回答要点：父库只读；训练问题 hash 划分 discovery/validation；Boundary sidecar 在 test 前冻结；不使用 OEA gold calls/answers/task id；OEA test 只用于最终 rollout evaluation。

### 备用页 C：PromptEvo Stage2 为何变差

回答要点：Stage2 是基于 paired trace 的回归修复，但额外补丁会带来探索和冗余调用；正因如此以 dev gate、accept/reject/rollback 作为流程约束。应报告非单调结果，而不是只挑最好阶段。

### 备用页 D：GRPO 为什么目前没有超过 Base

回答要点：online reward 只观察工具是否成功、是否产出 artifact、是否过长，并不知道工具在任务语义上是否正确；episode-level advantage 也无法定位某次错误产物绑定。因此下一步不是继续堆 reward，而是将来源与语义约束对齐到工具动作 token。

## 10. 指标来源与复核路径

- OEA v4-clean 主结果：`实验结果总览_20260725以来.md`、`论文表格_OEA主结果_20260824.md`、`ExperienceEvo_manuscript_state_20260915.json`。
- 本文新增的真实案例轨迹：`tmp/trajectories/longcat_base_matched_v4clean_dockerfixed_20260818/standard/results/oea_test_1000.json`、`tmp/trajectories/experience_evo_v4_clean_oea_train2000_longcat_eval_dockerfixed_20260816/standard/results/oea_test_1000.json`，以及各 adapted 方法的 `oea_test_1001.json`。
- PromptEvo 更新结果与严格性审计：`promptevo/20260901后实验周报整理.md`。
- PromptEvo 外部方法调研与适配边界：`promptevo/外部Agent数据集与自进化方法调研_20260917.md`、`promptevo/创新审计与条件协议演化方案_20260919.md`。
- PromptEvo API-Bank 外部 LongCat 分组结果：`tmp/promptevo_api_bank_experiments/api_bank_external_longcat_grouped_20260919/`；389 条为 full coverage（含优化 train/dev），`79` 条 `summary.json` 为 held-out。不同 subset 的 `api_bank_toolsearch_external_longcat_grouped_20260919_v2/` 不与主表混合。
- Boundary 当前方法、冻结规则与工作流：`experience_evo/README.md`、`evolution_store/experience_evo/oea_train2000_boundary_full_20260920_retry/boundary_manifest.json`、`tmp/agent_rl_runs/longcat_boundary/longcat_boundary_full_20260920_retry/boundary_workflow.json`。
- 标准 GRPO reward、评测与待实现算法：`ExperienceEvo_产物来源感知GRPO_小白说明与简历写法_20260920.md`（已合并标准 GRPO 对比与简历口径）。

本文件的边界：所有标为“评测中”或“研究计划”的部分不得在答辩中表述为已验证指标或已完成创新。

## 11. 已生成的答辩 PPTX 与复现入口

- 可直接打开的 PPTX：`ExperienceEvo_PromptEvo_中期答辩.pptx`
- 逐页讲稿：`ExperienceEvo_PromptEvo_中期答辩PPT_讲稿.md`
- PPTX 构建源：`tmp/ccfa-workfiles/ppt_midterm_20260921/source/build_midterm_ppt.py`

### 11.1 科研模板版（当前推荐）

- 参考模板重建版：`ExperienceEvo_PromptEvo_中期答辩_科研模板版.pptx`
- 逐页讲稿：`ExperienceEvo_PromptEvo_中期答辩_科研模板版_讲稿.md`
- 构建源：`tmp/ccfa-workfiles/ppt_midterm_template_20260921/source/build_template_midterm_ppt.py`
- 采用《袁老师联合基金PPT-参考模板(不外传).pptx》的母版/主题/16:9 页面结构，重新绘制章节页、相关工作机制图、ExperienceEvo/Boundary/PromptEvo/RL 方法图和指标表。
- PPTX 主要元素为原生 PowerPoint 文本、形状、连接线和表格（`p:pic=0`）；Boundary 页在构建时读取当前 `results/` 数量，只展示实时进度，不提前写入最终性能结论。
- 当前环境未安装 LibreOffice/soffice，已完成 `python-pptx` 结构 QA（15 页、617 个 shape、561 个文本 shape、无越界、无图片形状）；中文字体和箭头显示仍需在 PowerPoint/WPS 中打开后做一次渲染检查。

PPTX 为 16:9 宽屏，11 页，约 10 分钟。方法图、流程框、箭头、文字和指标柱形均使用 PowerPoint 原生对象，未将整页 raster 图作为背景；因此文字和主要图形可以在 PowerPoint 中继续编辑。Boundary 页读取工作流目录中的当前落盘数量，重新运行构建脚本可更新评测进度，但 Boundary 最终性能只有在 `watcher_summary.json` 等最终报告生成后才能补入。

本次 QA 已完成：PPTX 可被 `python-pptx` 重新打开；共 11 页；所有形状均在 16:9 页面边界内；无图片形状（`p:pic=0`）；文字保持为 live text。当前环境没有 LibreOffice/soffice，因此未能执行 Office 渲染级截图检查，使用 PowerPoint 打开后应重点检查中文字体替换、文本自动换行和不同版本 Office 的箭头显示。
