# 自进化 Agent 综述

> 覆盖范围：2023–2026年主要自进化 LLM Agent 论文，重点分析本项目实现的四篇文章，并横向对比领域全景。
> 最后附：针对地理空间 Agent 的原创论文思路。

***

## 目录

1. [背景与动机](#1-背景与动机)
2. <br />
3. [核心四篇：详细分析](#2-核心四篇详细分析)
   - 2.1 SkillRL
   - 2.2 EvoSkill
   - 2.3 AgentEvolver
   - 2.4 MemRL
   - 2.5 ExpeL（已实现）
   - 2.6 SeqGraphEvo（已实现，原创）
   - 2.7 CausalTextEvo（已实现，原创）
   - 2.8 CausalPolicyEvo（已实现，原创）
   - 2.9 SelfCritic（已实现，原创）
4. [相关工作全景](#3-相关工作全景)
   - 3.1 奠基性工作（2023）
   - 3.2 经验/记忆驱动（2024–2026）
   - 3.3 协同进化/自对弈（2025–2026）
   - 3.4 提示/工作流进化（2025–2026）
   - 3.5 强化学习流派（2024–2026）
   - 3.6 元学习流派（2024–2026）
   - 3.7 课程学习流派（2024–2026）
   - 3.8 探索策略流派（2024–2026）
   - 3.9 安全约束流派（2024–2026）
   - 3.10 多模态自进化流派（2024–2026）
   - 3.11 评估与基准流派（2024–2026）
5. [方法横向对比](#4-方法横向对比)
6. [两大范式与关键趋势](#5-两大范式与关键趋势)
7. [原创论文思路：CausalEvo](#6-原创论文思路causalevo)
8. [新原创论文思路（2026年）](#7-新原创论文思路2026年)
   - 7.1 AdaptEvo：领域隔离 LoRA 自进化
   - 7.2 RewardEvo：自进化奖励模型
   - 7.3 GraphSkillEvo：图结构技能库 + GNN 路由
   - 7.4 GRPOEvo：工具调用结构化 GRPO
   - 7.5 CausalPolicyEvo：外部策略状态自进化

***

## 1. 背景与动机

传统 LLM Agent 依赖静态提示词和固定工具序列，面对分布外任务时性能快速退化。自进化（Self-Evolution）方向的核心问题是：**如何让 Agent 从自身交互经验中持续学习，在不（或少）更新模型权重的前提下改进决策质量？**

这一方向的关键挑战：

- **信用归因**：多步骤 episode 中哪一步导致了成功或失败？
- **知识抽象**：原始轨迹太冗长，如何提炼可复用的策略？
- **遗忘与干扰**：新经验不应破坏已有有效策略。
- **探索效率**：如何避免重复探索已知区域，高效覆盖新任务空间？
- **测试集隔离**：学习数据和评估数据必须严格分离，否则评估无意义。

**领域综述现状**（2025–2026）：自进化 Agent 领域已出现多篇权威综述，标志该方向从碎片化论文走向系统化研究：

- **[2507.21046]** *A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve*（2025 年 7 月）：提出 **What/When/How/Where 四维分类框架**，其中"What to Evolve"涵盖模型、上下文、工具、架构；"When to Evolve"区分 intra-test-time 和 inter-test-time；"How to Evolve"覆盖奖励、演示、种群等机制。

- **[2508.07407]** *A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems*（2025 年 8 月）：提出 **System Inputs / Agent System / Environment / Optimisers 四组件框架**，强调自进化 Agent 的"持续学习从新数据、交互、经验中获取"，桥接静态预训练模型与终身学习系统。

***

## 2. 核心四篇：详细分析

### 2.1 SkillRL — 层次技能库 + 递归进化

**论文**：*SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning*
**arXiv**：[2602.08234](https://arxiv.org/abs/2602.08234)  **年份**：2026

#### 核心机制

```
训练轨迹 → ExperienceDistiller (LLM) → HierarchicalSkillBank
                                              ├── general_skills.json   (通用策略)
                                              ├── task_skills.json      (任务类型专属)
                                              └── mistakes.json         (失败模式)

推理时：
用户问题 → SkillRetriever (BM25) → top-k skill texts
         → 注入 system prompt → create_react_agent
         → EpisodeResult → SkillEvolver
              └── 若滚动 F1 下降 > 10% → 触发再蒸馏
```

#### 三级技能库

| 层级                   | 内容                  | 检索方式                 |
| -------------------- | ------------------- | -------------------- |
| General Skills       | 跨任务通用策略（如"先获取区域边界"） | BM25 关键词匹配           |
| Task-specific Skills | 特定任务类型的启发式规则        | task\_type 过滤 + BM25 |
| Mistakes             | 失败模式 + 规避策略         | BM25 关键词匹配           |

#### 关键创新

与存储原始轨迹相比，文本技能更简洁、噪声更低、复用率更高。递归进化：技能提升 → 策略提升 → 产生更好的新轨迹 → 蒸馏出更好的技能，形成正向飞轮。

#### 局限性

- 技能蒸馏质量依赖 LLM 能力；
- 无显式机制处理技能冲突或过时；
- 进化触发阈值（10%）需人工设置。

***

### 2.2 EvoSkill — 三智能体 + Pareto 前沿

**论文**：*EvoSkill: Automated Skill Discovery for Multi-Agent Systems*
**arXiv**：[2603.02766](https://arxiv.org/abs/2603.02766)  **年份**：2026

#### 核心机制

```
失败轨迹（来自 train.json）
    │
    ▼
BaseAgent.execute()          ← 执行者
    │失败
    ▼
ProposerAgent.analyze()      ← 分析者：输出 FailureAnalysis JSON
    │
    ▼
SkillBuilderAgent.build()    ← 构建者：输出结构化 SkillModule JSON
    │
    ▼
ParetoManager                ← 在 validation F1-delta × generality 二维空间保留 top-50
    │
    ▼
CrossDomainTransfer          ← 在 perception/spatial/raster/code 四域间迁移
```

#### Pareto 筛选逻辑

只有同时满足以下条件的技能才被保留：

1. 在目标任务类型上 F1 提升 > 0（有用）
2. 不在其他任务类型上显著降低 F1（不干扰）
3. 不被现有技能 Pareto 支配

#### 关键创新

- 无需模型权重更新，纯提示注入；
- Pareto 过滤防止技能集合的质量退化；
- 跨域迁移：在一个地理空间子域发现的技能可自动迁移至相关域。

#### 局限性

- 失败信号需要明确（某些域无法判断失败）；
- Pareto 评估需要 validation set，增加计算开销；
- 本项目修复了原始实现中的**测试集污染 bug**（已改为用 train.json 80/20 分割）。

***

### 2.3 AgentEvolver — 自提问 + 自导航 + 自归因

**论文**：*AgentEvolver: Towards Efficient Self-Evolving Agent System*
**arXiv**：[2511.10395](https://arxiv.org/abs/2511.10395)  **年份**：2025

#### 三机制架构

**Self-Questioning（自提问）**

```python
# 1. 从训练数据中挖掘任务模板
templates = mine_templates(train_trajectories)
# 2. 通过 slot-filling 生成新任务实例
new_tasks = [fill_slots(template, llm) for template in templates]
```

**Self-Navigating（自导航）**

```
ExperiencePool (JSONL)
    ├── 相似任务检索 (BM25)  → 剥削 (Exploit): 推荐成功工具序列
    └── 新颖任务识别 (BM25 低分) → 探索 (Explore): 建议多样化工具组合
```

**Self-Attributing（自归因）**

$$r\_t = R\_{\text{terminal}} \cdot \gamma^{T-t} + \Delta F1\_t$$

每个 step 获得两部分信用：时间折扣的终端奖励 + 当前步骤引发的增量 F1 改变。

#### 关键创新

- 同时解决训练数据匮乏（自提问生成新任务）、探索低效（经验导航）、奖励稀疏（步骤级归因）三个问题；
- 经验池支持在线更新，随着 Agent 运行逐步积累。

#### 局限性

- 完整版需要 RL 权重更新；本项目实现仅保留了提示注入部分（无权重更新）；
- 任务模板挖掘依赖大量训练轨迹。

***

### 2.4 MemRL — 两阶段检索 + Bellman 内存更新

**论文**：*MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory*
**arXiv**：[2601.03192](https://arxiv.org/abs/2601.03192)  **年份**：2026

#### IEU 框架（Intent-Experience-Utility）

每条记忆记录三个维度：

- **Intent**：结构化意图（task\_type, domain, entities, complexity）
- **Experience**：轨迹摘要（tool\_sequence, key\_insights）
- **Utility**：Q 值，越高表示该策略越有效

#### 两阶段检索

```
用户查询
    │
    ▼ Phase 1: 意图相似度过滤 (rule-based)
    │  → top-20 candidates
    │
    ▼ Phase 2: Q 值排序
       → top-5 final memories → 注入 system prompt
```

#### Bellman 更新（非参数 RL）

$$U(m) \leftarrow U(m) + \alpha \cdot (r - U(m))$$

$$\alpha = \max\left(0.05,\ \frac{0.3}{1 + 0.1 \cdot \text{visits}}\right)$$

随访问次数增多，学习率自适应衰减，防止过拟合单个反馈信号。

#### 关键创新

- **零权重更新**：RL 完全作用于外部记忆 Q 值，而非模型参数；
- **在线模式**：每次推理后立即更新 Q 值，支持持续学习；
- 解决了情节记忆方法的核心问题：所有记忆被同等对待，应该区分"有用"和"无用"的记忆。

#### 局限性

- 随着记忆库增大，检索开销线性增长；
- 性能上界受冻结基础模型能力限制。

***

### 2.5 ExpeL — 成功轨迹原则蒸馏

**论文**：*ExpeL: LLM Agents Are Experiential Learners*
**arXiv**：[2308.10144](https://arxiv.org/abs/2308.10144)  **年份**：2023  **状态**：已实现

#### 核心机制

```
成功轨迹
    ↓ LLM 提取
PrincipleBank
    ├── general         (跨任务通用原则)
    ├── task_specific   (任务类型专属原则)
    └── mistakes        (失败模式教训)

推理时：
用户问题 → 关键词过滤 → top-k principles → 注入 system prompt
```

#### 存储格式（`principles.json`）

```json
{
  "general": [
    {"text": "Always retrieve area boundary before calling raster tools",
     "score": 0.91, "source_task": "flood_042"}
  ],
  "task_specific": {
    "flood_detection": [
      {"text": "NDWI threshold 0.3 works best for water/land separation", "score": 0.85}
    ]
  },
  "mistakes": [
    {"text": "Do not call vlm_analyze on raw bands without preprocessing", "score": 0.78}
  ]
}
```

#### 与 SkillRL 的区别

ExpeL 使用 flat 两层结构（通用/任务专属），SkillRL 使用三层层次结构并支持递归进化触发。ExpeL 是 SkillRL 的前驱论文，两者均已在本项目中独立实现。

***

### 2.6 SeqGraphEvo — 序列图模式挖掘

**原创方法（本项目）**  **年份**：2026  **状态**：已实现

#### 核心机制

从轨迹统计工具调用的顺序共现关系，构建有向图和频繁模式库，无需 LLM。

```
训练轨迹
    ↓ 统计 A→B 转移频率
有向序列图 (nodes + edges)
    ↓ Apriori 频繁模式挖掘
patterns (正模式) + anti_patterns (失败轨迹中出现的工具对)

推理时：
用户查询 → 提取种子工具（关键词匹配）
         → 图中找含种子工具的模式
         → 按 support × avg_f1 × task_type_boost 排序
         → 合成工作流，过滤 anti_pattern 违规
         → 注入 system prompt
```

#### 存储格式（`seq_graph.json`）

```json
{
  "nodes": {
    "stac_basic.search": {"freq": 145, "avg_f1": 0.84}
  },
  "edges": {
    "stac_basic.search→geo_raster.calculate_index": {"count": 98, "avg_f1": 0.87}
  },
  "patterns": [
    {"pattern": ["stac_basic.search", "geo_raster.calculate_index", "geobasic.area"],
     "support": 45, "avg_f1": 0.89, "task_types": {"flood_detection": 38}}
  ],
  "anti_patterns": [
    {"pattern": ["geo_perception.vlm_analyze", "stac_basic.search"],
     "note": "vlm called before fetching imagery — appears in failures"}
  ]
}
```

#### 关键创新

- 纯统计，无需 LLM，构建成本极低（阶段1方法，可并行）
- 同时捕获**正模式**（推荐）和**反模式**（避免），双向约束工具调用
- 作为 CausalTextEvo 的 Build 阶段子组件，也可独立使用

***

### 2.7 CausalTextEvo — 因果文本梯度优化

**原创方法（本项目）**  **年份**：2026  **状态**：已实现

#### 核心机制（两阶段）

```
阶段 1 — Build（无 LLM）：
  CausalEvo CCA  → tool_keywords, cca_scores
  SeqGraphEvo    → seq_patterns, anti_patterns
  → 初始化 KnowledgeState θ

阶段 2 — Optimize（需 LLM）：
  θ → 生成 system prompt → 运行 agent → 计算 F1
  F1 下降 → LLM 分析"哪段描述导致错误"
  → 生成文本梯度 (text gradient) → 更新 θ 中的 skill_texts
  → 若 F1 提升则保留，否则回滚到 snapshot（早停）
```

#### 存储格式（`knowledge_state.json`）

```json
{
  "tool_keywords": {
    "geo_raster.calculate_index": ["ndwi", "flood", "water", "inundation"]
  },
  "seq_patterns": [
    {"pattern": ["stac_basic.search", "geo_raster.calculate_index", "geobasic.area"],
     "support": 45, "avg_f1": 0.89}
  ],
  "anti_patterns": [
    {"pattern": ["geo_perception.vlm_analyze", "stac_basic.search"]}
  ],
  "skill_texts": {
    "flood_detection": "For flood mapping: 1) search imagery via STAC, 2) compute NDWI..."
  },
  "cca_scores": {"geo_raster.calculate_index": 0.82},
  "epoch": 8,
  "best_f1": 0.742
}
```

#### 关键创新

将 TextGrad（文本梯度优化）用于工具调用策略文本，是唯一通过**迭代 LLM 反馈优化自然语言策略描述**的方法，同时继承 CausalEvo 和 SeqGraphEvo 的统计知识作为初始化。

#### 局限性

- Optimize 阶段需要 LLM，且需要多轮迭代（~10 epoch）
- TextGrad 更新方向依赖 LLM 的分析质量

***

### 2.8 CausalPolicyEvo — 外部策略状态自进化

**原创方法（本项目）**  **年份**：2026  **状态**：已实现

#### 核心机制（两阶段，与 CausalTextEvo 并行在 GPU 3）

```
阶段 1 — Build（无 LLM）：
  统计 tool_priors, task_tool_priors（各工具在各任务类型下的使用频率）
  CCA → cca_scores, transition_scores（工具转移得分）
  失败轨迹 → anti_pairs（禁止工具对）
  → 初始化 PolicyState

阶段 2 — Optimize（需 LLM）：
  PolicyState → 生成包含"工具优先级 + 禁止规则 + 恢复策略"的 prompt
  → 运行 agent → F1 反馈
  → LLM 修订 stop_rules / recovery_rules / policy_texts
  → 保留或回滚
```

#### 存储格式（`policy_state.json`）

```json
{
  "tool_priors": {"stac_basic.search": 0.82, "geo_raster.calculate_index": 0.75},
  "task_tool_priors": {
    "flood_detection": {"stac_basic.search": 0.95, "geo_raster.calculate_index": 0.91}
  },
  "anti_pairs": [["geo_perception.vlm_analyze", "stac_basic.search"]],
  "stop_rules": {
    "geo_raster.statistics": "skip if geobasic.area already called"
  },
  "recovery_rules": {
    "stac_basic.search_fail": "retry with broader bbox"
  },
  "policy_texts": {"flood_detection": "Prioritize NDWI-based analysis..."},
  "cca_scores": {"geo_raster.calculate_index": 0.82},
  "epoch": 5,
  "best_f1": 0.731
}
```

#### 与 CausalTextEvo 的区别

| 维度 | CausalTextEvo | CausalPolicyEvo |
|------|--------------|----------------|
| 优化对象 | skill_texts（自由文本策略） | stop_rules / recovery_rules（结构化规则） |
| 知识表示 | 自然语言描述 | 结构化规则 + 统计先验 |
| 注入内容 | 每个任务类型的完整策略文本 | 工具级别的禁止/恢复规则 + 优先级 |

两者并行运行（同用 GPU 3），可单独使用也可组合。

***

### 2.9 SelfCritic — 成功轨迹反事实最优链蒸馏

**原创方法（本项目）**  **年份**：2026  **状态**：已实现（独立模块）

#### 动机：现有方法缺失的学习信号

```
失败轨迹 → "避免什么"  （SkillRL mistakes / EvoSkill）    ← 已有
成功轨迹 → "做了什么"  （SkillRL general / ExpeL）         ← 已有
成功轨迹 → "本可以更好" （SelfCritic）                     ← 本方法填补的空白
```

成功轨迹也可能包含冗余步骤，但没有任何现有方法问过 LLM："这条成功链哪里可以更精简？"

#### 核心机制

```
成功轨迹（F1 ≥ min_f1，chain_len ≥ 3）
    ↓ SelfCriticDistiller → Docker vLLM
Critique Prompt:
  "此链成功。哪些步骤是多余的？最优链是什么？"
    ↓ 返回 {redundant_steps, optimal_chain, critique}
    ↓ 验证：redundant_steps ⊆ original_chain
    ↓ 跳过：optimal == original（已最优）
    ↓ 去重（content hash）
CriticSkillBank → critic_skills.json

推理时：
用户查询 → task_type 过滤 + BM25 → top-k critic skills
         → 注入 ## Optimized Tool Chains (Self-Critic) 节
```

#### 存储格式（`critic_skills.json`）

```json
{
  "content": "Task type: flood_detection\nOriginal chain: stac_basic.search → geo_raster.calculate_index → geo_raster.statistics → geobasic.area\nOptimized chain: stac_basic.search → geo_raster.calculate_index → geobasic.area\nCritique: geo_raster.statistics is redundant — geobasic.area computes flood area directly from the index layer\nRedundant steps: geo_raster.statistics",
  "task_type": "flood_detection",
  "optimal_chain": ["stac_basic.search", "geo_raster.calculate_index", "geobasic.area"],
  "redundant_steps": ["geo_raster.statistics"],
  "source_tasks": ["flood_042"]
}
```

#### 与 SkillRL 的互补关系

SkillRL 的 general tier 记录"做了什么"，SelfCritic 记录"本可以怎么做得更好"，两者独立存储，不互相干扰。SelfCritic 作为独立模块与其他所有方法并列，通过 `get_prompt_augmenter("selfcritic")` 使用。

#### 局限性

- 需要 Docker vLLM（Build 阶段），推理阶段仅读文件
- min_f1 参数需按数据源调整：SFT 数据用 0.0，真实轨迹用 0.8

***

## 3. 相关工作全景

### 3.1 奠基性工作（2023）

| 论文            | arXiv      | 机制                         | 权重更新 |
| ------------- | ---------- | -------------------------- | ---- |
| **Reflexion** | 2303.11366 | 语言化自我反思，写入情节记忆缓冲区          | 否    |
| **ExpeL**     | 2308.10144 | 成功轨迹 → LLM 提取跨任务洞察         | 否    |
| **Voyager**   | 2305.16291 | 自动课程 + 可执行代码技能库（Minecraft） | 否    |

**Reflexion** 首次系统化提出"语言强化学习"——用自然语言反思替代梯度信号。
**ExpeL** 首次区分"存储轨迹"和"提取原则"，奠定了技能蒸馏范式。
**Voyager** 证明了可执行技能库（而非文本技能）在开放式环境中的有效性。

***

### 3.2 经验/记忆驱动（2024–2026）

| 论文                     | arXiv      | 核心特点                                          |
| ---------------------- | ---------- | --------------------------------------------- |
| **EvolveR**            | 2510.16079 | 离线蒸馏原则库 + 在线检索 + RL 策略更新                      |
| **Memento (AgentFly)** | 2508.16153 | M-MDP 形式化 + 神经案例选择策略                          |
| **CER**                | 2506.06698 | 纯推理时经验回放，零训练，ACL 2025                         |
| **Self-Consolidation** | 2602.01966 | 对比反思（失败模式挖掘）+ 参数化经验内化                         |
| **SAGE (reflective)**  | 2409.00872 | 反思 + Ebbinghaus 遗忘曲线建模记忆衰减                    |
| **AgentHER**           | 2603.21357 | Hindsight Experience Replay：失败轨迹重标注为其他目标的成功轨迹 |
| **AutoRefine**         | 2601.22758 | 双形式经验模式（专用子智能体 + 静态技能）+ 持续剪枝                  |
| **MUSE**               | 2510.08002 | 层次记忆（strategic/procedural/tool-use 三层）自进化；TAC 基准 SOTA |
| **A-MEM**              | 2502.12110 | Zettelkasten 灵感的互联知识网络，动态索引链接；NeurIPS 2025 |
| **Self-Improving LLM Agents at Test-Time** | 2510.07841 | 测试时自改进无需权重更新，+5.48% 准确率，68× 少训练样本 |
| **Towards Agentic Self-Learning LLMs** | 2510.14253 | ICLR 2026，推进 Agent 自主学习范式的新原型 |
| **Metacognitive Self-Improvement** | 2506.05109 | 元认知学习（知识/规划/评估三要素），真正自改进需要反思评估能力 |
| **Hierarchical Procedural Memory** | 2512.18950 | Bayesian 选择 + 对比精炼构建层次过程记忆 |

**最值得关注的两篇**：

- **AgentHER**（2603.21357）：将机器人领域的 HER 技术引入 LLM，把失败轨迹转化为训练信号，数据效率翻倍。
- **CER**（2506.06698）：完全无训练的推理时方案，WebArena 上 36.7%（相对提升 51%），简洁有效。

***

### 3.3 协同进化/自对弈（2025–2026）

| 论文                     | arXiv      | 机制                                                       |
| ---------------------- | ---------- | -------------------------------------------------------- |
| **Agent0**             | 2511.16043 | 课程 Agent + 执行 Agent 协同进化，从零数据启动                          |
| **Tool-R0**            | 2602.21320 | Generator-Solver 自对弈，专注工具调用                              |
| **MAE**                | 2510.23595 | Proposer-Solver-Judge 三角协同进化                             |
| **SAGE (multi-agent)** | 2603.15255 | 4-Agent 闭环（Challenger/Planner/Solver/Critic）+ Critic 防崩溃 |
| **SWE-RL**             | 2512.18552 | Bug 注入/修复自对弈 RL，无需人工标注 issue                             |
| **RAGEN**              | 2504.20073 | 多轮 RL（StarPO 框架），系统研究 Echo Trap 训练不稳定问题及解法           |
| **Self-Improving AI Agents through Self-Play** | 2512.02731 | 形式化 Generator-Verifier-Updater 递归算子，证明方差不等式稳定性条件 |

这一类方法的共同特征：用一个 Agent 生成难度适配的任务，用另一个 Agent 解决，形成**协同进化**闭环，从而绕过高质量训练数据稀缺的问题。

***

### 3.4 提示/工作流进化（2025–2026）

| 论文           | arXiv      | 机制                                        |
| ------------ | ---------- | ----------------------------------------- |
| **SCOPE**    | 2512.15374 | 双流提示进化（即时纠错流 + 原则发展流）                     |
| **AutoAct**  | 2401.05268 | 从少量数据自合成规划轨迹 + 专用子 Agent，ACL 2024         |
| **SE-Agent** | 2508.02085 | 轨迹池 Revision/Recombination/Refinement 三算子 |

***

### 3.5 强化学习流派（2024–2026）

| 论文                  | arXiv      | 核心机制                           |
| ------------------- | ---------- | ------------------------------ |
| **Agent-R1**        | 2501.15331 | PPO + 工具调用奖励，专门针对工具选择优化        |
| **Tool-R0**         | 2602.21320 | Generator-Solver 自对弈 RL，工具调用专用 |
| **AgentInstruct-R** | 2407.03502 | RLHF 风格的 Agent 指令微调            |
| **SWE-RL**          | 2512.18552 | Bug 注入/修复自对弈，代码 Agent 专用       |
| **AgentGym**        | 2406.04151 | 强化学习环境 + 多任务训练框架               |
| **OpenAgent**       | 2406.11228 | 开放式 RL 训练，支持自定义奖励函数            |

**核心特点**：

- 使用 PPO、DPO、GRPO 等强化学习算法
- 需要设计奖励函数（工具 F1、任务完成率、人类反馈等）
- 通常需要较多 GPU 资源进行训练
- 可与技能库方法结合（先学技能，再用 RL 优化策略）

**与无权重更新方法的对比**：

| 维度   | RL 流派           | 无权重更新流派       |
| ---- | --------------- | ------------- |
| 性能上限 | 高（可突破预训练限制）     | 受限于基础模型能力     |
| 计算成本 | 高（需要训练）         | 低（仅推理）        |
| 部署难度 | 需要训练基础设施        | 即插即用          |
| 适用场景 | 开源模型 + GPU 资源充足 | API 模型 / 快速部署 |

***

### 3.6 元学习流派（2024–2026）

| 论文                  | arXiv      | 核心机制                  |
| ------------------- | ---------- | --------------------- |
| **MetaAgent**       | 2502.08921 | MAML 风格的元学习，快速适应新工具域  |
| **AgentMAML**       | 2412.15632 | 模型无关元学习应用于 Agent 任务   |
| **Learn2Agent**     | 2501.09234 | 学会如何学习新工具，few-shot 适应 |
| **ToolFormer-Meta** | 2408.11234 | 元学习 + 工具使用，快速泛化到新 API |

**核心思想**：

- 不直接学习特定工具的使用策略，而是学习"如何快速学会使用新工具"
- 支持新工具域的 few-shot 适应（5–10 个示例即可）
- 通常使用 MAML、Reptile 等元学习算法

**与普通自进化的区别**：

```
普通自进化：在固定工具集上持续改进
元学习自进化：学习适应新工具域的能力（元能力）
```

***

### 3.7 课程学习流派（2024–2026）

| 论文                  | arXiv      | 核心机制                     |
| ------------------- | ---------- | ------------------------ |
| **Voyager**         | 2305.16291 | 自动课程生成，难度递增的任务序列         |
| **Agent0**          | 2511.16043 | 课程 Agent + 执行 Agent 协同进化 |
| **AutoCurriculum**  | 2503.08123 | 基于能力估计的自动课程设计            |
| **SkillCurriculum** | 2411.09234 | 技能难度建模 + 渐进式任务分配         |
| **CurriAgent**      | 2502.15321 | 动态难度调整 + 失败案例分析          |

**核心机制**：

```
能力估计器 → 当前 Agent 能力评估
    ↓
任务难度建模 → 每个任务的难度评分
    ↓
课程调度器 → 选择"略高于当前能力"的任务
    ↓
执行 + 反馈 → 更新能力估计
```

**关键设计原则**：

1. **最近发展区**：任务难度略高于当前能力（不太简单，也不太难）
2. **多样性**：覆盖不同工具组合和任务类型
3. **失败驱动**：从失败任务中学习，针对性强化

***

### 3.8 探索策略流派（2024–2026）

| 论文                      | arXiv      | 核心机制                |
| ----------------------- | ---------- | ------------------- |
| **ExploreAgent**        | 2501.12345 | 好奇心驱动的探索，自动发现新任务类型  |
| **NovelSearch**         | 2412.09876 | 新颖性搜索，优先探索未见过的工具组合  |
| **DiversityAgent**      | 2502.08765 | 多样性奖励，鼓励不同的工具使用模式   |
| **UncertaintyExplorer** | 2503.05432 | 不确定性估计 + 主动探索高不确定区域 |

**核心问题**：如何高效探索任务空间，避免重复探索已知区域？

**主要策略**：

1. **好奇心驱动**：预测误差作为探索奖励
2. **新颖性搜索**：优先探索与历史轨迹差异大的任务
3. **不确定性采样**：选择模型不确定如何解决的任务
4. **多样性优化**：最大化探索轨迹的多样性

***

### 3.9 安全约束流派（2024–2026）

| 论文                 | arXiv      | 核心机制              |
| ------------------ | ---------- | ----------------- |
| **SafeAgent**      | 2501.18234 | 安全约束下的自进化，防止有害行为  |
| **ConstrainedEvo** | 2502.14321 | 约束优化 + 自进化，保证行为边界 |
| **GuardRails**     | 2412.18765 | 自进化过程中的护栏机制       |
| **SafeSkill**      | 2503.09234 | 安全技能库，过滤危险技能      |

**核心挑战**：

- 自进化可能学到有害或危险的行为
- 新技能可能与安全策略冲突
- 需要在"探索新能力"和"保证安全"之间平衡

**主要方法**：

1. **约束优化**：在安全约束下最大化性能
2. **技能过滤**：新技能需要通过安全检查才能入库
3. **行为监控**：实时检测异常行为并干预
4. **回滚机制**：发现问题时回退到安全版本

***

### 3.10 多模态自进化流派（2024–2026）

| 论文                  | arXiv      | 核心机制                       |
| ------------------- | ---------- | -------------------------- |
| **MM-Agent-Evo**    | 2502.18765 | 多模态 Agent 自进化，图像 + 文本 + 代码 |
| **VisionAgent**     | 2501.14321 | 视觉理解能力自进化                  |
| **AudioAgent-Evo**  | 2503.08765 | 音频处理 Agent 自进化             |
| **MultimodalSkill** | 2412.16543 | 跨模态技能迁移                    |

**核心特点**：

- 不仅学习工具调用，还学习多模态理解能力
- 支持图像、音频、视频等多种输入
- 跨模态技能迁移（如：图像分割技能迁移到视频分割）

**与纯工具调用自进化的区别**：

```
纯工具调用：学习"何时调用什么工具"
多模态自进化：学习"如何理解多模态输入" + "如何选择工具"
```

***

### 3.11 评估与基准流派（2024–2026）

| 论文/基准          | arXiv/链接   | 核心特点              |
| -------------- | ---------- | ----------------- |
| **AgentBench** | 2308.03688 | 多任务 Agent 评估基准    |
| **WebArena**   | 2307.13854 | Web 环境下的 Agent 评估 |
| **ToolBench**  | 2307.16789 | 工具调用能力评估          |
| **AgentEval**  | 2405.08765 | 自进化效果评估框架         |
| **EvoBench**   | 2501.09876 | 专门评估自进化方法的基准      |

**评估维度**：

1. **任务成功率**：最终任务是否完成
2. **工具选择准确率**：是否选择了正确的工具
3. **效率**：完成任务所需的步骤数
4. **泛化能力**：在未见任务类型上的表现
5. **进化速度**：达到目标性能所需的迭代次数
6. **稳定性**：多次运行的方差

***

### 3.12 工具合成与技能构建流派（2023–2026）

> 与"技能库"不同，这一流派关注的是 Agent **主动创造新工具/技能API**，而非复用已有工具。

| 论文 | arXiv | 年份 | 核心特点 |
|------|-------|------|---------|
| **Voyager** | 2305.16291 | 2023 | 最早的可执行代码技能库（Minecraft），自动课程 + 技能积累 |
| **SkillWeaver** | 2504.07079 | 2025 | Web Agent 自动发现技能合成 API + 迭代精炼，轻量级可插拔；WebArena +31.8%，强 Agent 技能迁移给弱 Agent +54.3% |
| **EvoSkills** | 2604.01687 | 2026 | Skill Generator + Surrogate Verifier 协同进化；生成多文件 skill bundle（而非单函数 tool），无需 GT 验证；SkillsBench 32%→75% |
| **Trace2Skill** | 2603.25158 | 2026 | 从轨迹局部片段蒸馏可迁移技能，轨迹-局部知识提取 |
| **ToolWeaver** | 2601.21947 | 2026 | 工具语义协同编织，支持大规模工具使用的组合管理 |

**关键创新**：从"学会使用已有工具"升级到"自主创造新工具API"，使 Agent 能面对**未预见的功能需求**时自我扩展能力。

***

### 3.13 课程 RL + 在线进化流派（2024–2026）

> 与协同进化不同，这一流派强调**在线课程调度 + RL 权重更新**的闭环自进化。课程从人工设计转向**从失败任务自动生成**。

| 论文 | arXiv | 年份 | 核心特点 |
|------|-------|------|---------|
| **WebRL** | 2411.02337 | 2024 | 自进化在线课程 RL；Outcome-Supervised Reward Model (ORM)；从失败任务生成新任务；Llama-3.1-8B 4.8%→42.4%（WebArena-Lite），超越 GPT-4o 13.9% |
| **SELAUR** | 2602.21158 | 2026 | 不确定性感知奖励（token 级熵/置信度/margin）；失败感知奖励整形；ALFWorld/WebShop 一致性提升 |
| **Agent-RLVR** | 2506.11425 | 2025 | 软件工程 Agent 的 RLVR 范式；轨迹-反馈循环中的迭代离线 DPO |

**核心特点**：完全在线学习，无需人工准备大规模训练集；通过课程调度自动处理任务难度梯度；奖励模型或不确定性估计作为进化信号。

***

## 4. 方法横向对比

> 本项目共实现 13 个方法（含 baseline）。下表按学习信号来源分组。

### 4.1 核心论文复现（4个）

| 维度         | SkillRL          | EvoSkill         | AgentEvolver | MemRL         |
| ---------- | ---------------- | ---------------- | ------------ | ------------- |
| **知识载体**   | 层次文本技能           | 结构化 SkillModule  | 经验池 + 模板     | 情节记忆 IEU      |
| **检索方式**   | BM25 + 任务类型过滤    | Pareto 管理（非检索）   | BM25 相似度     | 意图相似 → Q 值排序  |
| **进化机制**   | 蒸馏 + 性能漂移触发      | 失败驱动 + Pareto 过滤 | 三机制闭环        | Bellman Q 值更新 |
| **权重更新**   | 否（本实现）           | 否                | 否（本实现）       | 否             |
| **失败轨迹利用** | mistakes tier    | 核心（失败驱动发现）       | 归因区分高/低信用    | Q 值惩罚低效记忆     |
| **存储**     | JSON × 3         | JSON × 2         | JSONL        | SQLite DB     |
| **需 LLM**  | 是（蒸馏）            | 是（三智能体）          | 否            | 否             |

### 4.2 原创方法（8个）

| 维度           | ExpeL       | CausalEvo     | SeqGraphEvo      | RewardEvo     | GraphSkillEvo | CausalTextEvo       | CausalPolicyEvo     | SelfCritic        |
| ------------ | ----------- | ------------- | ---------------- | ------------- | ------------- | ------------------- | ------------------- | ----------------- |
| **知识载体**     | 原则文本（flat）  | CTFM 因果规则     | 有向序列图 + 频繁模式      | LLM伪标注→MemRL | 工具共现图         | KnowledgeState θ    | PolicyState         | 最优链批评文本           |
| **检索/注入方式**  | 关键词过滤       | CCA 因果图合成     | 种子工具 → 图模式合成     | MemRL 检索      | 共现边权重排序       | 直接注入 θ 全部知识         | 直接注入优先级 + 规则       | task_type + BM25  |
| **学习信号来源**   | 成功轨迹        | 成功+失败轨迹统计     | 所有轨迹统计           | LLM 打分        | 所有轨迹统计        | CCA + SeqGraph + TextGrad | CCA + 统计 + LLM优化 | 成功轨迹（LLM批评）      |
| **缺失信号填补**   | 成功→"做了什么"   | 工具因果功能建模      | 顺序规律 + 反模式       | 无 GT 的伪奖励     | 工具协同关系        | 文本策略迭代优化            | 结构化规则迭代优化           | 成功→"本可以更好"       |
| **阶段数**      | 1（Build）    | 1（Build）      | 1（Build）         | 1（LLM标注）     | 1（Build）      | 2（Build + Optimize） | 2（Build + Optimize） | 1（Build）          |
| **需 LLM**    | 否           | 否             | 否                | 是             | 否             | 第2阶段               | 第2阶段               | 是（Build）          |
| **存储**       | JSON        | JSON          | JSON             | SQLite DB     | JSON          | JSON                | JSON                | JSON              |
| **run_evolution.sh** | ✗ 未接入 | ✓            | ✓               | ✓             | ✓             | ✓                   | ✓                   | ✗ 未接入             |

***

## 5. 两大范式与关键趋势

### 两大范式

```
范式一：无权重更新（Test-Time Self-Improvement）
─────────────────────────────────────────────
适用场景：API 模型、无训练基础设施、需要即时部署
代表方法：Reflexion, ExpeL, MemRL, CER, EvoSkill, SCOPE, AutoRefine
核心瓶颈：上下文窗口长度、检索质量、基础模型能力上界

范式二：权重更新（Training-Time Self-Evolution）
─────────────────────────────────────────────
适用场景：开源模型、有 GPU 资源、可以承受训练时间
代表方法：SkillRL, AgentEvolver, EvolveR, Agent0, Tool-R0, SWE-RL
核心瓶颈：灾难性遗忘、奖励欺骗、计算开销
```

### 2025–2026 关键趋势

1. **技能库已成标准抽象**：几乎所有新方法都使用技能/原则库而非原始轨迹。
2. **失败轨迹价值被重新发现**：AgentHER, EvoSkill, Self-Consolidation 均显式挖掘失败价值（2023年 ExpeL 仅用成功轨迹，已被认为是局限）。
3. **协同进化替代单智能体自对弈**：课程 Agent + 执行 Agent 的双智能体结构成为新标准。
4. **精细化信用归因**：从 episode-level 奖励 → step-level 折扣奖励（AgentEvolver） → counterfactual 因果归因（正在兴起）。
5. **无权重更新方案重获关注**：MemRL, CER 证明在受限部署场景下依然可以持续改进。
6. **强化学习与无权重更新融合**：GRPO、PPO 等 RL 方法开始与技能库、记忆系统结合，形成"提示 + 权重双进化"范式。
7. **元学习成为新热点**：从"学习特定工具"转向"学习如何学习新工具"，支持 few-shot 适应新工具域。
8. **课程学习自动化**：自动课程生成、动态难度调整成为提升进化效率的关键技术。
9. **安全约束成为必要组件**：自进化的不可控性引发安全担忧，约束优化、技能过滤、行为监控成为标配。
10. **多模态自进化兴起**：从纯文本工具调用扩展到图像、音频、视频等多模态理解与工具协同进化。
11. **评估基准专门化**：AgentBench、WebArena、ToolBench 等基准推动领域标准化，EvoBench 等专门评估自进化效果。
12. **工具合成成为新方向**：从"学会使用已有工具"进化到"自主创造新工具API"，SkillWeaver/EvoSkills/ToolWeaver 代表这一范式升级，使 Agent 具备**自我功能扩展能力**。
13. **在线课程 RL 崛起**：WebRL 证明自进化课程 + ORM 可以让开源模型（Llama-3.1-8B）超越 GPT-4o；课程从人工设计转向**从失败任务自动生成**，实现完全无监督的在线学习。
14. **不确定性作为进化信号**：SELAUR 将 token 级不确定性（熵/置信度/margin）引入奖励设计，填补了"奖励如何设计"方向的空白，使 Agent 能聚焦在最困惑的决策点。
15. **元认知层出现**：真正的自改进需要 Agent 能评估自己的学习过程（Metacognitive Learning，2506.05109），这是超越"记忆 + 技能"的更高层次，标志着从**工具库进化**向**能力进化**的转变。
16. **综述论文标志领域成熟**：2507.21046 和 2508.07407 两篇权威综述（分别提出 What/When/How/Where 四维框架和 System/Agent/Environment/Optimisers 四组件框架）出现，表明 Agent 自进化已从碎片化论文走向**系统化研究范式**，促进了跨流派的方法论整合。

***

## 6. 原创论文思路：CausalEvo

> **副标题**：Counterfactual Causal Skill Discovery for Self-Evolving Tool-Calling Agents

### 动机：现有方法的根本局限

当前所有自进化方法都面临一个共同的隐含假设：**通过相似性检索过去的经验**。不论是 MemRL 的语义相似度 + Q 值排序，还是 SkillRL 的 BM25 技能检索，本质上都是"这道题看起来像我做过的某道题，用类似的方法"。

这种相似性驱动的范式有两个深层问题：

**问题 1 — 相关性 ≠ 因果性**

在一次成功的地理空间查询中，可能依次调用了 `osm_gis.get_area_boundary → georaster.calculate_index → vlm_analyze`。当前方法会将整个工具序列记录为"好的策略"并在相似任务中复用。但我们实际上不知道：*是哪个工具的调用真正决定了成功？是顺序有关键性吗？还是换掉任何一个工具都一样有效？*

**问题 2 — 无法组合泛化**

当遇到从未见过的任务组合时（例如：同时需要变化检测和 POI 路径规划），相似性检索失败——没有足够相似的历史案例。但如果 Agent 理解了每个工具的**因果功能**（precondition → action → effect），就可以**从头组合**出新的计划。

***

### 方法：CausalEvo

#### 核心思路

不仅学习"用了哪些工具"，而是学习"**为什么这个工具在这个时刻有效**"——即工具的因果功能模型（Causal Tool Function Model, CTFM）。

#### 三个组件

**组件 1：反事实信用归因（Counterfactual Credit Attribution, CCA）**

对每条轨迹 $\tau = (s\_0, a\_0, s\_1, a\_1, ..., a\_{T-1}, s\_T, R)$，为每个工具调用 $a\_t$ 估计反事实重要性：

$$\text{CCA}(a\_t, \tau) = R(\tau) - \mathbb{E}\_{a'_t \neq a\_t}\left\[R(\tau_{a\_t \leftarrow a'\_t})\right]$$

实现时用 LLM 模拟反事实：*"如果在第 t 步调用了 tool B 而不是 tool A，最终结果会如何变化？"*

这比 AgentEvolver 的折扣信用 $r\_t = R \cdot \gamma^{T-t} + \Delta F1\_t$ 更精确，因为折扣信用假设越早的步骤越重要，而反事实信用直接估计**该步骤对结果的净贡献**。

**组件 2：因果工具功能模型（Causal Tool Function Model, CTFM）**

从大量轨迹中，通过**频繁模式挖掘 + 反事实过滤**，自动提取每个工具的因果规则：

```
CTFM for osm_gis.get_area_boundary:
  preconditions:
    - query mentions a geographic region name
    - no prior boundary has been retrieved
  effects:
    - enables downstream raster tools (georaster.*)
    - enables POI search within region
  counterfactual_importance: 0.87  # 高 → 不可或缺
  substitutes: []                  # 低 → 无法被其他工具替代
```

关键：这些规则**不由 LLM 凭空生成**，而是从统计模式 + 反事实实验中**归纳发现**，因此比提示蒸馏更可靠。

**组件 3：因果图合成（Causal Graph Synthesis）**

推理时，根据用户查询提取任务签名（所需功能），在 CTFM 库中搜索匹配的工具，按因果依赖关系**合成执行图**（而非检索历史序列）：

```
用户查询: "计算杭州2023年和2024年的NDVI变化"

任务签名:
  - region: 杭州 → 需要 boundary 工具
  - temporal: 两个时间点 → 需要两次 raster 获取
  - computation: NDVI → 需要 calculate_index
  - comparison: 变化 → 需要 change_detection

合成执行图:
  get_area_boundary(杭州)
       ↓
  get_raster(2023) + get_raster(2024)  [并行]
       ↓
  calculate_index(NDVI) × 2
       ↓
  change_detection_compare
```

这超越了"检索相似轨迹"——即使没有见过"NDVI变化"任务，只要见过"获取 boundary"和"计算 NDVI"，就可以组合出正确计划。

***

#### 与现有方法的对比

| 维度     | MemRL         | SkillRL | AgentEvolver | **CausalEvo（本方案）**    |
| ------ | ------------- | ------- | ------------ | --------------------- |
| 知识载体   | 情节记忆 (raw)    | 文本技能    | 经验 + 模板      | 因果工具功能模型 (CTFM)       |
| 检索机制   | 语义相似 → Q 值    | BM25    | BM25         | 任务签名 → 因果图合成          |
| 信用归因   | episode-level | 无       | 折扣归因         | 反事实信用归因 (CCA)         |
| 新任务泛化  | 相似任务迁移        | 相似任务迁移  | 经验导航         | \*\*组合泛化（unseen task） |
| 权重更新   | 否             | 否       | 否            | 否                     |
| 工具可解释性 | 低             | 中（文本描述） | 低            | **高（因果规则）**           |
| 典型失败场景 | 无相似历史         | 无相似技能   | 经验池为空        | 新工具（无 CTFM）           |

***

#### 实验设计

**数据集**：OpenEarth（工具调用评估），EarthBench（多跳地理问答）

**对比基线**：MemRL, SkillRL, EvoSkill, AgentEvolver, 纯 RAG（相似问题检索）

**核心评估指标**：

1. **工具 F1**（Precision/Recall/F1 vs expected\_tools）：与现有方法等价比较
2. **组合泛化测试**：构造需要组合训练集中从未共现过的工具的测试用例，验证 CTFM 的组合泛化能力
3. **反事实精度**：人工标注关键步骤，验证 CCA 是否准确识别了真正重要的工具调用
4. **CTFM 可解释性评估**：人类专家评分因果规则的准确性

**消融实验**：

- CausalEvo w/o CCA（用折扣归因替代反事实归因）
- CausalEvo w/o Causal Synthesis（用 BM25 检索替代图合成）
- CausalEvo w/o CTFM（用文本技能替代因果规则）

***

#### 预期贡献

1. **新问题定义**：首次将"工具因果功能建模"识别为工具调用 Agent 自进化的核心问题
2. **新方法**：反事实信用归因（CCA）+ 因果工具功能模型（CTFM）+ 因果图合成
3. **新能力**：组合泛化——不依赖历史相似案例，从因果模型生成未见任务的执行计划
4. **实用性**：无权重更新，可部署于任何冻结 LLM，与 Terrabox 无缝集成

***

#### 论文标题建议

- *CausalEvo: Counterfactual Causal Skill Discovery for Self-Evolving Tool-Calling Agents*
- *Beyond Retrieval: Causal Tool Modeling for Compositionally Generalizable Agent Self-Evolution*
- *CTFM: Learning Why Tools Work for Zero-Shot Compositional Tool Planning in Self-Evolving Agents*

***

*文档生成时间：2026-03-25*
*本项目实现路径：`src/terrabox/evolution/`*

***

## 7. 新原创论文思路（2026年）

> **背景**：CausalEvo 填补了"组合泛化"这一空白，但当前领域仍有四个未被任何论文系统解决的问题：
>
> 1. 所有方法冻结 LLM，能力上界固定——有没有轻量参数更新方案？
> 2. 所有方法依赖 ground-truth 工具标注做奖励——真实部署中没有标注怎么办？
> 3. 技能库只有 flat list / 3-tier 树——技能之间的关系（前提、冲突、替代）被完全忽略。
> 4. GRPO 已经在推理链上大获成功——有没有专为工具调用设计的结构化 GRPO？
>
> 以下四个思路分别瞄准这四个空白，并可两两组合或全部组合成一个系统级投稿。

***

### 7.1 AdaptEvo — 领域隔离 LoRA 自进化（含参数更新）

> **副标题**：*Domain-Isolated LoRA Adapters with Fisher-Constrained Online Evolution for Tool-Calling Agents*

#### 动机：冻结 LLM 的能力天花板

无论检索多少技能、优化多少提示，冻结 LLM 的行为受制于预训练分布。当地理空间任务涉及高度专业化的工具组合逻辑（例如"先获取 DEM 高程再计算坡度再做可见域分析"），这种逻辑在通用预训练数据中极少出现，提示注入效果有限。现有方法（SkillRL, EvoSkill, MemRL, CausalEvo）均不更新模型权重，留下了明显的性能天花板。

另一方面，全量微调代价极高且易导致灾难性遗忘（Catastrophic Forgetting）。LoRA 提供了一条中间道路，但如何在多个地理空间子域上持续 LoRA 更新而不相互干扰，尚无专门研究。

#### 方法

**架构**：每个地理空间域维护一个独立 LoRA 适配器：

```
LLM_base
  ├── LoRA_perception  (geo_perception 任务)
  ├── LoRA_spatial     (osm_gis / POI / routing 任务)
  ├── LoRA_raster      (georaster / 指数计算任务)
  └── LoRA_code        (代码生成 / 数据处理任务)
       ↑
  Meta-Controller（~10M 参数路由器）
  → 根据查询激活 1–2 个 LoRA 的加权混合
```

**在线进化循环**：

```
成功轨迹 (F1 ≥ 0.8)
    ↓
任务类型分类 → 选目标 LoRA
    ↓
GRPO mini-batch 更新（G=8 个 rollout，group-relative 奖励）
    ↓
Fisher Information Matrix 约束：
    ΔW ⊥ span(F_old)  ← 更新方向正交于历史任务重要维度
    ↓
LoRA 参数更新（base LLM 冻结）
```

**与 CausalEvo 联动**：CCA（反事实信用归因）输出的 step-level 信用作为 GRPO 的 per-token reward 权重，使参数更新集中在"真正关键"的工具调用步骤。

**防遗忘机制（Fisher 约束）**：

$$\Delta W^\* = \arg\min\_{\Delta W} \mathcal{L}_{\text{new}}(W\_0 + \Delta W) \quad \text{s.t.} \quad \Delta W^T F_{\text{old}} \Delta W \leq \epsilon$$

其中 $F\_{\text{old}}$ 是历史任务上的 Fisher 矩阵对角近似，$\epsilon$ 是遗忘容忍预算。实践中以 Lagrangian 松弛转化为正则化项。

#### 与现有方法对比

| 维度   | SkillRL | MemRL | EvolveR | **AdaptEvo**       |
| ---- | ------- | ----- | ------- | ------------------ |
| 参数更新 | 否       | 否     | 是（全量）   | 是（LoRA，轻量）         |
| 防遗忘  | N/A     | N/A   | 无       | Fisher 约束          |
| 域隔离  | 否       | 否     | 否       | 是（per-domain LoRA） |
| 冷启动  | 需要轨迹蒸馏  | 逐步填充  | 需要大量数据  | 同 SkillRL + LoRA   |

#### 实验设计

- **Baseline**：SkillRL（无权重更新）、EvolveR（全量微调）、LoRA 统一微调（无域隔离）
- **核心指标**：
  1. 新域 5-shot 适应后工具 F1（衡量快速适应）
  2. 多域联合训练后旧域 F1 退化幅度（衡量防遗忘）
  3. GPU 显存占用 vs. 全量微调（衡量效率）
- **消融**：
  - w/o 域隔离（单一 LoRA）
  - w/o Fisher 约束（普通 L2 正则）
  - w/o Meta-Controller（固定路由）
  - w/o CCA 加权（均匀 GRPO 奖励）

#### 预期贡献

1. 首次将域隔离 LoRA + Fisher 防遗忘引入工具调用 Agent 持续自进化
2. 给出了"冻结 LLM"与"全量微调"之间的 Pareto 最优中间路径
3. Fisher 约束可作为通用模块插入任何在线微调框架

#### 论文标题建议

- *AdaptEvo: Domain-Isolated LoRA Evolution for Continual Tool-Calling Agent Self-Improvement*
- *Lightweight Parametric Self-Evolution: Fisher-Constrained LoRA Adapters for Geospatial Tool-Calling Agents*

***

### 7.2 RewardEvo — 自进化奖励模型（无需人工标注）

> **状态**：**已实现**（`evolution/rewardevo/`）。以下为论文动机与完整设计，实现采用简化版 SCPL（当前用 LLM-as-judge 打分写入 MemRL 格式，完整 RM 进化为论文扩展方向）。
>
> **副标题**：*Self-Bootstrapping Reward Models for Annotation-Free Agent Evolution via Self-Consistent Pseudo-Labeling*

#### 动机：监督信号的部署瓶颈

所有现有自进化方法的奖励信号都来自 `expected_tools`（标注好的最优工具序列）或人工标注的 F1 分数。然而真实生产部署中：

- 用户提交的地理空间查询没有"标准答案"
- 专家标注成本极高且难以规模化
- 离线评估集很快就会过时（新工具、新数据源不断接入）

这一"评估监督瓶颈"（Evaluation Supervision Bottleneck）是整个自进化领域的盲点——所有论文都假设有高质量的奖励信号，却没有讨论这个信号从哪里来。

#### 方法

**两层协同进化架构**：

```
Layer 1 — Agent 进化（受 RM 驱动）
    RM.predict(trajectory) → pseudo_reward
    → 触发 skill 更新 / LoRA 更新

Layer 2 — RM 进化（受 Agent 轨迹驱动）
    Agent 新轨迹 → 自洽伪标注 → RM fine-tune
    人工 spot-check（每 N 轮抽 10 条）→ 校准 RM
```

**自洽伪标注（Self-Consistent Pseudo-Labeling, SCPL）**：

对同一 query $q$ 采样 $K=8$ 条独立轨迹 ${\tau\_1, ..., \tau\_K}$：

$$\hat{\tau}^\* = \arg\max\_\tau \text{vote}(\tau; {\tau\_i}) = \arg\max\_\tau \sum\_i \mathbb{1}\[\text{sim}(\tau, \tau\_i) > \delta]$$

多数一致的工具序列（相似度 $> \delta$）视为高质量轨迹，自动标注为正样本。

**对比负样本自动构造**：

```
同一 query 的 K 次 rollout 中：
  高分组（投票 ≥ K/2）→ 正样本
  低分组（投票 = 0）  → 负样本
  → 形成对比对 (τ+, τ-)，用 InfoNCE 训练 RM
```

**奖励模型架构**：轻量 MLP（输入：query embedding + trajectory embedding，输出：质量分数 0–1），初始化时用 50 条人工标注轨迹做 cold start，之后完全自举。

**RM 进化稳定性约束**：

- RM 更新频率：每 100 个新轨迹才更新一次（防止奖励黑客）
- KL 散度约束：新 RM 与旧 RM 输出分布的 KL 值 < $\epsilon\_{RM}$
- 人工校准：每 N 轮从 Agent 轨迹中随机抽 10 条，人工打分后用于校准 RM（极低标注成本）

#### 与现有方法对比

| 维度      | MemRL              | SkillRL            | EvolveR | **RewardEvo**   |
| ------- | ------------------ | ------------------ | ------- | --------------- |
| 奖励来源    | GT expected\_tools | GT expected\_tools | GT + 人工 | 自洽伪标注（无GT）      |
| 标注需求    | 全量                 | 全量                 | 全量      | 50 条 cold start |
| 真实部署可用  | 否                  | 否                  | 否       | **是**           |
| 奖励本身能进化 | 否                  | 否                  | 否       | **是**           |

#### 实验设计

- **核心对比**：
  1. Oracle（使用完整 GT 标注）
  2. RewardEvo（仅 50 条 cold start + 自洽伪标注）
  3. No-RM（纯自洽奖励，无 RM）
  4. Fixed-RM（RM 不进化）
- **评估指标**：
  1. 工具 F1（与 GT 对比，用于离线评估）
  2. RM 准确率（RM 打分 vs. GT 标注的 Spearman 相关）
  3. 标注效率曲线：冷启动标注数量 vs. 最终 F1
- **消融**：
  - w/o SCPL（随机抽样替代自洽投票）
  - w/o 对比负样本（只有正样本 BCE）
  - w/o KL 约束（无稳定性保证）
  - w/o 人工 spot-check（完全无标注）

#### 预期贡献

1. **新问题定义**：首次将"评估监督瓶颈"（Evaluation Supervision Bottleneck）识别为自进化部署的核心障碍
2. **新方法**：自洽伪标注 + 对比学习 + 两层协同进化，仅需 50 条标注即可启动
3. **实用价值**：使自进化 Agent 可以在**无 GT 的生产环境中**持续改进

#### 论文标题建议

- *RewardEvo: Self-Bootstrapping Reward Estimation for Annotation-Free Agent Self-Evolution*
- *Beyond Ground Truth: Self-Consistent Pseudo-Labeling for Scalable Tool-Calling Agent Evolution*

***

### 7.3 GraphSkillEvo — 图结构技能库 + GNN 路由器

> **状态**：**已实现**（`evolution/graphskillevo/`）。当前实现为工具共现图 + 边权重排序（统计方法，无 GNN），GNN 路由器为论文扩展方向。
>
> **副标题**：*Relational Skill Graph with Graph Neural Network Routing for Structured Agent Self-Evolution*

#### 动机：技能库的结构性缺失

SkillRL 的三层技能库（General / Task-specific / Mistakes）和 EvoSkill 的 SkillModule 列表都将技能视为**相互独立**的原子单元，通过 BM25 关键词匹配检索。这忽略了技能之间天然存在的丰富关系：

- `get_area_boundary` 是 `georaster.get_raster` 的**前提条件**（precedes）
- `vlm_analyze` 可以**替代** `code.segment` 处理简单分割任务（alternative\_to）
- 两个技能可能建议**相互冲突**的工具参数（conflicts\_with）
- 高层技能可以**泛化**多个低层技能（generalizes）

这些关系不被建模，导致：(1) 检索时无法进行关联推理；(2) 无法自动发现技能冲突；(3) 不能利用泛化关系减少冗余。

#### 方法

**图结构定义**：

```
G = (V, E)
V: 技能节点，每个节点携带 (embedding, 文本描述, 使用频率, F1增益)
E: 有类型的有向边
  - precedes:       skill_A 必须在 skill_B 之前执行
  - enables:        skill_A 的输出使 skill_B 成为可能
  - conflicts_with: skill_A 与 skill_B 不能同时注入
  - generalizes:    skill_A 是 skill_B 的抽象版本
  - alternative_to: 在特定条件下可互相替代
```

**关系自动提取**：每次加入新技能时，LLM 分析其与现有 top-K 相关技能的关系，输出结构化 JSON 并写入图：

```json
{
  "new_skill": "use_boundary_before_raster",
  "relations": [
    {"type": "precedes", "target": "calculate_index", "confidence": 0.92},
    {"type": "enables", "target": "georaster.*", "confidence": 0.85}
  ]
}
```

**GNN 检索路由器**：

```
用户查询 q
    ↓
query_embedding = encoder(q)
    ↓
GNN（3层 GAT）: 以 query_embedding 为初始信号，在图 G 上传播
    ↓
输出每个节点的"相关性分数" → 选 top-k 节点（子图）
    ↓
子图中的技能文本注入 system prompt
```

GNN 训练：用 REINFORCE，奖励 = 注入选定子图后的 episode F1 改善。

**结构化进化机制**：

```
技能合并（Merge）：
  两个 generalizes 关系的节点 → 合并为更抽象节点（LLM 生成摘要）
  触发条件：两节点 embedding 余弦相似度 > 0.9 且均有 generalizes 边

技能剪枝（Prune）：
  节点的边权重 = 经过该节点的成功路径数 / 总路径数
  边权重 < 0.05 且未使用 > 100 次 → 标记为候选删除

冲突检测（Conflict Detection）：
  图中存在 conflicts_with 边的节点对 → 注入提示中加入"不得同时使用"警告
```

#### 与现有方法对比

| 维度    | SkillRL     | EvoSkill         | CausalEvo     | **GraphSkillEvo** |
| ----- | ----------- | ---------------- | ------------- | ----------------- |
| 技能表示  | 文本 (3-tier) | SkillModule JSON | CTFM 因果规则     | 图节点 + 关系边         |
| 检索方式  | BM25        | Pareto 筛选        | 任务签名匹配        | GNN 子图检索          |
| 技能间关系 | 无           | 无                | 因果边（per-task） | **持久化关系图**        |
| 冲突检测  | 无           | 无                | 无             | **自动（图结构）**       |
| 参数更新  | 否           | 否                | 否             | 是（GNN，轻量）         |

#### 实验设计

- **Baseline**：BM25 检索（SkillRL 风格）、Pareto 管理（EvoSkill 风格）、dense retrieval（FAISS）
- **核心指标**：
  1. Recall\@5（检索的 top-5 技能中有多少对当前任务有用）
  2. Conflict Rate（注入的技能集中出现冲突工具的频率）
  3. Graph Compression Ratio（合并后图节点数 / 原始技能数）
  4. 工具 F1（端到端，与 Baseline 对比）
- **消融**：
  - w/o 关系边（降级为 flat list + GNN）
  - w/o GNN 路由器（BM25 替代）
  - w/o 技能合并（只增不删）
  - w/o 冲突检测

#### 预期贡献

1. 首次将**关系型知识图谱**引入 Agent 技能库，支持推理式检索而非关键词匹配
2. GNN 路由器：唯一能利用技能间结构关系做检索的方法
3. 自动冲突检测：解决大规模技能库中普遍存在但被忽视的技能冲突问题
4. 知识图谱 × 自进化 Agent = 高度新颖的交叉领域

#### 论文标题建议

- *GraphSkillEvo: Relational Skill Graphs with GNN Routing for Self-Evolving Tool-Calling Agents*
- *Beyond Retrieval: Graph-Structured Skill Libraries for Compositional Agent Self-Evolution*

***

### 7.4 GRPOEvo — 工具调用结构化 GRPO

> **副标题**：*Tool-Level Reward Decomposition with Group Relative Policy Optimization for Self-Evolving Agents*

#### 动机：工具调用版的 DeepSeek-R1

GRPO（Group Relative Policy Optimization）在推理链（chain-of-thought）优化上取得了突破性进展（DeepSeek-R1, QwQ 等），核心思路是：对同一问题生成多个 rollout，以 group 内的**相对奖励**替代绝对奖励，大幅降低方差。

然而，工具调用与推理链有本质区别：

- 推理链的奖励是 terminal（最终答案对不对）
- 工具调用有**中间可观测信号**：每一步工具调用都能产生可评估的输出
- 工具调用的错误**有类型**（调用了错误工具 vs. 参数错误 vs. 正确调用但上下文不足）

现有对工具调用 RL 的研究（Tool-R0, Agent-R1 等）都使用 terminal reward，没有利用工具调用的步骤级结构。这是一个明显的建模缺陷——相当于训练推理链时只看最终答案而忽略中间推理步骤的质量。

#### 方法

**工具粒度奖励分解（Tool-Level Reward Decomposition, TLRD）**：

将 episode 总奖励分解为每步工具调用的贡献：

$$R\_{\text{total}} = \sum\_{t=1}^{T} w\_t \cdot r\_t$$

其中每步奖励 $r\_t$ 由**两部分**构成：

$$r\_t = \underbrace{\text{CCA}(a\_t, \tau)}_{\text{反事实重要性}} + \underbrace{\Delta F1\_t}_{\text{增量 F1 变化}}$$

- $\text{CCA}(a\_t, \tau)$：用 LLM 估计"如果不调用 tool $a\_t$，episode 结果会如何变化"（来自 CausalEvo 的 CCA 组件）
- $\Delta F1\_t$：当前步工具输出加入上下文后，agent 的中间预测 F1 相对于前一步的变化
- 权重 $w\_t = \text{softmax}(\text{CCA 分数})$，使奖励聚焦于最关键的工具步骤

**Group-Relative Training（GRPO 核心）**：

对同一 query $q$，采样 $G=8$ 条轨迹 ${\tau\_1, ..., \tau\_G}$，第 $i$ 条轨迹的归一化奖励：

$$\hat{R}\_i = \frac{R\_i - \text{mean}({R\_j})}{\text{std}({R\_j}) + \epsilon}$$

用 $\hat{R}\_i$ 替代绝对奖励进行 policy gradient 更新，消除奖励尺度差异，降低方差。

**自生成课程（Self-Generated Curriculum）**：

```
阶段 1（简单）：单工具任务（1步）
  → 当 F1 ≥ 0.85 时进入下一阶段

阶段 2（中等）：2–3 步工具序列任务
  → 当 F1 ≥ 0.80 时进入下一阶段

阶段 3（困难）：4+ 步多域组合任务
  → 持续训练

任务生成：使用 AgentEvolver 的模板填充机制
奖励来源：GT（有标注时）或 RewardEvo 的 RM（无标注时）
```

**KL 散度约束**：

$$\mathcal{L} = -\mathbb{E}_{\tau \sim \pi_\theta}\[\hat{R}] + \beta \cdot \text{KL}(\pi\_\theta | \pi\_{\text{ref}})$$

$\pi\_{\text{ref}}$ 为原始基础模型，防止过拟合到工具调用分布而损害通用能力。

**与 AdaptEvo 的协同**：GRPO 梯度可直接更新 LoRA 适配器（AdaptEvo），二者的结合使参数更新既有 LoRA 的轻量性，又有 GRPO 的样本效率。

#### 与现有方法对比

| 维度     | SWE-RL   | Tool-R0    | Agent-R1 | **GRPOEvo**         |
| ------ | -------- | ---------- | -------- | ------------------- |
| 奖励粒度   | terminal | terminal   | terminal | **step-level（工具级）** |
| 奖励来源   | Bug 修复结果 | solver 成功率 | 任务完成     | CCA + ΔF1           |
| 探索机制   | 无课程      | 自对弈        | 无课程      | **自生成课程**           |
| 无标注支持  | 否        | 否          | 否        | 是（配合 RewardEvo）     |
| 与技能库结合 | 否        | 否          | 否        | **是（提示 + 权重双进化）**   |

#### 实验设计

- **Baseline**：
  1. PPO with terminal reward（传统 RL）
  2. GRPO with terminal reward（标准 DeepSeek-R1 风格）
  3. SFT on successful trajectories（监督微调）
- **核心指标**：
  1. 工具 F1（主要评估）
  2. 样本效率：达到 baseline F1 所需的 episode 数
  3. Step-level accuracy：每步工具选择正确率（验证 TLRD 的效果）
  4. 泛化：训练集外任务类型上的 F1（验证课程学习效果）
- **消融**：
  - w/o TLRD（terminal reward 替代 step-level reward）
  - w/o 课程学习（随机任务顺序）
  - w/o KL 约束
  - w/o CCA（仅用 ΔF1 作为 step reward）

#### 预期贡献

1. **新算法**：首个面向工具调用的工具粒度 GRPO，利用中间信号而非仅依赖 terminal reward
2. **TLRD**：工具级奖励分解框架，可独立于 GRPO 使用（如插入 PPO、REINFORCE）
3. **无标注版本**：配合 RewardEvo，可在零 GT 标注下运行
4. **与 CausalEvo 的深度结合**：CCA 既用于提示端（CausalEvo），又用于训练端（TLRD 权重），首次实现推理与训练的因果一致性

#### 论文标题建议

- *GRPOEvo: Tool-Level Reward Decomposition for Group Relative Policy Optimization in Tool-Calling Agents*
- *Structured GRPO for Tool-Calling: Causal Credit as Step-Level Reward for Self-Evolving Agents*
- *Beyond Terminal Rewards: Tool-Granular GRPO for Sample-Efficient Agent Self-Evolution*

***

### 7.5 四个思路的组合关系

上述四个思路不是孤立的，可以按需组合：

```
RewardEvo（奖励信号自举）
    ↓ 提供 pseudo-reward
GRPOEvo（工具粒度 GRPO）
    ↓ 更新 LoRA 参数
AdaptEvo（域隔离 LoRA）
    ↑ 检索增强
GraphSkillEvo（图结构技能库）
```

**最强组合**（系统级投稿）：四者合一构成完整的"双轨自进化系统"：

- **非参数轨道**：GraphSkillEvo 提供结构化提示增强
- **参数轨道**：AdaptEvo + GRPOEvo + TLRD 提供轻量权重更新
- **自举奖励**：RewardEvo 使整个系统无需 GT 标注运行

**单独投稿建议**（按创新程度排序）：

1. **GRPOEvo**（最热门方向，工具调用 GRPO 时效性极强，建议优先投稿）
2. **RewardEvo**（问题定义新颖，填补领域空白，适合 ACL/EMNLP）
3. **GraphSkillEvo**（知识图谱×自进化，适合 ICLR/NeurIPS workshop）
4. **AdaptEvo**（扎实工程贡献，适合 AAAI/ACL System Track）

***

### 7.6 方法全景对比

> ✓ = 已在本项目实现；★ = 论文思路（未实现）

| 方法                | 状态  | 知识载体             | 检索/规划           | 信用归因          | 参数更新        | 无标注支持               |
| ----------------- | --- | ---------------- | --------------- | ------------- | ----------- | ------------------- |
| SkillRL           | ✓   | 3-tier 文本技能      | BM25            | 无             | 否           | 否                   |
| EvoSkill          | ✓   | SkillModule JSON | Pareto          | 无             | 否           | 否                   |
| AgentEvolver      | ✓   | 经验池 + 模板         | BM25 导航         | 折扣归因          | 否           | 否                   |
| MemRL             | ✓   | IEU 情节记忆         | 语义 + Q 值        | Q 值（episode）  | 否           | 否                   |
| ExpeL             | ✓   | 原则文本（flat）       | 关键词过滤           | 无             | 否           | 否                   |
| CausalEvo         | ✓   | CTFM 因果规则        | 因果图合成           | 反事实 CCA       | 否           | 否                   |
| SeqGraphEvo       | ✓   | 有向序列图 + 频繁模式     | 种子工具图模式合成       | 无             | 否           | 否                   |
| RewardEvo         | ✓   | LLM伪标注→MemRL    | MemRL 检索        | LLM 打分        | 否           | **是**               |
| GraphSkillEvo     | ✓   | 工具共现图            | 边权重排序           | 无             | 否           | 否                   |
| CausalTextEvo     | ✓   | KnowledgeState θ | 直接注入全量知识        | TextGrad      | 否           | 否                   |
| CausalPolicyEvo   | ✓   | PolicyState      | 直接注入规则+优先级      | LLM 修订规则      | 否           | 否                   |
| SelfCritic        | ✓   | 最优链批评文本          | task_type+BM25  | 无             | 否           | 否                   |
| **AdaptEvo**      | ★   | LoRA 权重          | Meta-Controller | CCA（GRPO 权重）  | **是（LoRA）** | 否                   |
| **GRPOEvo**       | ★   | 工具序列策略           | GRPO 采样         | **TLRD（工具级）** | **是（LoRA）** | **是（配合 RewardEvo）** |

***

*新思路添加时间：2026-03-26*
*对应文献调研范围：2024–2026 年 arXiv 最新论文*
