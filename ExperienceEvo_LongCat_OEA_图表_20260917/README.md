# ExperienceEvo × LongCat OEA 图表

2026-09-18 新增：训练与推理过程图见 [过程图使用说明](论文级/过程图使用说明.md)。`论文级/figures_v2/` 保存9组英文 PDF/PNG 和配套 CSV，包含完整 GRPO 训练诊断、工具序列前缀曲线、配对增益分布与带 bootstrap 区间的分层结果；复现入口为 `论文级/reproduce_figures.py`。

更新时间：2026-09-17

本目录保存从现有 ExperienceEvo train store 与 LongCat/OEA eval 结果直接生成的中文科研图。图表用于中期答辩、论文实验分析和补充材料；不新增实验、不修改原始结果。

## 数据来源

- 经验库：`evolution_store/experience_evo/oea_train2000_v4_clean_longcat_20260814/`
- 经验库审计：`src/terrabox/evolution/经验自进化_v4离线审计_20260815.md`
- ExperienceEvo eval：`tmp/trajectories/experience_evo_v4_clean_oea_train2000_longcat_eval_dockerfixed_20260816/standard/results/`

## 当前已生成图表

| 文件 | 中文含义 | 可用于说明的问题 |
|---|---|---|
| `经验库构建漏斗.pdf` | 从 rollout 事件到经验 family 的筛选过程 | 经验不是完整轨迹复制，而是经过过滤、聚合和支持度筛选 |
| `QNR经验分布.pdf` | 经验质量、支持度和风险的联合分布 | 为什么不能只按频率检索，需要 Q/N/R 联合排序 |
| `高频产物状态转移.pdf` | 经验库中的主要状态转移模式 | ExperienceEvo 的基本单位是产物状态转移 |
| `经验推荐采纳行为.pdf` | eval 中推荐工具的采纳与执行情况 | 经验提示是否真正进入模型决策链 |

PNG 文件是快速预览，PDF 文件是论文/答辩正式版本。所有原始 JSON/JSONL 数据保持在原目录，本目录不复制大结果文件。

## 本次生成的统计摘要

- 原始 rollout 事件：9062
- 非基础设施事件：8022
- 可归因风险事件：185
- 经验 family：803
- OEA eval 任务：1162
- ExperienceEvo trace 步数：5124
- 有推荐工具的步骤：3585
- 推荐列表内选择：2267
- 推荐列表外选择：1110

其中“推荐列表内选择”只表示模型选择的工具出现在 ExperienceEvo 推荐列表中，不等价于任务最终成功；后续如果需要，可以再按工具执行成功、目标产物形成和任务最终状态做更严格的校准图。

## 图表解读

### 经验库构建漏斗

用于说明经验库的构建过程：从真实 rollout 事件开始，过滤基础设施错误并按产物状态转移聚合，最终形成可检索的经验 family。它适合放在方法页，解释 ExperienceEvo 不是直接复制完整对话。

### QNR 经验分布

每个点代表一个经验 family。横轴是历史支持次数 `N`，纵轴是质量 `Q`，颜色表示风险 `R`。该图用于说明高频经验不一定等于高质量经验，因此需要联合考虑 Q/N/R，而不能只按出现频次排序。

### 高频产物状态转移

展示经验库中支持度最高的状态转移模式，例如 `task_request → gpkg`、感知结果到计算结果、图层到地图/绘图产物等。它适合支撑“产物状态转移是 ExperienceEvo 的基本经验单位”。

### 经验推荐采纳行为

统计 eval 中模型选择推荐列表内外工具的次数，说明经验提示实际进入了决策过程。该图不能单独证明推荐一定正确，需要与任务成功、工具执行成功和目标产物形成率结合解释。

## 口径说明

- `Nsig/Ntool` 是 train-side rollout 中的历史有效支持次数，不是 eval 阶段动态累计的使用次数。
- `经验推荐采纳行为.pdf` 中的“推荐内选择”表示 `selected_in_recommendations=True`，不等价于任务成功。
- 经验库漏斗中的基础设施错误只用于诊断，不提升经验风险。
- 图表不使用 OEA test 的 `expected_tools`、答案或评测指标构建经验库；gold 字段只允许用于事后诊断。

## 论文级图表

`论文级/figures/` 下的新版图表按 scientific-publication-plotter 规范生成：使用矢量 PDF、统一字体和字号、Wong 离散配色、viridis 连续配色、无图内标题，适合在论文中通过 caption 解释图意。

- `图1_主结果配对比较.pdf`：matched Base 与 ExperienceEvo 的多指标配对变化；每项指标单独归一化，仅用于比较方向和相对变化，不用于比较不同指标的绝对量纲。
- `图2_模块消融比较.pdf`：完整方法、Quse/QNR、step hint、verifier 和随机检索对照。
- `图3_QNR可靠性分布.pdf`：经验支持度 N、质量 Q、风险 R 的联合分布。
- `图4_经验推荐采纳率.pdf`：按决策步段统计推荐工具采纳率，点旁标注该步段样本数。
- `图5_产物状态转移支持度.pdf`：经验库中高支持度的产物状态转移。

图表对应的论文 caption 和数值口径保存在 `论文级/图表说明.json`。

## 后续可补充

1. matched Base/ExperienceEvo 主指标图；
2. 工具类别 F1 对比图；
3. fixed/regressed 任务矩阵；
4. Quse 分桶与工具执行成功率校准图；
5. 长尾工具分桶图；
6. Base 与 ExperienceEvo 的轨迹案例卡片。
