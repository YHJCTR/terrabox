# 自进化 Agent 论文草稿

本目录包含两份可独立导入 Overleaf 的 LaTeX 草稿。两份稿件共享研究背景，但不共享未经验证的性能结论。

| 工程 | 研究对象 | 当前定位 |
|---|---|---|
| `experienceevo_v4/` | 从真实 rollout 提炼的产物状态转移经验 | 方法主稿，包含实现对齐的方法、初步诊断、消融设计、机制审计和完整附录；另有源码审计补充材料 |
| `promptevo/` | 从配对轨迹演化的静态行为协议 | 独立方法稿，包含受限编译、对比归因、真实 rollout 接受门和负结果分析；另有 compiler/接受门补充材料 |

## 使用方式

分别将 `experienceevo_v4/main.tex` 和 `promptevo/main.tex` 上传到两个独立 Overleaf 项目，选择 `pdfLaTeX` 编译即可。两份文件自包含 TikZ 方法图和参考文献，不依赖仓库外图片、BibTeX 或自定义样式文件。若提交 supplement，请将同目录的 `supplementary.tex` 上传为第二个主文件；它同样是独立可编译的，不依赖外部文件。

## 写作边界

- 两份稿件已包含完整的相关工作、方法定义、可复现实验协议、失败/反事实分析、附录和经过原始论文/官方页面复核的核心参考文献。为避免制造证据，主结果表仍保留 `TBD`；ExperienceEvo 的历史单配置结果与 PromptEvo 的混合诊断均被清楚标为非最终证据。
- 不得在正式投稿前以模板取代真实实验数字、置信区间、独立答案核验或因果控制。
- `ExperienceEvo v4` 不能宣称一般文本答案 SOTA；其待检验假设是工具执行可靠性、状态衔接和产物完成度改善。
- `PromptEvo` 不能把静态候选选择称为验证通过。每次接受 prompt/patch 必须经过固定 dev 集上的真实 rollout。
- 第三方对照需在实验部分明确标注 `official`、`adapted` 或 `reimplemented`，以及与原论文的差异。

## 实验落表前检查

1. 固定 train/dev/test 划分、任务清单、模型版本、工具目录和随机种子。
2. 对 ExperienceEvo v4，补齐 `no-store soft-only` 与 `generic-guard only` 控制组，区分经验库收益和通用工程保护。
3. 对 PromptEvo，报告每个候选的 dev rollout、接受/拒绝记录、配对任务覆盖率和回归维度。
4. 为主指标计算配对 bootstrap 置信区间或合适的配对显著性检验；不要只报告单个聚合点估计。
5. 对 LLM judge 与 actor 同源的设置，补充独立 judge 或人工核查子集，并报告一致性。
6. 保存经验库/协议版本、代码 commit、环境变量、工具 schema 快照和完整运行 manifest。
7. Overleaf 编译后人工检查 TikZ 图、宽表格、引用编号、页码、匿名信息和最终标题/摘要是否与完成后的结果一致。

详细的目录结构、manifest 字段、任务级统计字段、ExperienceEvo provenance 审计和 PromptEvo candidate ledger 模板见 [投稿证据包模板](投稿证据包模板.md)。已有历史 rollout 的可用范围、主表字段来源和最终落表门槛见 [结果证据与落表映射](结果证据与落表映射_实验前.md)。从当前草稿到可公开上传所需的结果、源码、排版和发布审查见 [Overleaf 与 arXiv 上传检查单](Overleaf与arXiv上传检查单.md)。这些文档均用于实验完成后归档/审计证据，不能替代真实 rollout 或作为论文结果来源。

当前代码事实、投稿协议和最终实验结论之间的逐项边界见 [实现与论文证据追踪矩阵](实现与论文证据追踪矩阵_20260814.md)。

## 中期答辩材料

- [中期答辩 PPT（可编辑）](中期答辩_ExperienceEvo与PromptEvo_20260814.pptx)：18 页，合并讲述 ExperienceEvo 与 PromptEvo，但方法与实验结论保持分开。
- [PPT 生成源](中期答辩_ExperienceEvo与PromptEvo_20260814.py)：使用 PowerPoint 原生文本和形状生成，可按需修改后重新导出。
- [讲稿与页码索引](中期答辩讲稿与页码索引_20260814.md)：含建议时长、逐页讲述重点和答辩问答边界。

两篇稿件的当前权威来源、语义边界、源码对应、开放发布风险和状态由 [论文项目状态与权威来源](论文项目状态与权威来源.json) 统一记录。补结果、改实现或改变论文结论时，应先更新这一状态记录，再同步主稿、补充材料和证据包。

## 当前稿件规模与定位

当前两份 `main.tex` 均为约 6k+ 英文词的正文与附录初稿，远超过此前的短提纲，但仍低于部分 10k+ 词的长篇 arXiv 论文。是否需要继续扩写应由最终实验的复杂度、补充材料和目标会议页数决定，不能用无证据叙述填充。ExperienceEvo 更接近一篇以 OEA 为核心的系统/方法论文；PromptEvo 目前应定位为受控静态协议演化的实证方法论文，必须保留其混合/负结果，不得改写为普适性能提升论文。

## 当前源码对应

- ExperienceEvo v4 runtime: `src/terrabox/evolution/experience_evo/v4/runtime.py`
- Transition schema and builder: `src/terrabox/evolution/experience_evo/v2/models.py`, `src/terrabox/evolution/experience_evo/v2/builder.py`
- PromptEvo contrastive loop: `src/terrabox/evolution/promptevo/loop.py`
- Typed patch compiler: `src/terrabox/evolution/promptevo/protocol_patch.py`
