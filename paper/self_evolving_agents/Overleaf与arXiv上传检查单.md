# Overleaf 与 arXiv 上传检查单

本文档是两篇论文从当前的“方法与协议草稿”进入公开预印本的发布闸门。它不是实验计划的替代品：任何主结论都必须由冻结配置下的真实 rollout、任务级结果和可审计证据包支持。当前 `main.tex` 中的 `TBD`、诊断性数字和预注册式分析必须在此清单完成前保留，不能以估计值、旧 run 聚合值或图示占位替换。

## 0. 先做发布决策

- [ ] 明确本次发布的是 `ExperienceEvo`、`PromptEvo`，还是两篇独立预印本。二者共享背景，但研究对象、实现状态、评价设计和证据链不同；不能在同一篇中将未验证的模块拼接成一个统一方法结论。
- [ ] 为每篇论文确定目标类别和篇幅定位。arXiv 可使用通用格式；若同时面向会议投稿，必须切换到目标会议当年官方模板并满足其匿名、页数、补充材料和双盲规则。
- [ ] 为每篇论文分配唯一的 release tag、代码 commit、证据包目录和实验 manifest。论文 PDF、源文件、表格数字和证据包必须来自同一个冻结版本。

## 1. 结果与论证闸门

### 共同要求

- [ ] 冻结 train/dev/test 划分、任务 ID 清单、模型 API/权重版本、采样参数、工具 registry/schema、数据版本、代码 commit、Docker/服务镜像、cache/retry/timeout 策略和随机种子。
- [ ] 对每个主指标保存每任务结果，而非仅保存 `report.json` 聚合值。报告成功率、工具调用指标、普通答案准确率、生成型结果成功率、成本/时延及失败分类时，必须说明分母、排除规则和判定器。
- [ ] 使用配对比较和 bootstrap 置信区间或适当的配对检验。所有方法必须在相同任务、相同模型、相同工具集、相同服务配置和可比并行/重试策略下运行。
- [ ] 分离 actor 与 judge，或补充独立 judge/人工复核子集并报告一致性。生成图像、地图和文件等任务应将“最终工具成功”与“文本答案正确”分开报告。
- [ ] 将所有 `TBD` 替换为冻结实验结果后，逐段复核 Abstract、Introduction、Results、Discussion、Conclusion 和图注。每一个比较级、因果词和泛化声明都应能追溯到表格、任务级结果或明确限定的诊断。
- [ ] 保留负结果、失败模式和不确定性。不要只选择有利配置、筛选任务、删除失败 run 或将开发集结果写成测试集结果。

### ExperienceEvo 专项要求

- [ ] 修复并验证经验来源表述：由真实 rollout 构建的 `TransitionFamily` 与 v3 deterministic fallback 必须在 runtime 注入文本、检索结果和论文图表中清楚区分。fallback 不得被称为 rollout-derived experience。
- [ ] 扩展并冻结 step-level trace，使其至少记录 `retrieved_family_ids`、`family_provenance`、排序/检索策略、recommended tool、selected tool、selected-in-recommendations、product/contract check、guard event 和 verifier status。
- [ ] 以统一 manifest 重新运行最小因果矩阵：Base、Base + generic guard、No-store soft-only、Learned-store-only、Fallback-only、Full v4。缺少任一关键控制时，论文只能称为系统诊断，不可将效果归因给 learned experience。
- [ ] 核验不存在 train/test task、答案、gold trajectory 或 test artifact 泄漏。经验库构建输入、去重、过滤及 train/test 归属须可审计。
- [ ] 只有在 provenance trace 和控制实验完整后，才可以把机制审计中的“检索、遵从和成功”写成观察证据；它仍不是未经干预的因果证明。

### PromptEvo 专项要求

- [ ] 对每个进入 dev 的 candidate patch 保留 candidate ledger：输入轨迹、配对/归因记录、proposal、compiler 诊断、patch 文本、dev before/after、接受/拒绝和最终版本哈希。
- [ ] 把 patch 的 `evidence` 从当前可为空的声明字段升级为发布时的验证条件：每条被接受 patch 都必须非空，并可解析到保留 trace/attribution ID；无法解析的 candidate 不得称为 trace-backed。
- [ ] 在真实 rollout 的冻结 dev 集执行候选选择，并在一次不可再更新的 locked test 上评测已冻结协议。静态选择或无 runner 的 `accepted=False` 结果不能充当经验结论。
- [ ] 区分候选持久化与接受：`run.py accept` 可将 compiler-valid proposal 写入 prompt 文件，但不验证 dev 结果。每个 locked-test manifest 必须记录实际加载的 prompt fingerprint，并在 candidate ledger 中回链到 frozen-dev 的 accepted decision；仅有 CLI 文件或 compiler report 的版本不得送入 test。
- [ ] 如论文声称保护工具 F1、时延、调用数、风险或其他回归维度，接受门必须真实实现并记录这些条件。目前代码中的实际 acceptance gate 仅约束 success drop，不能把未执行的 protected-metric protocol 写成已实现机制。
- [ ] 报告未接受候选、回归和跨环境失败。现有 tau2、AgentDojo 与 OEA 历史结果只能作为 broad-stage diagnostics，不能替代 typed-patch 的冻结 dev/test 证据。

## 2. 论文源文件闸门

- [ ] 将 `Anonymous Authors`、`Anonymous Institution` 和任何匿名占位替换为真实作者、单位、联系邮箱、致谢/资金信息和所需的 arXiv 分类。若同时双盲投稿，维护匿名版本和公开 arXiv 版本，避免混用。
- [ ] 用官方 BibTeX 或原始论文页面逐项核对作者、标题、venue、年份、页码和 arXiv 编号。当前手写 bibliography 应在目标模板中迁移为标准 `.bib`/官方引用格式后再编译。
- [ ] 审查标题、摘要和贡献列表是否承诺了当前证据无法支持的 SOTA、泛化、效率或因果结论。所有限定语必须与最终实验设置一致。
- [ ] 检查图、表、算法和附录编号及每个引用。图表应使用最终数字；TikZ 图中不得残留实现内部路径、实验目录名、任务 ID、私有数据名或明显占位文本。
- [ ] 将补充材料作为可独立编译文件上传，检查其标题、作者信息、章节编号和 main paper 的交叉描述一致。

## 3. Overleaf 编译与视觉检查

- [ ] 在 Overleaf 选择 `pdfLaTeX`，编译 `main.tex` 与 `supplementary.tex`。当前仓库环境未安装 LaTeX 引擎，因此本地静态检查不能替代这一步。
- [ ] 修复所有编译 warning/error：未定义引用、overfull/underfull box、缺失字体/包、TikZ 定位、浮动体漂移和 bibliography warning。
- [ ] 在桌面 PDF 检查所有页面：方法图文字是否清晰、箭头是否遮挡节点、宽表是否超出页边距、数字/小数点是否对齐、标题和图注是否完整、参考文献是否分页异常。
- [ ] 用 100% 缩放审阅方法图；用双栏/单栏目标模板的实际版面审阅表格。不要仅依据 Overleaf 编辑器预览判断可读性。
- [ ] 生成干净的 submission source：只保留 `.tex`、`.bib`、必要图片与官方样式文件；删除 `.aux`、`.log`、临时构建文件、缓存、运行日志和未使用素材。

## 4. 公开发布与可复现性闸门

- [ ] 创建对应版本的 release manifest，覆盖提交日期、git commit、数据/任务哈希、split、依赖锁定、环境变量、模型/服务版本、命令、硬件与成本统计。
- [ ] 按 [投稿证据包模板](投稿证据包模板.md) 输出 `task_metrics`、失败统计、paired comparison、bootstrap 摘要；ExperienceEvo 还需 provenance audit，PromptEvo 还需 candidate ledger/evidence resolution audit。
- [ ] 确认附带材料不泄漏 API key、代理配置、个人路径、私有数据、未公开 benchmark 答案、攻击 payload、未脱敏轨迹或受限第三方文件。
- [ ] 明确代码、数据、模型访问和许可状态。不能公开的资产要在论文中说明可访问性限制，不得宣称完全可复现。
- [ ] 上传 arXiv 前逐项确认 metadata：标题、作者排序、评论、分类、license、报告编号和 DOI（若有）。上传后在 arXiv 生成 PDF 中再次检查字体、图、参考文献和补充材料链接。

## 5. 最终签字判断

仅当以下四项均为“是”时，该稿才可视为可公开上传：

1. 主表不含 `TBD`，且每个数字可追溯到冻结任务级结果。
2. 论文声明不超过实现与实验实际支持的范围。
3. Overleaf 目标格式成功编译，并完成逐页视觉检查。
4. 论文版本、代码 commit 和发布证据包具有唯一且一致的关联。

任一项为“否”时，当前稿仍是高完整度研究草稿，而不是应提交的 arXiv 版本。优先补齐阻塞证据，避免通过扩写、图形润色或选择性报告掩盖证据缺口。
