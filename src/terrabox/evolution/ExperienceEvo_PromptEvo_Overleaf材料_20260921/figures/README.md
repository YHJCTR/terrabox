# 图表说明

正文中的两张方法图使用 TikZ 内置绘制，不依赖外部图片：

1. ExperienceEvo：`rollout -> artifact transition extraction -> state-conditioned retrieval -> verifier/fallback -> next tool`;
2. PromptEvo：`paired base/candidate rollout -> Stage1 proposal -> Stage2 attribution -> typed patch compiler -> dev gate`.

如果答辩或论文需要替换为外部矢量图，可将 PDF/SVG 放入此目录，并替换正文中的 `figure` 环境。图中不要加入数据集 gold tool sequence、task ID 或具体路径；这些信息只能出现在证据附录。

建议绘制的结果图：

- OEA Base / v4-clean / Boundary 的 Success、Set-F1、Tool error 分组柱状图；
- v4-clean 消融的 Success 与 Set-F1；
- 消融 `answer_acc` 与 `answer_acc_w_gen` 的配对点图；
- PromptEvo 在 ToolBench、tau2、AgentDojo、OEA 上的主指标和成本指标双轴图；
- Swift GRPO 训练的 reward、KL、grad norm 曲线。

