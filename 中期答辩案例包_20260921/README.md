# 中期答辩案例包（2026-09-21）

这个目录是为 10 分钟中期答辩准备的可复制材料包，包含：

- 中期答辩主文档；
- ExperienceEvo 的失败/成功真实轨迹及可视化产物；
- PromptEvo 各场景与 Base 的指标对比；
- 已完成 RL 实验与 Qwen2.5-3B Base 的指标对比；
- 精选原始 `report.json`、`metrics_summary.json` 和实验来源说明。

## PPT 使用入口

- `ExperienceEvo_PromptEvo_中期答辩_模板风格副本.pptx`：基于学校宽屏模板副本制作的 10 页科研风格 PPT。
- `PPT逐页讲解与素材索引_20260921.md`：逐页讲解重点、建议时长、结果口径和素材路径。
- `source/build_ppt.py`：PPT 生成源脚本。
- `template/iOPEN-PPT模板-宽屏模板v1.1.pptx`：未修改的模板副本，用于参考和后续编辑。

## 推荐阅读顺序

1. `ExperienceEvo_PromptEvo_中期答辩PPT材料_20260921.md`：答辩内容、背景、问题、方法、实验和逐页 PPT 规划。
2. `案例说明_失败成功轨迹与产物_20260921.md`：可直接放入 PPT 的两个地理空间案例，以及 PromptEvo 的回归案例。
3. `指标汇总_PromptEvo与RL_20260921.md`：所有已记录场景的 Base 对比表、实验口径和可用结论。
4. `tables/`：可复制到 Excel、Origin 或绘图脚本的 CSV 表格。
5. `raw/`：精选原始结果文件，用于追溯数字；没有复制运行中的缓存和全量结果目录。

## 案例文件

`cases/experience_evo/` 中的四个 JSON 保留了完整任务描述、对话历史、工具调用序列、工具观察、指标和 `evolution_trace`。其中：

- `domestic_waste_base_failed.json` 与 `domestic_waste_v4_success.json` 是同一输入图像上的 Base / ExperienceEvo 对照；
- `kluane_base_osm_error.json` 与 `kluane_v4_success.json` 是同一 Kluane National Park 任务上的 Base / ExperienceEvo 对照；
- `TG_40010.jpg`、`kluane_v4_map.png` 和 `kluane_v4_boundary.gpkg` 是可用于 PPT 的输入/产物文件。

## 指标口径警告

- 不同表格可能来自不同日期、模型、服务版本、子集或重跑；每一节都标注了 `n`、split、模型和是否适合主表。
- API-Bank 的 389 条表包含优化 train/dev，不能当作独立测试集；79 条表是 held-out。
- AgentDojo 同时保留了周报中的 2026-09-04 汇总和本地可直接复核的 2026-09-15 raw rerun；两者不应混合计算。
- Swift GRPO 的标准 eval 有一版被 agent LLM 容器冲突污染，已明确标为不可作性能结论。
- veRL pure online GRPO 只跑到约 step 206，未完成最终 checkpoint 和正式 OEA eval，不能写成完成结果。
- 产物来源感知 GRPO 目前是设计中的算法，尚无正式训练结果；不要把它与已完成的标准 GRPO 混写。

## 不包含的内容

没有复制 `tmp/artifacts/artifact_index.json`、工具缓存、完整 `evaluations/`、全部 rollout 结果和模型权重，以避免材料包过大且混入运行时状态。需要追溯完整结果时，按指标文档中的原始路径回到仓库或 `tmp/`。
