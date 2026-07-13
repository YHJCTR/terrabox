# PromptEvo Experiment TODO

本文件记录当前 PromptEvo 在不同数据集 / agent 场景上的实验进度与后续待办。重点是跟踪每个场景是否已经完成 base / stage1 / stage2 三阶段实验，以及后续是否需要补跑 Qwen3 8B、LongCat2、think/no-think 或其它模型消融。

## 已完成 / 进行中

### API-Bank

- **场景**: API 调用选择与参数生成，偏静态 API call 预测、参数准确率和执行口径。
- **状态**: 已完成 base / stage1 / stage2 三阶段实验。
- **用途**: 当前最成熟的非地理工具调用参考场景，可作为其它 adapter 的实现和指标口径模板。
- **采用的三阶段实验组**: `quality_v2c_manual_general_20260629_210641`，最终 Stage2 采用该实验中的 `stage2copy` 结果。
- **实验记录与统一指标目录**:
  - `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_v2c_manual_general_20260629_210641/`
- **Base**:
  - 实验名: `base_original_merged`
  - 原始合并结果按运行文档应位于 `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/original_level1_3gpu/base_original_merged/`，但当前工作区已不再保留该原始目录。
  - 当前保留的 Base 权威指标来源: `quality_v2c_manual_general_20260629_210641/metrics_compare_base_stage1_stage2_stage2copy.json`。
- **Stage1**:
  - Prompt/版本名: `api_bank_api_call_quality_manual_general_stage1_20260629_210641`
  - Level1 结果目录: `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_v2c_manual_general_20260629_210641/level1_api_stage1/`
  - Level2 结果目录: `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_v2c_manual_general_20260629_210641/level2_toolsearch_api_stage1/`
- **Stage2**:
  - Prompt/版本名: `api_bank_api_call_quality_manual_general_stage2_copy_20260629_210641`
  - Level1 结果目录: `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_v2c_manual_general_20260629_210641/level1_api_stage2copy/`
  - Level2 结果目录: `src/terrabox/evolution/promptevo/adapters/api_bank/experiments/quality_v2c_manual_general_20260629_210641/level2_toolsearch_api_stage2copy/`
- **结果摘要**: Macro Success 为 Base `0.6192` → Stage1 `0.6398` → Stage2 `0.6470`。

### Terrabox / OEA - Qwen3 8B

- **场景**: OpenEarthAgent 多工具地理任务，覆盖感知、GIS、计算、逻辑组合。
- **状态**: 已完成 base / stage1 / stage2 三阶段实验。
- **用途**: 当前 OEA 本地模型主基线，用于和 LongCat2、DeepSeek 等外部 API 模型对比。
- **采用的同条件三阶段实验组**: 三阶段均使用 `data/oea_full_sft/openearth_test_tasks.json` 的同一批 `1162` 个任务，rollout agent 均为本地 Qwen3 8B。Stage2 静态提示词由 LongCat optimizer 生成，但实际 rollout 模型仍是 Qwen3 8B。
- **Base**:
  - 实验名: `oe_full_react_offline`
  - 权威结果目录: `tmp/trajectories/oe_full_react_offline/standard/results/`
- **Stage1**:
  - Prompt/记录名: `oea_stage1_sam2refresh_20260630_015032`
  - Rollout 实验名: `oea_stage1_sam2refresh_20260630_015032_rollout`
  - 权威结果目录: `tmp/trajectories/oea_stage1_sam2refresh_20260630_015032_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/oea_stage1_sam2refresh_20260630_015032/`
- **Stage2**:
  - Prompt/记录名: `oea_stage2_contrastive_sam2refresh_20260703_0945`
  - Rollout 实验名: `oea_stage2_contrastive_sam2refresh_20260703_0945_rollout`
  - 权威结果目录: `tmp/trajectories/oea_stage2_contrastive_sam2refresh_20260703_0945_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/oea_stage2_contrastive_sam2refresh_20260703_0945/`
- **结果摘要**:
  - Stage1 相比 Base 的主要提升在工具指标: set-F1 `0.667 → 0.705`、multiset-F1 `0.594 → 0.625`、exact match `24.78% → 38.64%`；success rate 则为 `89.16% → 87.61%`，不能表述为成功率提升。
  - Stage2 与 Stage1 基本接近: success rate 均为 `87.61%`，set-F1 `0.705 → 0.711`、multiset-F1 `0.625 → 0.632`、exact match `38.64% → 40.28%`。
  - 因此这组实验适合作为“Stage1 显著改善工具选择，Stage2 在同条件下保持并小幅继续优化”的 Qwen3 三阶段主实验。

### Terrabox / OEA - LongCat2 no-think

- **场景**: OEA 全量任务，agent LLM 使用 LongCat2 外部 API，关闭 thinking。
- **状态**: 已完成 base / stage1 / stage2 三阶段实验。
- **用途**: 外部 API no-think 主对照组。
- **Base**:
  - Rollout 实验名: `promptevo_oea_base_longcat2_20260704_031626_rollout`
  - 权威结果目录: `tmp/trajectories/promptevo_oea_base_longcat2_20260704_031626_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_base_longcat2_20260704_031626/`
- **Stage1**:
  - Prompt/记录名: `promptevo_oea_stage1_longcat2_20260705_033419`
  - Rollout 实验名: `promptevo_oea_stage1_longcat2_20260705_033419_rollout`
  - 权威结果目录: `tmp/trajectories/promptevo_oea_stage1_longcat2_20260705_033419_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_stage1_longcat2_20260705_033419/`
- **Stage2**:
  - Prompt/记录名: `promptevo_oea_stage2_longcat2_20260705_151500`
  - Rollout 实验名: `promptevo_oea_stage2_longcat2_20260705_151500_rollout`
  - 权威结果目录: `tmp/trajectories/promptevo_oea_stage2_longcat2_20260705_151500_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_stage2_longcat2_20260705_151500/`
- **结果摘要**: 三阶段结果目录均包含 `1162` 个任务文件，但 Stage1 当前只有 `1161` 条属于统一指标接受的有效状态，因此下表使用三阶段有效结果交集。Stage1 相比 Base 的 set-F1 和 multiset-F1 各约 `+0.01`，但 success rate `87.26% → 86.57%`；Stage2 与 Stage1 整体接近，但 operation/logic 指标有所回落，因此作为 LongCat2 no-think 对照组保留，不表述为稳定提升组。

#### LongCat2 no-think 三阶段工具指标

以下指标按三阶段有效结果的共同任务交集计算（`n=1161`）。set-F1 和 multiset-F1 使用 `0-1` 取值；其余指标使用百分数，差值为百分点。

| 指标 | Base | Stage1 | Stage2 | Stage1 - Base | Stage2 - Stage1 |
|---|---:|---:|---:|---:|---:|
| set-F1 | 0.696 | 0.704 | 0.700 | +0.008 | -0.004 |
| multiset-F1 | 0.633 | 0.642 | 0.633 | +0.009 | -0.009 |
| exact_match | 15.50% | 19.12% | 19.04% | +3.62 | -0.09 |
| ordered_exact | 11.97% | 14.73% | 14.21% | +2.76 | -0.52 |
| AnyOrder | 59.69% | 58.66% | 59.69% | -1.03 | +1.03 |
| SameOrder | 58.66% | 57.45% | 58.74% | -1.21 | +1.29 |
| Unique | 63.48% | 63.74% | 64.34% | +0.26 | +0.60 |
| F1 perception | 33.76% | 34.34% | 34.52% | +0.58 | +0.18 |
| F1 operation | 33.53% | 36.55% | 33.59% | +3.02 | -2.96 |
| F1 logic | 29.12% | 32.59% | 30.79% | +3.47 | -1.80 |
| F1 gis | 83.08% | 83.01% | 82.70% | -0.07 | -0.32 |

### Terrabox / OEA - LongCat2 think

- **场景**: OEA 全量任务，agent LLM 使用 LongCat2 外部 API，开启 thinking。
- **状态**: 已完成 base / stage1 / stage2 三阶段实验，三阶段均为 `1162/1162`。
- **注意**: 这组实验验证了带无进度 watchdog 的三流自动衔接可以完整结束；后续同类 watcher 仍必须保留 watchdog，防止 OSM/STAC/native/network 线程卡死。
- **Base**:
  - Rollout 实验名: `promptevo_oea_base_longcat2_think_20260706_133717_rollout`
  - 权威结果目录: `tmp/trajectories/promptevo_oea_base_longcat2_think_20260706_133717_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_base_longcat2_think_20260706_133717/`
- **Stage1**:
  - Prompt/记录名: `promptevo_oea_stage1_longcat2_think_20260707`
  - Rollout 实验名: `promptevo_oea_stage1_longcat2_think_20260707_rollout`
  - 权威结果目录: `tmp/trajectories/promptevo_oea_stage1_longcat2_think_20260707_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_stage1_longcat2_think_20260707/`
- **Stage2**:
  - Prompt/记录名: `promptevo_oea_stage2_longcat2_think_20260707`
  - Rollout 实验名: `promptevo_oea_stage2_longcat2_think_20260707_rollout`
  - 权威结果目录: `tmp/trajectories/promptevo_oea_stage2_longcat2_think_20260707_rollout/standard/results/`
  - 实验记录目录: `src/terrabox/evolution/promptevo/adapters/terrabox/experiments/promptevo_oea_stage2_longcat2_think_20260707/`
- **结果摘要**:
  - Stage1 相比 Base: set-F1 `0.684 -> 0.690`、multiset-F1 `0.617 -> 0.622`，但 success rate `85.89% -> 83.73%`，回合上限率 `8.69% -> 11.27%`。
  - Stage2 相比 Stage1: success rate `83.73% -> 83.30%`、set-F1 `0.690 -> 0.685`、multiset-F1 `0.622 -> 0.613`；operation/logic F1 分别下降 `3.33`/`3.65` 个百分点。
  - 当前结论是 thinking 三阶段链路已完整跑通，但 Stage1/Stage2 没有形成稳定净提升；后续应作为消融结果保留，而不是继续补跑同配置。

## 下一批待办

### 1. tau2-bench

- **场景**: 客服 / 任务型多轮对话 agent，关注最终 reward、任务完成度、对话策略和工具/状态交互。
- **现状**:
  - 上游源码已在 `/data1/yuhongjie2/tau2-bench`，数据位于其 `data/tau2/`。
  - adapter 已在 `src/terrabox/evolution/promptevo/adapters/tau2_bench/`，具备 prompt store、trajectory source、metric provider 和原生 rollout runner。
  - tau2 `.venv` 核心依赖检查已通过；历史 smoke 已验证本地 Qwen3 8B 的基本链路。
  - 正式 text/base 口径约 `375` 个任务：airline `50`、retail `114`、telecom base `114`、banking_knowledge `97`。
  - 历史全量尝试没有形成可复用的完整 Base；早期失败曾由 vLLM 缺少 `--enable-auto-tool-choice --tool-call-parser hermes` 导致，当前 agent LLM manager 已包含这两个参数。
  - `banking_knowledge` 正式服务保持 `max_model_len=32768`；少量请求达到约 34.6k，按明确口径记为隔离的 context/infrastructure failure，不升到 40k 冒整阶段 OOM 风险，也不允许阻断后续任务。Qwen/vLLM 偶发双重 JSON 编码的 tool arguments 已在 bootstrap 做传输层解码。
  - **当前状态**: Qwen3 8B Base `tau2_qwen3_8b_base_20260713` 已完成 `375/375`，并完成 LongCat2 no-think NL assertion 重评；Stage1 prompt 已生成，Stage1 rollout 尚未启动。
  - Base 结果目录:
    - `src/terrabox/evolution/promptevo/adapters/tau2_bench/experiments/tau2_qwen3_8b_base_20260713/airline_base/`
    - `src/terrabox/evolution/promptevo/adapters/tau2_bench/experiments/tau2_qwen3_8b_base_20260713/retail_base/`
    - `src/terrabox/evolution/promptevo/adapters/tau2_bench/experiments/tau2_qwen3_8b_base_20260713/telecom_base/`
    - `src/terrabox/evolution/promptevo/adapters/tau2_bench/experiments/tau2_qwen3_8b_base_20260713/banking_knowledge_base/`
  - Base 统一配置: Qwen3 8B no-think、4 个 domain 分别钉 GPU0-3、`max_model_len=32768`、Hermes tool parser、每 domain concurrency `1`、`max_steps=80`、每任务 timeout `900s`；banking 使用离线 `bm25` retrieval。
  - 可续跑 checkpoint 位于 `tmp/tau2_runtime/tau2_qwen3_8b_base_20260713/`，controller 日志为 `tmp/tau2_qwen3_8b_base_20260713.log`。
  - 原三阶段 watcher 已退出；当前无 Tau2 tmux。Base 后续链路曾在 LongCat 重评 JSON 解析处中断，该问题已修复，Base 重评已独立续完。
  - Stage1 实验/Prompt 版本: `tau2_qwen3_8b_stage1_20260713`；prompt 已保存到 `evolution_store/promptevo/tau2_bench/versions/tau2_qwen3_8b_stage1_20260713.txt`。
  - Stage2 实验/Prompt 版本: `tau2_qwen3_8b_stage2_20260713`。
  - 自动链路: Base 完成并释放四卡 -> LongCat2 no-think 离线重评 Base NL assertions -> Stage1 优化/四域 rollout -> 重评 Stage1 -> Base vs Stage1 配对式 Stage2 优化/四域 rollout -> 重评 Stage2。
  - LongCat 重评结果独立保存在各实验组的 `rejudged_longcat/`，不覆盖 tau2 原始结果；Stage1/Stage2 优化均读取重评口径。
- **待办**:
  - 从 Stage1 rollout 继续，再衔接 Stage1 重评、Stage2 优化和 Stage2 rollout。Codex 沙箱内直接 `nvidia-smi` 看不到 `/dev/nvidia*`，但 Docker GPU smoke 已确认宿主机 GPU0 可用；后续以 Docker 服务健康为准。
  - 维持 `max_model_len=32768`；超过窗口或发生单任务 OOM 时记录该任务失败并继续，不为少量长任务提升到 40k。
  - 检查 reward 是否提升，以及是否引入对话安全或任务流程退化。
  - 在 metricViewer 中按 `tau2_bench` 场景展示各 domain 及 macro/worst-domain 指标。

### 2. AgentDojo

- **场景**: 工具调用安全与鲁棒性 benchmark，重点关注 utility 和 security 的平衡。
- **现状**:
  - 上游源码已在 `/data1/yuhongjie2/agentdojo`，当前 checkout 为 package `0.1.35`、git `089ed468`，正式 benchmark 口径固定为 `v1.2.2`。
  - adapter 位于 `src/terrabox/evolution/promptevo/adapters/agentdojo/`，已具备 prompt store、trajectory source、utility/security metric provider、官方 CLI runner 和可续跑的 Base -> Stage1 -> Stage2 pipeline。
  - 正式配置为 Qwen3 8B no-think；GPU0-3/端口 9200-9203 启动四个等价 worker lane，动态领取 4 个 clean job 和按 injection task 拆分的 35 个 attack job，避免 workspace 固定占一卡造成尾部三卡空闲；使用 AgentDojo `vllm_parsed` + vLLM Hermes 原生工具调用，不走其 `local` 正则标签解析器，并通过 adapter-local `qwen_no_think` module 在每轮请求显式传 `enable_thinking=false`。
  - 每个 stage 都包含 clean utility 和 `important_instructions` attack 两个阶段；Stage1 分层采样 security failure、utility failure、success，Stage2 用 Base/Stage1 同任务配对归因并明确禁止用安全退化换取小幅 utility 提升。
  - 当前 v1.2.2 每阶段共 `1081` 条：workspace `614`、travel `167`、banking `169`、slack `131`；pipeline preflight 会从上游源码重新计算 expected total，避免版本升级后沿用旧总数。
  - 实验结果统一保存到 `src/terrabox/evolution/promptevo/adapters/agentdojo/experiments/<group>/`，prompt 版本保存到 `evolution_store/promptevo/agentdojo/versions/`；metricViewer 已按 `agentdojo` 场景发现顶层实验组。
  - 已完成纯离线合成测试：clean/attack 判定、稳定 task id、tool call/result 关联、resume CLI、GPU 钉卡/Hermes 参数和 metricViewer 发现均通过；没有启动正式 AgentDojo benchmark，也没有调用付费 API。
  - `cohere`、`deepdiff`、`google-genai` 已隔离安装到 `tmp/agentdojo_site_packages/`，不污染 `unsloth` 环境；adapter 会自动加入该目录。正式 preflight 已通过，识别到 4 个 suite 和每阶段 `1081` 条结果。
  - 当前没有代码或依赖阻塞；等待 Tau2 Stage1/Stage2 完成后启动四卡 smoke 和正式 rollout。Codex 沙箱内直接 `nvidia-smi` 不可用，但 Docker daemon 已验证能够分配 GPU。
- **待办**:
  - 宿主机 GPU 恢复后重新运行 preflight，并做同一正式配置的最小真实 smoke。
  - 用同一正式配置做最小真实 smoke：至少 1 个 clean user task + 1 个 attacked user/injection pair，确认 Qwen 原生工具调用、上游 evaluator、结果落盘和 GPU 清理。
  - smoke 通过后启动正式 Base，并同时启动 `chain-after-base`，自动接 Stage1 和 Stage2。
  - 汇报 clean utility、attacked utility、attacked security、attack success rate、balanced score；重点检查 PromptEvo 是否在提升 utility 的同时破坏 security。

### 3. ToolBench

- **场景**: 通用工具调用 benchmark，关注工具选择、API 调用轨迹和结果解析。
- **现状**: 当前 adapter 更偏 read-only，主要用于读取已有 ToolBench 结果做 trace / metric 分析。
- **待办**:
  - 先确认是否需要补 rollout runner。
  - 若只做离线分析，则先接入已有 ToolBench result 目录。
  - 若要完整三阶段，则需要明确如何用新 prompt 重跑 ToolBench。

### 4. Terrabox / OEA 其它模型消融

- **场景**: 同 OEA 任务，替换 agent LLM 或切换 think/no-think。
- **可选方向**:
  - Qwen3 8B think / no-think 对照。
  - DeepSeek no-think / think 对照。
  - LongCat2 不同配置或不同版本。
- **目标**: 区分 PromptEvo 的收益、模型能力差异、thinking 开关影响和工具基建问题。

### 5. 跨场景汇总报告

- 汇总 API-Bank、OEA、tau2-bench、AgentDojo、ToolBench 的 base / stage1 / stage2 指标。
- 按场景归一化展示:
  - 任务成功率或 reward
  - 工具选择 F1 / API call exact match
  - 安全指标
  - token、耗时、错误率
  - stage1 和 stage2 的净收益
- 目标是判断 PromptEvo 是否只对某个场景有效，还是能跨 agent 场景泛化。

## 推荐优先级

1. **P0: tau2-bench 三阶段**  
   tau2 和 OEA/API-Bank 差异最大，最能检验 PromptEvo 是否能泛化到多轮客服 agent。

2. **P1: AgentDojo 三阶段**  
   重点看安全指标，避免 PromptEvo 只提升任务完成但破坏安全约束。

3. **P2: ToolBench 梳理与 runner 决策**  
   先确认 read-only 是否足够；如果要完整闭环，再补 runner。

## PromptEvo 结构治理待办

### 当前问题盘点

1. **模块缺少唯一的人类入口文档**
   - `promptevo/` 根目录没有 `README.md`。
   - 当前 `CLAUDE.md` 同时包含架构设计、Stage1/Stage2 语义、模型配置、GPU 三流、watcher、监控命令和代码约束，已经接近运行手册与设计文档的混合体。
   - 同一类实验命令还散落在 `evolution/CLAUDE.md`、`ReAct/CLAUDE.md`、各 adapter README 和多个 `tmp/*.sh` 中。

2. **Terrabox adapter 迁移只完成了一半**
   - 旧实现仍在 `promptevo/adapters_terrabox.py`。
   - 新入口 `promptevo/adapters/terrabox/core.py` 只是重新导出旧文件里的类。
   - `run.py`、`interfaces.py` 和部分文档仍直接引用 `adapters_terrabox.py`，导致“旧单文件”和“新 adapter 目录”两套入口同时存在。

3. **完整实验编排没有正式入口**
   - `promptevo.run` 目前只提供 `mine/propose/contrastive/accept/validate`，不负责 base → stage1 → stage2 的 rollout 编排。
   - 本地双流、外部 API 三流、GPU 钉卡、服务生命周期、watchdog、resume 和 stage 间衔接主要复制在多个 `tmp/*.sh` 中。
   - 当前至少有 8 个 OEA/LongCat/Stage1/Stage2 watcher 或续跑脚本，长度约 128 到 343 行，重复定义 `run_p0/run_p1/run_osm/cleanup/result_count` 等逻辑。

4. **文档存在机械重复**
   - `adapters/AGENT.md` 与 `adapters/CLAUDE.md` 基本相同，仅互相引用的文件名不同。
   - `tau2_bench/AGENT.md` 与 `tau2_bench/CLAUDE.md` 完全相同。
   - 同步维护两份全文容易产生遗漏，应明确一个源文件或自动生成策略。

5. **实验目录缺少统一契约**
   - 早期实验将日志、prompt、比较结果全部平铺在实验根目录；新实验使用 `logs/`，但文件名仍不完全一致。
   - `run_info.txt`、prompt path、proposal、contrastive result、状态、指标和比较报告并非每个实验都有。
   - prompt version 既存在按 adapter 分组的 `evolution_store/promptevo/terrabox|api_bank/versions/`，也存在历史公共 `evolution_store/promptevo/versions/`，新旧口径容易混用。

6. **缺少 PromptEvo 正式测试**
   - 当前没有发现针对 PromptEvo core、Terrabox adapter、Stage1/Stage2 状态机和 watcher 恢复逻辑的正式测试。
   - 现有验证主要依赖真实长实验，一旦编排写错，发现成本很高。

### 目标文档结构

建议后续将文档职责固定为：

- `promptevo/README.md`：唯一的人类入口。只写模块定位、目录地图、最短 quick start、当前支持的 adapter，并链接到其它文档。
- `promptevo/RUNBOOK.md`：唯一实验运行手册。集中记录 base/stage1/stage2、local 双流、外部 API 三流、think/no-think、watchdog、resume、GPU 释放和指标查看命令。
- `promptevo/CLAUDE.md` 与对应 `AGENTS.md`：只保留编码约束、Stage1/Stage2 不可混用规则、文档同步要求，不再承载完整运行命令。
- `DESIGN_v2_contrastive.md`：只保留算法、接口和设计决策，不继续追加服务器运行细节。
- `EXPERIMENT_TODO.md`：只记录实验状态、治理计划和优先级。
- `adapters/README.md`：只描述跨 adapter 的统一接口和目录契约；每个 adapter 的数据集、指标、依赖和运行差异写在自己的 README。

### 目标代码与运行入口

1. **建立一个正式 pipeline 状态机**
   - 将 base → stage1 propose/accept → stage1 rollout → stage2 contrastive → stage2 rollout 的状态判断收进 PromptEvo 正式 Python 模块。
   - pipeline 应读取 manifest 判断已完成阶段，天然支持 `--resume`，不再依赖脚本名称和日志文本猜状态。
   - provider、thinking、prompt token budget、任务文件、总任务数和 rollout 拆流策略都由参数或配置对象传入。

2. **统一 rollout orchestration**
   - 抽取一个公共编排层，复用本地双流和外部 API 三流的 GPU/service 环境配置。
   - OSM 无进度 watchdog、容器清理、服务锁、`TERRABOX_TOOL_SERVICE_SCOPE=call` 和 `TERRABOX_KEEP_VLM_WARM=1` 只维护一份。
   - adapter runner 只描述任务如何启动，不再复制完整 shell watcher。
   - 若最终需要新增 `scripts/` 启动脚本，先按仓库约束确认命名和参数，并同步 `scripts/SCRIPTS_GUIDE.md`；正式逻辑仍放在 Python 模块中，脚本只做薄入口。

### 项目级 Skill：按需加载 PromptEvo 实验运维能力

这部分适合做成项目级 Skill，但必须采用“Skill 操作层 + Python 执行层”的两层结构。

#### 可行性与边界

- Skill 的 `name`/`description` 元数据保持常驻，用于判断用户是否在请求启动、续跑、监控、恢复或清理 PromptEvo 实验。
- `SKILL.md` 只在触发后加载；双流、三流、watchdog、恢复和清理的详细参考可以继续拆到 `references/`，由 Skill 按 provider/场景选择性读取。
- Skill 适合保存 Codex 需要遵循的操作流程、决策规则、检查清单和薄辅助脚本。
- Skill **不应**成为实验状态机本身。真正的并发控制、PID 管理、watchdog、resume、manifest 更新和 GPU/container 清理必须由仓库内可测试的 Python orchestration 执行。
- 实验能否正确继续不能依赖 Codex 是否准确执行了一段长 Markdown；Skill 只负责选择配置并调用确定性的正式 CLI。

#### 建议目录

项目级 Skill 建议命名为 `terrabox-run-promptevo-pipelines`，目录目标为：

```text
.codex/skills/terrabox-run-promptevo-pipelines/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── local-dual-flow.md
│   ├── external-api-three-flow.md
│   ├── recovery-and-cleanup.md
│   └── experiment-manifest.md
└── scripts/
    └── inspect-promptevo-state.sh
```

实施时先验证当前 Codex surface 是否自动发现仓库 `.codex/skills/`；如果当前版本只发现用户级 Skill，则再选择安装到 `~/.codex/skills/` 或封装为项目 plugin。Skill 创建前按 `skill-creator` 流程用 `init_skill.py` 初始化，不手写缺少 metadata 的目录。

#### Skill 的触发范围

`description` 应明确覆盖以下请求：

- 启动或续跑 PromptEvo base/stage1/stage2。
- 在本地 Qwen/vLLM 双流与 LongCat/DeepSeek 外部 API 三流之间选择运行模式。
- 查看进度、判断卡死、恢复 OSM/nogpu 子流。
- 检查 GPU 钉卡、服务锁、容器生命周期和实验结束后的显存释放。
- 根据 manifest 判断下一阶段，而不是根据实验名或聊天上下文猜测。

一般的 PromptEvo 算法开发、optimizer 修改和 adapter 代码 review 不应自动触发这个运维 Skill，避免加载无关 GPU/服务器上下文。

#### SKILL.md 核心工作流

1. 读取项目 `AGENTS.md`、PromptEvo `RUNBOOK.md` 和当前实验 `manifest.json`。
2. 只读检查 `tmux ls`、相关 Python PID、Docker 容器、GPU、结果数量和最后更新时间。
3. 根据 provider 与 manifest 选择：
   - local provider → 本地双流/3+1；
   - external provider → 外部 API 三流；
   - 已有部分结果 → `--resume`；
   - 无进度超过阈值 → 调用正式 pipeline 的分支恢复命令。
4. 在启动前执行正式 CLI 的 `plan` 或 `dry-run`，打印将使用的 prompt、父实验、GPU lane、端口、结果目录和清理策略。
5. 调用正式 orchestration CLI，不在 Skill 内拼接几百行临时 shell。
6. 启动后验证 tmux/PID、结果增长和日志路径。
7. 实验结束后验证报告落盘、进程退出、容器归属和 GPU 释放。

#### Skill references 的职责

- `local-dual-flow.md`：本地 agent LLM 的双流/3+1 拆分、agent GPU 与感知 GPU 关系。
- `external-api-three-flow.md`：LongCat/DeepSeek 三流、`service_scope=call`、VLM warm 和误调用感知工具时的锁等待。
- `recovery-and-cleanup.md`：无进度 watchdog、failed JSON 与 missing result 的区别、恢复分支、tmux/container/GPU 清理。
- `experiment-manifest.md`：stage、provider、thinking、parent experiments、prompt version、任务数和完成条件。

这些细节只放在 references 中，不和 `SKILL.md`、`RUNBOOK.md` 重复全文。`SKILL.md` 只保留选择规则和读取导航。

#### Skill scripts 的限制

- Skill 内脚本只做确定性、低风险的状态检查，例如汇总 tmux/PID/GPU/results/mtime。
- 状态检查脚本默认只读，不得删除 result、停止未知进程或修改实验目录。
- 启停和恢复必须调用仓库正式 orchestration CLI，由 CLI 校验 manifest 和进程归属。
- 不在 Skill 中保存 API key、服务器密码、固定 PID、当前实验名或一次性时间戳。
- 不复制当前 300 多行 watcher；当前 watcher 中验证有效的逻辑应迁移到 Python orchestration 后再由 Skill 调用。

#### 实施步骤

1. 先实现并测试正式 `promptevo orchestration`：manifest、plan、start、resume、recover、status、cleanup。
2. 使用 fake runner 验证双流/三流状态机、watchdog 和 Stage1 → Stage2 衔接，不调用真实 GPU/API。
3. 明确 `RUNBOOK.md` 中的唯一 CLI，并冻结新临时 watcher。
4. 用户确认 Skill 位置后，使用 `skill-creator/scripts/init_skill.py` 初始化 `terrabox-run-promptevo-pipelines`。
5. 编写精简 `SKILL.md` 和四份按需 references；状态检查脚本复用正式 CLI 的只读接口。
6. 使用 `quick_validate.py` 校验 Skill frontmatter、命名和目录结构。
7. 新开 Codex 会话验证项目级 Skill 能被发现，并用一条 fake/smoke 实验进行 forward test。
8. 验证通过后，才逐步归档历史 watcher；当前正在运行的 LongCat think watcher仍保留到实验结束。

#### 不建议的做法

- 不把完整 watcher shell 原样塞进 `SKILL.md`。
- 不让 Skill 直接根据 `nvidia-smi` 利用率猜测服务是否可用；VLM 常驻时应依赖服务锁和 manifest。
- 不让 Skill 自动删除 failed 结果；必须先区分 LLM 错误、基建污染和 missing result。
- 不让 Skill 自行发明实验名、prompt parent 或 Stage2 输入；这些由 manifest 和正式 CLI 决定。
- 不把 Skill 当作后台 daemon。持续监控仍由 tmux 中的正式 pipeline/watchdog 进程负责。

3. **完成 Terrabox adapter 迁移**
   - 将 `adapters_terrabox.py` 拆入 `adapters/terrabox/prompts.py`、`traces.py`、`metrics.py`、`runner.py`。
   - `adapters/terrabox/__init__.py` 作为唯一公共导入入口。
   - 旧 `adapters_terrabox.py` 暂时保留兼容转发并标记 deprecated；所有内部代码和文档迁移完成后再删除。

4. **统一实验 manifest 和目录**
   - 每个 adapter experiment 固定包含：
     - `manifest.json`：dataset、provider、model、thinking、stage、parent experiments、prompt version、task file、rollout experiment、创建时间和状态。
     - `prompt.txt` 与 `prompt.meta.json`。
     - `proposal.json` 或 `contrastive_result.json`。
     - `logs/`：只放分流和 watcher 日志。
     - `metrics/`：单实验指标和 A/B 对比。
   - `tmp/trajectories/<experiment>/<mode>/results/` 继续作为逐任务结果唯一权威来源，adapter experiment 目录只保存配置、日志和派生报告。

5. **统一版本库口径**
   - 新实验只写 `evolution_store/promptevo/<adapter>/versions/`。
   - 根目录历史 `evolution_store/promptevo/versions/` 标记为 legacy，只读保留；完成引用审计后再决定迁移或归档。

### 脚本清理策略

1. `tmp/watch_longcat2_think_base_stage1_stage2_20260707.sh` 对应链路已完成；在正式 orchestration 吸收其 watchdog 和清理逻辑前只读保留，之后可归档或删除。
2. 其它 `tmp/` watcher 先建立清单，记录对应实验、是否仍有 tmux/process、是否包含唯一修复逻辑。
3. 将仍有效的差异合并进正式 orchestration 后：
   - 已完成且可由 manifest/日志复现的 watcher 移入一次性归档或删除。
   - smoke、旧失败重试和硬编码实验名的脚本不进入 `scripts/`。
   - 不再创建新的几百行实验专属 watcher；新实验只创建配置/manifest。
4. 清理前必须检查 `tmux ls`、相关 Python PID 和实验状态，避免删除正在运行任务依赖的脚本。

### 测试补齐

- Core 单元测试：Stage1 单边采样、Stage2 base-vs-stage1 配对、candidate selection、thinking 内容清理、prompt growth guard。
- Adapter contract 测试：PromptStore、TrajectorySource、MetricProvider 的 task_id 对齐和路径解析。
- Pipeline 状态机测试：base 已完成、stage1 部分完成、failed JSON 隔离、watchdog 重启、stage2 自动衔接。
- 轻量集成测试：使用 fake runner / 小型固定轨迹验证三阶段流程，不调用真实 GPU 和付费 API。
- 每个真实 provider 全量实验前只保留 1 到 3 条 smoke，成功后将 smoke 标记为可清理，避免污染 experiment viewer。

### 推荐治理顺序

1. **P0：冻结新的临时 watcher 复制**，先建立 `README.md`、`RUNBOOK.md` 和统一实验 manifest 规范。
2. **P0：启动 tau2-bench Qwen3 8B 三阶段主实验**，先完成 GPU 健康检查和正式配置 smoke，再进入 Base 全量。
3. **P1：抽取正式 orchestration/pipeline**，把 watchdog、三流和 stage 衔接收成一份实现。
4. **P1：完成 Terrabox adapter 目录化迁移**，内部导入全部切换到 `adapters.terrabox`。
5. **P2：补 core/adapter/pipeline 测试**，以后先用 fake runner 验证流程再跑付费或多卡实验。
6. **P2：审计并归档旧 watcher、legacy versions 和 smoke experiments**，最后再做物理删除。
