# PromptEvo 与 RL 实验指标总表

> 生成日期：2026-09-21。本文档只汇总仓库中已经完成、已有结果文件或已有实验周报明确记录的结果；没有把未完成实验或污染结果包装成正向结论。

## 1. 读表规则

- 百分比均为百分数；`F1`、`precision`、`recall` 等保留原始小数或百分数口径，并在表头说明。
- `full` 表示该次实验声明覆盖的完整评测范围；`held-out` 表示独立保留集；`matched` 表示只在交集任务上比较。
- API-Bank 的 389 条 full coverage 包括 train/dev/test（232/78/79），因此只能作为全覆盖诊断；79 条才是当前可报告的 held-out 口径。
- AgentDojo 的周报快照和 2026-09-15 raw rerun 采用了不同实验批次，单独列出，不混合平均。

## 2. PromptEvo：各场景与 Base 对比

### 2.1 ToolBench / StableToolBench strict paper split（n=160，推荐主表）

这是冻结的 paper split test，LongCat，回归门控版本。工具指标由同一 adapter 统计。

| 方法 | Success | Avg tools | No-finish | Give-up | Tool error | Repeat |
|---|---:|---:|---:|---:|---:|---:|
| Base | 86.25% | 5.14 | 11.88% | 4.38% | 14.37% | 10.62% |
| PromptEvo Stage1 | **88.12%** | **4.67** | 11.88% | **0.62%** | **7.50%** | **9.38%** |
| PromptEvo Stage2 | 84.38% | 5.59 | **15.62%** | 0 | 10.00% | 8.75% |

Stage1 是比较干净的正向结果；Stage2 反而回退，说明候选 patch 必须经过 dev gate、对比归因和回滚保护，不能只看单一平均分。

来源：`raw/promptevo/toolbench/test_*_metrics_summary.json`，原始实验目录为 `tmp/promptevo_toolbench_experiments/promptevo_stabletoolbench_longcat_regression_gated_20260911_230524_fixed_server_lowq_test_*`。另有 `split_manifest.json` 记录冻结 split。

### 2.2 API-Bank full coverage（n=389，含优化 train/dev；诊断表，不是独立 test）

#### 与外部方法的 exact call accuracy

| 方法 | Correct / 389 | Exact call accuracy |
|---|---:|---:|
| Base | 322 | 82.776% |
| GEPA | 322 | 82.776% |
| SCOPE | 325 | **83.548%** |
| AHO | 322 | 82.776% |
| EvoTool | 301 | 77.378% |
| EvoTool modular base | 281 | 72.237% |
| PromptEvo final | 322 | 82.776% |
| PromptEvo Stage1 | 322 | 82.776% |

#### PromptEvo Stage1/Stage2 的结构指标

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| Success | 82.78% | 82.01% | **83.55%** |
| Tool F1 | .9469 | .9444 | **.9572** |
| Parse rate | **97.94%** | 96.66% | 97.69% |
| API name accuracy | 95.63% | 95.63% | **96.92%** |
| Missing argument | 4.63% | 3.60% | **3.34%** |
| Extra argument | 4.37% | 3.08% | **2.83%** |
| Runtime error | 0 | 0 | 0 |

结论：Stage2 在工具结构指标上有改善，但 389 条不是独立测试；不能用它证明泛化能力。

### 2.3 API-Bank held-out（n=79，推荐泛化口径）

| 方法 | Correct / 79 | Exact call accuracy |
|---|---:|---:|
| Base | 64 | 81.013% |
| GEPA | 64 | 81.013% |
| SCOPE | 64 | 81.013% |
| AHO | 64 | 81.013% |
| EvoTool | 57 | 72.152% |
| PromptEvo | 64 | 81.013% |

当前 PromptEvo held-out 与 Base 持平，答辩中应表述为“协议 patch 在不损伤 held-out 的前提下改善了部分结构指标/回归风险”，不要伪造成 held-out 提升。

来源：`raw/promptevo/api_bank/*_summary.json`（这些是 79 条 held-out 的 compact summary）；389 条 full coverage 的 split 信息在原始目录 `full_set_summary.json` 中。

### 2.4 AgentDojo（n=1081）

#### 周报中的 2026-09-04 LongCat 汇总快照

这是当前答辩材料曾采用的完整 1081 条快照，来源为 `raw/promptevo/20260901后实验周报整理.md`。它与 2026-09-15 raw rerun 不同，不能混合。

| 阶段 | Utility | Security | Balanced | Attacked Utility | Attacked Security | Attack Success | Tool Error | Avg Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 73.45% | 35.43% | 61.53% | 71.55% | 26.45% | 73.55% | 13.51% | 10.10 |
| Stage1 | 72.43% | 35.34% | 62.04% | 70.07% | 26.34% | 73.66% | 14.25% | 10.08 |
| Stage2 | 71.97% | 35.15% | 61.79% | 69.55% | 26.13% | 73.87% | 14.06% | 10.15 |
| Stage3 | **74.19%** | 34.14% | **62.28%** | **72.18%** | 24.97% | 75.03% | 14.80% | 10.08 |

Stage3 的 balanced 略高，但 security 下降、attack success 上升；只能说明效用—安全权衡，不应声称全面优于 Base。

#### 本地可直接复核的 2026-09-15 raw rerun

| 阶段 | Utility | Security | Balanced | Attacked Utility | Attacked Security | Attack Success | Tool Error | Avg Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 74.75% | 32.38% | 60.56% | 73.13% | 22.97% | 77.03% | 16.93% | 10.06 |
| Stage1 | 71.88% | 34.32% | 60.86% | 69.76% | 25.18% | 74.82% | 14.06% | 10.03 |
| Stage2 | 74.10% | 33.58% | 61.11% | 72.39% | 24.34% | 75.66% | 14.43% | 10.00 |
| Stage3 | 74.65% | 33.21% | 61.18% | 73.02% | 23.92% | 76.08% | 16.00% | 10.11 |

两组数字不同是因为实验日期/配置不同。论文或 PPT 必须选定一组并写清 manifest，不要跨批次拼接。

### 2.5 tau2-bench（Qwen3-8B，n=1500，LongCat 外部 agent）

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| Success / avg reward | 8.47% | 9.20% | 9.20% |
| avg_db_reward | 10.27% | 10.33% | **11.13%** |
| avg_action_reward | 1.13% | 1.27% | **1.33%** |
| avg_communicate_reward | **10.67%** | 9.67% | 10.40% |
| avg_nl_reward | 17.33% | 18.13% | **19.27%** |
| db_failure_rate | **47.40%** | 49.87% | 50.40% |
| communicate_failure_rate | 0.80% | 1.33% | 1.33% |
| max_steps_rate | 14.20% | 14.67% | 15.73% |
| error_termination_rate | 28.13% | 25.13% | **22.73%** |
| infrastructure_error_rate | 2.80% | **1.00%** | 1.07% |
| avg_messages | 43.24 | 45.42 | 47.66 |
| avg_tool_calls | 11.10 | 11.76 | 11.84 |
| avg_duration_s | 60.58 | 65.69 | 77.59 |

领域成功率（airline / banking / retail / telecom）：

| 方法 | airline | banking | retail | telecom |
|---|---:|---:|---:|---:|
| Base | 17.50% | 0.26% | 16.23% | 3.73% |
| Stage1 | 15.00% | 1.29% | **19.30%** | 3.29% |
| Stage2 | 15.00% | **2.84%** | 17.54% | 3.73% |

tau2 的 reward 有局部上涨，但数据库失败、平均调用和时延没有同步改善，适合用于“规则 patch 的收益具有场景依赖”这一分析。

### 2.6 OEA LongCat2 think（n=1162）

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| success_rate | **86.1%** | 83.7% | 83.3% |
| set-F1 | .685 | **.690** | .685 |
| multiset-F1 | .618 | **.622** | .613 |
| exact_match | 15.4% | 17.0% | 17.0% |
| ordered_exact | 10.4% | 13.0% | 12.8% |
| AnyOrder | 58.1% | **59.3%** | 58.6% |
| SameOrder | 56.7% | 57.6% | **57.7%** |
| Unique | 62.7% | **63.7%** | 63.3% |
| F1 perception | **31.21** | 30.68 | 29.98 |
| F1 operation | **34.29** | 33.95 | 30.62 |
| F1 logic | 32.09 | **32.78** | 29.12 |
| F1 gis | 80.87 | **82.11** | 81.11 |
| cap_rate | **8.7%** | 11.3% | 12.0% |
| errors/task | .57 | **.56** | .63 |
| tools/task | 6.62 | 6.71 | 6.83 |
| llm/task | 7.53 | 7.60 | 7.71 |
| total tokens | 65,590,209 | 68,611,408 | 70,794,650 |
| avg_duration_s | 100.8 | 127.3 | 114.3 |

Stage2 相比 Base：success -2.58 pp，set-F1 持平，exact +1.64 pp，AnyOrder +0.69 pp，GIS F1 +0.30；同时成功任务净减少 30。不能只挑 exact 指标下结论。

### 2.7 OEA Qwen3 matched contrastive（n=906）

这是交集任务上的对比口径，来源于中期材料，不与 1162 条 LongCat2 think 表混合。

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| Set F1 | .628 | .653 | **.661** |
| Exact | 22.85% | 32.23% | **34.55%** |
| Ordered | 16.11% | 20.97% | **23.18%** |
| Success | **86.87%** | 未作为主要结论 | 85.21% |

### 2.8 StableToolBench 旧口径（n=765，历史诊断，不作主表）

| 阶段 | Success | Avg calls | No-finish | Give-up | Tool error | Transient error |
|---|---:|---:|---:|---:|---:|---:|
| Base | 59.22% | 6.80 | 17.91% | 30.33% | 70.20% | 68.76% |
| Stage1 | 58.82% | 6.75 | 18.43% | 30.20% | 69.15% | 67.84% |
| Stage2 | 57.52% | 7.08 | 18.82% | 29.93% | 69.28% | 68.37% |

Stage2 主要出现 `request invalid/status 500`，平均调用增加，说明静态提示补丁无法替代参数构造和运行时恢复。

### 2.9 AgentDojo 旧 meta-v2（历史结果）

| 指标 | Base | Stage1 | Stage2 |
|---|---:|---:|---:|
| success_rate | 7.40% | 7.40% | 7.31% |
| utility_rate | 30.90% | **31.17%** | 30.80% |
| security_rate | 24.05% | 23.96% | 24.05% |
| balanced_score | 27.49% | **27.90%** | 27.49% |
| clean_utility_rate | 40.21% | **41.24%** | 40.21% |
| injection_task_utility_rate | **62.86%** | 60.00% | 60.00% |
| attacked_utility_rate | 28.77% | **29.08%** | 28.77% |
| attacked_security_rate | 13.49% | 13.38% | 13.49% |
| attack_success_rate | 86.51% | 86.62% | 86.51% |
| runtime_error_rate | 1.67% | **1.57%** | 1.76% |
| tool_error_rate | 18.87% | **18.50%** | 18.50% |
| avg_turns | 10.16 | **10.15** | 10.19 |
| avg_tool_calls | 3.81 | 3.81 | 3.84 |
| avg_duration_s | 11.83 | **11.58** | 11.93 |

## 3. 已完成 RL 与 Qwen2.5-3B Base 的 OEA 对比

### 3.1 统一 OEA full test（n=1162）

Base 的 raw 来源是 `A_Base_3B_standard_20260903`；Pure GRPO 是 Swift checkpoint-1900 的稳定重跑；ExperienceEvo 是同一 checkpoint 加 v4-clean 运行时经验。三者不是同一种模型条件：后两者是训练后的 checkpoint，ExperienceEvo 还额外注入经验，因此应按实验链路解释。

| 指标 | Qwen2.5-3B Base | Pure GRPO ckpt-1900 | Pure GRPO + ExperienceEvo v4-clean |
|---|---:|---:|---:|
| n | 1162 | 1162 | 1162 |
| Success rate | 71.86% | 70.65% | **73.75%** |
| status completed / failed / limited | 1138 / 24 / 0 | 1130 / 30 / 2 | 1140 / 19 / 3 |
| Set precision / recall / F1 | .588 / .493 / **.5142** | .583 / .488 / .5102 | .581 / **.506** / .5137 |
| Multiset precision / recall / F1 | 未在 Base raw report 中记录 | .502 / .474 / **.4514** | .478 / .460 / .4300 |
| Exact / ordered exact | 未在 Base raw report 中记录 | 12.39% / 6.28% | **12.82% / 6.37%** |
| AnyOrder / SameOrder / Unique | .0723 / .0706 / .1368（legacy ToolOrder） | 15.06% / 14.54% / 18.67% | **16.01% / 15.06% / 22.55%** |
| Tool error | **48.45%** | 49.74% | **34.17%** |
| F1 perception / operation / logic / GIS | .3198 / .3025 / n/a / .7025 | 23.07 / 20.88 / 25.30 / **63.82** | **34.51** / 16.53 / 24.92 / 59.65 |
| Empty-call rate | 未在 Base raw report 中记录 | 5.94% | 13.51% |
| Cap rate | 未在 Base raw report 中记录 | 2.07% | **1.29%** |
| Errors / tools / LLM per task | 未在 Base raw report 中记录 | 1.57 / 3.75 / 4.73 | **1.14 / 3.63 / 4.74** |
| Tokens / time per task | 36,197.8 / 未统一记录 | 35,608 / 103.65s | 42,081 / 86.30s |

注：Base 的 `metrics_suite.json` 用 legacy `ToolOrder_*` 字段记录顺序指标，Pure GRPO/ExperienceEvo 用新的 `any_order/same_order/unique` 字段；这几列不能当成严格同口径的逐项比较。Base 的原始总 token 是 42,061,802，表中 36,197.8 是按 1162 条任务换算的平均值。

### 3.2 RL 结果解读

- Pure GRPO 本身没有稳定超过 Base：Success -1.21 pp，Set-F1 -0.0040，Tool error +1.29 pp。
- 加入 ExperienceEvo 后，Success 比 Base +1.89 pp、比 Pure GRPO +3.10 pp；Tool error 比 Base -14.28 pp、比 Pure GRPO -15.57 pp，但 Set-F1 只比 Base -0.0005，说明主要改善了工具错误/执行效率，不应写成所有答案质量指标均提升。
- ExperienceEvo 版本的 empty-call rate 更高，GIS F1 低于 Base/Pure GRPO；这提示经验注入可能造成部分任务过早停止或领域偏置，需要后续分领域诊断。

### 3.3 Swift online GRPO 训练过程指标（n=1900 steps）

实验目录：`tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_catalogprompt_obplaceholder_turn6_resp128_train2000_autofallback_20260912`。

| 指标 | 值 |
|---|---:|
| Base model | Qwen2.5-3B-Instruct |
| Backend | MS-Swift / TRL GRPO |
| OEA train / val | 1900 / 100 |
| num_generations | 2 |
| batch / grad accumulation | 1 / 4 |
| LoRA rank | 8 |
| learning rate | 1e-6 |
| max turns / completion length / max length | 6 / 128 / 8192 |
| local rollout forward batch | 1 |
| completed steps | 1900 / 1900 |
| stitched reward mean | .7964 |
| v4-only reward mean | .9575 |
| recent 20 reward mean | 1.0380 |
| recent 50 reward mean | 1.0449 |
| KL mean / recent 50 | .00316 / .00341 |
| recent 50 grad norm | .7845 |
| peak/latest memory | 22.32 GiB |
| speed | 84.49 s/it |
| final checkpoint | step 1900，已保存并合并 HF |

### 3.4 已做但不能作为正式能力结论的 RL 结果

#### Swift checkpoint 的污染 eval

目录：`tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_ckpt1900_oea_standard_eval_tok24576_stable3_20260914`。1162 个文件齐全，但有 569 条 agent LLM 容器冲突/退出 exception。

| 指标 | Qwen3B Base | 污染 Swift eval |
|---|---:|---:|
| completed / failed / exception | 1138 / 24 / 0 | 582 / 9 / 569 |
| Success | 71.9% | 28.7% |
| Tool error | 48.5% | 13.3%（受 exception 污染） |
| Set P/R/F1 | .588/.493/.514 | .216/.178/.186 |
| Multiset P/R/F1 | .498/.479/.450 | .193/.156/.158 |
| Exact / Ordered | 13.7% / 7.1% | 4.3% / 1.8% |
| OEA Any / Same / Unique | 16.6% / 16.1% / 19.9% | 3.4% / 3.4% / 6.1% |
| Empty-call / tools per task | 5.9% / 3.84 | 51.3% / 1.38 |

这版只能作为基础设施故障诊断，不能用来比较模型能力。

#### 早期离线/规则 reward 与 action eval

| 实验 | Trace 数 | Parse | Reward mean |
|---|---:|---:|---:|
| Pure GRPO + QLoRA | 7600 | 97.82% | .934 |
| ExperienceEvo reward + GRPO + QLoRA | 3648 | 96.11% | .133 |

Pure GRPO action eval（1162 条离线动作，不是真实工具 rollout）：Parse 54.99%，Single action 49.05%，Legal tool 54.91%，Required args 54.88%，First-tool accuracy 30.12%，First-tool in gold 39.67%，Set-F1 .234，Multiset-F1 .186，Exact .95%，Reward mean .0186，耗时约 5.91h。它只能说明离线 action 口径与真实 rollout 不一致，不能代替正式 OEA eval。

#### veRL pure online GRPO

目录：`tmp/experience_evo_rl/qwen25_3b_pure_online_grpo_tp2_noactoroff_tok8192_util015_full1900_train2000_20260909`。已保存 step 50/100/150/200，约运行到 step 206，未完成最终 checkpoint 和正式 OEA 评测；状态为未完成，不列入能力主表。

## 4. 对答辩最稳妥的主结论

1. PromptEvo：ToolBench Stage1 在冻结 split 上成功率由 86.25% 提升到 88.12%，工具错误由 14.37% 降到 7.50%；Stage2 回退，证明 patch gate/rollback 是必要的。API-Bank held-out 与 Base 持平，因此主张“稳健性和回归控制”，不要主张全面泛化提升。
2. ExperienceEvo + RL：标准 GRPO 单独没有稳定超过 Base；同一训练 checkpoint 加 ExperienceEvo 后，Success 73.75% 高于 Base 71.86%，Tool error 34.17% 明显低于 Base 48.45%，但 Set-F1 基本持平，需诚实表述为执行可靠性改善。
3. 产物来源感知 GRPO 是下一步算法方案，不应与已完成的标准 GRPO 结果混为已验证创新。
