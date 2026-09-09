# EvolveR lifecycle strict adapted baseline

本模块用于在 Terrabox/OEA 上做 EvolveR 风格的第三方对比实验。它是 **official-source-guided adapted**，不是完整官方 EvolveR RL 复刻。

## 官方机制对齐点

- 参考官方仓库 `Edaizi/EvolveR`，当前本地源码位于 `/data1/yuhongjie2/EvolveR`，审计 HEAD 为 `63834b727ee6e7af3410657de36eb845814249ba`。
- 保留 EvolveR 的核心 lifecycle 抽象：历史 trajectory → `guiding` / `cautionary` principles → semantic retrieval → runtime prompt injection。
- 保留官方 principle schema 的关键字段：`principle_id`、`type`、`description`、`structure`、`metric_score`、`usage_count`、`success_count`、`successful_trajectory_ids`、`failed_trajectory_ids`。
- 保留官方 distillation prompt 的输出格式：`[DESCRIPTION]:` + `[STRUCTURE]:`。

## 适配边界

- 不复刻官方 veRL / GRPO 训练循环。
- 不复刻 NQ/HotpotQA 专用 retriever、VDB server 和 `<search_experience>` reward shaping。
- 不训练或替换 OEA agent LLM；只做 frozen-agent prompt augmentation。
- 因此论文/周报中应命名为 `EvolveR lifecycle (adapted, strict no-label)`，不能写成完整官方 EvolveR。

## strict no-label 口径

- 经验库不保存 `task_id`、`task_type`、`expected_tools`、gold answer、metrics/F1、最终答案事实、绝对路径和 OEA train/test identifier。
- 原始训练来源只在 manifest 中记录为 `LongCat OEA train2000 base rollout`，不写具体结果目录路径。
- 成功/失败只用 rollout 可见状态判定：completed 且无 tool error 生成 guiding principle；非基础设施 tool-use failure 生成 cautionary principle；API/网络/Docker/OOM/context 等基础设施失败跳过。
- 正式 strict rollout 必须设置 `TERRABOX_EVOLVER_RETRIEVAL=qwen` 并存在 `qwen_embedding_index.json`，避免 lexical fallback。

## 典型流程

先做小样本构建和泄漏检查：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.evolver_lifecycle.runner build-strict \
  --source-results tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --store-dir evolution_store/evolver_lifecycle/oea_train2000_longcat_strict_nolabel_20260831_smoke \
  --limit 5 \
  --llm-provider longcat
```

Qwen embedding 服务可用后补索引：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.evolver_lifecycle.runner build-strict \
  --source-results tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --store-dir evolution_store/evolver_lifecycle/oea_train2000_longcat_strict_nolabel_20260831_smoke \
  --limit 5 \
  --llm-provider longcat \
  --embedding-backend qwen
```

检索 smoke：

```bash
TERRABOX_EVOLVER_RETRIEVAL=qwen \
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.evolver_lifecycle.runner retrieve-smoke \
  --store-dir evolution_store/evolver_lifecycle/oea_train2000_longcat_strict_nolabel_20260831_smoke \
  --query "Generate a burn severity summary from pre-fire and post-fire imagery."
```

OEA rollout smoke 通过后，正式 full eval 应沿用外部 API 分 lane 规则：`gpu --workers 1` 与 `nogpu --workers 2-4` 两条并行、共享结果目录并 `--resume`。
