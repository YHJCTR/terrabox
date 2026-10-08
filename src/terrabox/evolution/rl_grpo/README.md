# Pure GRPO 真实工具基线

`rl_grpo/` 是纯 GRPO baseline 的方法目录。它不默认注入 ExperienceEvo 经验，也不打开 QNR/Quse 等后续自定义 reward 组件。

公共训练框架、真实工具环境、Swift/veRL 适配、上下文压缩和 trace schema 统一由 `src/terrabox/evolution/agent_rl/` 提供。本目录只负责把方法名固定为 `rl_grpo`，避免 pure baseline 的实验产物混到 `experience_evo_rl/`。

## 推荐入口

```bash
PYTHONPATH=src python -m terrabox.evolution.rl_grpo.runner prepare-data \
  --experiment qwen25_3b_swift_grpo_train2000_YYYYMMDD \
  --limit 2000

PYTHONPATH=src python -m terrabox.evolution.rl_grpo.runner write-swift-command \
  --experiment qwen25_3b_swift_grpo_train2000_YYYYMMDD
```

等价公共入口为：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner prepare-data \
  --method rl_grpo \
  --experiment qwen25_3b_swift_grpo_train2000_YYYYMMDD \
  --limit 2000
```

## 输出位置

默认输出到：

```text
tmp/agent_rl_runs/rl_grpo/<experiment>/
  configs/
  data/
  metrics/
  artifacts/
  checkpoints/
```

训练完成后仍需在固定 OEA test manifest 上做完整真实工具 rollout，训练 reward 不能直接写作 OEA 主指标。

## 当前稳定配置记录

2026-09-12 的 Swift pure GRPO 正式重跑使用：

```text
tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_catalogprompt_compobs_vllm35_train2000_20260912
```

关键配置：`--limit 2000`、1900 train / 100 val、`num_generations=2`、`max_length=8192`、`max_completion_length=1024`、`vllm_gpu_memory_utilization=0.35`、`vllm_max_num_seqs=1`、`dataloader_num_workers=0`、`dataset_num_proc=1`。工具观察压缩由公共层控制：raw observation 完整落盘，模型上下文只接收 1024-char 以内的结构化摘要；长 mask/数组/大 JSON/长文本使用 `<omitted_*:hash>` 占位符，避免多轮真实工具 rollout 在 GRPO backward 阶段触发显存峰值 OOM。
