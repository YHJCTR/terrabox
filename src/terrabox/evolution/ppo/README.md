# GRPO / PPO 基线

该目录用于把 `data/newdata/sft_train_strict.jsonl` 转成 veRL 可用的数据，并启动 Qwen3-8B LoRA GRPO 基线。

第一版 GRPO reward 是静态工具策略 reward：它根据模型生成的工具调用文本和 gold 工具序列打分，不在 veRL 采样阶段真实执行 Terrabox 工具。训练后的模型需要再回到 ReAct runner 中做真实工具环境评测。

准备数据：

```bash
PYTHONPATH=src \
/home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.ppo.runner prepare-data \
  --strict-data data/newdata/sft_train_strict.jsonl \
  --experiment grpo_qwen3_8b_strict \
  --parquet
```

生成训练命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src \
/home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.ppo.runner train-grpo \
  --experiment grpo_qwen3_8b_strict \
  --verl-dir /data1/yuhongjie2/verl \
  --max-steps 20
```

