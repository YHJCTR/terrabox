# ReAct 基线

该目录用于跑“无记忆、无训练”的原生 Qwen3-8B ReAct 真实工具调用轨迹。

输入建议使用 `data/newdata/sft_train_strict.jsonl`。runner 会先生成不包含 SFT gold assistant/tool 消息的 `tasks.json`，再调用仓库已有的 `scripts/run_trajectory_experiment.py`。

小规模测试：

```bash
PYTHONPATH=src no_proxy=localhost,127.0.0.1 \
/home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.ReAct.runner smoke \
  --strict-data data/newdata/sft_train_strict.jsonl \
  --limit 5 \
  --port 9100 \
  --agent-gpu 0 \
  --tool-gpu 1
```

输出默认放在 `src/terrabox/evolution/ReAct/exp/{experiment}/`。

