# ExperienceEvo-guided RL（strict no-label）

该模块包含两条口径不同的实验链路：

1. **离线首动作 GRPO 诊断**：只训练/评测单步 JSON action，不执行 Terrabox 工具。该链路只用于格式、显存和 reward 工程诊断，不能作为正式 OEA agent RL 主结果。
2. **online veRL 真实工具 RL**：通过 veRL multi-turn `ToolAgentLoop` 接入真实 Terrabox 工具，模型生成动作后执行工具、回填 observation，并由完整 episode 的可观察事件计算 reward。这是后续 Qwen 3B/4B RL 的正式主线。

## 口径

- 训练 prompt 只包含任务公开文本、公开输入文件和公开工具目录；可选注入 ExperienceEvo 的 rollout-derived 过程经验摘要。
- 离线 reward 只使用工具名合法性、参数 schema 合法性、单 action 约束、ExperienceEvo 高 Q/低 risk 工具策略匹配和重复/高风险惩罚。
- online reward 由真实工具 episode 的可观察事件组成：工具调用合法性、工具执行状态、artifact 产出、空输出/错误/超时、轮数和 token 长度；ExperienceEvo prior 只能作为显式消融项加入，不得静默混入 pure baseline。
- 不读取或序列化 `task_id`、`task_type`、`expected_tools`、`gold_tool_calls`、`ground_truth`、gold-derived `metrics/F1/reward`。
- 不设置 `TERRABOX_TOOL_GPU_DEVICES`，避免覆盖仓库约定的 per-service GPU 钉卡。

## 当前主线

正式主线已切到 online veRL：

- 工具包装：`online_tools.py` 中的 `TerraboxOeaTool`，把 veRL function name 映射回 Terrabox canonical slug，并通过 `AgentToolExecutor.execute()` 执行真实工具。
- Agent loop：`online_agent_loop.py` 中的 `TerraboxToolAgentLoop`，继承 veRL `ToolAgentLoop`，在 episode 结束时写入 `rm_scores` 和 `metrics/online_episode_traces.jsonl`。
- 数据生成：`data_builder.py --online` 生成 veRL multi-turn rows；prompt 只含公开任务、公开文件和公开工具 schema，不含 test/train gold 字段。
- 配置生成：`runner.py write-online-configs` 生成 `configs/terrabox_oea_tool_config.yaml` 和 `configs/terrabox_agent_loop_config.json`。
- 训练命令：`runner.py train-grpo --online` 默认只写 `run_grpo_command.sh`；只有显式 `--launch` 才启动 veRL。
- 显存策略：online LoRA 训练启用 veRL 的 `layered_summon=True`。它在 actor 向 colocated vLLM 同步 adapter 时逐层 materialize FSDP 参数，避免为了导出 LoRA 而构造完整模型 state dict；这是 24GB GPU 下保持 10,240 token 多轮上下文的必要配置。
- 长上下文显存策略：pure online GRPO 的 `entropy_coeff=0` 且 `calculate_entropy=False`。本地 veRL 的 old-log-prob 路径已修正为遵循该配置，因而不会在系数为零时无意义地构造完整词表 entropy；这不改变 reward、GRPO 损失或 agent 行为。若后续实验启用 entropy 正则，则可改为 `calculate_entropy=True` 并使用 chunking/checkpointing 控制峰值显存。
- 服务并发策略：online OEA 固定 `agent.num_workers=1`，因为 GPU 感知服务以跨进程锁串行。感知工具使用 420 秒超时，避免将服务冷启动或同 lane 等待错误写成负奖励；每个 GRPO group 内的多个 rollout 仍会完整执行。
- validation 安全策略：online OEA 不依赖 veRL 默认的全量同批 validation。默认应显式设置 `data.val_batch_size`、`data.val_max_samples` 和 `data.dataloader_num_workers`；在 24GB GPU + 126GiB CPU 内存机器上，2000 条训练的安全起点是 `val_max_samples=8`、`val_batch_size=1`、`dataloader_num_workers=0`、`val_before_train=False`。这只限制训练期健康检查，不改变完整 1900 条 train 数据，也不替代最终 OEA test 全量评测。
- 记忆压缩：工具原始 observation 先完整写入 `metrics/raw_tool_observations.jsonl`，再由 `online_tools.py` 的确定性压缩器送回模型。压缩器保留头部、尾部以及错误、artifact 路径、统计值和图层等关键行，默认上下文预算为 8192 字符（可用 `TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS` 调整）。`metrics/online_episode_traces.jsonl` 记录每次调用的原始/压缩字符数和保留行数。训练与评测必须使用同一预算；改变预算后旧 online 结果不可直接混合。
- 显存配置：pure online GRPO 关闭 actor KL/entropy 分支（`use_kl_loss=False`）。实验不使用 KL reward，因此不改变 reward；关闭它可避免 24GB 卡在 old-logprob 阶段为全词表 entropy 额外申请数 GB 显存。online 训练默认将 `actor_ppo_max_token_len_per_gpu`、`rollout_log_prob_max_token_len_per_gpu` 和 `ref_log_prob_max_token_len_per_gpu` 设为 8192；如果再次在 old-logprob 或 actor update 阶段 OOM，优先降低这些训练侧 token 上限，不要先降低任务 prompt、response 或工具 observation 的语义预算。
- 长序列配置：如果真实 rollout 序列超过训练侧 token 上限，veRL 会在 old-logprob 动态切 batch 前直接断言失败；此时应把训练侧 token 上限升到覆盖实际序列长度（当前 3B online 安全值 10240），并同步降低 `train_batch_size` 到 1 控制显存。不要用低于实际序列长度的 max-token 上限伪装“降并发”。
- 不要给 online veRL/vLLM 路径设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；当前 colocated vLLM 会走 V1 memory pool，该配置会让 engine core 初始化失败。
- RemoteSAM：online runner 默认将 `/data1/yuhongjie2/RemoteSAM/pretrained_weights` 作为 `REMOTESAM_CHECKPOINT_HOST` 挂载，并关闭 EPOC，保证离线加载本地 BERT 权重；若路径不存在，应先中止实验而不是让服务错误进入 reward。

当前已生成的 pure online GRPO 数据/配置：

```text
tmp/experience_evo_rl/qwen25_3b_online_grpo_train2000_true_tools_20260905/
  verl_data/train.jsonl          # 1900 条
  verl_data/val.jsonl            # 100 条
  verl_data/train.parquet
  verl_data/val.parquet
  configs/terrabox_oea_tool_config.yaml      # 23 个 OEA 工具
  configs/terrabox_agent_loop_config.json
  metrics/online_episode_traces.jsonl        # 训练启动后追加
  run_grpo_command.sh                         # veRL online GRPO 命令
```

启动前必须确认至少一张训练卡和一条工具/感知 lane 可用。若 GPU0--2 被其它实验占用，只允许在 GPU3 做不含感知的工程检查；不能把这种检查写成完整 OEA online RL 结果。

## 历史离线链路

历史 Qwen2.5-3B 离线诊断使用 TRL + PEFT + bitsandbytes 的单卡 QLoRA GRPO，不走 veRL colocated vLLM/FSDP。它可以稳定完成 QLoRA forward/backward，但不执行真实工具，因此不再作为正式主线。

资源约束：训练固定 `CUDA_VISIBLE_DEVICES=3`，GPU0/1/2 保留给 Terrabox 工具和感知服务；训练不启动 vLLM、不执行 Terrabox 工具、不调用 LongCat/DeepSeek。

已完成的 20-step pilot 显示：

| Variant | Eval-50 parse | Eval-50 first-tool acc. | Eval-50 set F1 | 结论 |
|---|---:|---:|---:|---|
| 3B base schema | 32.0% | 24.0% | 0.163 | 未训练基线 |
| Pure GRPO 20-step | 38.0% | 30.0% | 0.191 | 有小幅正向 |
| ExperienceEvo prompt only | 14.0% | 8.0% | 0.064 | 长经验 prompt 明显伤格式 |
| ExperienceEvo GRPO no prompt | 42.0% | 28.0% | 0.184 | parse/reward 更稳，工具命中略弱于 pure |
| ExperienceEvo GRPO + prompt | 18.0% | 10.0% | 0.083 | 不适合直接放大 |

因此当前优先路线是 `ExperienceEvo reward-only / no-prompt`：prompt 不注入检索经验，reward policy 使用 top-1 ExperienceEvo 推荐工具。这样保留经验作为训练信号，同时避免长上下文扰乱 3B 小模型的 JSON 格式遵循。

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.experience_evo_rl.runner prepare-data \
  --experiment qwen25_3b_expevo_reward_noprompt_grpo_qlora_s100_20260901 \
  --limit 500 \
  --top-k 0 \
  --prompt-top-k 0 \
  --reward-top-k 1

tmux new -d -s expevo_rl_qwen25_3b_reward_noprompt_s100_20260901 \
  "cd /data1/yuhongjie2/terrabox && \
   export CUDA_VISIBLE_DEVICES=3 && \
   export PYTHONPATH=src && \
   export TOKENIZERS_PARALLELISM=false && \
   export WANDB_DISABLED=true && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
     -m terrabox.evolution.experience_evo_rl.train_qlora_grpo \
     --experiment qwen25_3b_expevo_reward_noprompt_grpo_qlora_s100_20260901 \
     --model-path /data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct \
     --max-steps 100 \
     --num-generations 2 \
     --per-device-train-batch-size 2 \
     --gradient-accumulation-steps 4 \
     --max-completion-length 192 \
     --learning-rate 1e-5 \
     --lora-rank 8 \
     --lora-alpha 16 \
     --logging-steps 1 \
     --save-steps 25 \
     2>&1 | tee tmp/experience_evo_rl/qwen25_3b_expevo_reward_noprompt_grpo_qlora_s100_20260901/train_tmux.log"
```

评测入口为离线 action 评测：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.experience_evo_rl.eval_actions \
  --experiment qwen25_3b_eval50_20260901 \
  --limit 50 \
  --max-new-tokens 192 \
  --temperature 0
```

该评测只在 held-out test 上生成第一步 action，并在生成后才使用 gold 工具序列计算离线指标；gold 不进入 prompt/reward。

## veRL smoke 备选

当前环境中 `flash_attn` 未安装，因此 runner 默认关闭 veRL 的
`actor_rollout_ref.model.use_remove_padding`。1.5B 是当前最稳的闭环模型；
3B/4B 可以完成数据构建、FSDP 和 vLLM 初始化，但在 24GB 3090 上容易卡在
colocated vLLM + FSDP LoRA 的权重同步显存峰值。

注意：`top-k` 建议先设为 1，避免 ExperienceEvo 经验块过长导致 veRL prompt
过滤把训练集清空。正式 pure online GRPO baseline 应使用 `--prompt-top-k 0 --reward-top-k 0`，避免经验进入 prompt 或 reward。24GB 3090 上优先用 `--train-batch-size 2 --rollout-n 2` 跑通完整一轮；稳定后再把 `rollout_n` 或 batch 回升。

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.experience_evo_rl.runner prepare-data \
  --experiment qwen25_15b_expevo_rl_smoke_b8_20260831 \
  --limit 100 \
  --top-k 1

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.experience_evo_rl.runner reward-dry-run \
  --experiment qwen25_15b_expevo_rl_smoke_b8_20260831 \
  --limit 5

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.experience_evo_rl.runner train-grpo \
  --experiment qwen25_15b_expevo_rl_smoke_b8_20260831 \
  --model-path /data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_1.5B_Instruct \
  --n-gpus 4 \
  --cuda-visible-devices 0,1,2,3 \
  --max-steps 3 \
  --rollout-backend vllm \
  --train-batch-size 8 \
  --max-prompt-length 2048 \
  --max-response-length 128 \
  --rollout-n 1 \
  --rollout-gpu-memory-utilization 0.35 \
  --rollout-max-model-len 2048 \
  --rollout-max-num-batched-tokens 2048 \
  --rollout-max-num-seqs 1 \
  --lora-rank 8 \
  --lora-alpha 16
```

最后一步默认只写 `tmp/experience_evo_rl/<experiment>/run_grpo_command.sh`，不会自动启动。确认 GPU 空闲后再追加 `--launch` 或手动用 tmux 启动脚本。
