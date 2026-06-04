# SFT Baseline

这个目录保存 Terrabox 当前 strict 数据上的 SFT baseline。目标是训练一个后续方法可复用的 Qwen3-8B 冷启动模型，让模型先学会当前 Terrabox 工具名、JSON action 格式、常见调用顺序和任务风格。

## 当前推荐路线

正式训练推荐使用 **veRL FSDP SFT 后端**，不是原来的 Unsloth 单进程后端。

原因：

- Unsloth/TRL 路线虽然能看到 4 张卡，但日志显示 `Data Parallel GPUs = 1`，不能算真正 4 卡联合训练。
- 第一版 compact SFT 数据最长约 11467 tokens；16k Unsloth backward 已实测 OOM。
- veRL 的 `verl.trainer.sft_trainer` 支持 `torchrun + FSDP`，更适合 4 张 3090 联合训练。

当前正式实验使用：

```text
experiment: shuffle_seed42_sft90_10_compact_fulltools
train: 7781 条
val/test: 865 条
max_length: 13312
backend: veRL FSDP
GPU: 0,1,2,3
base_model: /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/
```

## 当前可跑配置：3090-safe

直接用 13k、`train_batch_size=4`、LoRA rank 16 的第一版 veRL 配置会在 `loss.backward()` OOM。实测 OOM 时 GPU1-3 只剩约 0.3-1.2GiB 空闲。

当前已验证可以进入训练的配置是 `--preset 3090-safe`：

```text
max_length: 13312
max_token_len_per_gpu: 13312
nproc_per_node: 4
train_batch_size: 1
micro_batch_size_per_gpu: 1
lora_rank: 8
lora_alpha: 16
lora_target_modules: all-linear
sequence_parallel_size: 4
param_offload: True
optimizer_offload: True
activation_offload: True
use_torch_compile: False
checkpoint.save_contents: ['model', 'extra']
save_freq: 1000
test_freq: -1
```

这里默认只保存 veRL/FSDP shard checkpoint，不在训练过程中保存 HuggingFace 完整权重。原因是 8B 模型的 `hf_model` 保存会触发 CPU full gather，上一次实测会把 126GiB CPU 内存打满并导致服务器卡死。

这不是“只保存了一部分模型”。`['model','extra']` 会保存完整的 SFT 训练 checkpoint，只是以 FSDP 分片形式保存：

- `model_world_size_4_rank_*.pt`：4 张卡各自的模型 shard，合起来是完整 LoRA/SFT checkpoint。
- `extra_state_world_size_4_rank_*.pt`：训练器/优化器/调度器等额外状态，用于断点续训。
- `lora_train_meta.json`、tokenizer/config 等元数据：用于后续受控转换或恢复。

区别是：这个 checkpoint 不能直接作为 HuggingFace/vLLM 模型加载。后续如果要真实 rollout，需要单独做“FSDP shard -> HF/LoRA checkpoint”的受控转换，并在转换时限制 CPU 内存占用。

验证记录：

```text
experiment: shuffle_seed42_sft90_10_compact_fulltools_safe13k
tmux: sft_verl_safe13k
log: src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_safe13k/verl_sft_train.log
step 1 loss: 1.2994
step 2 loss: 1.1939
first-step max_memory_allocated_gb: 16.92
first-step max_memory_reserved_gb: 22.09
runtime memory after training starts: about 13.0GiB per GPU in nvidia-smi
```

这个配置的代价是慢：日志中的早期估算约 27 小时完成 1 epoch。它适合作为“先跑通且保存完整 shard checkpoint”的保守方案；如果要加速，再考虑缩短 max_length、进一步压缩数据或做更激进的 batch/LoRA 设置。

注意：`test_freq=-1` 表示不按固定步数做中途 validation；如果命令仍提供 `data.val_files`，veRL 在最后一步仍会做一次 validation。正式训练保留 validation 文件，用于得到训练结束时的 val/loss。

## 数据压缩策略

原始 strict SFT 中有 EarthBench 长时序/光谱任务，单条样本可能包含上百个 TIFF 路径和很长的工具返回列表。直接训练会导致上下文过长和 OOM。

当前 compact 数据的策略是：

- 不压缩 system 中的完整工具 catalog。
- 只压缩超长 `OBSERVATION` 文本。
- 只压缩 assistant action arguments 中的超长字符串和长列表。
- 每条样本保留 `sft_compaction` 元数据，便于审计。

第一版 compact 数据参数：

```text
--compact-long-context
--max-observation-chars 600
--max-string-chars 400
--max-list-items 4
--no-compact-system-catalog
```

精确 tokenizer 统计：

```text
train rows: 7781, p50 7235, p90 7672, p95 7763, p99 8237, max 11467
val rows: 865, p50 7235, p90 7685, p95 7766, p99 8254, max 11207
```

因此正式 veRL 训练使用 `max_length=13312`，并设置 `data.truncation=error`，保证不会静默截断。

## 生成 compact SFT 数据

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/terra/bin/python \
  -m terrabox.evolution.sft.runner prepare-data \
  --experiment shuffle_seed42_sft90_10_compact_fulltools \
  --train-start 865 \
  --train-limit 7781 \
  --val-start 0 \
  --val-limit 865 \
  --compact-long-context \
  --max-observation-chars 600 \
  --max-string-chars 400 \
  --max-list-items 4
```

输出：

```text
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/sft_data/train.jsonl
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/sft_data/val.jsonl
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/eval_tasks.json
```

## 转换 veRL Parquet

veRL SFT 需要 Parquet。`terra` 环境当前没有 pyarrow，推荐用 `unsloth` 环境转换：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.sft.runner prepare-verl-data \
  --experiment shuffle_seed42_sft90_10_compact_fulltools
```

输出：

```text
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/verl_data/train.parquet
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/verl_data/val.parquet
```

## 启动 veRL 多卡 SFT

生成脚本：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/terra/bin/python \
  -m terrabox.evolution.sft.runner train-verl \
  --experiment shuffle_seed42_sft90_10_compact_fulltools_safe13k \
  --train-file src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/verl_data/train.parquet \
  --val-file src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools/verl_data/val.parquet \
  --output-model-dir src/terrabox/evolution/sft/model/shuffle_seed42_sft90_10_compact_fulltools_safe13k_verl \
  --cuda-visible-devices 0,1,2,3 \
  --nproc-per-node 4 \
  --max-length 13312 \
  --max-token-len-per-gpu 13312 \
  --preset 3090-safe
```

运行脚本：

```bash
tmux new-session -d -s sft_verl_safe13k \
  'cd /data1/yuhongjie2/terrabox && bash src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_safe13k/run_verl_sft_train.sh'
```

日志：

```text
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_safe13k/verl_sft_train.log
```

模型输出：

```text
src/terrabox/evolution/sft/model/shuffle_seed42_sft90_10_compact_fulltools_safe13k_verl/
```

veRL checkpoint 会保存到：

```text
src/terrabox/evolution/sft/model/<experiment>_verl/global_step_*/
```

默认目录下会有 FSDP shard 文件，不一定有可直接加载的 `huggingface/` 完整权重目录。后续 Terrabox/vLLM rollout 需要先把 shard checkpoint 受控转换成 HuggingFace/LoRA 格式，或者训练时显式加 `--save-hf-model`，但正式长训练不推荐这么做，因为它会触发 CPU full gather。

## 当前半量正式实验

为了避免 8B HF full checkpoint 保存时占满 CPU 内存，当前正式训练使用半量训练集和 shard-only 保存：

```text
experiment: shuffle_seed42_sft90_10_compact_fulltools_half_shard
train: shuffle seed 42 后从第 865 条开始取 3890 条
val/test: shuffle seed 42 后前 865 条
max_length: 13312
backend: veRL FSDP
GPU: 0,1,2,3
checkpoint: FSDP shard only, checkpoint.save_contents=['model','extra']
save_freq: 1000
test_freq: -1，中途不做 validation，最后仍会因提供 val_files 做一次 validation
```

生成数据：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/terra/bin/python \
  -m terrabox.evolution.sft.runner prepare-data \
  --experiment shuffle_seed42_sft90_10_compact_fulltools_half_shard \
  --train-start 865 \
  --train-limit 3890 \
  --val-start 0 \
  --val-limit 865 \
  --compact-long-context \
  --max-observation-chars 600 \
  --max-string-chars 400 \
  --max-list-items 4

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.sft.runner prepare-verl-data \
  --experiment shuffle_seed42_sft90_10_compact_fulltools_half_shard
```

生成训练脚本：

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/terra/bin/python \
  -m terrabox.evolution.sft.runner train-verl \
  --experiment shuffle_seed42_sft90_10_compact_fulltools_half_shard \
  --output-model-dir src/terrabox/evolution/sft/model/shuffle_seed42_sft90_10_compact_fulltools_half_shard_verl \
  --cuda-visible-devices 0,1,2,3 \
  --nproc-per-node 4 \
  --max-length 13312 \
  --max-token-len-per-gpu 13312 \
  --preset 3090-safe
```

启动：

```bash
tmux new-session -d -s sft_half_shard \
  'cd /data1/yuhongjie2/terrabox && bash src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_half_shard/run_verl_sft_train.sh'
```

查看：

```bash
tail -n 80 src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_half_shard/verl_sft_train.log
tmux attach -t sft_half_shard
```

## 2 卡尝试记录

服务器不稳定后，进一步缩小了数据量：

```text
experiment: shuffle_seed42_sft90_10_compact_fulltools_quarter_2gpu_shard
train: shuffle seed 42 后从第 865 条开始取 1945 条
val/test: shuffle seed 42 后前 432 条
```

数据已经生成到：

```text
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_quarter_2gpu_shard/
```

但是当前 veRL/FSDP 后端无法只用 2 张 3090 跑 Qwen3-8B SFT。2 卡 smoke 在 FSDP 初始化阶段、进入训练 step 之前 OOM：

```text
engine.ulysses_sequence_parallel_size=2
trainer.n_gpus_per_node=2
torch.OutOfMemoryError: Tried to allocate 2.32 GiB
GPU 0 free: about 0.95 GiB
GPU 1 free: about 0.46 GiB
```

这个 OOM 和训练/验证数据条数无关，因为它发生在模型 FSDP shard 初始化阶段。减少样本数量不能解决；需要：

- 继续用 4 卡，但设置较低 GPU power limit，降低跳闸风险；
- 或改用真正 4bit/QLoRA 后端；
- 或换更小基础模型；
- 或研究 veRL 是否支持更激进的 CPU/ZeRO/offload 配置。

## 4 卡 250W quarter 实验

2 卡 FSDP 初始化 OOM 后，当前推荐回到 4 卡，但降低 GPU 功耗并缩小数据量：

```text
experiment: shuffle_seed42_sft90_10_compact_fulltools_quarter_4gpu_250w_shard
train: 1945
val/test: 432
GPU: 0,1,2,3
power limit: 250W per GPU
save_freq: 200
checkpoint: FSDP shard only, checkpoint.save_contents=['model','extra']
```

设置功率限制需要 sudo/root 权限：

```bash
sudo nvidia-smi -i 0,1,2,3 -pl 250
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits
```

数据和脚本已生成：

```text
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_quarter_4gpu_250w_shard/
src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_quarter_4gpu_250w_shard/run_verl_sft_train.sh
```

功率限制确认生效后启动：

```bash
tmux new-session -d -s sft_quarter_4gpu_250w \
  'cd /data1/yuhongjie2/terrabox && bash src/terrabox/evolution/sft/exp/shuffle_seed42_sft90_10_compact_fulltools_quarter_4gpu_250w_shard/run_verl_sft_train.sh'
```

## 测试 SFT 模型

训练完成后，用真实 Terrabox 工具链评估 prompt-only eval tasks：

```bash
PYTHONPATH=src no_proxy=localhost,127.0.0.1 /home/yuhongjie/miniconda3/envs/unsloth/bin/python \
  -m terrabox.evolution.sft.runner rollout \
  --experiment shuffle_seed42_sft90_10_compact_fulltools \
  --model-path src/terrabox/evolution/sft/model/shuffle_seed42_sft90_10_compact_fulltools_verl/global_step_*/huggingface \
  --agent-gpu 0 \
  --tool-gpu 1 \
  --limit 865 \
  --launch
```

注意：上面 `global_step_*` 需要替换成实际最新 checkpoint 目录。

## SFT Loss 和真实工具调用

SFT 训练阶段不执行真实工具，也不根据真实工具返回重新计算 loss。

当前流程是：

1. 读取 strict SFT 的 `messages`。
2. 对超长 observation/action 参数做可审计压缩。
3. 转成 veRL MultiTurnSFTDataset Parquet。
4. 用 next-token cross entropy 训练 assistant 输出。
5. 训练完成后再通过真实 Terrabox rollout 测试工具调用效果。

所以 SFT 的意义是冷启动：让模型学会当前系统的工具接口和任务风格。它不是 RL，也不自动验证最终答案语义正确性。

## Unsloth 后端状态

`train_lora.py` 仍保留用于小规模 smoke 或单机调试，但不推荐作为正式多卡 SFT。

已知限制：

- 当前日志显示 `Data Parallel GPUs = 1`。
- 第一版 compact 数据用 16k 训练时 backward OOM。
- 8192 对第一版 compact 数据不够，会触发长度预检失败。

因此正式实验请优先使用 veRL FSDP 后端。
