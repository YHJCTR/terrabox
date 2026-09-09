# Qwen 3B online GRPO 安全超参数说明

更新时间：2026-09-10

## 背景

最新一次在线实验 `qwen25_3b_pure_online_grpo_compressed_v4_train2000_20260907` 已在 step 0 后因 CUDA OOM 中止。失败发生在 veRL 计算 old log-prob / entropy 阶段，不是 Terrabox 工具执行阶段。该实验已经保存了部分真实工具轨迹，但没有产出最终 checkpoint，不能作为完整 OEA RL 结果。

本轮调整目标是：不改变 OEA 任务、工具目录、真实工具执行和上下文压缩口径，只降低 veRL 训练侧显存峰值和训练期 validation 的批量内存峰值，使完整 2000 条数据准备、1900 条 train 的 online GRPO 可以稳定跑起来。

补充说明：上一次 `safe8192_noalloc` 版本没有再触发 CUDA OOM，而是在 veRL 训练前 validation 阶段触发 Ray CPU 内存阈值。根因不是 Terrabox 工具 worker 太多；online OEA 已固定 `agent.num_workers=1`。问题是 veRL 默认 `data.val_batch_size=null` 时会把全部 100 条 val 一次性送入 online validation，并且 `trainer.val_before_train=True` 会在第一个训练 update 前就做这次大批量 validation。

## GPU 分配

| 资源 | 默认分配 | 说明 |
|---|---|---|
| veRL 训练 / policy rollout | `CUDA_VISIBLE_DEVICES=2,3` | 训练进程只看物理 GPU 2/3；日志里的 `GPU 0/1` 是可见设备内的逻辑编号。 |
| VLM 工具服务 | `VLM_GPU_DEVICES=0` | GPU0 保留给 `geo_perception.vlm_analyze`。 |
| 其它重感知工具 | `SAM2/RemoteSAM/RemoteCLIP/InstructSAM/ChangeOS=1` | GPU1 保留给分割、检测、ChangeOS 等工具。 |
| 动态工具兜底 | `TERRABOX_GPU_ALLOWED_DEVICES=0,1` | 防止工具服务误落到训练卡。 |

该分配保持“训练卡”和“工具卡”隔离，避免真实工具调用和 veRL 反向传播互相抢显存。

## 已验证稳定调整

| 参数 | 旧值 | 新值 | 中文释义 | 影响 |
|---|---:|---:|---|---|
| `data.train_batch_size` | 4 | 1 | 每个 GRPO step 取多少条任务 prompt | 显著降低 rollout、old-logprob、actor update 和 checkpoint 阶段的峰值内存；完整一轮 step 数为 1900，训练更慢但稳定性最高。 |
| `actor_rollout_ref.rollout.n` | 4 | 2 | 每条任务采样几个 rollout 作为 GRPO group | 降低 rollout、log-prob 和 reward 阶段显存/时间峰值；组内相对优势仍可计算，但方差会比 `n=4` 大。 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | 12288 | 8192 | actor 更新阶段每张 GPU 允许处理的最大 token 数 | 主要降低训练更新显存峰值；不直接截断 prompt/response。 |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | 12288 | 8192 | old log-prob 计算每张 GPU 最大 token 数 | 针对本次 OOM 的直接修复点；不改变工具执行结果。 |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | 12288 | 8192 | reference log-prob 计算每张 GPU 最大 token 数 | 降低 ref log-prob 的显存峰值；pure online 当前不使用 KL reward，但保守同步降低。 |
| `data.max_prompt_length` | 6144 | 6144 | 模型输入 prompt 最大 token | 保持不变，避免任务、工具 schema 或 system instruction 被更早过滤。 |
| `data.max_response_length` | 4096 | 2048 | 单次生成 response 最大 token | 降低长 response 带来的显存和 CPU 峰值；部分复杂任务可能触顶截断，因此需要在过程指标中持续观察 `response_length/clip_ratio`。 |
| `multi_turn.max_tool_response_length` | 8192 | 6144 | 单次工具 observation 回填模型前的最大长度 | 保留关键路径/数值/图层证据，同时减少上下文膨胀；原始 observation 仍单独保存，便于后续复核。 |
| `TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS` | 8192 | 8192 | 工具 observation 压缩后的字符预算 | 保持训练/后续评测一致；原始 observation 仍完整保存。 |
| `data.val_max_samples` | -1/100 条 | 0 / 禁用训练期 test | 训练期间不跑大批 validation | 当前正式训练设置 `trainer.test_freq=0`、`trainer.val_before_train=False`，避免 validation 先占用 CPU/Ray 内存；最终 OEA test 另跑全量评测。 |
| `data.val_batch_size` | null/一次性全 val | 1 | validation dataloader 每批多少条 | 避免 100 条 online episode 同批进入 Ray rollout manager。 |
| `data.dataloader_num_workers` | 8 | 0 | dataloader 子进程数 | 降低 CPU 内存和进程开销；训练速度略降。 |
| `trainer.val_before_train` | True | False | 是否训练前先跑 validation | 避免还没更新就被大 validation 杀掉。 |
| `trainer.save_freq` | 50 | 50 | 每多少 step 保存一次 checkpoint | 保留中断恢复点；配合跳过 optimizer checkpoint，已验证通过 step 50/100/150/200。 |
| `TERRABOX_VERL_SKIP_OPTIM_CKPT` | 0 | 1 | 保存 checkpoint 时跳过 optimizer state | 避免 FSDP optimizer state 在 CPU 汇聚时触发 Ray/CPU OOM；代价是这些中间 checkpoint 更适合作为模型评测/导出点，不适合作完整 optimizer 续训点。 |

## 可以继续调整的参数

| 参数 | 建议范围 | 何时调整 | 风险 |
|---|---:|---|---|
| `rollout_log_prob_max_token_len_per_gpu` | 6144--8192 | 如果再次在 old log-prob OOM，优先降它 | 太低可能导致 veRL 动态 batch 切分更碎，速度下降。 |
| `actor_ppo_max_token_len_per_gpu` | 6144--8192 | 如果 actor update OOM | 主要影响速度，不改变任务语义。 |
| `ref_log_prob_max_token_len_per_gpu` | 6144--8192 | 如果 ref log-prob OOM | 当前 pure GRPO 影响较小。 |
| `train_batch_size` | 1--4 | 如果 step 0 仍 OOM，降到 1；若稳定且显存富余，可回升到 4 | 降低会变慢；升高可能再次 OOM。 |
| `rollout_n` | 2--4 | 如果稳定后想提高 GRPO 组内信号，可升到 4 | 升高会显著增加 rollout 与 log-prob 压力。 |
| `rollout_gpu_memory_utilization` | 0.15--0.30 | vLLM rollout engine 初始化或 sleep/offload 不稳时调整 | 过高会挤压 FSDP/log-prob；过低可能降低吞吐。当前稳定值为 0.15。 |
| `max_prompt_length` | 4096--6144 | 只有确认 prompt 过长且训练集大量被过滤时才降 | 会直接影响模型可见信息，不建议优先动。 |
| `max_response_length` | 2048--4096 | 如果 response 过长导致显存或速度不可接受时才降；若 clip ratio 长期偏高，再谨慎升到 3072/4096 | 升高会增加显存和 old-logprob 压力；降低可能让工具调用链不完整。 |
| `max_tool_response_length` | 4096--8192 | 如果工具 observation 导致上下文过长才降；若工具证据缺失，再谨慎升高 | 可能丢失数值、路径、layer 信息；必须同步记录新口径。 |
| `online_max_turns` | 8--12 | 如果任务经常空转或上下文增长过快可降 | 太低会让真实多步 GIS/VLM 任务未完成。 |
| `lora_rank` | 4--16 | 如果参数显存仍紧，可降到 4；如果稳定可升到 16 | 降低表达能力，升高显存和优化器状态。 |
| `val_max_samples` | 4--32 | 如果 Ray CPU 内存仍高，降到 4；如果稳定后想看更平滑训练期 val，升到 16/32 | 只影响训练期监控，不是最终 OEA test 指标。 |
| `val_batch_size` | 1--4 | 如果稳定且 CPU 内存富余，可升到 2/4 | 升高会增加同时在内存里的 online episode。 |
| `val_before_train` | False/True | 只有稳定后才打开 | 打开会先消耗一轮 validation 时间和内存；不影响最终训练数据覆盖。 |

## 不建议先调整的内容

- 不隐藏任何 OEA 工具；否则和 LongCat/OEA 主表不是同一评测口径。
- 不把 `gpu-class` 过滤用于训练数据；online RL 训练应看到完整工具目录和完整任务分布。
- 不先降低 `max_prompt_length` 或 `max_tool_response_length` 来硬截断信息；本次 OOM 的直接原因是 log-prob/entropy 显存峰值，不是工具 observation 原文太长。
- 不把工具服务放到 GPU2/3；这会和 veRL 训练互相抢显存。

## 当前已能正常运行的复现配置

实验名：`qwen25_3b_pure_online_grpo_tp2_noactoroff_tok8192_util015_full1900_train2000_20260909`

- tmux：`qwen3b_online_grpo_tp2_noactor_tok8192_util015_full1900_20260909`。
- 数据：OEA train 2000 条拆分为 `1900 train / 100 val`；当前训练覆盖完整 1900 train，最终 OEA test 另跑全量真实工具评测。
- 模型：`/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct`。
- 框架：`/data1/yuhongjie2/verl`，入口 `verl.trainer.main_ppo`，`algorithm.adv_estimator=grpo`，LoRA rank 8。
- GPU：训练固定 `CUDA_VISIBLE_DEVICES=2,3`，rollout tensor parallel `TP=2`；工具服务固定在 GPU0/1，避免与训练抢显存。
- 并发：`data.train_batch_size=1`，`actor_rollout_ref.rollout.n=2`，`actor_rollout_ref.rollout.agent.num_workers=1`，`data.dataloader_num_workers=0`。
- token：`data.max_prompt_length=6144`，`data.max_response_length=2048`，`actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192`，`actor_rollout_ref.rollout.max_model_len=8192`，`actor_rollout_ref.rollout.max_num_batched_tokens=8192`。
- vLLM：`actor_rollout_ref.rollout.gpu_memory_utilization=0.15`，保持 rollout engine 常驻，避免 vLLM V1 sleep/free-cache 兼容性问题。
- FSDP/offload：`actor.fsdp_config.param_offload=False`，`actor.fsdp_config.optimizer_offload=True`，`ref.fsdp_config.param_offload=True`。
- checkpoint：`trainer.save_freq=50`，`trainer.max_actor_ckpt_to_keep=1`，`TERRABOX_VERL_SKIP_OPTIM_CKPT=1`；已验证保存 `global_step_50/100/150/200` 正常。
- validation：`trainer.test_freq=0`，`trainer.val_before_train=False`；训练期间不把 validation 指标当正式 OEA test 指标。
- 工具上下文：`max_assistant_turns=12`，`multi_turn.max_tool_response_length=6144`，原始工具 observation、episode trace、resource monitor 分开保存。

截至 2026-09-10 02:01，该配置已运行到 `training/global_step=206` / progress `207/1900`，无 `OOM`、`OutOfMemory`、`CUDA error`、`Traceback`、`RayOutOfMemory` 或 `Killed`，并已连续通过 4 次 checkpoint。过程指标正常，但仍需持续观察两类风险：真实工具慢调用导致 step 时间波动，以及 `max_response_length=2048` 下部分任务出现 response clip。

## 已废弃或失败启动尝试

注意：当前 veRL 的 colocated vLLM 路径会使用 vLLM V1 memory pool，不能设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，否则 vLLM engine core 初始化会失败。当前 runner 会在启动脚本中显式清掉 `PYTORCH_CUDA_ALLOC_CONF`；该 allocator 配置只适合部分纯 PyTorch/FSDP 场景，不适合本 online GRPO 路径。

废弃启动尝试：`qwen25_3b_pure_online_grpo_compressed_safe8192_train2000_20260908` 曾短暂启动，但因为设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，在 vLLM engine 初始化阶段触发 memory pool 兼容性断言失败，没有进入有效训练，不作为实验结果记录。

废弃启动尝试：`qwen25_3b_pure_online_grpo_compressed_safe8192_noalloc_train2000_20260908` 已进入真实工具 rollout，但在训练前 validation 阶段被 Ray 因 CPU 内存超过 95% 阈值杀掉；它没有完成第一个有效 train update，不作为正式训练结果记录。其失败推动了当前 `valsafe` 参数：只限制训练期 validation 并发，保留完整 train 数据和完整工具目录。

废弃启动尝试：`qwen25_3b_pure_online_grpo_compressed_safe8192_valsafe_train2000_20260908` 已验证 validation 限流生效，但在 veRL `sleep_replicas()` 调用 vLLM V1 `sleep(level=1)` 时触发 CUDA invalid argument。当前正式重跑使用 `--keep-rollout-loaded` 生成 `free_cache_engine=False`，避免进入该 sleep/free-cache 路径。代价是 rollout engine 常驻显存，后续如果在 actor update 阶段显存不足，优先把 `train_batch_size` 降到 1 或把训练侧 token 上限降到 6144。

废弃启动尝试：`qwen25_3b_pure_online_grpo_compressed_safe8192_valsafe_keepvllm_train2000_20260908` 已进入真实 online rollout，并产生 episode/tool observation，但首个 old-logprob 阶段出现配置断言：实际 rollout 序列长度约 8.4k--8.9k token，大于 8192 的训练侧 max-token 上限。该失败不是 OOM。

废弃启动尝试：`qwen25_3b_pure_online_grpo_entropyfix_gpu23_train2000_20260908` 能跑到 step 49，但保存 step 50 checkpoint 时触发 CPU/Ray OOM；后续通过 `TERRABOX_VERL_SKIP_OPTIM_CKPT=1` 跳过 optimizer state 保存解决。

废弃启动尝试：多个 2026-09-09 `tp2_*_ckptcheck` / `2step` 目录用于定位 actor 参数 offload、vLLM KV cache 和 checkpoint 兼容性问题；其中 `tp2_noactoroff_tok8192_util015_ckptverify_2step` 应保留作复现依据，其余无 checkpoint 的小目录可在当前 full run 完成后归档或删除。
