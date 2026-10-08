# Agentic RL 公共适配层

`agent_rl` 是 Terrabox evolution 下的真实工具 Agentic RL 公共基础设施目录，不代表具体算法方法。

它的职责是统一：

- 真实 Terrabox 工具交互环境；
- 多轮 rollout 的上下文压缩和 artifact 状态；
- Swift / veRL 训练框架适配；
- public-view 数据导出；
- 真实工具 reward 组件；
- episode trace、resource log 和 checkpoint manifest。

当前已提供：

- `env.py`：统一真实工具环境 `TerraboxAgentEnv`；
- `rewards.py`：公共真实工具 reward 组件；
- `observability.py`：JSONL trace、observation 压缩、内存 trim；
- `data.py`：veRL public rows 到 Swift Gym-env rows 的转换；
- `adapters/swift.py`：MS-Swift `external_plugins` / `gym_env` 适配；
- `adapters/verl.py`：现有 veRL online 实现的中性薄封装；
- `runner.py`：统一 `prepare-data`、`write-swift-command`、`write-verl-command` 入口。

具体方法仍然放在独立目录，例如：

- `rl_grpo/`：pure GRPO baseline；
- `experience_evo_rl/`：ExperienceEvo + RL；
- `qnr_rl/` 或 `rewardevo_rl/`：后续 QNR/Quse 或 reward 创新。

详细计划见 `Swift_veRL_双框架适配计划_20260911.md`。

## 使用方式

Pure GRPO baseline：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner prepare-data \
  --method rl_grpo \
  --experiment qwen25_3b_swift_grpo_train2000_YYYYMMDD \
  --limit 2000

PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner write-swift-command \
  --method rl_grpo \
  --experiment qwen25_3b_swift_grpo_train2000_YYYYMMDD

PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner write-verl-command \
  --method rl_grpo \
  --experiment qwen25_3b_verl_grpo_train2000_YYYYMMDD
```

Swift GRPO trainer 的 RLOO 优势估计变体：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner write-swift-command \
  --method rl_grpo \
  --experiment qwen25_3b_swift_rloo_train2000_YYYYMMDD \
  --advantage-estimator rloo
```

`--advantage-estimator` 可选 `grpo`、`rloo`、`reinforce_plus_plus`。这些选项仍使用同一 Swift `grpo` trainer、同一真实工具 Gym 环境和同一 reward，只改变优势估计方式；生成的 `swift_command_manifest.json` 会记录具体 estimator。`num_generations=2` 时 RLOO 的留一基线方差较大，若改为 4 必须同时记录额外 rollout budget，不能把采样数量带来的提升归因于 estimator 本身。

ExperienceEvo + RL：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner prepare-data \
  --method experience_evo_rl \
  --experiment qwen25_3b_swift_expevo_grpo_train2000_YYYYMMDD \
  --limit 2000 \
  --prompt-top-k 5 \
  --reward-top-k 5
```

Swift 当前通过 `/data1/yuhongjie2/ms-swift` 源码目录接入；若环境未安装 `json_repair`、`dacite` 等 Swift 依赖，`adapters/swift.py` 会在 import 时明确报错。由于当前环境没有 `swift` CLI 进 PATH，生成脚本默认使用：

```bash
python -m swift.cli.main rlhf ...
```

当前 `json_repair` 与 `dacite` 已安装到仓库临时依赖目录：

```text
tmp/python_deps/unsloth
```

生成的 Swift 脚本会自动把该目录加入 `PYTHONPATH`。默认 Swift 命令不传 `--deepspeed`，避免当前环境缺 `deepspeed` 时无法启动；如后续要使用 ZeRO2，需要先在合适环境安装 deepspeed，再显式传 `--deepspeed zero2`。

Qwen2.5-3B 本地模型目录在 Swift 中需要显式指定：

```text
--model_type qwen2 --template qwen2_5
```

低资源默认使用 `--vllm_mode colocate`。注意：colocate 模式不要传 `--vllm_server_host` / `--vllm_server_port`，否则 Swift 会误按 external vLLM server 初始化并等待 8000 端口。

2026-09-11 当前 Codex sandbox 视角下 `nvidia-smi` / `torch.cuda.is_available()` 不可用，因此从 Codex tmux 直接启动会在 Swift 参数初始化阶段报 `Your setup doesn't support bf16/gpu`。这不是数据或 adapter 失败；当前可用的启动方式是在宿主 GPU 可见上下文中执行生成的 `configs/run_swift_grpo_with_status.sh`，例如：

```bash
systemd-run --user --wait --collect --pipe /bin/bash -lc \
  'cd /data1/yuhongjie2/terrabox && CUDA_VISIBLE_DEVICES=2 bash tmp/agent_rl_runs/rl_grpo/<experiment>/configs/run_swift_grpo_with_status.sh'
```

如果 `systemd-run --user --collect --unit=...` 或 `--scope` 报 `Failed to create bus connection`，使用上面的 `--wait --collect --pipe` 前台方式。

当前 Swift 低资源默认配置：单卡、LoRA、`vllm_mode=colocate`、`vllm_gpu_memory_utilization=0.35`、`vllm_max_model_len=8192`、`vllm_max_num_seqs=1`、`dataloader_num_workers=0`、`dataset_num_proc=1`、关闭 persistent dataloader workers。该配置优先保证能跑通，不追求最快吞吐。`vllm_gpu_memory_utilization=0.25` 在 Qwen2.5-3B + GRPO colocate 下可能导致 vLLM 无法分配 KV cache，不建议作为默认值。

已经跑完 1900/1900 的 Qwen2.5-3B Swift online GRPO 默认复用配置如下，后续 pure GRPO / ExperienceEvo-GRPO / reward ablation 默认先从这组参数开始，只在资源或算法目标变化时单独改动：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `model_path` | `/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct` | 基座模型 |
| `train/val` | `1900 / 100` | OEA train-side 2000 切分 |
| `tool catalog` | 23 个 OEA 工具，文本注入 system prompt | 不使用 Swift 顶层 `tools` 字段 |
| `max_length` | `8192` | 训练输入总长度 |
| `max_completion_length` | `128` | Swift 单步 completion 上限；多轮由环境继续驱动 |
| `max_turns` | `6` | 控制在线工具轨迹长度，降低显存/CPU 峰值 |
| `num_generations` | `2` | GRPO 最小组内采样数，平衡成本和相对优势估计 |
| `steps_per_generation` | `4` | 与 Swift/TRL GRPO 对齐的生成-更新节奏 |
| `per_device_train_batch_size` | `1` | 单卡稳定优先 |
| `gradient_accumulation_steps` | `4` | 有效 batch 累积 |
| `learning_rate` | `1e-6` | 低学习率，避免在线 reward 抖动下更新过猛 |
| `local_rollout_forward_batch_size` | `1` | 降低 logprob/backward 显存峰值；比默认 64 慢但稳定 |
| `vllm_gpu_memory_utilization` | `0.35` | colocate 场景给训练侧留显存 |
| `vllm_max_model_len` | `8192` | 与训练 max length 对齐 |
| `vllm_max_num_seqs` | `1` | 避免 KV cache 峰值过高 |
| `save_steps` / `save_total_limit` | `50 / 2` | 周期 checkpoint，保留最近版本 |
| `observation compression` | `1024 chars / threshold 900` | 原始 observation 落盘，模型上下文只看结构化摘要 |

该配置的最终成功训练目录为：

```text
tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_catalogprompt_obplaceholder_turn6_resp128_train2000_autofallback_20260912
```

最终 checkpoint：

```text
checkpoints/swift/v4-20260913-152155/checkpoint-1900
```

注意：这次训练中间发生过 optimizer/scheduler state 续跑兼容问题和一次 backward OOM，最终采用 `resume_only_model` 从 checkpoint 继续完成；因此它可作为“资源受限单卡 online GRPO 默认工程配置”，但论文记录中应说明中途是模型权重续跑，不是完整 optimizer state 无缝续跑。

在线工具观察采用两级压缩：原始工具返回完整写入 `metrics/raw_tool_observations.jsonl`；回传给模型上下文的 observation 默认限制为 `TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS=1024`，超过 `TERRABOX_ONLINE_TOOL_COMPACT_THRESHOLD_CHARS=900` 时优先压成结构化摘要。摘要保留 `status/error`、artifact 路径、图层/CRS、计数、统计量、短 bbox/坐标、detection 样例和 OCR 摘要；长 mask、长数组、大 JSON 和长文本改写为 `<omitted_*:hash>` 占位符，并记录长度与少量 sample。这个机制是通用上下文压缩，不读取 gold label，也不改变 raw trace 和后续离线指标统计。

占位符语义：模型能看到“这里有一段长 mask / 长文本 / 大列表被省略、原始内容已保存、hash 是什么、长度是多少、样例是什么”，但不会把完整数组或大块文本塞回上下文。例如：

```json
{
  "mask": {
    "placeholder": "<omitted_numeric_list:...>",
    "type": "numeric_list",
    "raw_saved": "metrics/raw_tool_observations.jsonl",
    "length": 20000,
    "sample": [1, 1, 1, 1, 1, 1, 1, 1]
  },
  "output_path": "/tmp/artifacts/sample/out.png"
}
```

因此后续工具仍能使用真实 artifact/path，模型也能判断工具是否成功；只有不可直接放入上下文的冗长证据被占位。

此前 `qwen25_3b_swift_grpo_catalogprompt_train2000_clean_20260912`、`qwen25_3b_swift_grpo_catalogprompt_compobs_train2000_20260912`、`qwen25_3b_swift_grpo_catalogprompt_compobs_vllm35_train2000_20260912` 等目录只作为诊断或中间失败目录，不作为默认配置来源。

OEA 数据中的图片/文件路径只保留在 `env_config.images` / `env_config.data_files` 中供 Terrabox 工具使用；Swift 训练样本不写顶层 `images` 字段，避免文本 Qwen2.5-3B 被 Swift/vLLM 误判为多模态模型。

当前环境的 `datasets==4.3.0` 与 Swift dev 版本的特征检测不完全兼容，生成脚本默认设置 `TERRABOX_SWIFT_DATASETS_COMPAT=1`，并通过 `tmp/python_deps/unsloth/sitecustomize.py` 做局部兼容处理。该处理只在本实验脚本环境生效，不改全局 Python 包。

`run_swift_grpo_with_status.sh` 会把训练日志写入 `train_swift_tmux.log`，并后台采样 CPU/GPU/进程资源到 `metrics/resource_monitor.jsonl`；采样器随训练脚本退出自动停止。

启动前检查：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner check-env
```

## Swift 训练后标准 OEA 评测

Swift/PEFT 训练产物默认是 LoRA adapter，不能直接作为本地 ReAct/vLLM 的 `AGENT_LLM_MODEL_PATH`。训练完成后先合并 adapter，再用固定 OEA test manifest 跑真实工具 rollout：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner export-swift-adapter \
  --checkpoint tmp/agent_rl_runs/rl_grpo/<train_run>/checkpoints/swift/<version>/checkpoint-1900 \
  --output-dir tmp/agent_rl_runs/rl_grpo/<train_run>/merged_hf/checkpoint-1900 \
  --exist-ok --launch

PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner write-oea-eval-command \
  --eval-experiment <eval_run_name> \
  --model-path tmp/agent_rl_runs/rl_grpo/<train_run>/merged_hf/checkpoint-1900 \
  --output-dir tmp/agent_rl_runs/rl_grpo/<eval_run_name> \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --port 9103 --agent-gpu 3 --vlm-gpu 0 --tool-gpu 1 \
  --agent-model-len 24576 --agent-context-length 24576 \
  --max-completion-tokens 4096 --agent-gpu-memory-utilization 0.80 \
  --gpu-workers 1 --nogpu-workers 1
```

原生 tool-call 的 Base/GRPO/RLOO/REINFORCE++ checkpoint 保持默认 `--tool-protocol native`。对 OpenEarthAgent 文本 ReAct/JSON-actions 的 SFT merge，必须显式传训练数据同版本生成的 system prompt，生成器会把协议、prompt 路径和 SHA-256 写入 `configs/oea_eval_manifest.json`，并把同样的参数传给 GPU/nogpu 两条 lane：

```bash
PYTHONPATH=src python -m terrabox.evolution.agent_rl.runner write-oea-eval-command \
  --eval-experiment qwen25_3b_replayed_sft_oea_eval \
  --model-path tmp/agent_rl_runs/sft/<run>/merged_hf/checkpoint-148 \
  --output-dir tmp/agent_rl_runs/sft/qwen25_3b_replayed_sft_oea_eval \
  --tool-protocol sft-json \
  --sft-system-prompt-file tmp/agent_rl_runs/sft/<run>/system_prompt.txt \
  --agent-model-len 24576 --agent-context-length 24576 \
  --max-completion-tokens 4096 \
  --gpu-workers 1 --nogpu-workers 1
```

`sft-json` rollout 会从这个 system prompt 的 `Tool catalog` JSON 读取运行时工具目录，并校验每个条目都存在于当前 registry。这样即使只跑一个小分片，模型看到的 23 工具目录仍与实际可调用工具一致；不会再按该分片的 `expected_tools` 缩成几项 OSM 工具，造成模型调用“提示中存在、运行时不存在”的工具。native rollout 继续按任务文件工具并集生成 allow-list。若 SFT prompt 缺少可解析的工具目录或 registry 缺少目录中的工具，runner 会在写入结果前直接失败，避免污染正式指标。

生成的 `configs/run_oea_eval.sh` 固定使用 `standard` 模式、OEA test、真实工具、`--resume`、GPU/nogpu 两段 lane，并显式打开 `--no-skip-osm/bing/vlm/changeos`。本地 Qwen2.5-3B watcher 默认把 vLLM 服务 context、runner context 和 completion budget 对齐到 24576/24576/4096；runner 不会为了凑输出预算静默截断历史，context overflow 会作为失败结果记录。默认会复用健康且 model/context/GPU utilization 匹配的 agent LLM 容器，只有需要强制重建时再设置 `TERRABOX_AGENT_LLM_CLEAN_START=1`。本地 merged checkpoint 评测默认 `gpu-workers=1`、`nogpu-workers=1`，避免多个 worker 共享同一个本地 agent LLM 容器时把一次容器/端口级基础设施故障扩散成大量 task-level `exception` 结果；外部 API 模型可按 OEA watcher 规范单独提高 no-GPU lane 并发。watcher 每条 lane 结束后要求 `run_status.json` 为 `complete` 且 `invocation_complete=true` 才继续，runner 非零退出或状态缺失/不完整都会停止队列。完成后再用 `rollout_report` / `rollout_metrics` 和 answer judge 生成正式 OEA 主表；训练 reward、离线 action eval 和中止轨迹不能替代完整 OEA 指标。

## 当前边界

- veRL 不重写：保留已经跑通到 checkpoint 的 `experience_evo_rl` online veRL 路径，只把通用 reward、trace 和压缩逻辑抽到公共层。
- Swift 需要新适配：当前已有 Gym env 插件和命令生成器，但正式训练前必须先补齐 Swift 运行环境依赖并做 1-2 条 interface check。
- 当前 `check-env` 显示 Swift、TRL、PEFT、vLLM、Ray 和关键 rlhf 参数可用；若当前 shell 中 `nvidia-smi` 不可见，优先使用宿主 `systemd-run --user --wait --collect --pipe` 启动，不要把 GPU 不可见误判为 adapter 失败。
- 跨框架 checkpoint 不是无损续训：veRL FSDP 分片需要先导出为标准 HF/PEFT adapter，Swift 只能作为 `init_from_adapter`，不能写成从 veRL step 无损恢复 optimizer/scheduler/RNG。
