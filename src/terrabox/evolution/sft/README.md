# SFT Baseline

这个目录保存 Terrabox 当前 strict 数据上的 SFT baseline。目标是训练一个后续方法可复用的 Qwen3-8B 冷启动模型，让模型先学会当前 Terrabox 工具名、JSON action 格式、常见调用顺序和任务风格。

> 训练的是**纯文本 Qwen3-8B**(工具调用能力),不是 VL。ReAct/Reflection 的 agent LLM(port 9100)就是这个文本 8B(vLLM 挂载 HF 目录跑 `--model /model`);感知交给工具(instructsam 等),图片不进 agent LLM。`scripts/train/train_sft.py` 与 `/data1/yuhongjie2` 下的 SFT 是**另一套 VL 训练**,与本模块无关,勿混用。

## ✅ 最新实验(2026-06-15 实跑):`oe_full` —— OE 全量 + 23 工具聚焦 catalog + 单卡 QLoRA

主线升级:**OE 与 EB 分开训练**,先单独用 **OpenEarth 全量**数据冷启动。相比 `v2_sft`(OE+EB 混合、47 工具 catalog、过滤子集),本次:
- **OE/EB 分离**:只用 OE 数据;catalog 只放 **OE 的 23 个可调用工具**(不再把 40+ 工具塞进上下文)→ 训练上下文 = OE rollout 工具列表,对齐更紧。
- **全量 + 去塌缩**:14538 条(此前被过滤的 osm/gpkg/search/changeos/count/region_attr 等任务**全部纳入**,因为这些工具现已全部真实实现)。
- **工具吃 OEA 原生参数契约**:转换器逐字保留 gold 参数(只解析 `img_N`→绝对路径),**每个 gold 调用的参数都匹配工具 schema**(`argument_status` 全 `adapted`,0 未知参数)→ SFT 目标 = rollout 能执行的调用。
- **0 丢弃**:聚焦 catalog 使 system prompt 19.5K(v2 是 38.8K),序列中位 5492 / 最长 6362 token → `max_seq 8192` 下仅丢 3/14338(0.02%)。
- **AddIndexLayer** 接 `{year,month}` → 免账号 earth-search Sentinel-2 STAC 实时取数(轻量窗口读);其余 23 工具均已端到端验证可跑(24 工具覆盖 rollout:8/10 成功,含 changeos/strip-rcnn docker)。

### 数据怎么来
```bash
# 1) 构建全量 OE/EB 分离数据集(OE catalog=23 工具;输出 data/oea_full_sft/{openearth,earthbench}/{train,test}.jsonl)
PYTHONPATH=src python scripts/build_oea_full_sft.py
# 2) 切 train/val(固定 seed shuffle,verbatim 不压缩)
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner prepare-data \
  --experiment oe_full --strict-data data/oea_full_sft/openearth/train.jsonl \
  --val-start 0 --val-limit 200 --train-start 200 --train-limit 100000
#  → train 14338 / val 200 ; exp/oe_full/sft_data/{train,val}.jsonl
```

### 运行指令(2026-06-15 实跑,GPU3,tmux)
```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
# 必须在 tmux 里跑(关终端不杀);断路器:--temp-target 80 软件温控 + --dataset-num-proc 8 + 频繁存档/resume
tmux new-session -d -s oe_full_sft "source /home/yuhongjie/miniconda3/etc/profile.d/conda.sh && conda activate unsloth && \
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src $PY -m terrabox.evolution.sft.train_lora \
  --train-file src/terrabox/evolution/sft/exp/oe_full/sft_data/train.jsonl \
  --val-file  src/terrabox/evolution/sft/exp/oe_full/sft_data/val.jsonl \
  --model-path /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/ \
  --output-dir src/terrabox/evolution/sft/model/oe_full \
  --max-seq-length 8192 --drop-overlength --lora-rank 32 --lora-alpha 64 --learning-rate 1e-4 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --num-train-epochs 1 \
  --save-steps 50 --save-total-limit 2 --save-merged-model \
  --dataset-num-proc 8 --temp-target 80 --temp-resume 75 --cooldown-sec 0 --resume \
  > src/terrabox/evolution/sft/exp/oe_full/sft_train.log 2>&1"
# 共 1792 步(14335÷8);跳闸/重启后重跑同命令 --resume 自动从最近 checkpoint 续
# 监控:tmux attach -t oe_full_sft  |  tail -f .../oe_full/sft_train.log  |  grep telemetry 日志看温度
# 产物:src/terrabox/evolution/sft/model/oe_full/merged (= ReAct agent LLM 挂载格式)
```

### SFT 后评测(分基准,带 JSON-actions 适配器)
与下方 v2 的 ④ 完全相同,只是把 `MERGED` 换成 `model/oe_full/merged`、system prompt 取 `data/oea_full_sft/openearth/train.jsonl` 第 0 条;OE 评测用 `data/oea_full_sft/openearth/test.jsonl`(1162 条,已剔除 7 条 train/test 重叠)。**不再需要 `--skip-osm/--skip-bing`**(联网工具已就绪:OSM、Serper(`.env` 里 `SERPER_API_KEY` 自动加载 + SQLite 缓存)、STAC 全可用)。对比基线:`--model-path` 换 base 8B,其余不变。

---

## 最新实验(2026-06-09 实跑):`v2_sft` —— v2 数据 + unsloth 单卡 QLoRA

主线 SFT,**用于 RL 冷启动 + SFT 后重跑 ReAct 对比**。目标:SFT 后模型在 ReAct 上比原生 Qwen3-8B 更好(原生主要败在选错工具/格式不对,SFT 教正确 slug + `{thought,actions}` 格式 + 调用顺序)。

### 设计要点:verbatim + 丢弃超长,**不做有损压缩**(关键)
- **不压缩**(去掉了 `--compact-long-context`)。原因:之前的压缩会把 **gold 工具调用的参数**(如几十文件的 `output_path` 列表)压成 `{__sft_compacted_list__}` 占位符 → 模型学到**跑不通的工具调用** → SFT 后 ReAct 反而变差。**args 必须原样保留。**
- **与真实 rollout 对齐**:rollout 的"压缩"其实是**历史摘要**(上下文超长才总结旧轮),不是逐观测截断;单条全量进 prompt,硬上限是 vLLM `max_model_len=24576`。所以 SFT 也 **verbatim**,靠总长上限兜底 → train≈infer,不伤效果。
- **超长行丢弃而非截断**:`--drop-overlength` 把超过 `max_seq_length` 的行**整条丢掉**(不在工具调用中间截断)。实跑丢 **85/7195(1.2%)** train + 2/200 val,都是几十波段的极端 EB 批任务(它们的根治是工具改吃目录,另说;现在丢掉不影响其余)。

### 用什么数据
- 训练:`data/fixdata_decollapse_v2/sft_train_strict.jsonl`(v2 训练池 7395)→ `prepare-data` 切 val 200 / train 7195 → 训练时 `--drop-overlength` 丢超 16384 的 85 行 → **实际 train 7110 / val 198**,**verbatim 未压缩**。
- 测试**不在这里**:`data/fixdata_decollapse_v2/eval_openearth.json`(1647)、`eval_earthbench.json`(202),SFT 后用 ReAct rollout 评测、**分基准报**。

### 产生什么模型
文本 Qwen3-8B QLoRA,**merge 后 bf16 满精度**(无推理量化损失);产物 `src/terrabox/evolution/sft/model/v2_sft/`:`adapter/`(LoRA)+ `merged/`(完整 HF,= ReAct agent LLM 挂载格式,`AGENT_LLM_MODEL_PATH` 指它即可测)。

### LoRA 参数 / 冻结层(本次实跑)
| 项 | 值 |
|---|---|
| 量化 | 4-bit NF4 QLoRA,**base 全冻结** |
| 可训练 | 仅 LoRA 适配器,~87M / 8.28B = **1.05%** |
| **冻结** | embedding / 所有 layernorm / lm_head / 全部 base 权重 |
| target_modules | `q,k,v,o,gate,up,down`(36 层全部注意力+MLP 线性层) |
| rank / alpha | **32 / 64**(scaling=2);dropout 0;bias none |
| loss | **assistant-only**(`train_on_responses_only`,Qwen3 ChatML 标记) |
| lr / scheduler | **1e-4**(LoRA 冷启动用 1e-4~2e-4;默认 2e-5 太保守) |
| epoch / batch | 1 epoch / 有效 batch 8(`bs1 × accum8`) |
| max_seq / 超长 | **13312** + `--drop-overlength`(16384 实测 OOM,见下) |
| 步数 / ETA | **一条多轮轨迹=1 样本** → 7110÷8 ≈ **889 步** / 单卡 ~24–30h |
| checkpoint | `--save-steps 300 --save-total-limit 2`(总 ~889 步,1000 存不到;**只存 adapter+优化器,不 CPU 满载**;末尾 merge 一次 bf16) |
| 显存 | **13312 实测安全**(最长样本 forward+backward 峰值 14.87G/24G);**16384 实测会 OOM**(长 EB 样本的 LM-head logits 151936×16384×4≈10G,叠加基底 12.4G 超 24G,在 step~50 撞到长样本即崩) |

### 运行指令(本次实跑,照抄即可)
```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
EXP=v2_sft

# ① 准备数据(切 train/val,verbatim 不压缩)
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner prepare-data \
  --experiment $EXP --strict-data data/fixdata_decollapse_v2/sft_train_strict.jsonl \
  --val-start 0 --val-limit 200 --train-start 200 --train-limit 100000

# ② 训练(单卡 GPU0;assistant-only + drop 超长 + merged 默认产出)
tmux new-session -d -s v2_sft "
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner train \
  --experiment $EXP --model-path /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/ \
  --cuda-visible-devices 0 --max-seq-length 13312 --drop-overlength --lora-rank 32 --lora-alpha 64 \
  --learning-rate 1e-4 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --num-train-epochs 1 --save-steps 300 --save-total-limit 2 --drop-overlength --launch"
#   产物: src/terrabox/evolution/sft/model/$EXP/merged ;日志 exp/$EXP/sft_train.log

# ③ SFT 后用 ReAct 评测(分基准),与原生 ReAct 对比
PYTHONPATH=src no_proxy=localhost,127.0.0.1 $PY -m terrabox.evolution.sft.runner rollout \
  --experiment $EXP --model-path src/terrabox/evolution/sft/model/$EXP/merged \
  --task-file data/fixdata_decollapse_v2/eval_openearth.json --limit 300 \
  --agent-gpu 0 --tool-gpu 1 --launch
# 原生基线:同 task-file,--model-path 换成 /data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/
```

### ④ SFT 模型怎么真正跑进 ReAct(必读:JSON-actions 适配器)

> 2026-06-11:实跑发现 **SFT 模型不能直接走原生 ReAct**。原因:本仓库的 fixdata SFT 数据是 **OpenEarthAgent 风格的"文本 ReAct"**——工具调用是写在 **assistant content 里的 JSON**(`{"thought":..., "actions":[{"tool","function_name","arguments"}]}`),**不是原生 `tool_calls`**;且 OE 部分还有"**先出空 actions 的 Plan turn → user 把问题重问一遍 → 才行动**"的结构(EB 部分则直接行动)。原生 ReAct 用 vLLM `--tool-call-parser hermes` 解析 `<tool_call>` 标签,**解析不到 content 里的 JSON → tools=[]、1 步即停**。

**解决:开适配器**(`TERRABOX_SFT_JSON_ACTIONS=1`,默认关,不影响原生模型)。它让 `eval_modes` 的 sequential ReAct 循环:① 不 `bind_tools`、从 content 解析 `{thought,actions}`;② 历史按训练格式(assistant 存纯 JSON、observation 存 `HumanMessage("OBSERVATION:\n...")`);③ Plan turn(空 actions 且无 `final_answer`)不当结束,补一句重新问让它继续;并配 `TERRABOX_SFT_SYSTEM_PROMPT_FILE` 用训练时的 system prompt。**这仍然是 ReAct(同一套 rollout 代码/数据),只是让循环听懂 SFT 模型的方言。**

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
MERGED=$(pwd)/src/terrabox/evolution/sft/model/v2_sft/merged
# system prompt 直接取训练数据第 0 条(catalog 恒定);若无则:
#   $PY -c "import json;open('SP.txt','w').write(json.loads(open('.../sft_data/train.jsonl').readline())['messages'][0]['content'])"
SP=$(pwd)/src/terrabox/evolution/sft/exp/v2_sft/sft_system_prompt.txt
OUT=src/terrabox/evolution/sft/exp/v2_sft/react_eval_earthbench   # 指标落 SFT 目录,不进原生 ReAct

AGENT_LLM_MODEL_PATH=$MERGED TERRABOX_SFT_JSON_ACTIONS=1 TERRABOX_SFT_SYSTEM_PROMPT_FILE=$SP \
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
$PY -m terrabox.evolution.ReAct.runner rollout \
  --task-file data/fixdata_decollapse_v2/eval_earthbench.json \
  --experiment v2_sft_react_eval_earthbench --output-dir $OUT \
  --mode standard --port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 \
  --exclude-tools ipython.execute
# OE(1647 条,重度依赖 instructsam/VLM):--task-file 换 eval_openearth.json,--output-dir 换 react_eval_openearth
# 断点续跑:--resume 默认开;跳闸后重跑同命令自动跳过已完成
# 公平对比基线(可选):AGENT_LLM_MODEL_PATH 换成 base 8B、其余不变(base 也走适配器=同协议同数据,OEA 论文比法)
```

校验适配器生效:首条结果应 **tool_calls 非空、llm_calls>1、F1 有值**;system prompt 是 `You are a Terrabox geospatial tool-use agent...`。代码见 `agent/eval_modes/common.py`(`parse_sft_actions` / plan-turn)、`agent/eval_modes/standard.py`(`_resolve_system_prompt`)、`scripts/run_trajectory_experiment.py`(`extract_tool_calls` 也解析 SFT content)。

### 指令限制 / 注意
- 必须 `unsloth` conda 环境;**`--cuda-visible-devices 0` 单卡**(unsloth 本就单卡,且无跳闸风险)。
- **`--drop-overlength` 与 verbatim 配套**:不压缩 → 64 工具 catalog(8880)+ 长轨迹会有 ~1.2% 超 16384 → 必须 drop,否则预检报错(默认禁止静默截断)。
- **不要量化真实 rollout**:rollout 保持 bf16 / 24576;推理量化(AWQ/GPTQ 4-bit)会真降工具调用精度,且救不了超长任务(那是上下文长度问题,留给"工具吃目录")。
- 步数 ≈ 889(**多轮轨迹=1 样本**,不是按轮次展开成万步);想更充分→**先跑完 1 epoch 评测,不够再从 checkpoint 续 +1 epoch**,比一上来 2 epoch(~50h)省。
- SFT 占 GPU0;`rollout` 评测才需 docker(runner 自动加 `--use-docker`)。

## ⚠️ 2026-06-08 更新(数据对齐 / 3 卡 / HF 转换 / checkpoint 频率)

配合数据集对齐和 ReAct/Reflection 流程改造,本模块同步更新:

- **默认数据换成对齐后的 de-collapse 数据**:`runner.py` `DEFAULT_DATA = data/fixdata_decollapse/sft_train_strict.jsonl`(44 工具、参数可执行、与 ReAct/Reflection **同源**)。旧的 `data/newdata/` 是对齐前的,不要再用。仍可用 `--strict-data` 覆盖。
- **veRL 默认 3 卡**:`train-verl` 默认 `--cuda-visible-devices 0,1,2 --nproc-per-node 3`(原来 4 卡会跳闸;8B FSDP 在 2 卡初始化就 OOM,3 卡是稳定甜点;`3090-safe` preset 会让 `sequence_parallel_size=nproc=3`)。
- **新增 `convert-hf` 子命令**:把 veRL **FSDP shard checkpoint 离线转成 ReAct 可服务的 HF 模型**(标准 `config.json + safetensors 分片 + tokenizer`,即 agent LLM Docker 挂载的 `/model` 格式)。用 `--use_cpu_initialization` 在 **CPU 上离线转换**,避开训练时内联 `--save-hf-model` 触发的 126GiB full-gather 把服务器搞挂的问题。
- **checkpoint 频率可配**:
  - unsloth `train`:新增 `--save-steps`(默认 50)、`--save-total-limit`(默认 2)、`--eval-steps`(默认 50)。
  - veRL `train-verl`:沿用 `--save-freq` / `--max-ckpt-to-keep`。
  - 这样可以**用几条数据 + 小 `--save-steps`/`--max-steps` 跑通整套流程并验证中途存 ckpt**。
- **保存格式现状**:unsloth 路径 `--save-merged-model` **默认开**,直接产出 `merged/`(HF 格式,ReAct 可服务);veRL 路径只存 FSDP shard,**必须再跑 `convert-hf`** 得到 HF 模型。

### 把 SFT 模型放进 ReAct 测试(完整链路)

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
EXP=<your_experiment>

# 路线 A:unsloth(单卡, merged/ 直接可服务)
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner train \
  --experiment $EXP --cuda-visible-devices 0 --save-steps 50 --launch
#  → 产物: src/terrabox/evolution/sft/model/$EXP/merged

# 路线 B:veRL(3 卡 FSDP) → 离线转 HF
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner train-verl \
  --experiment $EXP --preset 3090-safe --save-freq 200 --launch     # 训练(3 卡)
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner convert-hf \
  --experiment $EXP --launch                                        # 取最新 global_step_*, CPU 转 HF
#  → 产物: src/terrabox/evolution/sft/model/${EXP}_verl/<step>_hf

# 评测:rollout 会把 model-path 设成 AGENT_LLM_MODEL_PATH 喂给 ReAct standard 模式
PYTHONPATH=src no_proxy=localhost,127.0.0.1 $PY -m terrabox.evolution.sft.runner rollout \
  --experiment $EXP --model-path <merged 或 <step>_hf 目录> \
  --agent-gpu 0 --tool-gpu 1 --launch
```

### 全量数据 + 与 ReAct 对齐的切片

eval(val)用与 ReAct 测试集**完全相同**的 shuffle[0:216];train 用其余全部:

```bash
PYTHONPATH=src $PY -m terrabox.evolution.sft.runner prepare-data \
  --experiment sft_full_decollapse \
  --val-start 0 --val-limit 216 \
  --train-start 216 --train-limit <剩余条数或不传以取到末尾>
```

> **loss 口径(已按 veRL 标准对齐)**:unsloth 路径默认 `--mask-prompt`,用 `train_on_responses_only`(Qwen3 ChatML 标记 `<|im_start|>user\n` / `<|im_start|>assistant\n`)**只对 assistant 轮算 loss**,多轮 ReAct 轨迹的每个 assistant 轮都受监督、system/user/observation 被掩码 —— 与 veRL `MultiTurnSFTDataset` 一致。需要旧的整段 loss 时用 `--no-mask-prompt`。


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
