# 交接指南:在新服务器上跑 `oe_full` SFT(给新 Claude 会话照做)

你(新 Claude 会话)的任务:在这台新服务器上,用 **OpenEarth 全量数据**对 **Qwen3-8B 文本模型**做一次 **QLoRA SFT**,产出可用于 ReAct 的 merged 模型。本文件自包含,按顺序执行即可;不需要原会话的上下文。

---

## 0. 背景(为什么这么跑,30 秒)
- 训练的是**纯文本 Qwen3-8B**(学工具名 + JSON action 格式 + 调用顺序),**不是 VL**。
- **训练阶段不读图片**:数据里的图像路径只是 assistant 参数里的文本 token。所以**不需要 OEA 图像、不需要 docker 工具服务、不需要 vLLM**(这些只在后续 ReAct 评测时才用)。
- 数据是 OEA 全量(14538 train),catalog 只含 OE 的 23 个工具(聚焦),gold 参数 verbatim 且全部匹配工具 schema。已验证结构与历史上"SFT 后 ReAct 提升"的 v2 实验字节级一致 → 预期 SFT 后效果优于原生 Qwen3-8B。

## 1. 需要从原服务器拷过来的三样(其余靠 git)
| 内容 | 原服务器路径 | 大小 | 放到新机哪 |
|------|------------|------|-----------|
| 基座模型 | `/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/` | 16GB | 任意,记作 `$BASE`(也可从 HF 下 `Qwen/Qwen3-8B` 基座等价) |
| 训练数据 | `src/terrabox/evolution/sft/exp/oe_full/sft_data/train.jsonl` | 369MB | 仓库同路径 |
| 验证数据 | `src/terrabox/evolution/sft/exp/oe_full/sft_data/val.jsonl` | 5.2MB | 仓库同路径 |

> 数据自包含(verbatim,无需图像/网络)。若懒得保留目录结构,把两个 jsonl 放任意位置,训练时用 `--train-file/--val-file` 指过去即可。
> 备选:只拷 `data/oea_full_sft/openearth/train.jsonl`(374MB)到新机,再跑 `prepare-data`(见附录)自己切 train/val。

## 2. 代码
```bash
git clone git@20.205.243.166:YHJCTR/terrabox.git && cd terrabox && git checkout evolve
```
SFT 全在 `src/terrabox/evolution/sft/`。本指南也在该目录。

## 3. 环境(conda)
```bash
conda create -n unsloth python=3.10 -y && conda activate unsloth
pip install unsloth==2026.1.1          # 会带出兼容的 torch/transformers/trl/peft/bitsandbytes
# 原机精确版本(如需对齐):
#   torch==2.9.0+cu128  transformers==4.57.3  trl==0.24.0  peft==0.18.0
#   bitsandbytes==0.49.0  accelerate==1.12.0  datasets==4.3.0  python==3.10.19
```
**硬件**:1 张 ≥24GB GPU(RTX 3090 同级)。max_seq 8192 实测峰值 ~15GB,单卡足够。需 CUDA 驱动支持 cu128(torch 2.9)。

验证环境 OK:
```bash
python -c "import unsloth, torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. 运行训练(tmux,断路器安全)
> ⚠️ 本项目有**断路器/降频强约束**(见仓库 `CLAUDE.md`):长训练必须 `--temp-target`(GPU 温度节流,压 80°C 下)+ `--dataset-num-proc` 低值(≤16,这里 8)+ `--save-steps` 频繁 + `--resume`。命令里都带好了,**不要去掉**。无 root 设不了 `nvidia-smi -pl` 时就靠 `--temp-target` 软件温控兜底。**必须在 tmux 里跑**(关终端不杀;随时可能跳闸,要能续)。

```bash
PY=$(which python)            # 确认是 unsloth env 的 python
BASE=/path/to/Qwen3-8B        # ← 改成你拷过来的基座路径
GPU=0                         # ← 改成空闲卡号

tmux new-session -d -s oe_full_sft "
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src $PY -m terrabox.evolution.sft.train_lora \
  --train-file src/terrabox/evolution/sft/exp/oe_full/sft_data/train.jsonl \
  --val-file  src/terrabox/evolution/sft/exp/oe_full/sft_data/val.jsonl \
  --model-path $BASE \
  --output-dir src/terrabox/evolution/sft/model/oe_full \
  --max-seq-length 8192 --drop-overlength --lora-rank 32 --lora-alpha 64 --learning-rate 1e-4 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --num-train-epochs 1 \
  --save-steps 50 --save-total-limit 2 --save-merged-model \
  --dataset-num-proc 8 --temp-target 80 --temp-resume 75 --cooldown-sec 0 --resume \
  > src/terrabox/evolution/sft/exp/oe_full/sft_train.log 2>&1"
```
- 共 **1792 步**(14335÷8,会先丢 ~3 条超 8192 的)。单卡约 **1–2 天**。
- 跳闸/重启后:**重跑同一条命令**,`--resume` 自动从最近 checkpoint 续(存档在 `model/oe_full/checkpoint-*`)。

## 5. 监控
```bash
tmux attach -t oe_full_sft           # Ctrl-b d 脱离
tail -f src/terrabox/evolution/sft/exp/oe_full/sft_train.log
grep "'loss':"  src/terrabox/evolution/sft/exp/oe_full/sft_train.log   # loss 应总体下降
grep telemetry  src/terrabox/evolution/sft/exp/oe_full/sft_train.log   # 每步 GPU 温度/功率(温度应被压在 80 下)
nvidia-smi -i $GPU
```
健康判据:开训后日志出现 `Total steps = 1,792`;loss 从 ~1.x 往下走;每 50 步出一个 `checkpoint-*`;telemetry 温度 <80°C。

## 6. 产物 & 下一步
- 训练末自动 merge:`src/terrabox/evolution/sft/model/oe_full/merged/`(完整 bf16 HF)= ReAct agent LLM 的挂载格式。
- 评测(可选,需图像 + 工具服务,见仓库 `src/terrabox/evolution/sft/README.md` 顶部 `oe_full` 章节的"SFT 后评测"):SFT 模型走 ReAct 要开 `TERRABOX_SFT_JSON_ACTIONS=1` 适配器 + 训练时的 system prompt,和原生 base 8B 对比,看 F1/成功率是否提升。

## 附录:从源数据自己切 train/val(若没拷 sft_data)
```bash
PYTHONPATH=src python -m terrabox.evolution.sft.runner prepare-data \
  --experiment oe_full --strict-data data/oea_full_sft/openearth/train.jsonl \
  --val-start 0 --val-limit 200 --train-start 200 --train-limit 100000
# → 生成 exp/oe_full/sft_data/{train,val}.jsonl(固定 seed shuffle,verbatim 不压缩)
```
（若连 `data/oea_full_sft/` 都没有:在装好工具的环境里 `PYTHONPATH=src python scripts/build_oea_full_sft.py` 重新生成,但那需要 `/data1/.../OpenEarthAgent/data` 的原始 OEA 数据——直接拷 sft_data 最省事。）
