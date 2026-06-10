# DPO Baseline（独立对比实验）

在文本 Qwen3-8B 上做 **DPO**，作为 SFT / GRPO / 自研方法之外的一条独立 baseline。**A 模式**：`chosen = gold 轨迹`，`rejected = 模型自己在同一道题上的 rollout`。

> **独立实验,不依赖任何 evolution 方法(MemRL/Reflection 等)。** 只复用所有 baseline 共享的基础设施:`full_shared.sft_schema`(gold 样本)、`reflection.data_split` 的确定性 shuffle(**保证和 ReAct/Reflection/SFT 同一切分,可比**)、`ReAct.runner.build_rollout_env`(评测环境)。技术栈 = `trl DPOTrainer + unsloth`(和 SFT 的 `train_lora.py` 同栈;veRL 主打在线 PPO/GRPO,无现成离线 DPO,故不用)。

## DPO 的配对到底要什么(纠正常见误解)

DPO 的单元是 **`(prompt, chosen, rejected)`** —— **同一个 prompt 上的两条不同轨迹,且有质量差**。
- **按 prompt(task_id)配对,不是按"轨迹相似/一致"配对。** "rollout 和 gold 不一致"恰恰是好事。
- 唯一要丢的是 **reward gap = 0** 的对(两条一样好/坏 → 没梯度),`pair_builder` 已自动跳过 `chosen==rejected` 的。

## A 模式 + rejected 策略(你问的"全是坏轨迹跟 gold 比,行不行")

可以,很方便,但要选对 `--rejected-policy`,因为**约一半 rollout 其实 real_success=True**(用不同工具把题做对了):

| 策略 | 含义 | 适用 / 风险 |
|------|------|------|
| **`below-gold`(默认)** | rollout `f1 < 1.0`(没复现 gold 工具集)→ 当 rejected | 对应你说的"基本都不匹配 gold→全是坏轨迹"(实测保留 ~98%)。本质是**向 gold 工具用法对齐的 imitation-anchored DPO**。**风险**:会把"换了别的工具但做对了"的成功轨迹也当负例。 |
| `failed-only` | 只有 `real_success=False` 当 rejected | **最干净的正确性信号**,绝不惩罚"另一种对的解法"。实测保留 ~一半。 |
| `all` | 所有匹配到的 rollout 都当 rejected | 连成功的也罚,慎用。 |

**我的建议**:
- 想要"工具调用向 gold 看齐"的效果 → `below-gold`(你的原意),但论文里要写清楚这是"以 gold 为锚的偏好",并报告它惩罚了部分成功轨迹这一点。
- 想要更稳、更像"真偏好学习" → `failed-only`(gold 对 vs 模型错,干净)。
- 两个都跑成消融(`below-gold` vs `failed-only`)最稳妥。

> 注意:A 模式的 rejected 来自模型已有 rollout(**off-policy**),且 chosen 永远是 gold。这更接近"用偏好损失加强 SFT",不是从模型自身好/坏对比里学。审稿角度它是合理的廉价 baseline;真·on-policy 偏好需要每题采样多条再排序(成本=k×rollout,另一条路线)。

## 数据 / 切分 / 环境对齐

- 数据:`data/fixdata_decollapse/sft_train_strict.jsonl`(44 工具,与所有 baseline 同源)。
- 切分:`val = shuffle[0:216]`(与 ReAct 测试集**逐条相同**),`train = shuffle[216:...]`。
- rejected 来源:复用**已有 ReAct/Reflection rollout 缓存**(`--rollout-dir`),零额外 rollout 成本。
- 评测:`rollout` 子命令复用 `build_rollout_env`(单卡 VLM 防跳闸 + instructsam service),与 baseline 同环境。

## 流程

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
EXP=dpo_modeA_belowgold
# 复用 reflection train(和/或 ReAct)rollout 缓存做 rejected 来源
CACHE=src/terrabox/evolution/reflection/exp/shuffle_seed42_reflection_decollapse/train

# 1) 构造偏好对(纯数据,不耗 GPU)
PYTHONPATH=src $PY -m terrabox.evolution.dpo.runner prepare-pairs \
  --experiment $EXP --rollout-dir $CACHE \
  --rejected-policy below-gold \
  --val-start 0 --val-limit 216 --train-start 216

# 2) DPO 训练(unsloth 单卡;--save-steps 控制中途 checkpoint;merged 默认产出)
PYTHONPATH=src $PY -m terrabox.evolution.dpo.runner train \
  --experiment $EXP --cuda-visible-devices 0 --beta 0.1 --save-steps 50 --launch
#  → 产物: src/terrabox/evolution/dpo/model/$EXP/merged  (HF 格式,vLLM 可直接挂)

# 3) 在 216 测试集上评测(环境与 ReAct/Reflection 一致)
PYTHONPATH=src no_proxy=localhost,127.0.0.1 $PY -m terrabox.evolution.dpo.runner rollout \
  --experiment $EXP --model-path src/terrabox/evolution/dpo/model/$EXP/merged \
  --start-index 0 --limit 216 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --launch
```

## 子命令

| 子命令 | 作用 |
|--------|------|
| `prepare-pairs` | 从 gold + rollout 缓存按 task_id 构造 `(prompt, chosen, rejected)`,写 train/val pairs + prompt-only `eval_tasks.json` |
| `train` | trl DPOTrainer + unsloth QLoRA;存 adapter + merged(HF)+ 中途 checkpoint |
| `rollout` | 用训练后的 merged 模型在真实 Terrabox 工具链上评测 |

## 备注

- `train` 默认单卡(`--cuda-visible-devices 0`):unsloth/PEFT 本就单卡,`ref_model=None` 靠"关掉 adapter"得到参考模型,不复制第二份 8B。
- merged 目录 = ReAct agent LLM(9100)挂载的同款 HF 格式,评测时经 `AGENT_LLM_MODEL_PATH` 注入。
- checkpoint:`--save-steps/--save-total-limit` 控制;少量数据 + 小 `--max-steps` 可先跑通整套流程。
- VLM 容器全局单例:DPO 评测不能和 ReAct/Reflection/SFT/MemRL 评测同时跑。
