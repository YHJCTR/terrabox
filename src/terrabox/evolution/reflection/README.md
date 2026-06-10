# Reflection 基线（Reflexion 式自反思记忆）

在 ReAct 之上加“自反思记忆”：train 阶段跑轨迹 → LLM 对**自己的轨迹**反思出教训 → eval 阶段检索注入提示词。子命令：`prepare-tasks` / `rollout` / `build-memory` / `stats`。

## ⚠️ 2026-06-08 重写：忠实的 Reflexion（无 gold 泄漏）

旧实现是**模板拼接**,且把 **gold `expected_tools` 和 `f1`** 写进经验("Missing expected tools: instructsam")再注入 eval = **泄漏标准答案(作弊)**。已改为正常 Reflexion：

- **LLM 自己反思**(`reflector.py` 用 `EvolutionLLMClient` 调 Docker vLLM):只看**自己的轨迹**(question、自己调的工具、**自己观察到的报错**[tool_error/oom/timeout/file_not_found/no_tool_call/repeated_tool_call]、自己的最终答案),写 1-2 条"下次怎么做更好"的教训。
- **绝不碰 gold**:经验条里 `expected_tools=[]`、`f1=0`,**eval 注入时只显示 LLM 写的教训 + "上次你用了哪些工具"**,不显示任何 gold/F1/task_type。
- 成功/失败只用**可观察信号**判定(有没有报错/有没有调工具),不靠 gold。

> build-memory 需要 Docker vLLM 在线(EvolutionLLMClient)。若离线,会退化成"只列自观察问题"的非泄漏模板(仍不碰 gold),并告警。

## 与 ReAct 完全对齐(复用其代码)

reflection runner `from ..ReAct.runner import build_rollout_env`、`from ..ReAct.metrics import write_metrics`,自动继承 ReAct 的：输出重定向兜底(`TERRABOX_ARTIFACT_OUTPUT_DIR`)、新系统提示("能用工具就别自己想")、单卡 VLM 逻辑、对齐数据、全套指标(precision/recall、set/multiset/relaxed-F1、answer-accuracy、分类别)。

rollout 默认已对齐 ReAct：`--agent-gpu 0 --tool-gpu 1 --vlm-gpus 2`(单卡轻量档,防跳闸)、`--exclude-tools None`(fixdata 已无 ipython)、instructsam **service 内核**(`TERRABOX_INSTRUCTSAM_BACKEND=service`)。

> VLM 容器全局单例,**ReAct 与 Reflection 不能同时跑**。

## 数据(与 ReAct 同一份、同 seed42、同 44 工具)

用 **`data/fixdata_decollapse/sft_train_strict.jsonl`**(非坍缩 44 工具;坍缩用 `data/fixdata/`,36 工具)。
**关键**:用**全量 task 文件 + `--start-index/--limit` 切片**(不要用切好的小文件),这样 `allowed` 按全量算 = 44,和 ReAct 一致。
- eval = shuffle[0:216](与 ReAct 测试集**完全相同**)
- train = shuffle[216:648](432,与 eval **零重叠**,防泄漏)

## 一键流水线(推荐,可选择性拼接阶段)

`pipeline` 子命令把 **train → build-memory → eval → stats** 串起来,docker 服务在各阶段间**保活**(agent LLM 9100 留给 build-memory 的 LLM 自反思),最后统一停。

```bash
EXP=shuffle_seed42_reflection_decollapse
DIR=src/terrabox/evolution/reflection/exp/$EXP
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python

tmux new-session -d -s refl_pipe "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.reflection.runner pipeline \
  --experiment $EXP --task-file $DIR/tasks_full.json \
  --phases train,build-memory,eval,stats \
  --train-start 216 --train-limit 432 --eval-start 0 --eval-limit 216 \
  --vlm-gpus 2 2>&1 | tee $DIR/logs/pipeline.log"
```

`--phases` 任选子集,灵活续跑(train 都是 `--resume`,不浪费已跑):
- 只重建记忆+评测:`--phases build-memory,eval,stats`(需 agent LLM 9100 在线)
- 只评测:`--phases eval,stats`
- 只看 train 的 ReAct 口径指标:`--phases stats`(配 `stats --rollout-dir $DIR/train`)

> train 阶段 = **在 shuffle[216:648] 上跑的纯 ReAct**(同一 `run_trajectory_experiment.py --mode standard`,只是不注入记忆),一样产出全套 ReAct 指标;反思增益只在 **eval 阶段**(注入记忆后)体现。

## ✅ v2 数据运行指令（2026-06-09,EarthBench 恢复,训练池+两套测试分开)

v2 = `data/fixdata_decollapse_v2/`(统一 **64 工具**;训练池与测试已**预切分**,不是同一 shuffle 文件的切片),所以**不用 pipeline 的单文件切片模式**,用**手动分步** + `--output-dir` 把两套测试分目录:

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
EXP=v2_reflection
DIR=src/terrabox/evolution/reflection/exp/$EXP
V2=data/fixdata_decollapse_v2

# 0) 从 v2 训练池生成 train 任务文件(经验提取用;rollout 贵,--limit 取子集如 432)
PYTHONPATH=src $PY -c "from terrabox.evolution.reflection.data_split import write_task_slice; \
write_task_slice('$V2/sft_train_strict.jsonl','$DIR/train_tasks.json',seed=42,start=0,limit=None,split_name='train')"

# 1) train rollout(无记忆,产轨迹;--limit 控成本)
tmux new-session -d -s refl_v2_train "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.reflection.runner rollout --experiment $EXP --phase train \
--task-file $DIR/train_tasks.json --start-index 0 --limit 432 \
--port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --resume"

# 2) build-memory(LLM 自反思;需 agent LLM 9100 在线)
PYTHONPATH=src no_proxy=localhost,127.0.0.1 $PY -m terrabox.evolution.reflection.runner build-memory \
  --experiment $EXP --trajectory-dir $DIR/train

# 3a) eval ① OpenEarth(注入记忆;--output-dir 分目录;OE 大→--limit 抽样)
tmux new-session -d -s refl_v2_eval_oe "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.reflection.runner rollout --experiment $EXP --phase eval \
--task-file $V2/eval_openearth.json --output-dir $DIR/eval_oe --start-index 0 --limit 300 \
--evolution-store $DIR --port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --resume"

# 3b) eval ② EarthBench(全量 202)
tmux new-session -d -s refl_v2_eval_eb "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.reflection.runner rollout --experiment $EXP --phase eval \
--task-file $V2/eval_earthbench.json --output-dir $DIR/eval_eb --start-index 0 --limit 202 \
--evolution-store $DIR --port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --resume"

# 4) 指标:OpenEarth / EarthBench 分开报
PYTHONPATH=src $PY -m terrabox.evolution.reflection.runner stats --experiment $EXP --rollout-dir $DIR/eval_oe
PYTHONPATH=src $PY -m terrabox.evolution.reflection.runner stats --experiment $EXP --rollout-dir $DIR/eval_eb
```

- **v1 ↔ v2 切换**:v1 用上面的 `pipeline`(单 shuffle 文件 `[0:216]`/`[216:648]` 切片);v2 用本节(训练池 + 两套预切测试,eval 分目录、分基准报)。GPU/内核/单卡 VLM/无 gold 泄漏 全部一致。
- 验证日志:`Built 64 LangChain tools`;eval 前仍做记忆"非空 + 无 gold + 确实注入"两项验证(见下)。

## 手动分步流程

```bash
EXP=shuffle_seed42_reflection_decollapse
DIR=src/terrabox/evolution/reflection/exp/$EXP
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
FULL=$DIR/tasks_full.json   # = 全量 shuffle 文件(可直接复用 ReAct 的 tasks_all_shuffled_seed42.json)

# 0) 全量任务文件(8646,seed42 shuffle)
PYTHONPATH=src $PY -c "from terrabox.evolution.reflection.data_split import write_task_slice; \
write_task_slice('data/fixdata_decollapse/sft_train_strict.jsonl','$FULL',seed=42,start=0,limit=None,split_name='all')"

# 1) train 阶段 rollout（432 = shuffle[216:648]；单卡 VLM、service 内核）
tmux new-session -d -s refl_train "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.reflection.runner rollout --experiment $EXP --phase train \
--task-file $FULL --start-index 216 --limit 432 --port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --resume"

# 2) 构建反思记忆（LLM 自反思；需 Docker vLLM 在线 → 先确保 agent LLM 容器在 9100）
PYTHONPATH=src no_proxy=localhost,127.0.0.1 $PY -m terrabox.evolution.reflection.runner build-memory \
  --experiment $EXP --trajectory-dir $DIR/train

# 3) eval 阶段 rollout（注入记忆；--evolution-store 必须 = 实验目录,旧 README 写 .../store 是错的）
tmux new-session -d -s refl_eval "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.reflection.runner rollout --experiment $EXP --phase eval \
--task-file $FULL --start-index 0 --limit 216 --evolution-store $DIR \
--port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --resume"

# 4) 指标（与 ReAct 同口径,直接和 ReAct 的 216 结果对比）
PYTHONPATH=src $PY -m terrabox.evolution.reflection.runner stats --experiment $EXP
```

## eval 前必做的两项验证(避免"记忆没生效"被误读为"反思无用")
1. `head $DIR/memory/reflection_memory.jsonl` —— 确认**非空**、reflection 是 LLM 写的、**不含 gold**。
2. 抽一条 eval 结果的 conversation,确认系统提示里有 `## Lessons from your past similar attempts` 段(记忆**确实注入**)。
