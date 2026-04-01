# Self-Evolution Experiments

自进化模块实验脚本说明。所有脚本均在项目根目录下运行，需设置 `PYTHONPATH=src`。

> **Python 环境**：必须使用 `unsloth` conda 环境（含 langchain/langgraph/vllm 等依赖）：
> ```bash
> /home/yuhongjie/miniconda3/envs/unsloth/bin/python
> ```
> 系统 `python3` 缺少必要依赖，会报 `No module named 'langchain_core'`。

---

## 快速开始

```bash
cd /data1/yuhongjie2/terrabox
PYTHON=/home/yuhongjie/miniconda3/envs/unsloth/bin/python

# 1. 构建知识库（无需 vLLM，读取 train.json）
PYTHONPATH=src $PYTHON src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase build --build-limit 5000 --reset

# 2. 用真实 vLLM 评估（加 --online 开启在线自进化）
PYTHONPATH=src $PYTHON src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase eval --online --eval-limit 50

# 3. 对比所有方法（全流程 + 在线自进化）
PYTHONPATH=src $PYTHON src/terrabox/evolution/experiments/run_real_evolution.py \
    --method all --phase both --online --build-limit 5000 --eval-limit 100
```

---

## 脚本一览

| 脚本 | 用途 |
|------|------|
| `run_real_evolution.py` | **主脚本**：真实数据 + 真实 vLLM，支持所有方法 |
| `test_causalevo_unit.py` | CausalEvo 单元测试（无需 vLLM，合成数据）|
| `compare_all_methods.sh` | Shell 版方法对比（封装各 runner CLI）|
| `ablation_causalevo.sh` | CausalEvo 消融实验 Shell 脚本 |

---

## run_real_evolution.py 详解

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `causalevo` | 运行哪个方法。可选：`baseline` / `causalevo` / `memrl` / `skillrl` / `agentevolver` / `evoskill` / `all` |
| `--phase` | `both` | `build`：仅离线建库（无需 vLLM）；`eval`：仅在线评估（需要 vLLM）；`both`：两阶段都跑 |
| `--train-data` | `data/openearth/train.json` | 训练数据路径 |
| `--eval-data` | `data/openearth/eval.jsonl` | 评估数据路径 |
| `--store-dir` | `evolution_store` | 知识库根目录（自动创建） |
| `--build-limit` | `5000` | build 阶段读取的训练轨迹数量上限 |
| `--eval-limit` | `50` | eval 阶段评估的案例数量 |
| `--top-k` | `5` | 检索 top-k 条知识/记忆 |
| `--ablation` | 无 | CausalEvo 消融模式：`no_cca` / `no_synthesis` / `no_ctfm` |
| `--online` | False | eval 时实时更新知识库（在线学习） |
| `--timeout` | `120` | 每个 agent 任务的超时时间（秒）|
| `--output` | 自动 | 结果 JSON 路径，默认 `evolution_store/results/{method}.json` |
| `--reset` | False | build 前清空已有知识库（重新开始）|
| `--verbose` | False | 输出 DEBUG 级别日志 |

### 在线自进化支持情况

| 方法 | --online 效果 | 持久化方式 |
|------|--------------|-----------|
| causalevo | ✅ 更新 CTFM 中每个工具的 success_rate 计数器 | JSON 文件 |
| memrl | ✅ 更新最近检索记忆的 Q 值（Bellman 更新） | SQLite DB |
| skillrl | — 无在线更新（离线蒸馏后固定） | — |
| agentevolver | — 无在线更新（经验池只在 build 阶段填充） | — |
| evoskill | — 无在线更新（技能发现需 LLM 多轮对话） | — |
| baseline | — 无知识库 | — |

> `--online` 只影响 causalevo 和 memrl；其他方法忽略该标志（不报错）。

### 运行流程

```
Build 阶段（无需 vLLM）
  ↓ 读取 train.json 训练轨迹
  ↓ 构建/更新各方法的知识库
  ↓ 保存到 evolution_store/{method}/

Eval 阶段（需要 vLLM）
  ↓ 连通性检测（自动启动 Docker/subprocess 服务）
  ↓ 读取 eval.jsonl 评估案例
  ↓ augmenter.augment(question) → 注入知识到 system prompt
  ↓ create_react_agent(llm, tools, prompt=augmented_prompt)
  ↓ ToolMatchEvaluator 计算 F1（对比 expected_tools）
  ↓ 每 10 条保存 partial 结果，防中断丢数据
  ↓ 输出汇总表
```

### 输出文件

```
evolution_store/
├── causalevo/
│   └── causal_tool_models.json       # CTFM 知识库
├── memrl/
│   └── episodic_memory.db            # SQLite 情节记忆
├── skillrl/
│   ├── general_skills.json
│   ├── task_skills.json
│   └── mistakes.json
├── agentevolver/
│   └── experience_pool.jsonl
├── evoskill/
│   └── pareto_frontier.json
└── results/
    ├── baseline.json                 # 各方法评估结果
    ├── causalevo.json
    ├── causalevo.partial.json        # 中间结果（断点续传参考）
    ├── comparison.json               # 汇总对比
    └── ...
```

### 评估指标

| 指标 | 说明 |
|------|------|
| `precision` | 预测工具中命中 expected_tools 的比例 |
| `recall` | expected_tools 中被覆盖的比例 |
| `f1` | 调和平均，主要参考指标 |
| `exact_match` | 预测工具集与 expected_tools 完全一致的比例 |
| `errors` | 超时或异常失败的任务数 |
| `delta F1` | 相对 baseline 的 F1 提升量 |

---

## 各方法说明

### Baseline（无进化）
直接用基础 system prompt 运行 ReAct agent，不注入任何知识。作为对比基准。

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method baseline --phase eval --eval-limit 100
```

### CausalEvo（因果进化，本文提出）
三个核心组件：
- **CCA**（反事实信用归因）：量化每个工具对成功的因果贡献
- **CTFM**（因果工具函数模型）：每个工具的先验知识（前置关键词、下游工具、成功率）
- **CGS**（因果图合成）：对新查询生成带拓扑排序的工具执行计划

```bash
# 建库
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase build --build-limit 10000 --reset

# 评估
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase eval --eval-limit 100

# 在线学习（eval 同时更新知识库）
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase eval --online --eval-limit 100

# 消融实验
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase eval --ablation no_cca --eval-limit 100

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase eval --ablation no_synthesis --eval-limit 100

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method causalevo --phase eval --ablation no_ctfm --eval-limit 100
```

也可以使用专用消融脚本（需要 vLLM）：
```bash
bash src/terrabox/evolution/experiments/ablation_causalevo.sh
```

### MemRL（情节记忆 + Bellman 更新）
IEU 框架（Intent-Experience-Utility）：两阶段检索 + Q 值在线更新。

```bash
# 建库（把训练轨迹存入 SQLite）
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method memrl --phase build --build-limit 10000 --reset

# 评估（含在线 Q 值更新）
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method memrl --phase eval --online --eval-limit 100
```

也可以直接调用 runner：
```bash
python -m terrabox.evolution.memrl.runner populate \
    --train-data data/openearth/train.json \
    --eval-data data/openearth/eval.jsonl \
    --memory-db evolution_store/memrl/episodic_memory.db \
    --limit 5000

python -m terrabox.evolution.memrl.runner eval \
    --eval-data data/openearth/eval.jsonl \
    --memory-db evolution_store/memrl/episodic_memory.db \
    --top-k 5 --limit 100 --output evolution_store/results/memrl.json
```

### SkillRL（层次技能库 + 递归进化）
三级技能库（通用/任务特定/错误模式），BM25 检索，性能下降时触发重新蒸馏。

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method skillrl --phase build --build-limit 5000 --reset

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method skillrl --phase eval --eval-limit 100
```

或直接调用 runner：
```bash
python -m terrabox.evolution.skillrl.runner distill \
    --train-data data/openearth/train.json --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/skillrl --limit 5000

python -m terrabox.evolution.skillrl.runner eval \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/skillrl --top-k 3 --limit 100
```

### AgentEvolver（自提问 + 自导航 + 自归因）
三机制：从训练数据挖掘任务模板、相似轨迹导航、步骤级折现信用归因。

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method agentevolver --phase build --build-limit 5000 --reset

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method agentevolver --phase eval --eval-limit 100
```

### EvoSkill（多智能体 + Pareto 前沿）
三智能体（执行/提议/构建），在失败分析后生成结构化技能，Pareto 管理技能库。

```bash
PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method evoskill --phase build --build-limit 200 --reset

PYTHONPATH=src /home/yuhongjie/miniconda3/envs/unsloth/bin/python src/terrabox/evolution/experiments/run_real_evolution.py \
    --method evoskill --phase eval --eval-limit 100
```

> **注意**：EvoSkill 的 build 阶段内部调用 LLM 进行多智能体分析，因此 build 也需要 vLLM。`--build-limit` 控制参与发现的 episode 数量（建议 30-200，太大会很慢）。

---

## 单元测试（无需 vLLM）

```bash
PYTHONPATH=src python src/terrabox/evolution/experiments/test_causalevo_unit.py
```

测试内容：
1. 统计 CCA 计算正确性
2. CTFM 构建 + JSON 存储轮回
3. 因果图合成（知识库 → 执行计划）
4. Prompt 注入器（Full + 3 种消融模式）
5. 无效消融模式报错

预期输出：`✅ All 5 tests passed!`

---

## 方法对比（Shell 版）

对比所有 5 种方法 + baseline：
```bash
cd /data1/yuhongjie2/terrabox
EVAL_LIMIT=100 BUILD_LIMIT=5000 \
    bash src/terrabox/evolution/experiments/compare_all_methods.sh
```

> 该脚本对每个方法独立运行，单个失败不影响其他方法，结果保存到 `evolution_store/comparison/`。

---

## 常见问题

### Q: vLLM 健康检测失败怎么办？
脚本会通过 `get_llm(config)` 自动启动 Docker 容器。若仍失败：
- 检查 Docker 是否运行：`docker ps`
- 检查 `agent_config.yaml` 中 `local_llm_port`（默认 9100）
- 手动测试：`curl http://127.0.0.1:9100/v1/models`
- 先跑 build（无需 vLLM）：`--phase build`

### Q: 如何断点续传 eval？
每 10 条自动保存 `evolution_store/results/{method}.partial.json`。
目前不支持自动从断点继续（重新运行会覆盖），但可以用 `--eval-limit` 缩小范围分批运行。

### Q: build 后重新 eval 不需要 --reset 吗？
是的，`--reset` 只在 build 阶段生效，会清空知识库从头建。eval 阶段不受 `--reset` 影响。

### Q: EvoSkill build 需要 vLLM 吗？
是的，EvoSkill 的 discover 阶段需要调用 LLM 进行失败分析和技能生成。其他方法（causalevo/memrl/skillrl/agentevolver）的 build 阶段均不需要 vLLM。

### Q: 评估结果中 errors 很高怎么办？
- 增大 `--timeout`（默认 120 秒，可试 `--timeout 180`）
- 减小 `--eval-limit`，先跑少量验证环境是否正常
- 查看日志中每个失败任务的报错原因
