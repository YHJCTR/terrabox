# MemRL（原方法）胶水模块

这个目录是**胶水代码**，让 `/data1/yuhongjie2/MemRL` 的**原始 MemRL 方法**能在 Terrabox 的数据集和运行环境上跑起来，用于和 baseline（ReAct/Reflection）以及后续自研方法对比。

> **不是 MemRL 的重实现。** `source_*` 这条路径通过 `ExternalMemRLSourceService` 直接 import 并调用原仓库的 `memrl.service.memory_service.MemoryService` / strategies / value-driven（Two-Phase Retrieval、Q 更新等都走原方法）。本模块只负责：① 把 Terrabox 数据转成 MemRL 记录；② 把 rollout 跑在和 baseline 相同的环境上；③ 在 rollout worker 侧提供可移植的记忆检索索引。MemRL 算法逻辑保持原样。

## 两条路径（别混用）

| 路径 | 文件 | 说明 |
|------|------|------|
| **source（原方法，用这个对比）** | `source_runner.py` / `source_memory_service.py` / `source_adapter.py` / `source_prompt_injector.py` | 调用 `/data1/yuhongjie2/MemRL` 原始实现 |
| light（Terrabox 侧简化版，**非原方法**） | `runner.py` / `memory_builder.py` / `prompt_injector.py` | SQLite 简化 episodic memory，仅作快速预览/兜底，不要用于"MemRL 原方法"对比 |

## 后端：external_memrl vs lite

`create_memrl_source_service(backend=...)`：
- **`external_memrl`**：真正的原始 MemRL（需要原仓库依赖 + LLM/embedding 端点就绪）。**做原方法对比必须用它。**
- `lite`：可移植的 token-overlap 检索 + Q 更新兜底，纯本地、无外部依赖。仅用于冒烟/端点不可用时。
- `auto`（默认）：先试 external，失败自动降级 lite，并在 manifest 里写 `fallback_reason`。

> ⚠️ 做正式对比时**显式传 `--backend external_memrl`**，不要用 `auto`，否则一旦端点没配好会静默退化成 lite，得到的就不是原方法的结果。

## 原方法的端点要求（关键）

原始 MemRL 需要一个 **OpenAI 兼容的 chat LLM** 和一个 **embedding 端点**（`source_memory_service.py` 里通过环境变量读取）：

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `MEMRL_LLM_BASE_URL` | `http://localhost:9100/v1` | chat 模型 = Terrabox agent LLM（9100） |
| `MEMRL_LLM_MODEL` | `terrabox-local` | 见 `curl localhost:9100/v1/models` 里的 id（实际是 `/model`） |
| `MEMRL_LLM_API_KEY` | `sk-local` | 占位即可 |
| `MEMRL_EMBED_BASE_URL` | 同 LLM | **embedding 端点** |
| `MEMRL_EMBED_MODEL` | `text-embedding-3-large` | embedding 模型名 |
| `MEMRL_TOP_K` / `MEMRL_BUILD_STRATEGY` / `MEMRL_RETRIEVE_STRATEGY` / `MEMRL_UPDATE_STRATEGY` | `5` / `trajectory` / `query` / `adjustment` | 透传给原方法 strategy |

> **坑**：9100 跑的是 Qwen3-8B **chat** 模型，不等于 embedding 服务。直接用默认 `MEMRL_EMBED_BASE_URL=http://localhost:9100/v1` 通常会因为 `/v1/embeddings` 不存在而失败；若用 `auto` 会退化成 lite。正式对比必须另起 embedding 服务，并显式设置 `MEMRL_EMBED_BASE_URL/MEMRL_EMBED_MODEL`。`MEMRL_LLM_MODEL` 也要改成 9100 实际暴露的模型 id（通常是 `/model`）。

### embedding 模型与维度

原 MemRL 当前在 `/data1/yuhongjie2/MemRL/memrl/service/memory_service.py` 里把 Qdrant `vector_dimension` 写死为 **3072**。因此：

- 如果用 `text-embedding-3-large`，默认 3072 维，与当前原代码匹配。
- 如果用本机已下载的 Qwen3 Embedding 4B：`/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding`，默认维度是 **2560**（`hidden_size=2560` / `word_embedding_dimension=2560`），和 3072 不匹配；需要先把原 MemRL 的 `vector_dimension` 改成可配置，或让 embedding 服务固定输出一个与 Qdrant 配置一致的维度。
- 如果用 Qwen3 Embedding 8B，默认最大维度是 4096，也同样需要改维度配置。
- 每次更换 embedding 模型或输出维度，都应使用新的 `--store-dir` 重新建库，不要复用旧 Qdrant/memory snapshot。

推荐下一步：把 `vector_dimension` 做成环境变量（如 `MEMRL_EMBED_DIM`），然后用小样本先检查：

```bash
curl http://localhost:<EMBED_PORT>/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"<embedding-model>","input":["test"]}'
```

确认返回的 `embedding` 长度与 `MEMRL_EMBED_DIM` 一致后，再跑 `populate-source`。当前如果使用 Qwen3 Embedding 4B，建议设置 `MEMRL_EMBED_DIM=2560`（需要先给原 MemRL 加这个环境变量支持）。

## Python 环境建议

`populate-source --backend external_memrl` 会 import `/data1/yuhongjie2/MemRL` 的原仓库依赖（`memos`、`qdrant_client` 等），因此建议用原 MemRL 的 conda 环境：

```bash
PY=/home/yuhongjie/miniconda3/envs/memoryrl/bin/python
PYTHONPATH=src:/data1/yuhongjie2/MemRL $PY -m terrabox.evolution.memrl_full.source_runner --help
```

如果缺少 Terrabox 胶水代码需要的包，再在 `memoryrl` 环境里补最小依赖；不要为了 `populate-source` 去改正在跑 SFT 的 `unsloth` 环境。当前已验证：

```bash
env -u ALL_PROXY -u all_proxy \
PYTHONPATH=src:/data1/yuhongjie2/MemRL \
no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
/home/yuhongjie/miniconda3/envs/memoryrl/bin/python \
  -m terrabox.evolution.memrl_full.source_runner --help
```

`eval-source` 会启动真实 Terrabox rollout，仍建议用 `unsloth` 环境（默认 `--python-bin /home/yuhongjie/miniconda3/envs/unsloth/bin/python`）。

本机代理注意：MemRL 原方法本身不要求外网代理；如果 LLM 和 embedding 都是本地 OpenAI-compatible 服务，整个实验应走 localhost。问题是 Python HTTP 客户端会自动读取 shell 里的 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，而本机曾出现 `ALL_PROXY=socks5://127.0.0.7897` 这类非法代理值，`memos` import 时会被 `ollama/httpx` 读取并直接报错。跑本地模型时建议清掉代理变量，只保留本地绕过：

```bash
env -u ALL_PROXY -u all_proxy \
  no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
  PYTHONPATH=src:/data1/yuhongjie2/MemRL \
  $PY -m terrabox.evolution.memrl_full.source_runner populate-source ...
```

## 原 MemRL 的记忆生命周期

原 MemRL 不是“只离线建一个固定库，然后永远只读库评测”的方法。它的核心是 runtime learning：

1. 当前任务开始前，从已有 memory 中检索相关经验。
2. 把检索到的 memory context 注入 agent prompt。
3. 跑完任务后，根据环境反馈/成功失败更新被检索 memory 的 Q-value。
4. 把本轮新轨迹通过 `add_memories()` 写回 memory，供后续任务继续检索。
5. 周期性保存 memory checkpoint/snapshot，便于恢复或评测固定 checkpoint。

代码上可以在原仓库 runner 里看到这个闭环：`retrieve_query(...)` → 组装 `memory_context` → 执行任务 → `update_values(...)` → `add_memories(...)`。因此它既可以从空库开始在线积累，也可以从已有 train/history 轨迹预填充一个初始库，再在 online 阶段继续更新。

Terrabox 当前建议采用两阶段对比：

- **populate-source**：先用当前 SFT train gold 轨迹预填充初始 memory，相当于给 MemRL 一个和 SFT 训练集对齐的经验库。
- **eval-source**：在 OpenEarth / EarthBench eval 上只读注入这个经验库，先做和 ReAct/Reflection 可比的离线评测。
- **online-source**：需要评估 MemRL runtime learning 时，再让 eval 产生的新轨迹写回库并做 Q 更新。

## 当前 Terrabox 胶水的 prompt 注入方式

Terrabox rollout 中，每个 task 会调用 `get_prompt_augmenter("memrl_full_source", store_dir=...)`，再执行 `augment(question)` 生成一段 MemRL 经验文本。这段文本会和基础系统提示、工具列表、当前任务一起进入 agent prompt。

当前 `source_*` 胶水有一个实现细节需要注意：

- `populate-source --backend external_memrl` 会真正 import 并调用原 MemRL 的 `MemoryService.add_memories(...)`，同时写出一个可移植的 `memory_index.jsonl`。
- `eval-source` 的 prompt 注入目前读取的是 `memory_index.jsonl`，由 `source_prompt_injector.py` 里的便携检索器选出若干条经验注入 prompt。
- 因此，目前 eval 阶段确实能“走 MemRL 经验库”注入经验，但检索排序还不是原 MemRL service 的完整 `retrieve_query` / Q-value two-phase 路径。若要严格复现原方法的 runtime retrieval，应进一步让 prompt injector 使用原 MemRL snapshot/service，或在 populate/online 后把原 MemRL 的 query/Q 排序状态持久化到可供 rollout worker 使用的索引里。

## 数据与环境对齐（和 baseline 可比）

- **v2 / 当前 SFT 对齐（推荐）**：当前 `v2_sft` 使用 `data/fixdata_decollapse_v2/sft_train_strict.jsonl` 切分而来；实际 train 是 `src/terrabox/evolution/sft/exp/v2_sft/sft_data/train.jsonl`，val 是 `.../val.jsonl`。MemRL 建记忆若要和当前 SFT train 严格一致，应使用这个 train JSONL，而不是全量 strict（否则会把 SFT val 200 条也写进 memory）。
- **v1 旧默认**：`DEFAULT_SFT = data/fixdata_decollapse/sft_train_strict.jsonl`（44 工具）仍可用于旧 ReAct/Reflection v1 对比，但不要和 v2 结果混报。
- **运行环境**：`eval-source` 现在复用 `ReAct.runner.build_rollout_env`，自动继承单卡 VLM 轻量档（`--vlm-gpus 2`，防四卡跳闸）、instructsam **service 内核**（`--instructsam-backend service`）、artifact 输出重定向、tool-GPU 绑定 —— 与 ReAct/Reflection **完全一致**。
- **v2 评测集**：OpenEarth 和 EarthBench 必须分开跑、分开报：`data/fixdata_decollapse_v2/eval_openearth.json`、`data/fixdata_decollapse_v2/eval_earthbench.json`。
- **v1 评测集**：用与 ReAct 测试集相同的 shuffle[0:216]。eval 用全量 shuffle 任务文件 + `--start-index 0 --limit 216`（与 reflection 的 `tasks_full.json` 一致），记忆用 train 切片构建。

## 标准流程：v2 / 当前 SFT 对齐（推荐）

### 0) 准备本地服务

需要两个本地 OpenAI-compatible 服务：

- chat LLM：Terrabox agent LLM，例如 `http://localhost:9100/v1`，模型 id 通常是 `/model`。
- embedding：单独的 `/v1/embeddings` 服务，例如 Qwen3 Embedding 4B/8B 或其他 embedding 模型；确认输出维度和 MemRL Qdrant 配置一致。

```bash
no_proxy=localhost,127.0.0.1 curl http://localhost:9100/v1/models
no_proxy=localhost,127.0.0.1 curl http://localhost:<EMBED_PORT>/v1/models
```

### 1) 用当前 SFT train 构建 MemRL 原方法记忆

```bash
PY=/home/yuhongjie/miniconda3/envs/memoryrl/bin/python
STORE=evolution_store/memrl_full_source_v2_sft_train

env -u ALL_PROXY -u all_proxy \
PYTHONPATH=src:/data1/yuhongjie2/MemRL \
no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
MEMRL_LLM_BASE_URL=http://localhost:9100/v1 \
MEMRL_LLM_MODEL=/model \
MEMRL_LLM_API_KEY=sk-local \
MEMRL_EMBED_BASE_URL=http://localhost:<EMBED_PORT>/v1 \
MEMRL_EMBED_MODEL=<embedding-model-id> \
MEMRL_EMBED_DIM=2560 \
$PY -m terrabox.evolution.memrl_full.source_runner populate-source \
  --backend external_memrl \
  --memrl-root /data1/yuhongjie2/MemRL \
  --sft src/terrabox/evolution/sft/exp/v2_sft/sft_data/train.jsonl \
  --store-dir $STORE \
  --snapshot-id final
```

正式结果前检查：

```bash
cat $STORE/manifest.json
```

确认 `service.backend` 是 `external_memrl`，且没有 `fallback_reason`。

### 2) 分基准 eval（OpenEarth / EarthBench 分开）

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
STORE=evolution_store/memrl_full_source_v2_sft_train

PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
$PY -m terrabox.evolution.memrl_full.source_runner eval-source \
  --task-file data/fixdata_decollapse_v2/eval_openearth.json \
  --start-index 0 --limit 300 \
  --store-dir $STORE \
  --experiment v2_memrl_oe \
  --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --instructsam-backend service

PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
$PY -m terrabox.evolution.memrl_full.source_runner eval-source \
  --task-file data/fixdata_decollapse_v2/eval_earthbench.json \
  --start-index 0 --limit 202 \
  --store-dir $STORE \
  --experiment v2_memrl_eb \
  --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --instructsam-backend service
```

### 3) 汇总指标

```bash
PYTHONPATH=src $PY -m terrabox.evolution.memrl_full.source_runner stats \
  --store-dir $STORE --experiment v2_memrl_oe \
  --task-file data/fixdata_decollapse_v2/eval_openearth.json --limit 300

PYTHONPATH=src $PY -m terrabox.evolution.memrl_full.source_runner stats \
  --store-dir $STORE --experiment v2_memrl_eb \
  --task-file data/fixdata_decollapse_v2/eval_earthbench.json --limit 202
```

## 旧 v1 流程（仅复现旧对比时使用）

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
STORE=evolution_store/memrl_full_source
TASKS=src/terrabox/evolution/reflection/exp/shuffle_seed42_reflection_decollapse/tasks_full.json  # 全量 shuffle 任务文件

# 0)（仅 external）确保 9100 agent LLM 在线 + 配好 embedding 端点环境变量
no_proxy=localhost curl http://localhost:9100/v1/models

# 1) 用 train 切片(gold 轨迹)构建 MemRL 原方法记忆
PYTHONPATH=src $PY -m terrabox.evolution.memrl_full.source_runner populate-source \
  --backend external_memrl --memrl-root /data1/yuhongjie2/MemRL \
  --sft data/fixdata_decollapse/sft_train_strict.jsonl \
  --store-dir $STORE --snapshot-id final

# 2) 在 216 测试集上 eval（注入 MemRL 记忆；环境与 ReAct/Reflection 一致）
PYTHONPATH=src $PY -m terrabox.evolution.memrl_full.source_runner eval-source \
  --task-file $TASKS --start-index 0 --limit 216 \
  --store-dir $STORE --experiment memrl_source_eval216 \
  --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 --instructsam-backend service

# 3) 在线模式（跑完把生成轨迹写回记忆并做 Q 更新，原方法的 runtime RL）
PYTHONPATH=src $PY -m terrabox.evolution.memrl_full.source_runner online-source \
  --backend external_memrl --task-file $TASKS --start-index 0 --limit 216 \
  --store-dir $STORE --top-k 5 --q-alpha 0.1

# 4) 汇总指标（rollout 口径 + 记忆统计）
PYTHONPATH=src $PY -m terrabox.evolution.memrl_full.source_runner stats \
  --store-dir $STORE --experiment memrl_source_eval216 \
  --task-file $TASKS --limit 216
```

## 子命令

| 子命令 | 作用 |
|--------|------|
| `populate-source` | 从 SFT gold / 历史 rollout 轨迹构建 MemRL 记忆（写 `memory_index.jsonl` + 调原方法 `add_memories`） |
| `eval-source` | 真实 Terrabox rollout（注入记忆），环境对齐 baseline |
| `online-source` | eval 后把新轨迹写回记忆 + Q 更新（原方法的 runtime RL 闭环） |
| `stats` | 汇总 rollout 指标和记忆统计 |

## 注意

- VLM 容器全局单例：**MemRL eval 不能和 ReAct/Reflection 同时跑**。
- 与 baseline 对比时务必：相同 `--task-file` + `--start-index/--limit`、相同 `--vlm-gpus/--instructsam-backend`、`--backend external_memrl`。
- `auto` 降级到 lite 会在 `manifest.json` 标注 `fallback_reason`；正式结果前先确认 `service.backend == "external_memrl"`。
