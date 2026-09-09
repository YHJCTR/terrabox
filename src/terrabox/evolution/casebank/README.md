# Memento-style CaseBank（严格无标签适配）

本模块是 Memento 的非参数 Case Memory / Case-Based Reasoning 思路在 Terrabox OEA 真实工具
rollout 上的适配，复用正负案例、语义检索和压缩案例指导。它不是官方完整复现：官方仓库
`/data1/yuhongjie2/external_repos/Memento` 的 Meta-Planner、独立 Executor MCP runtime、在线
case-selection policy 与 Terrabox 的 LangGraph 单 agent rollout 接口不兼容，未被直接接入。
因此实验和论文中必须称为 `Memento-style CaseBank (adapted)`。

## 严格无标签口径

`--strict-nolabel` 只使用训练 rollout 对 actor 可见的任务文本、真实工具序列、完成状态和
可观察工具错误。它会：

- 不读取 `metrics`、gold、`expected_tools`、任务标签或答案 judge；
- 将明确终态的基础设施/API/Docker/OOM/网络失败排除出案例库；已恢复的历史工具 timeout 不会整条丢弃，也不会单独提高负向权重；
- 从序列化 case、embedding 文档和 prompt 中移除 task id、task type、最终答案和原始路径；
- 将可识别的地名、文件和 artifact 引用替换为占位符；
- 用完成状态、正常 final、实际工具调用和明确工具错误计算 reward；
- 强制 Qwen embedding retrieval，不允许 lexical fallback。

严格 store 的 case ID 只是建库顺序号，不是数据集 task ID。运行时传入的兼容 `task_type`
参数会被忽略。

## 建库

先确认 embedding 服务暴露的模型与配置一致且维度为 2560。默认 URL 是
`http://127.0.0.1:9101/v1/embeddings`，可通过以下变量显式指定：

```bash
export TERRABOX_CASEBANK_EMBEDDING_URL=http://127.0.0.1:9101/v1/embeddings
export TERRABOX_CASEBANK_EMBEDDING_MODEL=/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding
export TERRABOX_CASEBANK_EMBEDDING_DIM=2560

PYTHONPATH=src python -m terrabox.evolution.casebank.builder \
  --results-dir tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --output-dir evolution_store/casebank/oea_train2000_longcat_strict_nolabel_20260828 \
  --strict-nolabel \
  --embedding-backend qwen
```

建库时会检查 `/v1/models`、实际 embedding 响应和向量维度；任一检查失败会立即退出，绝不
静默退化。建库后的 `manifest.json` 必须记录 `reproduction_scope=adapted`、embedding 配置和
过滤数量；`cases.jsonl` 与 embedding index 都需要进行禁用字段扫描。

## 评测

严格 store 必须设置：

```bash
export TERRABOX_CASEBANK_RETRIEVAL=qwen
```

然后用统一 rollout 入口传入：

```text
--evolution-method memento_casebank
--evolution-store evolution_store/casebank/oea_train2000_longcat_strict_nolabel_20260828
```

外部 LongCat 的 OEA 全量需按 `gpu --workers 1` 与 `nogpu --workers 2-3` 分 lane，使用独立结果
目录，并沿用 `TERRABOX_TOOL_SERVICE_SCOPE=call`、`TERRABOX_KEEP_VLM_WARM=1` 和跨进程限流。
