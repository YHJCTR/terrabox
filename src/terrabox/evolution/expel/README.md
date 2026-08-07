# ExpeL 适配

本目录保留旧的离线工具序列预测入口，同时提供 `expel_live` 真实 agent 评测模式。

## 真实 OEA 链路

```text
已有 train2000 LongCat base rollout
  -> 过滤基础设施错误，按 task_type + 工具顺序覆盖选择成功轨迹
  -> LongCat 批量抽取可迁移 principles
  -> 保存脱敏成功 episode 摘要，作为官方 ExpeL 风格 few-shot memory
  -> principles.json
  -> OEA test 真实 LongCat agent
  -> query lexical retrieval: principles + 1 条成功 episode
  -> 真实 Terrabox 工具执行
  -> rollout_report + LongCat answer judge
```

构建命令：

```bash
PYTHONPATH=src python -m terrabox.evolution.expel.runner build-live \
  --source-results tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --store-dir evolution_store/expel/oea_train2000_longcat_official_20260807 \
  --max-success 240 --max-failure 120 --batch-size 8
```

真实评测使用统一入口的 `--evolution-method expel_live`。它不读取 test 的
`expected_tools`、gold answer、task id 或 gold trajectory。旧 `expel` 方法和旧
离线输出不会被覆盖。

`expel_live` 支持两种检索后端：`lexical` 是可审计的 token overlap 基线；`qwen`
使用 Qwen3-4B-Embedding 预计算 rule/episode 向量，并在 eval 时只对 query 请求
embedding。正式 watcher 使用后者，embedding 服务固定在不与主 rollout 共用的 GPU2。
构建时加 `--embedding-backend qwen`，eval 时设置 `TERRABOX_EXPEL_RETRIEVAL=qwen`。

构建过程会原子保存每个已处理 batch。LongCat 偶发返回不合法 JSON 时，该 batch 会有限重试；仍无法解析时只记录到 manifest 的 `llm_parse_failed_batches` 并继续构建，避免单个格式错误中断整轮真实评测。
