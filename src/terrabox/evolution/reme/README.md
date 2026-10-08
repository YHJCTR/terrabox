# ReMe 程序性经验 OEA 适配器

统一入口：`get_prompt_augmenter("reme", store_dir=...)`，也接受 `reme_procedural`。与 ReasoningBank、ExperienceEvo 独立存放。

## 源码版本与实现边界

- 官方仓库：https://github.com/agentscope-ai/ReMe
- 主分支快照：`/data1/yuhongjie2/ReMe`，提交 `5231f3970c550bb0bc5d1bc7fbc85534274ca530`。
- 本适配器使用的程序性经验快照：`/data1/yuhongjie2/ReMe-procedural`，`reme_v3` 对应提交 `2f37a159b72a04ac1885a7db7f1a663a833e7791`。两套源码分别保留，均为官方提交归档，不含 git 历史。
- 实际读取 `reme/extension/procedural_memory/{summary,retrieve}/*.yaml` 官方提示词，不把当前文件记忆系统误称为论文的程序性经验方法。

当前管线为：真实轨迹 → 可见证据自评 → LLM 分段 → 成功/失败经验提取 → 同问题多轨迹高低分对比（有配对时）→ LLM 质量验证 → 确定性去重 → 冻结库 → 语义召回 → LLM 重排 → 注入标准 ReAct。

经验保存 `when_to_use/content/confidence/validation_score/source`；embedding 文档使用 `when_to_use + content`，与 ReasoningBank 的历史 query 检索不同。官方质量验证阈值使用 0.5。检索不额外加入 QNR、工具标签或 gold bonus。

必须报告为 **ReMe (official-prompt adapted, frozen procedural memory)**，差异包括：

- 复用官方 prompt 与流程机制，但不直接启动 ReMe 的 service/向量数据库 runtime；使用本仓库统一 LLM 客户端与显式 embedding HTTP 服务。
- 轨迹分数改为不读 gold 的 LLM 自评；成功布尔值决定成功/失败提取支路，标量分数用于比较。
- 显式启用官方分段组件；按提示词规定的零起始、含末尾下标分段，修正上游代码与 prompt 的边界歧义。
- 比较支路只在相同可见问题有不同分数轨迹时工作；若现有 train 每题仅一条轨迹，这个支路不会产生经验，不能宣称已测试完整比较学习收益。
- 当前去重是经验文本精确去重，未移植官方可选语义去重；尚无在线 utility/frequency 更新、删除与增库。完整在线 ReMe 应另设实验，避免测试反馈污染冻结主表。
- 默认语义召回至少 10 条，再用官方 LLM rerank 选 Top-K；上游允许关闭 rerank，本适配明确启用。可选 rewrite 默认关闭。
- JSON 格式错误立即报错，避免上游宽松解析或 fallback 静默改变实验机制。

## 命令

在项目根目录、`unsloth` 环境执行：

```bash
PYTHONPATH=src python -m terrabox.evolution.reme.runner preflight

PYTHONPATH=src python -m terrabox.evolution.reme.runner build \
  --results <train真实rollout的results目录> --source-split train \
  --train-manifest <train任务JSON或JSONL> --eval-manifest <eval任务JSON或JSONL> \
  --store evolution_store/reme/oea_train --provider longcat --resume

PYTHONPATH=src python -m terrabox.evolution.reme.runner index \
  --store evolution_store/reme/oea_train \
  --embedding-endpoint http://127.0.0.1:9101/v1/embeddings \
  --embedding-model /data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding
```

示例 embedding 端口需换成实际服务，不自动启动或占用 GPU。建库的 LLM 预算随分段数和候选数增加，默认逐条处理并原子缓存。长输入超过 `--max-input-chars` 明确报错，不截断；续跑要求输入、provider/模型、prompt 和源码哈希一致。bank 完成与 index 完成是不同阶段，两者都完成才能 rollout。

正式评测在原有 OEA 命令追加：

```text
--evolution-method reme --evolution-store evolution_store/reme/oea_train
```

外部 LongCat 评测需显式设置 `TERRABOX_REME_PROVIDER=longcat`，用于重排；否则默认 local。`TERRABOX_REME_SOURCE` 可指定源码位置，prompt 哈希必须匹配建库版本。embedding 地址/模型覆盖变量与 ReasoningBank 相同。

Python 工厂支持 `top_k`、`recall_k`、`rewrite`，以及测试依赖注入 `client/embedding`。当前任务开始时检索；数据集 task_type/gold 不参与。`record_outcome()` 不更新 bank，`last_retrieval` 保存最近候选。正式运行应记录重排 provider/模型、recall_k/top_k/rewrite 与源码版本；重排增加额外 API 请求，不应与纯 actor token 成本混为一谈。

数据划分校验、基础设施错误过滤、冻结索引一致性、缓存续跑与测试说明见 [ReasoningBank README](../reasoningbank/README.md)。接口测试不替代真实建库、真实 embedding 服务验证和完整 OEA 评测。
