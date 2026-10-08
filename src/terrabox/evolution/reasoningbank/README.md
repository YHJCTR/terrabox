# ReasoningBank OEA 适配器

统一入口：`get_prompt_augmenter("reasoningbank", store_dir=...)`，也接受 `reasoning_bank`。一个方法一个目录，不依赖 ExperienceEvo 的 QNR、状态过滤或词法检索。

## 来源与复刻范围

- 官方仓库：https://github.com/google-research/reasoning-bank
- 本机源码：`/data1/yuhongjie2/reasoning-bank`，固定提交 `ed80611788292ea739f1effd31f16c53823b8a0d`。
- 下载方式是 GitHub 提交归档快照，不包含 `.git` 历史；已逐文件校验归档与解压内容一致。
- 直接读取官方 `WebArena/prompts/memory_instruction.py` 的成功/失败蒸馏提示词，只将领域描述从 web navigation 改成 geospatial tool use，并增加不复制历史答案/文件名的要求。
- 保留每条轨迹最多 3 条 `title/description/content` 经验，按照**历史任务 query 的 embedding**召回其经验组；不是用 ExperienceEvo 检索器代替。
- 标记为 **ReasoningBank (official-prompt adapted, frozen)**。OEA 工具环境、自评 judge、embedding 服务和冻结 train→test 流程是适配差异；未实现 MaTTS 多轨迹搜索和测试时在线增库，不能称完整官方复现。

## 建库与索引

在项目根目录、`unsloth` 环境执行（路径中的占位参数需替换）：

```bash
PYTHONPATH=src python -m terrabox.evolution.reasoningbank.runner preflight

PYTHONPATH=src python -m terrabox.evolution.reasoningbank.runner build \
  --results <train真实rollout的results目录> --source-split train \
  --train-manifest <train任务JSON或JSONL> --eval-manifest <eval任务JSON或JSONL> \
  --store evolution_store/reasoningbank/oea_train --provider longcat --resume

PYTHONPATH=src python -m terrabox.evolution.reasoningbank.runner index \
  --store evolution_store/reasoningbank/oea_train \
  --embedding-endpoint http://127.0.0.1:9101/v1/embeddings \
  --embedding-model /data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding
```

示例端口不代表服务已启动；应使用当前实际 embedding 服务，不能覆盖正在运行的 agent LLM。建库每条轨迹会调用自评与蒸馏 LLM，费用独立于正式 rollout。

输入必须含 `question` 和真实 `conversation_history`。train/eval manifest 只取 question 做划分核查，不传入其标签。所有输入问题须属于 train 且不出现在 eval；有意做 transductive 实验应另设方法，不能绕过检查。

建库只白名单读取真实 user/assistant/tool 对话及 assistant 工具调用；不读取 evaluator success、expected tools、task_type、gold 或 F1。成功/失败由独立可见轨迹 LLM 自评，**不保证自评等于任务真实正确性**。明确的工具基础设施错误被过滤。未含真实工具 observation 的输入直接报错，不能拿 action smoke 或 SFT gold 当作 rollout。

`--max-input-chars` 默认 100000，约束每次蒸馏请求；超限报错，不悄悄截断。它是字符预算，不替代服务的 token 上限。`--resume` 要求输入、模型配置、源码/提示词哈希一致。每条成功处理结果原子保存于 `build_cache/`，完成后才发布 `bank.json`；`index.json` 校验 bank 哈希、模型名、维度和覆盖数量。索引错误或服务不可用均明确失败，不回退词法检索。

## 统一 rollout 接入

在现有正式 OEA rollout 命令中追加：

```text
--evolution-method reasoningbank --evolution-store evolution_store/reasoningbank/oea_train
```

其余任务集、standard 模式、完整 OEA 工具目录、服务配置、轮数和统计口径保持与对照一致。完整运行仍遵循模块 AGENTS 的感知/nogpu 分 lane 规则。

`top_k` 是历史 query 组数，每组最多 3 条经验；默认 5 不等于 5 条经验。可通过 Python 工厂参数修改。`augment()` 返回标准 BASE_SYSTEM 加经验块；`record_outcome()` 不写库。当前在任务开始时注入，不新增每步检索。可通过 `last_retrieval` 读取最近一次候选用于审计。

可用 `TERRABOX_MEMORY_EMBEDDING_ENDPOINT` 切换同一模型的服务地址；`TERRABOX_MEMORY_EMBEDDING_MODEL` 与已建索引不一致时拒绝运行。

## 产物与验证

每个 store 保存 `manifest.json`（模型、输入划分、提示词/适配源码哈希与复刻差异）、中断缓存、`bank.json`、`index.json`。这些是实验产物，不应提交源码仓库。上游许可证保留在源码目录，适配器运行时读取原 prompt，不复制整个上游项目。

接口测试：`PYTHONPATH=tmp/adapter_test_deps:src python -m pytest tests/test_frozen_memory_adapters.py -q`。测试用可控 LLM/embedding 替身验证流程与失败条件，不代表正式建库或完整 OEA 实验已完成。

2026-09-18 验证：两套上游 prompt 预检通过，接口测试 12 项通过；只读解析现有 `longcat_oea_train2000_seed42_base_nightly_20260724/standard/results` 的 2000 条记录全部成功，问题均属于 train、与 eval 问题交集为零。工具观察基础设施错误筛查命中 496 条；这是建库过滤诊断，不是任务准确率。尚未执行付费全量蒸馏或真实 embedding/OEA 端到端评测。

测试依赖单独安装于 gitignored `tmp/adapter_test_deps/`（pytest 8.4.2、packaging 25.0），不改变训练环境的已安装包。统一 rollout 对这两个适配器的检索/重排失败会抛错停止，避免静默跑成 Base。
