# Evolution 实验模块 — Codex/Agent 上下文(模块级,自动加载于本目录树)

本目录是独立实验模块。在此干活时,除项目根 `CLAUDE.md` 外,本文件也会自动生效。**详细跑法 / GPU 配置 / 查看指标在各子模块 README**(只有 `CLAUDE.md`/`AGENTS.md` 自动加载,README 需主动看)。

> 本文件与 `CLAUDE.md` 内容保持一致,仅子文档引用指向各自格式(`*/AGENTS.md`)。改其一务必同步另一份。
> 本目录新增或修改方案、TODO、实验记录、README 等 Markdown 文档时默认用中文；命令、变量名、API 名、论文/方法英文名和引用原文可保留英文。
> 人类阅读入口优先看 `文档索引.md`;旧长文和过期英文草稿集中放在 `文档归档/`。

## 当前主力:oea_full(OEA 全量,仅 OpenEarth)
- 数据:`data/oea_full_sft/`(`scripts/build_oea_full_sft.py` 生成),catalog 仅 **23 个 OE 工具**,参数对齐真实 schema。`openearth_test_tasks.json`(测试 1162)、`openearth/{train,test}.jsonl`(SFT 含 gold)。EarthBench 暂不涉及。
- base ReAct 基线:`tmp/trajectories/oe_full_react_offline/`。
- **完整文档**:`ReAct/AGENTS.md`(基线 rollout)、`reflection/AGENTS.md`(自反思 train→test)。

## 第三方方法复刻口径
- 跑论文/第三方 baseline 时,默认优先使用对方官方源码、官方 prompt、官方数据流程或论文中明确描述的机制;实验文档/manifest 需记录源码路径、commit/版本、关键配置和引用依据。
- 若 OEA 工具环境、数据接口、成本、运行依赖或官方代码缺失导致无法完整复刻,必须向用户说明并在 store/manifest/实验记录中标注 `official` / `adapted` / `reimplemented` 口径及差异,不得把弱化实现静默当作完整复现。
- 主表对比尽量使用机制级或源码级复刻;只能低成本近似的版本应命名为 adapted baseline,并把不可复刻部分列为后续补实验项。

## 经验检索与 task_type 口径
- 自进化方法及第三方经验/记忆 baseline 的 train 建库与 eval runtime **不得**把 benchmark 数据集提供的 `task_type`/`type` 当作经验分桶键、family ID、索引字段、候选过滤条件、排序 bonus、路由键或 prompt 注入条件；同样不得以 task id、`expected_tools`、gold answer 或其他仅数据集标注可见字段替代。
- 经验检索必须遵循原作者公开论文或官方源码的检索方式。若原方法没有使用逐样本 `task_type`，适配器不得自行加入该捷径；若因接口限制改成其他检索逻辑，必须在 manifest 和实验记录中标为 `adapted` 或 `reimplemented` 并说明差异。
- 严格的 rollout-derived self-evolution 中，actor 与经验蒸馏 LLM 只能看到：任务文本、任务提供的图像/数据文件、公开工具说明、自己实际调用的工具、参数、observation、最终自然语言回答和可观测执行错误；不得把 `expected_tools`、`gold_tool_calls`、`ground_truth`、gold 派生 F1/reward 或数据集 `task_type` 送入 LLM prompt、经验文本、经验质量值或检索索引。
- 可使用任务文本、当前可见 artifact/product state、可用工具、实际 rollout 轨迹，以及原方法规定的可见上下文建立 BM25、embedding、关键词或状态转移索引。模型从这些可见信息自行推断的意图特征可以使用，但必须不读取数据集标签，并在实验记录中说明其来源。
- `task_type`、`expected_tools`、gold answer 和 gold 指标仅可用于实验结束后的独立评测、报表统计，或显式标为 oracle coverage/teacher upper-bound 的数据诊断；不可混入主 self-evolution store。若用 `expected_tools` 的序列覆盖挑选 train2000，必须单列为 oracle-selected subset，不能声称该筛选本身是无监督自进化。
- 统一 rollout 即使为兼容接口传入 `task_type`，augmenter 和 builder 必须忽略该数据集标签；新增或修改方法时应以任务可见信息和各原方法规定的检索输入作为唯一依据。
- Memento/CaseBank 在严格 OEA 对比中必须使用 `casebank.builder --strict-nolabel --embedding-backend qwen` 重建独立 store；严格记录不序列化 task id/task type/final answer，过滤 infrastructure/API/Docker/OOM 轨迹，reward 只能使用 rollout 可见执行状态和工具错误，embedding 文档同样不含数据集标签。严格 store 强制 `TERRABOX_CASEBANK_RETRIEVAL=qwen`，endpoint、模型和 2560 维检查失败必须退出，禁止 lexical fallback。该模块是 `Memento-style CaseBank (adapted)`，不是官方 planner/executor runtime 完整复现；历史 `oea_train2000_longcat_semantic_20260810` store 仅作旧 adapted 诊断，不应与严格主表混用。
- MemRL 严格 OEA 对比必须走 `memrl_full.source_runner --strict-nolabel --backend external_memrl`；strict 只能消费 rollout trajectories，不允许 SFT/gold bootstrap，不序列化 task id/task type/expected tools/metrics/F1/gold/final answer/绝对路径，过滤明确基础设施/API/Docker/OOM/context/quota/auth/billing 失败，reward 只由 completed、最终回答存在性、实际工具调用和可观测工具错误生成。`--strict-nolabel` 禁止 `backend=auto` 降级；Qwen3 Embedding 4B 需显式 `MEMRL_EMBED_DIM=2560` 或 `MEMRL_VECTOR_DIMENSION=2560`，维度/外部后端失败必须中止。正式结果命名为 `MemRL (official-source adapted, strict rollout-only)`，offline fixed-memory 与 online runtime-learning 必须分开报。

## SkillRL 严格 rollout-only 适配
- `get_prompt_augmenter("skillrl_rollout")` / `skillrl_strict` / `skillrl_nonrl` 是独立于旧 `skillrl` 的无训练适配模式,不覆盖原实现或历史结果。
- build 只允许消费 train agent 的实际可见 rollout 字段(任务文本、实际工具调用、完成状态、可观测工具错误),不得读取或传递数据集 `task_type`、`expected_tools`、`gold_tool_calls`、`ground_truth`、metrics/F1 或 task ID。
- 使用 Qwen embedding 服务做语义 skill retrieval；embedding 服务不可用时应明确中止,不得静默退化为 lexical retrieval。该模式只复现 SkillRL 的 frozen-policy offline skill-bank arm,不含论文中的 teacher SFT 与 GRPO,结果须标为 `adapted_non_rl`。详细命令见 `skillrl/README.md`。

## experience_evo 外部经验库
- 位置:`experience_evo/`,与 `promptevo/` 同级;不要把产物转移经验库混入 PromptEvo 的静态 prompt 优化目录。
- MVP 是离线经验自进化:历史 rollout `results/` → artifact transition 抽取 → LongCat/DeepSeek/local 蒸馏 → `evolution_store/experience_evo/...` JSONL+SQLite store → `get_prompt_augmenter("experience_evo")` 检索注入。
- 默认可用已有 LongCat OEA base `tmp/trajectories/promptevo_oea_base_longcat2_20260704_031626_rollout/standard/results` 建库;这适合 transductive smoke/case-library 测试。严格对比应后续用同一 train subset 分别构建 reflection / memrl_full_source / experience_evo，再到 OEA test 评估。
- 当前 ExperienceEvo 的严格主口径应是 rollout-derived offline self-evolution:只从 train/base rollout 的实际 `conversation_history` 工具调用、参数、observation、自然语言回答与可观测执行反馈抽取经验。历史 v1/v2/v3/v4 store 曾以数据集 `task_type` 分 family，且以 gold F1 生成 reward/Q 值；该 store 与其对应结果属于非严格历史口径，正式主表前必须重建为 task-type-free、gold-free store。gold replay store 只能作为 teacher upper-bound/diagnostic,不要和主结果混用。
- Gold 验证分两层:`gold-audit` 只做静态 schema 审计,默认读取 live Terrabox registry 以匹配当前真实可执行工具接口,显式传 `--catalog` 才按旧 catalog 快照复现;`gold-replay` 才是真实 teacher-forced 工具执行。`gold-replay` 不调用 LongCat actor,会在执行前把 OEA symbolic artifact alias(`gpkg_N`/`tif_N`/`img_N`,以及少量命名 GeoPackage alias 如 `marienplatz_gpkg_1`/`gpkg_jeronimos_1`)绑定到样本输入或前序工具产物,也会把 OEA 原始 observation 中的固定产物名(`out.tif`/`out.png`/`dummy_generated_image.jpg`)绑定到最近一次真实生成的图像或栅格产物但不改写输出参数本身,每条任务独立 artifact 目录,实际重跑某条任务前会清空该任务自己的 artifact 子目录,输出标准 `results/<task_id>.json`;`final_answer_full` 是 replay observation evidence,需要再跑 `scripts/judge_answers.py` 才能判断是否支持 `ground_truth` 正确结论。`gold-replay --resume` 会跳过干净 completed 和 OOM 终态,但会自动重跑 failed 以及旧版误写成 completed、observation 中仍含工具错误的脏结果;`--max-transient-retries` 默认 5,只重试 infra/provider/timeout 类瞬时 replay 失败;train2000 这类 subset 应用 `--data` 指向完整 train/test JSONL,`--subset-file` 只按 task_id 过滤,不要把不含 `gold_tool_calls` 的 subset 文件直接当 `--data`;精确复测旧失败用可重复的 `--task-id <task_id>`。直接跑 `gold-replay` 时模块会默认设置单卡 VLM(`VLM_TENSOR_PARALLEL_SIZE=1`,`VLM_MAX_MODEL_LEN=8192`,`VLM_MIN_IMAGE_MODEL_LEN=8192`,`VLM_GPU_MEMORY_UTILIZATION=0.95`,`VLM_MAX_NUM_SEQS=1`,`TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS=4096`),避免 `agent_config.yaml` 中双卡 VLM 配置污染单 lane gold replay；8192 是服务上下文,默认输出预算仍先设为 4096 以减少 context retry；InstructSAM Docker 默认传 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解 24GB 卡上的碎片 OOM,可用 `INSTRUCTSAM_PYTORCH_CUDA_ALLOC_CONF` 覆盖；OEA `CountGivenObject` 的 `bbox`/`region` 兼容当前支持全图宽高互换归一化与轻微越界 clamp,用于避免把历史 gold bbox 约定差异误判为数据错误。
- 注入内容不得包含 `expected_tools`、gold answer、task id、数据集 `task_type` 或精确历史文件路径;gold F1/reward 不得用于经验筛选或 Q/N/Risk 统计，严格主口径只能由实际轨迹的可观测结果与独立 checker 基于该轨迹给出反馈；蒸馏 examples 必须先把历史问题、地点名、文件路径、layer 名和自由文本参数替换为 `<named_area>` / `<artifact_reference>` 等占位符。
- 当前 ExperienceEvo MVP 检索不是全量塞上下文:先用停用词过滤+OEA domain alias 的 lexical retriever 找签名级产物转移,再在同一 `input_signature -> output_signature` 下找工具级经验并计算 `Quse = lambda*Qtool + (1-lambda)*Qsig`;v1/v2 仍是任务开始前静态 two-stage prompt block。
- v2/v3 作为并行模式保留,不覆盖旧方案:`get_prompt_augmenter("experience_evo")` 仍读 v1 `experiences.jsonl`;`get_prompt_augmenter("experience_evo_v2")` 读 v2 `events_v2.jsonl`/`families_v2.jsonl`/`experience_evo_v2.sqlite`;`get_prompt_augmenter("experience_evo_v3")` 复用 v2 store。
- v4 作为独立并行模式保留,不覆盖 v1/v2/v3:`get_prompt_augmenter("experience_evo_v4")` 复用 v2 store,但关闭 v3 的 answer-ready hard guard、premature-final guard 和 synthetic final,只保留当前运行图片路径与 calculator schema 两类硬约束；其余产物状态判断以软提示和 verifier checkpoint 形式提供。可选 `TERRABOX_EXPEVO_V4_LLM_CHECKER=1` 使用 LongCat 做独立 checker,默认关闭以避免全量请求量翻倍。v4 消融只用环境变量切换,默认行为不变:`TERRABOX_EXPEVO_V4_DISABLE_STEP_HINT=1` 关闭逐步检索,`TERRABOX_EXPEVO_V4_DISABLE_QUSE=1`/`TERRABOX_EXPEVO_V4_DISABLE_TOOL_RANKING=1` 关闭 Quse 排序展示,`TERRABOX_EXPEVO_V4_DISABLE_VERIFIER=1`/`TERRABOX_EXPEVO_V4_DISABLE_VERIFICATION=1` 关闭 verifier checkpoint。
- `experience_evo_v4_no_store_soft_only` 与 `experience_evo_v4_generic_guard` 是 v4-clean 的显式因果对照,不得覆盖或替代 `experience_evo_v4_clean`：前者只保留无经验库的通用当前运行证据检查,不打开 store、不给出历史 transition/工具建议；后者只保留与数据集标签无关的图片路径和 calculator schema guard,不注入任何经验/提示。二者用于量化通用执行框架与经验库的独立贡献。
- `experience_evo_v5` 是 v4-clean 的独立增强版,不得覆盖或替代 `experience_evo_v4_clean`：它复用同一严格 rollout-only store、step hint、Quse/risk 和 soft verifier,只新增一次 final-answer verifier 子 agent。该 verifier 只能读取当前运行的 artifact state、成功/失败工具调用、observation 与 draft answer,用于检查最终答案中的数值、单位、阈值、最近/最远、计数、实体选择和产物存在性是否被当前证据支持；不得读取 train/eval gold、`task_type`、`expected_tools`、task id 或历史完整轨迹。关闭开关为 `TERRABOX_EXPEVO_V5_DISABLE_FINAL_VERIFIER=1`。
- `experience_evo_v5_hybrid_qwen` 是 v5 的独立检索增强版,不得覆盖或替代 `experience_evo_v5` / `experience_evo_v4_clean`：它保留 v5 final-answer verifier,但在 v4-clean 的状态硬过滤之后加入 Qwen embedding semantic rank,并与 BM25 lexical rank、structured artifact-state rank 做 RRF 融合。embedding 文档只能由 rollout-derived family 的 intent/product state/product-tool experience/输入输出规则/恢复建议和公开工具名构成,不得包含 `task_type`、task id、`expected_tools`、gold answer、gold calls 或 gold metrics。Qwen embedding 服务和 `qwen_family_embedding_index.json` 必须显式可用且模型名一致;index 缺失、覆盖不完整或模型不一致时必须失败退出,不得静默降级为纯 BM25/v5。embedding 只参与候选排序,不能替代 product-state hard filter。
- v3 runtime 先推断当前初始产物状态(`task_request`/`input:image`/`input:raster`/`input:gpkg` 等),只注入 preconditions 已满足的第一步产物转移,并用 query intent、输入形态、available_tools 过滤明显错配经验；standard eval 的 sequential loop 每次工具 observation 后会用 `agent.artifacts` 更新 product state,再调用 `step_hint()` 检索下一步 family。v3 允许极窄 runtime fallback 修补离线 store 覆盖缺口:当前在 index change 任务已有 1 个 `add_index_layer` 产物时继续推荐第二个 `add_index_layer`,已有 2 个 index layer 后才推荐 `compute_index_change`;multi-target 图像测量任务会在已有 1 个 `instructsam`/`calculator` 结果但仍缺目标计数时继续推荐对应下游工具;属性/健康/状态判断次数不足时继续推荐 `region_attribute_description`;像素阈值/GSD 任务会按 `sam2_segment -> compute.solver -> compute.calculator -> compute.solver -> compute.calculator` 的当前状态轮转补齐阈值、面积和百分比;per-object count、size-selection draw、localized attribute+count 任务都有窄口径 runtime fallback;fallback 不读取 gold/expected_tools/answer/task id,也不写回经验库。
- `evolution_trace` 会记录 product_state、recommended_tools、answer_ready、selected_tool、selected_in_recommendations、blocked_tool_calls、guard_preview 和 tool_result_status,用于审计经验是否真的被用到；若当前状态已有可回答结果(`compute.calculator`/`compute_route_dist`/`compute_index_change`/OCR/属性描述等),v3 会触发 answer-ready guard,拦截额外工具调用并要求 final answer；同一 answer-ready 状态下模型连续两次请求额外工具时,sequential loop 会用当前成功 observation 收敛成 final answer,避免 max-turn 空转。multi-target 任务会保留重复产物计数,只完成一个对象的属性描述或一次计算不会被当作整题完成。
- v3 对 GSD/像素面积/距离测量任务额外启用窄口径 tool guard:首步若模型想调用 `vlm_analyze` 或 `strip_rcnn_detect`,会在真实执行前拦截;单目标/局部对象测量要求先用 `geo_perception.instructsam` 取得可测量像素/掩码证据;显式 `segment all`/`sum pixel areas`/`combined ground area` 这类 bulk all-object segmentation 任务例外,允许并优先提示 `geo_perception.sam2_segment`;显式少于/阈值/占用像素类任务必须先 `sam2_segment`,之后只允许 `compute.solver`/`compute.calculator` 继续数值链路。`vlm_analyze` 不视为可回答结果。guard 还会拦截非当前 task image/当前运行产物的图片路径,并用 `successful_call_records` 区分同工具同 target 的重复调用和 multi-target 下不同 `text`/target 的必要重复调用；若最近一次 `instructsam` 成功但返回 0 个对象,允许换不同 target 再定位,同 target 仍会被拦截；对 damage/symmetry/health 这类已定位、下一步应做属性描述的任务,重复 `instructsam` 会被明确导向 `geo_perception.region_attribute_description`。非 visual 请求会过滤 `add_text`/`draw_bboxes`/display/plot 等可视化产物转移；明确失败 observation(如 `Error in calculator:`)不得进入成功 product state,截断但前缀含 `status: success` 的成功 JSON 会作为成功 observation 保留 product state。OSM `marketplace(s)` / `{"shop":"marketplace"}` 会归一为 `{"amenity":"marketplace"}`。
- v2/v3 的经验单元都是 `input_product_state -> target_product_state` family,含 product-level Qsig/Nsig/Rsig 与 tool-level Qtool/Ntool/Rtool,eval prompt 中按 `Quse = lambda*Qtool + (1-lambda)*Qsig` 给工具排序。`distill-v2` 会按 `--progress-every` 增量写回并在重启时跳过已完成 family;LongCat 402/额度/空响应结果需先从 rollout `results/` 备份移出,再用 `--resume` 补跑。
- `infra_error` 尺度:timeout/429/网络/provider overload/OOM/context/Docker/service-health 只过滤,不抬高 Risk;只有 missing required、invalid argument、文件/图层/产物引用不存在等 LLM 工具使用错误才计入经验 Risk。
- GeoPackage append-style 工具（包括 `add_index_layer`、`compute_index_change`、`add_pois_layer` 与 `compute_route_dist`）必须先检查 `gpkg_contents`/SQLite catalog，并在实际 GDAL 写入前使用同一 GeoPackage 的进程锁和锁内 recheck；完整的同类型 raster/vector layer 可复用，不完整/类型冲突/检查失败/写入失败必须返回明确错误，不能删除、覆盖或伪造成功。Raster window 也必须裁剪到源影像范围。这些 artifact 契约修复不改变任何工具 timeout；OSM 无匹配、网络慢、AOI 过大和 LLM 参数错误仍按原有失败处理。
- 运行说明见 `experience_evo/README.md`;模块 CLI 走 `PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner ...`,不新增 `scripts/`。

## experience_evo_rl 真实工具 RL
- 位置:`experience_evo_rl/`;该模块有两条口径:历史离线首动作 GRPO 诊断,以及正式 online veRL 真实工具 RL。离线链路不执行 Terrabox 工具,不得写作完整 OEA agent RL 结果。
- 正式 online 链路使用 `TerraboxOeaTool` + `TerraboxToolAgentLoop`:Qwen policy 生成 veRL function tool call,真实执行 `AgentToolExecutor`,回填 observation,episode 结束由可观察工具状态写入 `rm_scores` 并记录 `metrics/online_episode_traces.jsonl`。
- online 工具 observation 采用确定性记忆压缩后再回填模型；原文写入 `metrics/raw_tool_observations.jsonl`，压缩统计写入 episode trace，默认上下文预算 8192 字符，训练/评测必须保持一致。
- pure online GRPO 默认关闭 actor KL/entropy 分支（不使用 KL reward），避免 24GB 卡在 old-logprob 阶段产生额外全词表显存峰值；当前本地 veRL 的 `RayPPOTrainer._compute_old_log_prob` 已修正为遵循 `actor.calculate_entropy`/`entropy_coeff`，系数为零时不得无条件 materialize entropy。online runner 默认将 actor PPO / rollout old-logprob / ref log-prob 的 per-GPU token 上限设为 8192，若再次 OOM 优先调低这些训练侧上限，不先截断任务 prompt、response 或工具 observation 语义预算。
- online veRL 默认 validation 容易把全部 val 同批送入 Ray rollout manager；正式训练需显式控制训练期 validation 并发（建议 `val_max_samples=8`、`val_batch_size=1`、`dataloader_num_workers=0`、`val_before_train=False`）。这只影响训练期健康检查，不减少完整 train 数据，也不能替代最终 OEA test 全量评测。
- 若真实 rollout 序列长度超过训练侧 token 上限，veRL 会在 old-logprob 阶段断言失败而不是自动截断；需把 actor / old-logprob / ref log-prob token 上限升到覆盖实际序列（当前 3B online 可用 10240），同时用 `train_batch_size=1` 控制显存。
- online runner 默认挂载本地 RemoteSAM checkpoint（`REMOTESAM_CHECKPOINT_HOST=/data1/yuhongjie2/RemoteSAM/pretrained_weights`）并关闭 EPOC，避免离线实验因联网下载 BERT 失败。
- online GRPO 数据仍必须是 strict no-label public view:prompt 只能包含任务文本、公开图片/数据文件和公开工具 schema;不得把 eval/test 的 `expected_tools`、gold answer、task id、`task_type` 或 metrics 写入 prompt、experience 或训练 reward。训练阶段若显式使用 train gold/oracle reward,必须在实验名和文档中标注,不能混入 strict no-label 主线。
- `runner.py prepare-data --online` 生成 1900/100 train/val parquet/jsonl;`write-online-configs` 生成 veRL 工具/agent-loop 配置;`train-grpo --online` 默认只写 `run_grpo_command.sh`,只有显式 `--launch` 才启动训练。启动前必须确认训练卡和工具/感知 lane,不能在其它实验占用 GPU0--2 时把 GPU3 单卡工程检查写成完整 OEA online 结果。

## promptevo 外部 adapter
- 外部项目适配放在 `promptevo/adapters/`,详细入口见 `promptevo/CLAUDE.md` 与
  `promptevo/adapters/README.md`。API-Bank 是当前轻量外部场景:默认优化静态 API-call
  任务说明,用 exact/API-name/argument/error-bucket 指标做稳定接受门;`execute_api_calls=True`
  仅作 API-Bank evaluator-style 执行诊断,会受网络、随机/状态 API 和可选依赖影响。
- PromptEvo 的可选 `--proposal-format patch` 只允许通用类型化协议补丁
  (`tool_selection`/`argument_validation`/`error_recovery`/`termination_and_repetition`)；不得把
  项目工具名、任务 ID、路径、基准名、gold 或固定 workflow 编译入静态提示词。编译器确定性拒绝
  task ID、路径和已知 benchmark 标识。patch 候选必须
  用固定 dev task 的真实 rollout 验证才能接受，禁止静态选择或 gold 轨迹替代验证。

## rollout 共用脚本 `scripts/run_trajectory_experiment.py` 的关键 flag
- `--llm-provider {local,deepseek,longcat}`:可把做任务的 agent LLM 从本地 vLLM 切到外部 API。
  外部 provider 仍可配合 `--use-docker` 使用本地感知/OSM 工具服务,但不会启动本地 agent vLLM、
  不占 agent GPU、不套本地 `local_llm_max_model_len` 上下文上限;仍受 `--max-iterations` 控制轮次。
- `--no-skip-mock/bing/osm/vlm/changeos`:这些 `--skip-*` **默认全 True**;oea_full 含 osm/bing/vlm/changeos,**务必显式带 5 个 `--no-skip-*`**,否则静默丢任务。
- `--skip-online` / `--only-online`:离线 / 在线分两遍跑(在线需本地转发窗口)。
- `--gpu-class {any,gpu,nogpu}`:在线再分——`gpu`=gold 含 GPU 感知服务、`nogpu`=gold 不含 GPU 感知服务。只分流任务,不改变可见工具目录。
- `--workers N`:单个 rollout 进程内任务级**进程**并发,脚本默认 1；但 LongCat/DeepSeek 等外部 API 的 OEA 全量 eval / baseline 复刻 / overnight watcher 默认不要单 worker,必须显式设置多 worker。**默认跑法是 GPU/no-GPU 分 lane**：`--gpu-class gpu --workers 1` 跑含感知工具任务，`--gpu-class nogpu --workers 2-4` 跑 OSM/API/compute 任务；两条 lane 写入同一实验 `results/` 时必须 `--resume` 且结果文件按 task id 独立。每个 worker 独立初始化 agent/LLM/经验检索器,避免线程共享 `TERRABOX_TASK_DATA_*`。带 `--evolution-method` 时,统一 rollout 会把 `question`、`task_type`、`images`、`data_files`、`available_tools` 传给 augmenter,但不传当前 eval 的 `expected_tools`、gold answer 或 task id。新写 watcher 必须显式写 `--workers` 或环境变量（如 `TERRABOX_OEA_REPRO_WORKERS`），不要依赖脚本默认 1。
- **单感知 lane 限制**：若一个 watcher 只配置一条 VLM/感知服务 lane（例如 `VLM_GPU_DEVICES=0` 且所有重感知服务在同一套 GPU/端口），GPU 感知批必须设 `--workers 1`。不能用多个 worker 让它们排队抢同一把服务锁：通用工具 timeout 从 handler 开始计时，会把锁等待误判成工具失败并污染方法指标。需要加速时，优先按 `--gpu-class gpu` 与 `--gpu-class nogpu` 拆成两条并行 rollout lane；只有多套独立 VLM/感知端口和 GPU 时，才把 GPU 感知 lane 扩成双 lane/三流。纯 online-nogpu 批才可单独提高 worker 数。
- **VLM / InstructSAM 预热（正式感知 eval 必需）**：外部 API rollout 在首条任务前必须显式调用 Docker `vllm_manager.start_service()` 并等待 `/health` 成功；不能把首次下载/加载多模态模型的数分钟冷启动塞进默认 120 秒工具 timeout。预热进程不能在就绪后立即退出，因为 `BaseServiceManager` 会在 `atexit` 自动释放它启动的容器；watcher 必须保留 service keepalive 到 rollout 结束，再由该进程统一清理。预热失败时 watcher 必须停止，不能带着冷启动超时继续写正式 `results/`。单卡配置建议 `TERRABOX_VLM_STARTUP_TIMEOUT_SECONDS=900`，并给 `geo_perception.vlm_analyze`/`region_attribute_description` 设置至少 360 秒的实际推理 timeout；预热完成后保持 `TERRABOX_KEEP_VLM_WARM=1`。InstructSAM 默认保持 call-scoped；只有在其 Docker create/start 稳定、且被钉到独占 GPU/独立端口时，才允许额外设置 `TERRABOX_KEEP_INSTRUCTSAM_WARM=1` 并在 rollout 前预热 `/health`。若 InstructSAM 预热出现 Docker create/start 卡死或健康检查失败，必须改回 call-scoped 并保持 GPU lane 单 worker，不要让整条 watcher 卡在预热阶段。Docker 管理器以受限时的 `docker create` + `docker start` 启动，`TERRABOX_DOCKER_START_TIMEOUT_SECONDS=90` 同时约束两个阶段；超时后清理同名残留并让 watcher 失败，禁止空转。
- **Docker 控制面异常处理**：若日志出现 `docker create/start ... timed out`、`context canceled`、`TimeoutExpired`、健康检查长期失败，且 `nvidia-smi` 显示目标 GPU 空闲，不要把它当作模型 OOM 或方法失败继续跑。应先暂停/失败退出当前 watcher，确认 `docker version`、`docker ps`、目标容器 labels/ports、`perception_keepalive.log` 和 `docker logs`；仅清理本实验创建且已确认归属的 `terrabox-*` 残留容器。必要时重启 Docker daemon，但不得擅自重启机器或处理他人容器。正式结果目录必须从干净 `results/` 重新开始或明确隔离旧脏目录。
- **Docker GPU runtime 与 InstructSAM 冷启动**：daemon 默认保持 `runc`；GPU 服务必须显式使用 `--gpus device=...`，不要把全局默认 runtime 改成 `nvidia`。InstructSAM 健康等待由 `TERRABOX_INSTRUCTSAM_STARTUP_TIMEOUT_SECONDS` 控制，默认 420 秒；首次 SAM2/GeoRSCLIP 加载可能接近 3 分钟。健康超时前 manager 会记录 `docker inspect` 和最近容器日志，再清理本实验容器并让 watcher 失败。
- `--max-transient-retries`(默认5)+`--max-transient-retry-seconds`(默认1200):provider/API 连接类抖动导致整条失败→**丢弃该次整条重跑**;预算耗尽后保留当前失败并继续后续任务。确定性错误(上下文超限、选错工具、missing required/invalid argument/不存在文件或图层/API 402、quota/auth/billing 或额度不足错误)不重试;已经进入工具循环后的 `Tool execution timeout`/max-turn 失败按本条轨迹失败处理,避免少数 OSM 大范围/坏参数任务卡住 watcher。配合 `--resume` 时,已有 `results/{task_id}.json` 若是 transient failed 且未耗尽预算,会自动复跑;可用 `--no-retry-existing-transient` 关闭。

## GPU 钉卡(踩过坑,跑感知任务必看)
跑含 GPU 感知工具的批次(离线感知 / `--gpu-class gpu`):
- **绝不设 `TERRABOX_TOOL_GPU_DEVICES`**;用 per-service 变量钉到非 agent 卡:`VLM_GPU_DEVICES`(+`VLM_TENSOR_PARALLEL_SIZE`)、`INSTRUCTSAM_/SAM2_/REMOTESAM_/STRIP_RCNN_/REMOTECLIP_GPU_DEVICES`,`AGENT_LLM_GPU_DEVICES` 单独给 agent(独占)。
- VLM Docker 必须只暴露 leased GPU(`--gpus device=...`),不要再用 `--gpus all`+容器内 `CUDA_VISIBLE_DEVICES` 做隔离；否则容器内 cuda:0 可能撞上 agent LLM。
- 感知 rollout 默认 OCR 走 CPU(`TERRABOX_OCR_USE_GPU=0`),避免 EasyOCR 和 VLM/InstructSAM 抢显存；`sam2_segment`/RemoteSAM 外层 timeout 默认应高于 120s(推荐 420s)以覆盖冷启动。OSM 网络查询工具 `get_area_boundary` / `add_pois_layer` 默认外层 timeout 为 240s；landmark+`buffer_m` 会先走短超时点 geocode(`TERRABOX_OSM_GEOCODE_TIMEOUT` 默认 20s)，并用 `TERRABOX_OSM_MAX_BUFFER_M`(默认 10000m) / `TERRABOX_OSM_MAX_AOI_KM2`(默认 50000km²，设 0 关闭)快速拒绝明显过大 AOI；`add_pois_layer` 会把 `bar`/`schools`/`bus stops` 等常见类别字符串归一为 OSM tag dict，未知字符串仍按 POI 名称兼容处理。
- RemoteSAM 默认离线从 `/checkpoints/bert-base-uncased` 读取 BERT，并默认关闭 EPOC/DirectSAM；迁移服务器时只要 `/data1/yuhongjie2/RemoteSAM/pretrained_weights/bert-base-uncased` 与 Swin checkpoint 存在即可，不依赖 `/home` 下 HF cache。
- 不钉卡时动态分配显存紧张会**兜底砸 GPU0**(撞 agent LLM)→ CUDA OOM;`reflection.runner` 走 `ReAct.runner.build_rollout_env` 已正确钉卡。
- `--gpu-class nogpu` 只按 gold expected tools 分流任务,不允许隐藏工具目录中的 GPU 感知工具；否则 agent 评测口径会变化。若 agent 误调用感知工具,必须 per-service 钉到已有感知 lane 并共用 `TERRABOX_SERVICE_LOCK_DIR`,让它等待同 lane 服务而不是默认砸 GPU0。
- 四张 3090 跑 OEA 全量时按 provider 切换:本地 agent LLM 用 ReAct 文档里的
  **双流/3+1**(GPU0/1/2 跑感知流, GPU3 并行跑 `--only-online --gpu-class nogpu`);
  外部 API agent LLM 用 **三流**(GPU0/1 与 GPU2/3 各跑一条感知流, OSM/nogpu 单独跑)。
  外部 API 三流必须用 `TERRABOX_TOOL_SERVICE_SCOPE=call TERRABOX_KEEP_VLM_WARM=1`,只保留 VLM 常驻,其它感知服务调用后释放,避免同一卡多个重服务常驻导致 CUDA OOM。仅当 InstructSAM 被明确钉到不与其它重感知服务共享的独占 GPU 时，才允许额外设 `TERRABOX_KEEP_INSTRUCTSAM_WARM=1`；否则必须保持 call-scoped。
  P0/P1 两条感知 lane 除了钉 `*_GPU_DEVICES` 外，也要给非 VLM 服务分配不同端口(如 lane1 用 `SAM2_PORT=9012 REMOTECLIP_PORT=9013 REMOTESAM_PORT=9014 STRIP_RCNN_PORT=9015 INSTRUCTSAM_PORT=9016 CHANGEOS_PORT=9017`)，避免两个进程同时重建同名 `terrabox-*` 容器。
  不要先用 GPU0/1/2 三分片跑完 nogpu 再跑感知批,那会空置 GPU3 并延后慢感知任务。
- 多 worker 是外部 API 全量实验的默认策略,不是只用于 smoke 后补跑；但单感知 lane 上不能直接 `--workers 3` 跑全量。OEA 全量默认拆成 `gpu --workers 1` + `nogpu --workers 2-4` 并行跑，既避免 InstructSAM/VLM/SAM2 等服务锁等待污染工具 timeout，又能让 LongCat/DeepSeek 在 OSM/API 慢等待期间保持利用率。单 worker 仅用于 smoke/debug、明确限流或用户指定,不要让 LongCat/DeepSeek 全量 watcher 长时间空转。

## 工具结果缓存(感知 + 安全 OSM 只读工具,跨实验共享)
慢且可安全复用的工具(8 个 GPU 感知模型 + 少量安全 OSM 只读工具,当前 `osm_gis.get_bbox_from_raster`)走**内容哈希结果缓存**(`agent/tool_result_cache.py`,挂在 `AgentToolExecutor.execute` 最外层):
- **key** = `sha256(slug + 所有非输出参数)`,**文件路径参数替换为其内容 sha256** → 命中需"工具+输入图像内容+所有参数"全一致(`output_path` 等输出键排除);换图/换 text 必不命中,文件名相同但内容不同不会撞。
- **命中即返回**:不起 docker、不加载模型、不联网;缓存的产物文件(掩码/标注图等)**还原到原路径**供下游读。
- **只缓存成功结果**(error/OOM/timeout 文本不存),不污染瞬时重试。
- 默认目录为仓库内 `cache/tool_result_cache/`(gitignored),**全实验共享**(base/reflection/SFT 互相复用;测试集固定+图像路径稳定→跨实验命中率高)。口径中性(返回的就是工具本会返回的内容,只更快,不改 tool-F1)。
- 会创建/追加/渲染 GeoPackage 的链式 `osm_gis` 工具默认不缓存,避免旧 `.gpkg` artifact 污染后续 layer。开关:`TERRABOX_TOOL_RESULT_CACHE=0` 关;`..._DIR` 改目录;`..._SLUGS` 显式覆盖白名单。compute/绘图(亚秒级)不缓,bing 用自带 `.db` 缓存。
- 多流共享 `TERRABOX_ARTIFACT_OUTPUT_DIR` 时，`artifact_index.json` 使用进程间文件锁写入；运行中不要删除 `.lock` 文件。

## 查看进度与指标
- **看进度/指标首选本节命令,不要优先用 `scripts/run_trajectory_experiment.py stats`**:后者只是旧的轨迹统计入口,不含当前统一口径的进度/ETA、Table4、分类别 F1 和跨实验配对对比。
- **结果权威来源**:`results/<task_id>.json` 是 rollout/judge/指标重算的唯一权威原始记录。当前 `run_trajectory_experiment.py` 会在结束时从完整 `results/` 重建 `report.json`、`trajectories.jsonl`、`trajectories_full.jsonl`;历史实验遗留的旧派生文件仍可能只覆盖最后一批或已过期。正式指标用 `rollout_report`/`rollout_metrics` 从 `results/` 只读重算;不要把旧 `report.json` 当权威。
- **统一 CLI(各实验首选)**:`evolution/shared/rollout_report.py` —— 自动解析 ReAct/promptevo 的 `tmp/trajectories/<exp>/standard/results` 与 reflection/后续方法的 `src/terrabox/evolution/<method>/exp/<exp>/<phase>/results`,统一打印进度、完整指标与配对对比。
  - 单实验进度+指标:`PYTHONPATH=src python -m terrabox.evolution.shared.rollout_report status <experiment-or-results-dir> --scope all --total 1162`
  - 指定 results 目录:`PYTHONPATH=src python -m terrabox.evolution.shared.rollout_report status --results-dir <某实验>/results --scope all --total 1162`
  - 任意两实验配对对比:`PYTHONPATH=src python -m terrabox.evolution.shared.rollout_report compare <cur-exp-or-results-dir> <base-exp-or-results-dir>`
  - 例:`... compare oe_full_reflection_div1k oe_full_react_offline`;只统计两边都已完成的同一批 task_id,适合实验未全量完成时随时看。
- **公共权威库(指标实现)**:`evolution/shared/rollout_metrics.py` —— 工具维度指标(set/multiset P/R/F1、exact/ordered/lcs、**OEA Table4 的 AnyOr/SameO/Uni**、分类别 micro F1)+ 4 种剪枝口径(raw/oea15/failcap3/both)+ 进度/ETA + 状态/资源。只读重算,跨实验同口径。
  - CLI(任意实验):`PYTHONPATH=src python -m terrabox.evolution.shared.rollout_metrics [online|offline|all|all-scopes] --results-dir <某实验>/results [--total N] [--legend]`(`--total` 给定则附进度/ETA)。
  - 库函数:`load_tasks` / `score(tasks,transform)` / `print_phase(label,results_dir,total)` / `report(results_dir,scope)` / `progress`。多阶段(train+eval)对每个 results 目录各调一次。
- `scripts/monitor_parallel.sh` / `scripts/monitor_parallel_experiments.sh` 是早期 merged-train 两段切分实验的交互监控脚本,默认实验名、总数和日志路径都按 `merged_*` 写死或通过 `EXP*_NAME` 调整。它们不适合作为 OEA/promptevo/reflection 这类 3+1 同一实验目录写入的默认监控入口;这类实验仍优先用上面的 `rollout_report status/compare` 或 `rollout_metrics` 从 `results/` 只读重算。
- 旧临时助手:`tmp/online_progress.py`、`tmp/compare_experiments.py` 仅保留历史兼容,路径写死 `tmp/trajectories/...`;新实验不要再依赖它们。
- 其它**独立口径**(各有用途,勿混):`scripts/score_rollout_metrics.py`(OpenEarthAgent-style 独立统计)、`ReAct/metrics.py`(`runner stats` 用,set/multiset + 错误桶)。
- 指标口径:per-category F1 = **按工具类别**(perception/operation/logic/gis),不是任务类别;Table4=e2e、Table3=teacher-forced 单步。新增/修改指标逻辑应改**公共库**,薄包装自动受益。

> 改本模块代码后:同步更新对应子模块 README 与本文件(项目根 `CLAUDE.md`/`AGENTS.md` 只放项目级事实)。

## ExpeL 真实 OEA 适配
- `expel` 的旧入口仍是离线工具序列预测；新增 `expel_live`/`expel_official` 是独立模式，不覆盖旧 store 或旧结果。
- `build-live` 复用已有 train2000 LongCat base rollout 的真实 `conversation_history`。严格口径不得读取 gold/expected tools、gold F1 或数据集 `task_type` 作为蒸馏输入、候选分桶或覆盖选择；历史按 `F1>=0.8`、`task_type + tool_sequence` 选择的 store 仅作非严格诊断。失败原则仅允许模型可归因工具错误，timeout/OOM/network/quota/context/service 错误过滤掉。
- live store 保存 LongCat 批量 principles 和脱敏成功 episode 摘要。评测时通过 `get_prompt_augmenter("expel_live")` 检索原则加最多一条成功 episode，再进入真实 OEA 工具执行和 LongCat answer judge。
- ExpeL 旧主实验明确使用可审计 lexical retrieval；如后续改用 Qwen embedding，必须单独重建 store/manifest 并标注为新检索器消融，不能把旧结果静默改称 embedding retrieval。
