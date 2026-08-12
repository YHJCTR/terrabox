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

## SkillRL 严格 rollout-only 适配
- `get_prompt_augmenter("skillrl_rollout")` / `skillrl_strict` / `skillrl_nonrl` 是独立于旧 `skillrl` 的无训练适配模式,不覆盖原实现或历史结果。
- build 只允许消费 train agent 的实际可见 rollout 字段(任务文本、任务类型、实际工具调用、完成状态、可观测工具错误),不得读取或传递 `expected_tools`、`gold_tool_calls`、`ground_truth`、metrics/F1 或 task ID。
- 使用 Qwen embedding 服务做语义 skill retrieval；embedding 服务不可用时应明确中止,不得静默退化为 lexical retrieval。该模式只复现 SkillRL 的 frozen-policy offline skill-bank arm,不含论文中的 teacher SFT 与 GRPO,结果须标为 `adapted_non_rl`。详细命令见 `skillrl/README.md`。

## experience_evo 外部经验库
- 位置:`experience_evo/`,与 `promptevo/` 同级;不要把产物转移经验库混入 PromptEvo 的静态 prompt 优化目录。
- MVP 是离线经验自进化:历史 rollout `results/` → artifact transition 抽取 → LongCat/DeepSeek/local 蒸馏 → `evolution_store/experience_evo/...` JSONL+SQLite store → `get_prompt_augmenter("experience_evo")` 检索注入。
- 默认可用已有 LongCat OEA base `tmp/trajectories/promptevo_oea_base_longcat2_20260704_031626_rollout/standard/results` 建库;这适合 transductive smoke/case-library 测试。严格对比应后续用同一 train subset 分别构建 reflection / memrl_full_source / experience_evo，再到 OEA test 评估。
- 当前 ExperienceEvo 主口径是 rollout-derived offline self-evolution:只从 train/base rollout 的实际 `conversation_history` 工具调用、参数、observation 抽取经验;gold `expected_tools`/`gold_tool_calls` 只用于 train 覆盖选择、指标评分和后续 gold replay audit,不直接进入主 store 蒸馏。若构建 gold replay store,只能作为 teacher upper-bound/diagnostic,不要和主结果混用。
- Gold 验证分两层:`gold-audit` 只做静态 schema 审计,默认读取 live Terrabox registry 以匹配当前真实可执行工具接口,显式传 `--catalog` 才按旧 catalog 快照复现;`gold-replay` 才是真实 teacher-forced 工具执行。`gold-replay` 不调用 LongCat actor,会在执行前把 OEA symbolic artifact alias(`gpkg_N`/`tif_N`/`img_N`,以及少量命名 GeoPackage alias 如 `marienplatz_gpkg_1`/`gpkg_jeronimos_1`)绑定到样本输入或前序工具产物,也会把 OEA 原始 observation 中的固定产物名(`out.tif`/`out.png`/`dummy_generated_image.jpg`)绑定到最近一次真实生成的图像或栅格产物但不改写输出参数本身,每条任务独立 artifact 目录,实际重跑某条任务前会清空该任务自己的 artifact 子目录,输出标准 `results/<task_id>.json`;`final_answer_full` 是 replay observation evidence,需要再跑 `scripts/judge_answers.py` 才能判断是否支持 `ground_truth` 正确结论。`gold-replay --resume` 会跳过干净 completed 和 OOM 终态,但会自动重跑 failed 以及旧版误写成 completed、observation 中仍含工具错误的脏结果;`--max-transient-retries` 默认 5,只重试 infra/provider/timeout 类瞬时 replay 失败;train2000 这类 subset 应用 `--data` 指向完整 train/test JSONL,`--subset-file` 只按 task_id 过滤,不要把不含 `gold_tool_calls` 的 subset 文件直接当 `--data`;精确复测旧失败用可重复的 `--task-id <task_id>`。直接跑 `gold-replay` 时模块会默认设置单卡 VLM(`VLM_TENSOR_PARALLEL_SIZE=1`,`VLM_MAX_MODEL_LEN=8192`,`VLM_MIN_IMAGE_MODEL_LEN=8192`,`VLM_GPU_MEMORY_UTILIZATION=0.95`,`VLM_MAX_NUM_SEQS=1`,`TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS=4096`),避免 `agent_config.yaml` 中双卡 VLM 配置污染单 lane gold replay；8192 是服务上下文,默认输出预算仍先设为 4096 以减少 context retry；InstructSAM Docker 默认传 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解 24GB 卡上的碎片 OOM,可用 `INSTRUCTSAM_PYTORCH_CUDA_ALLOC_CONF` 覆盖；OEA `CountGivenObject` 的 `bbox`/`region` 兼容当前支持全图宽高互换归一化与轻微越界 clamp,用于避免把历史 gold bbox 约定差异误判为数据错误。
- 注入内容不得包含当前 eval 的 `expected_tools`、gold answer、task id 或精确历史文件路径;历史 F1/reward 只用于经验筛选和 Q/N/Risk 统计;蒸馏 examples 必须先把历史问题、地点名、文件路径、layer 名和自由文本参数替换为 `<named_area>` / `<artifact_reference>` 等占位符。
- 当前 ExperienceEvo MVP 检索不是全量塞上下文:先用停用词过滤+OEA domain alias 的 lexical retriever 找签名级产物转移,再在同一 `input_signature -> output_signature` 下找工具级经验并计算 `Quse = lambda*Qtool + (1-lambda)*Qsig`;v1/v2 仍是任务开始前静态 two-stage prompt block。
- v2/v3 作为并行模式保留,不覆盖旧方案:`get_prompt_augmenter("experience_evo")` 仍读 v1 `experiences.jsonl`;`get_prompt_augmenter("experience_evo_v2")` 读 v2 `events_v2.jsonl`/`families_v2.jsonl`/`experience_evo_v2.sqlite`;`get_prompt_augmenter("experience_evo_v3")` 复用 v2 store。
- v4 作为独立并行模式保留,不覆盖 v1/v2/v3:`get_prompt_augmenter("experience_evo_v4")` 复用 v2 store,但关闭 v3 的 answer-ready hard guard、premature-final guard 和 synthetic final,只保留当前运行图片路径与 calculator schema 两类硬约束；其余产物状态判断以软提示和 verifier checkpoint 形式提供。可选 `TERRABOX_EXPEVO_V4_LLM_CHECKER=1` 使用 LongCat 做独立 checker,默认关闭以避免全量请求量翻倍。v4 消融只用环境变量切换,默认行为不变:`TERRABOX_EXPEVO_V4_DISABLE_STEP_HINT=1` 关闭逐步检索,`TERRABOX_EXPEVO_V4_DISABLE_QUSE=1`/`TERRABOX_EXPEVO_V4_DISABLE_TOOL_RANKING=1` 关闭 Quse 排序展示,`TERRABOX_EXPEVO_V4_DISABLE_VERIFIER=1`/`TERRABOX_EXPEVO_V4_DISABLE_VERIFICATION=1` 关闭 verifier checkpoint。
- v3 runtime 先推断当前初始产物状态(`task_request`/`input:image`/`input:raster`/`input:gpkg` 等),只注入 preconditions 已满足的第一步产物转移,并用 query intent、输入形态、available_tools 过滤明显错配经验；standard eval 的 sequential loop 每次工具 observation 后会用 `agent.artifacts` 更新 product state,再调用 `step_hint()` 检索下一步 family。v3 允许极窄 runtime fallback 修补离线 store 覆盖缺口:当前在 index change 任务已有 1 个 `add_index_layer` 产物时继续推荐第二个 `add_index_layer`,已有 2 个 index layer 后才推荐 `compute_index_change`;multi-target 图像测量任务会在已有 1 个 `instructsam`/`calculator` 结果但仍缺目标计数时继续推荐对应下游工具;属性/健康/状态判断次数不足时继续推荐 `region_attribute_description`;像素阈值/GSD 任务会按 `sam2_segment -> compute.solver -> compute.calculator -> compute.solver -> compute.calculator` 的当前状态轮转补齐阈值、面积和百分比;per-object count、size-selection draw、localized attribute+count 任务都有窄口径 runtime fallback;fallback 不读取 gold/expected_tools/answer/task id,也不写回经验库。
- `evolution_trace` 会记录 product_state、recommended_tools、answer_ready、selected_tool、selected_in_recommendations、blocked_tool_calls、guard_preview 和 tool_result_status,用于审计经验是否真的被用到；若当前状态已有可回答结果(`compute.calculator`/`compute_route_dist`/`compute_index_change`/OCR/属性描述等),v3 会触发 answer-ready guard,拦截额外工具调用并要求 final answer；同一 answer-ready 状态下模型连续两次请求额外工具时,sequential loop 会用当前成功 observation 收敛成 final answer,避免 max-turn 空转。multi-target 任务会保留重复产物计数,只完成一个对象的属性描述或一次计算不会被当作整题完成。
- v3 对 GSD/像素面积/距离测量任务额外启用窄口径 tool guard:首步若模型想调用 `vlm_analyze` 或 `strip_rcnn_detect`,会在真实执行前拦截;单目标/局部对象测量要求先用 `geo_perception.instructsam` 取得可测量像素/掩码证据;显式 `segment all`/`sum pixel areas`/`combined ground area` 这类 bulk all-object segmentation 任务例外,允许并优先提示 `geo_perception.sam2_segment`;显式少于/阈值/占用像素类任务必须先 `sam2_segment`,之后只允许 `compute.solver`/`compute.calculator` 继续数值链路。`vlm_analyze` 不视为可回答结果。guard 还会拦截非当前 task image/当前运行产物的图片路径,并用 `successful_call_records` 区分同工具同 target 的重复调用和 multi-target 下不同 `text`/target 的必要重复调用；若最近一次 `instructsam` 成功但返回 0 个对象,允许换不同 target 再定位,同 target 仍会被拦截；对 damage/symmetry/health 这类已定位、下一步应做属性描述的任务,重复 `instructsam` 会被明确导向 `geo_perception.region_attribute_description`。非 visual 请求会过滤 `add_text`/`draw_bboxes`/display/plot 等可视化产物转移；明确失败 observation(如 `Error in calculator:`)不得进入成功 product state,截断但前缀含 `status: success` 的成功 JSON 会作为成功 observation 保留 product state。OSM `marketplace(s)` / `{"shop":"marketplace"}` 会归一为 `{"amenity":"marketplace"}`。
- v2/v3 的经验单元都是 `input_product_state -> target_product_state` family,含 product-level Qsig/Nsig/Rsig 与 tool-level Qtool/Ntool/Rtool,eval prompt 中按 `Quse = lambda*Qtool + (1-lambda)*Qsig` 给工具排序。`distill-v2` 会按 `--progress-every` 增量写回并在重启时跳过已完成 family;LongCat 402/额度/空响应结果需先从 rollout `results/` 备份移出,再用 `--resume` 补跑。
- `infra_error` 尺度:timeout/429/网络/provider overload/OOM/context/Docker/service-health 只过滤,不抬高 Risk;只有 missing required、invalid argument、文件/图层/产物引用不存在等 LLM 工具使用错误才计入经验 Risk。
- 运行说明见 `experience_evo/README.md`;模块 CLI 走 `PYTHONPATH=src python -m terrabox.evolution.experience_evo.runner ...`,不新增 `scripts/`。

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
- `--workers N`:单个 rollout 进程内任务级**进程**并发,脚本默认 1；但 LongCat/DeepSeek 等外部 API 的 OEA 全量 eval / baseline 复刻 / overnight watcher 默认不要单 worker,必须显式设置多 worker（通常 `--workers 3` 起步，纯 online-nogpu/慢 OSM 补跑可用 2-4）。每个 worker 独立初始化 agent/LLM/经验检索器,避免线程共享 `TERRABOX_TASK_DATA_*`。GPU 感知批次仍以 lane 钉卡 + 服务锁分流为主。带 `--evolution-method` 时,统一 rollout 会把 `question`、`task_type`、`images`、`data_files`、`available_tools` 传给 augmenter,但不传当前 eval 的 `expected_tools`、gold answer 或 task id。新写 watcher 必须显式写 `--workers` 或环境变量（如 `TERRABOX_OEA_REPRO_WORKERS`），不要依赖脚本默认 1。
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
  外部 API 三流必须用 `TERRABOX_TOOL_SERVICE_SCOPE=call TERRABOX_KEEP_VLM_WARM=1`,只保留 VLM 常驻,其它感知服务调用后释放,避免同一卡多个重服务常驻导致 CUDA OOM。
  P0/P1 两条感知 lane 除了钉 `*_GPU_DEVICES` 外，也要给非 VLM 服务分配不同端口(如 lane1 用 `SAM2_PORT=9012 REMOTECLIP_PORT=9013 REMOTESAM_PORT=9014 STRIP_RCNN_PORT=9015 INSTRUCTSAM_PORT=9016 CHANGEOS_PORT=9017`)，避免两个进程同时重建同名 `terrabox-*` 容器。
  不要先用 GPU0/1/2 三分片跑完 nogpu 再跑感知批,那会空置 GPU3 并延后慢感知任务。
- 多 worker 是外部 API 全量实验的默认策略,不是只用于 smoke 后补跑；含 GPU 感知任务通常 `--workers 3` 起步并依赖 per-service 钉卡与服务锁控制显存,纯 online-nogpu/慢 OSM 补跑可用 2-4。单 worker 仅用于 smoke/debug、明确限流或用户指定,不要让 LongCat/DeepSeek 全量 watcher 长时间空转。

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
- `build-live` 复用已有 train2000 LongCat base rollout 的真实 `conversation_history`，不读取 gold/expected tools 作为蒸馏输入；成功源按 `F1>=0.8`、`task_type + tool_sequence` 覆盖选择，最多 240 条；失败原则仅允许模型可归因工具错误，timeout/OOM/network/quota/context/service 错误过滤掉。
- live store 保存 LongCat 批量 principles 和脱敏成功 episode 摘要。评测时通过 `get_prompt_augmenter("expel_live")` 检索原则加最多一条成功 episode，再进入真实 OEA 工具执行和 LongCat answer judge。
- 当前没有可用的 Qwen embedding HTTP 服务，主实验明确使用可审计 lexical retrieval；Qwen embedding 服务准备好后再单独做检索器消融，不能静默声称已使用 embedding。
