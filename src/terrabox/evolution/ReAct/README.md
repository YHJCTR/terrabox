# ReAct 基线

该目录用于跑“无记忆、无训练”的原生 Qwen3-8B ReAct 真实工具调用轨迹。runner 会先生成不含 SFT gold assistant/tool 消息的任务文件，再调用仓库已有的 `scripts/run_trajectory_experiment.py`。

## ✅ 2026-06-07 更新：数据↔工具已对齐 + 两套数据

> 经系统排查并修复，数据集里 gold 调用的**工具名与参数已与真实工具 schema 完全对齐**（两份数据 0 缺必填 / 0 多余键 / 0 空参数）。修复内容：①参数键名翻译（input_nir_path→band_a_path、file_list→input_paths、output_filename→output_path…）；②`mean/max_value_and_index/min_value_and_index` 修正为真正的**数字列表归约工具**（参数 `values`）；③无对应工具的纯逻辑(argmax/日期/条件均值)在 `compute.solver` 中合成可运行 `command`，数据已 **0 ipython**；④EarthBench 的**批量调用**（一串文件列表）由通用 `auto_batch`（`toolkits/_batch_util.py`）兼容，单图工具自动逐张处理并返回 `{output_paths,results}`，相关 19 个工具描述已注明。
>
> 系统提示已加入通用规则：**能用工具算的必须调用工具，不得自己臆想结果**（`agent/session.py` + `evolution/shared/prompt_builder.py`，全实验通用）。

**两套数据(行序一致，seed42 同序):**
| 数据 | 路径 | 工具数 | 何时用 |
|------|------|------|------|
| **坍缩** | `data/fixdata/sft_train_strict.jsonl` | 36 | 通用工具基线 |
| **非坍缩** | `data/fixdata_decollapse/sft_train_strict.jsonl` | 44 | 细粒度具名工具（EarthBench 风格）|

重新生成：`PYTHONPATH=src python scripts/build_fixdata.py --mode collapse|decollapse`。
指标(`ReAct/metrics.py`，Reflection 也复用)已含 **precision/recall、set/multiset/relaxed-F1、ordered/exact-match、answer-accuracy(需 `score-answers` 离线打分)、分 task_type breakdown**。

## ✅ 最新默认运行指令（2026-06-07，单一事实来源，照抄即可）

> GPU 布局：agent LLM→GPU0(port 9100)、感知工具→GPU1、VLM(instructsam 计数后端)→**GPU2 单卡**(短上下文,防四卡跳闸)。
> instructsam 默认走 **service 内核**（SAM2+GeoRSCLIP+VLM 计数）；跑完自动停 docker。
> 数据已 **0 ipython**，故**不再需要** `--exclude-tools`；数据与工具已对齐(见上节)。
> ⚠️ ReAct 与 Reflection **不能同时跑**（共用 VLM 容器 + GPU2）；要对比就用**同一批测试数据**。

```bash
EXP=<实验名>                       # 如 shuffle_seed42_test216_decollapse_aligned
# 选数据:非坍缩(44工具) 或 坍缩(36工具)——二选一
DATA=data/fixdata_decollapse/sft_train_strict.jsonl
# DATA=data/fixdata/sft_train_strict.jsonl
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python

# 1) 生成 seed42 shuffle 任务文件（全量；--limit 在 rollout 阶段截取）
PYTHONPATH=src $PY -c "from terrabox.evolution.reflection.data_split import write_task_slice; \
write_task_slice('$DATA','src/terrabox/evolution/ReAct/exp/$EXP/tasks_all_shuffled_seed42.json',\
seed=42,start=0,limit=None,split_name='all_shuffled_seed42')"

# 2) rollout（前 216 条；service 内核 + VLM 单卡 + 跑完自动停 docker）
tmux new-session -d -s react_svc "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.ReAct.runner rollout \
--task-file src/terrabox/evolution/ReAct/exp/$EXP/tasks_all_shuffled_seed42.json \
--experiment $EXP --output-dir src/terrabox/evolution/ReAct/exp/$EXP \
--port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 \
--start-index 0 --limit 216 --resume \
2>&1 | tee src/terrabox/evolution/ReAct/exp/$EXP/run.log
"
```

VLM 单卡轻量档：`--vlm-gpus 2`(单卡→自动短上下文 4096,只用 3 张卡)；要双卡长上下文用 `--vlm-gpus 2,3`，上下文用 `--vlm-max-model-len` 调。
切换 instructsam 内核：加 `TERRABOX_INSTRUCTSAM_BACKEND=vlm`（默认 `service`），其余不变。
保活/不停 docker：加 `--keep-services`（默认跑完自动停掉所有 terrabox 容器）。

**工具数说明**：
- 非坍缩 active = **44 工具**，坍缩 active = **36 工具**；catalog 从实时 registry 重建,**== rollout 绑定的工具**(同 slug、同描述)。
- 数据已 **0 ipython / 0 mock / 0 osm**；gold 参数已与工具 schema 对齐(0 缺必填/0 多余键)。
- 验证日志:非坍缩应是 `Built 44 LangChain tools`、`44 total / 44 allowed`(坍缩为 36)。

## ✅ v2 数据运行指令（2026-06-09，EarthBench 恢复+扩充,分基准评测）

v2 = `data/fixdata_decollapse_v2/`(OE 8234 + EarthBench 202 base×5=1010,统一 **64 工具**)。已**预切好 prompt-only 测试文件**,无需再生成 shuffle/slice,**直接 `--task-file` 指它**,OpenEarth / EarthBench **分两次跑、分开报指标**。

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
V2=data/fixdata_decollapse_v2

# 测试① OpenEarth(共 1647,全跑 ~42h → 用 --limit 抽样,如 300)
tmux new-session -d -s react_v2_oe "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.ReAct.runner rollout \
--task-file $V2/eval_openearth.json \
--experiment v2_react_oe --output-dir src/terrabox/evolution/ReAct/exp/v2_react_oe \
--port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 \
--start-index 0 --limit 300 --resume \
2>&1 | tee src/terrabox/evolution/ReAct/exp/v2_react_oe/run.log
"

# 测试② EarthBench(共 202,可全跑 ~5h)
tmux new-session -d -s react_v2_eb "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
TERRABOX_INSTRUCTSAM_BACKEND=service \
$PY -m terrabox.evolution.ReAct.runner rollout \
--task-file $V2/eval_earthbench.json \
--experiment v2_react_eb --output-dir src/terrabox/evolution/ReAct/exp/v2_react_eb \
--port 9100 --agent-gpu 0 --tool-gpu 1 --vlm-gpus 2 \
--start-index 0 --limit 202 --resume \
2>&1 | tee src/terrabox/evolution/ReAct/exp/v2_react_eb/run.log
"
```

- **v1 ↔ v2 切换**:v1 用上面"最新默认运行指令"(自己 shuffle+slice);v2 直接用预切的 `eval_openearth.json` / `eval_earthbench.json`。其余(GPU 布局、service 内核、单卡 VLM)完全一致。
- 验证日志:应为 `Built 64 LangChain tools`、`64 total / 64 allowed`(v2 统一工具集)。
- OE 与 EB **必须分开统计**(`stats` 各自指向对应 exp 目录),不要合成一个数。

## 数据集：用 fixdata（不要再用 newdata）

> 2026-06-05 起，推荐输入 **`data/fixdata/sft_train_strict.jsonl`**，不要再用 `data/newdata/sft_train_strict.jsonl`。

原因：newdata 把 OpenEarth 的 `Calculator / Solver / Plot` 三个独立计算工具**合并成了 `ipython.execute`**，导致 `ipython` 占工具调用 67%、工具多样性失真。fixdata 用每条 gold 自带的 `raw_tool/raw_arguments` 把它们**精确还原**为真实工具 `compute.calculator / compute.solver / compute.plot`（实现见 `src/terrabox/toolkits/compute.py`），schema 与 newdata 完全一致，是 drop-in 替换。生成脚本：`scripts/build_fixdata.py`。

fixdata 与 newdata 行序一致 → 同 `seed=42` shuffle 得到**完全相同的任务顺序**（前 N 条是同一批任务），所以历史实验可在 fixdata 上 1:1 复现，只是 `expected_tools` 从 `ipython.execute` 变为 `compute.*`。

> 2026-06-07 更新：`scripts/build_fixdata.py` 已重写——现在把**所有** `ipython.execute` 占位都还原成真实工具（OpenEarth Calculator/Solver/Plot→`compute.*`；EarthBench `calculate_ndwi/ndti/...`→`geo_raster.calculate_index`；批量统计→`geo_statistics.batch_raster_stats`；纯 Python 辅助→`compute.solver`），**fixdata 已无 ipython**。同时 system-prompt 的工具 catalog 改为**从实时 registry 重建、只含数据集 active 工具集**，所以 **SFT catalog == rollout 绑定的工具**（同 slug、同描述），且不含 mock/osm/bing/vlm。坍缩版 active = **33 工具**。

## 坍缩 vs 非坍缩数据集（两种粒度，可切换）

Terrabox 的工具是**通用参数化**设计（如 `geo_statistics.scalar_arithmetic(operation=...)`、`geo_raster.calculate_index`），而 OpenEarth/EarthBench 源用的是**具体命名**工具（`multiply`、`calculate_ndwi`…）。据此提供两份数据，用 `--strict-data`/`--task-file` 选择，**模型看到的工具随数据自动一致**：

| 数据 | 目录 | active 工具数 | 说明 |
|------|------|------|------|
| **坍缩**（默认/推荐基线） | `data/fixdata/` | **33** | 多个具体工具合并到通用工具；干净 |
| **非坍缩** | `data/fixdata_decollapse/` | **44** | 还原 EarthBench 风格的具名工具，tool-selection 粒度更细 |

- 非坍缩新增的 15 个具名工具在 `src/terrabox/toolkits/granular_tools.py`，**全部复用现有真实 handler**（`calculate_index`/`scalar_arithmetic`/`batch_raster_stats`/`mean_of_means`/`stats_lst_by_ndvi`），**无 mock、无新未测代码**：
  - `geo_raster.calculate_ndwi/ndti/ndsi`
  - `geo_statistics.subtract/divide/multiply`
  - `geo_statistics.batch_image_mean/max/sum/mean_max_min`、`max_value_and_index`、`min_value_and_index`、`mean`
  - `earth_sci.mean_lst_by_ndvi/max_lst_by_ndvi`
- 重新生成：`PYTHONPATH=src python scripts/build_fixdata.py --mode collapse`（→`data/fixdata`）/ `--mode decollapse`（→`data/fixdata_decollapse`）。
- **跑非坍缩的指令**：把上面默认指令里的 `DATA=data/fixdata/sft_train_strict.jsonl` 换成 `DATA=data/fixdata_decollapse/sft_train_strict.jsonl`，`EXP` 另起名（如 `shuffle_seed42_test216_decollapse`），其余不变。日志应是 `Built 44 LangChain tools`、`44 total / 44 allowed`。
- ⚠️ **OOM 注意**：非坍缩绑定 44 个工具（catalog ≈ 6.5k tokens，比坍缩多 ~1k），agent LLM 上下文 24576 仍够用；若长轨迹接近上限再调小 `--limit` 或精简描述。
- ⚠️ **可比性**：坍缩 vs 非坍缩的工具集不同，**只能各自纵向比**（如各自的 ReAct vs Reflection），不要直接横比两者的 tool-F1。

## ⚠️ 必须关闭 source-schema 别名

跑 fixdata 实验时**务必设** `TERRABOX_ENABLE_SOURCE_SCHEMA_TOOL_ALIASES=false`。

否则：rollout 默认开启该别名，且 fixdata 中 earthbench 的占位符仍让 `ipython.execute` 留在 allowed 工具里，会触发 `src/terrabox/agent/tools.py` 把 `Calculator/Solver/Plot → ipython.execute` 的旧别名重新加进 LangChain 工具列表。结果模型**同时看到** `compute__calculator`（真实）和 `Calculator`（→ipython），可能继续走 ipython，**架空 fixdata 的修复**，且 tool-F1 会误判。

验证方法：日志里应是 `Built 34 LangChain tools`（含别名时是 43）；allowed 列表含 `compute.calculator/solver/plot` 且**不含** `Calculator/Solver/Plot`。

## 建议：排除 ipython.execute（`--exclude-tools ipython.execute`）

`ipython.execute` **不在 OpenEarth/EarthBench 源数据里**，是 Terrabox 转换的产物。fixdata 还原计算工具后，gold 里只剩 44 条 earthbench 占位符任务（占 0.5%，203 测试集中仅 1 条）还带 ipython，且它们本就是未实现工具的占位符。但未训练模型会**优先抓 ipython 且用错格式**（它要求 `<python>`/```python 包裹，模型发裸代码会被拒），使基线虚低、且让别名有机可乘。

因此推荐 rollout 时加 `--exclude-tools ipython.execute`：模型只能用更鲁棒的 `compute.*` 做计算；ipython 离开 allowed 后，别名也不会被触发（双保险）。验证：`Built 33 LangChain tools`、`6 excluded / 33 allowed`、allowed 不含 ipython。
`--exclude-tools` 是逗号分隔的通用排除参数（已加到 `run_trajectory_experiment.py` 与本 runner）。

## instructsam 内核已换成 VLM（Qwen3-VL）— 需 `--vlm-gpus`

> 2026-06-06：原 InstructSAM 检测服务在 rollout 中 **100% 返回 0 个目标**（服务损坏）。已把 `geo_perception.instructsam` 的**内核**换成 Qwen3-VL（`src/terrabox/toolkits/compute.py` 同级，实现见 `geo_perception.py` 的 `instructsam_handler`，原实现已注释）。对模型/指标**仍是 `geo_perception.instructsam`**，VLM 对模型不可见（错误信息也清洗成 InstructSAM 措辞）。

要点（`build_rollout_env` 已自动处理，传 `--vlm-gpus` 即可）：
- VLM 是 bf16 ~16.6GB，**单卡 24G 装不下 16384 上下文的 KV cache**，必须 **2 张卡**：传 `--vlm-gpus 2,3`（tensor-parallel=卡数）。
- VLM 容器**整轮常驻保活**（`TERRABOX_KEEP_VLM_WARM=1`，自动设），避免每次 instructsam 调用冷启动~2min；其它感知工具仍在 tool-gpu 上按调用启停。
- 首次冷启动给了 600s 超时（`TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_INSTRUCTSAM`，自动设）。
- 坐标已按原图尺寸**裁剪**（VLM 会输出越界坐标），尺寸无关。
- 单卡配置见 `agent_config.yaml`：`vlm_gpu_devices/vlm_tensor_parallel`（当前为 2 卡）。

GPU 布局：agent LLM→GPU0(port 9100)、感知工具→GPU1、VLM→GPU2,3。

参考基线（已跑完）：`exp/shuffle_seed42_first432_fixdata_vlm_warm/`（432 题；instructsam count>0 占 40%，相对原来 0% 是质变；但 8B VLM 定位精度有限，exact_match 仍低）。

## 小规模冒烟

```bash
PYTHONPATH=src no_proxy=localhost,127.0.0.1 \
TERRABOX_ENABLE_SOURCE_SCHEMA_TOOL_ALIASES=false \
/home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.ReAct.runner smoke \
  --strict-data data/fixdata/sft_train_strict.jsonl \
  --limit 5 --port 9100 --agent-gpu 0 --tool-gpu 1
```

## 用固定 shuffle 任务文件复现历史实验（203 条测试）

先生成与历史一致的 shuffle 任务文件（同 seed42、全量），再用 `--start-index/--limit` 取片：

```bash
# 1) 生成 fixdata 的 shuffle 任务文件（expected_tools 自动变为 compute.*）
PYTHONPATH=src python -c "from terrabox.evolution.reflection.data_split import write_task_slice; \
write_task_slice('data/fixdata/sft_train_strict.jsonl', \
'src/terrabox/evolution/ReAct/exp/<EXP>/tasks_all_shuffled_seed42.json', \
seed=42, start=0, limit=None, split_name='all_shuffled_seed42')"

# 2) 跑 rollout（前 203 条）
tmux new-session -d -s react_fixdata "
PYTHONPATH=src no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
TERRABOX_ENABLE_SOURCE_SCHEMA_TOOL_ALIASES=false \
/home/yuhongjie/miniconda3/envs/unsloth/bin/python -m terrabox.evolution.ReAct.runner rollout \
--task-file src/terrabox/evolution/ReAct/exp/<EXP>/tasks_all_shuffled_seed42.json \
--experiment <EXP> --output-dir src/terrabox/evolution/ReAct/exp/<EXP> \
--port 9100 --agent-gpu 0 --tool-gpu 1 --start-index 0 --limit 203 --resume \
2>&1 | tee src/terrabox/evolution/ReAct/exp/<EXP>/run.log
"
```

可选：设 `TERRABOX_TOOL_SERVICE_SCOPE=run` 让感知工具容器整轮常驻（不每次启停），显著提速；默认 `call` 每次调用启停。

输出默认放在 `src/terrabox/evolution/ReAct/exp/{experiment}/`，跑完 runner 自动写 `metrics/metrics.json`。

## 已知参考基线

- 旧（newdata, execalias）：`src/terrabox/evolution/ReAct/exp/shuffle_seed42_train400_test203_fulltools_promptfix_execalias/`
- 新（fixdata, 别名关闭）：`src/terrabox/evolution/ReAct/exp/shuffle_seed42_train400_test203_fixdata_compute/`
