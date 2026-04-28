#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# 自进化 + 离线评测管道（支持多数据集）
#
# 用法：bash scripts/run_evolution.sh --experiment <name> [选项]
#
# 选项：
#   --experiment <name>   实验名（必填），结果输出到 evo_res/<name>/
#   --dataset <ds>        数据集：disaster（默认）| openearth
#   --methods <m1 m2 ..>  运行指定方法（默认全部 9 个）
#   --phase <1|2|all>     仅运行指定阶段（默认 all）
#   --no-eval             仅自进化，跳过离线评测步骤
#
# 支持的方法：
#   agentevolver  memrl  causalevo  seqgraphevo  causalpolicyevo(build) ← 阶段 1（无 LLM，并行）
#   causaltextevo  causalpolicyevo(optimize)                             ← 阶段 1 build + 阶段 2 optimize（需 LLM）
#   skillrl  rewardevo  graphskillevo                                     ← 阶段 2（需 LLM，GPU 并行）
#
# 执行流程：
#   阶段 1（无 LLM，并行）：AgentEvolver / MemRL / CausalEvo / SeqGraphEvo / CausalPolicyEvo(build)  (~5min)
#   阶段 2（需 LLM，GPU 并行，4 GPU）：
#     GPU 0 (port 9100): SkillRL 蒸馏  (~2h)
#     GPU 1 (port 9101): RewardEvo LLM-judge  (~1.5h)
#     GPU 2 (port 9102): GraphSkillEvo  (~30min)
#     GPU 3 (port 9103): CausalTextEvo + CausalPolicyEvo optimize queue  (~45min)
#   汇总：离线评测结果对比（--no-eval 时跳过）
#
# 示例：
#   # 灾害数据全量运行
#   bash scripts/run_evolution.sh --experiment disaster4
#
#   # OpenEarthAgent 数据，仅自进化不评测
#   bash scripts/run_evolution.sh --experiment openearth --dataset openearth --no-eval
#
#   # 仅运行阶段1方法
#   bash scripts/run_evolution.sh --experiment openearth --dataset openearth --phase 1 --no-eval
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ─── 参数解析 ────────────────────────────────────────────────────────────────

EXPERIMENT=""
DATASET="disaster"
PHASE="all"
SKIP_EVAL=false
LIMIT=""
ALL_METHODS=(agentevolver memrl causalevo seqgraphevo causalpolicyevo causaltextevo skillrl rewardevo graphskillevo)
METHODS=()

print_usage() {
    cat << 'EOF'
用法：bash scripts/run_evolution.sh --experiment <name> [选项]

选项：
  --experiment <name>      实验名（必填），输出到 evo_res/<name>/
  --dataset <ds>           数据集: disaster（默认）| openearth
  --methods <m1 m2 ..>     运行指定方法（默认全部 9 个）
                           可选: agentevolver memrl causalevo seqgraphevo causalpolicyevo causaltextevo
                                 skillrl rewardevo graphskillevo
  --phase <1|2|all>        仅运行指定阶段（默认 all）
                           1 = 无 LLM 方法，2 = 需 LLM 方法，all = 全部
  --no-eval                仅自进化，跳过离线评测步骤
  --limit <N>              限制自进化使用的轨迹数（默认全量）

示例：
  bash scripts/run_evolution.sh --experiment disaster4
  bash scripts/run_evolution.sh --experiment openearth --dataset openearth --no-eval
  bash scripts/run_evolution.sh --experiment openearth --dataset openearth --methods skillrl --phase 2
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-e) shift; EXPERIMENT="$1"; shift ;;
        --dataset|-d)    shift; DATASET="$1"; shift ;;
        --phase)         shift; PHASE="$1"; shift ;;
        --no-eval)       SKIP_EVAL=true; shift ;;
        --limit)         shift; LIMIT="$1"; shift ;;
        --methods)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 == --* ]]; do
                METHODS+=("$1"); shift
            done ;;
        --help|-h) print_usage; exit 0 ;;
        *) echo "未知参数: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "错误：必须指定 --experiment <name>"
    print_usage
    exit 1
fi

if [[ "$DATASET" != "disaster" && "$DATASET" != "openearth" ]]; then
    echo "错误：--dataset 只支持 disaster 或 openearth"
    exit 1
fi

[[ ${#METHODS[@]} -eq 0 ]] && METHODS=("${ALL_METHODS[@]}")

# --limit: 构造传给各 runner 的参数片段
LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"

OUT="evo_res/${EXPERIMENT}"

# ─── 数据路径（按 dataset 切换）──────────────────────────────────────────────

if [[ "$DATASET" == "openearth" ]]; then
    TRAIN_DATA="data/openearth/trajectories.json"
    EVAL_DATA="data/openearth/eval.jsonl"
    AE_DATA="data/openearth/agentevolver.jsonl"
    SFT_DATA=""          # openearth 不需要 SFT augmented 数据
    IMAGE_MAPPING=""     # openearth 不需要图像映射
else
    TRAIN_DATA="data/disaster_trajectories.json"
    EVAL_DATA="data/disaster_eval.jsonl"
    AE_DATA="data/disaster_agentevolver.jsonl"
    SFT_DATA="data/disaster_sft_augmented.json"
    IMAGE_MAPPING="data/sft_image_mapping.json"
fi

# conda
PY="conda run --no-capture-output -n unsloth python"

# 辅助函数
G='\033[0;32m'; B='\033[0;34m'; Y='\033[1;33m'; R='\033[0;31m'; M='\033[0;35m'; N='\033[0m'
ts()      { echo -e "${B}[$(date '+%H:%M:%S')]${N} ${G}$1${N}"; }
section() { echo -e "\n${M}━━━ $1 ━━━${N}\n"; }
ok()      { echo -e "${G}✓ $1${N}"; }
err()     { echo -e "${R}✗ $1${N}"; }

has_method() { [[ " ${METHODS[*]} " == *" $1 "* ]]; }

# ─── LLM 服务配置 ──────────────────────────────────────────────────────────
MODEL_PATH="/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
DOCKER_IMAGE="terrabox/agent-llm:latest"
MAX_MODEL_LEN="24576"

check_llm() {
    python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('127.0.0.1', $1))
    s.sendall(b'GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n')
    d = s.recv(128)
    s.close()
    sys.exit(0 if b'200 OK' in d else 1)
except SystemExit: raise
except: sys.exit(1)
" 2>/dev/null
}

start_llm() {
    local gpu=$1 port=$2
    local name="terrabox-agent-llm-${port}"

    if check_llm "$port"; then
        ok "LLM 已运行: GPU ${gpu}, port ${port}"
        return 0
    fi

    docker rm -f "$name" 2>/dev/null || true

    ts "启动 LLM: GPU ${gpu} → port ${port} (model: ${MODEL_PATH})..."
    docker run -d \
        --name "$name" \
        --gpus all \
        -e "CUDA_VISIBLE_DEVICES=${gpu}" \
        -e "TRANSFORMERS_OFFLINE=1" \
        -e "HF_HUB_OFFLINE=1" \
        -p "${port}:8000" \
        -v "${MODEL_PATH}:/model:ro" \
        --shm-size=8g \
        "$DOCKER_IMAGE" \
        --model /model \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --enforce-eager

    local w=0
    while ! check_llm "$port"; do
        if ! docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null | grep -q true; then
            err "LLM 容器已退出 (GPU ${gpu})"
            docker logs --tail 30 "$name" 2>&1
            return 1
        fi
        sleep 10; w=$((w+10))
        [ $w -ge 600 ] && { err "LLM 启动超时 (GPU ${gpu}, ${w}s)"; return 1; }
        echo -ne "\r  等待就绪... ${w}s/600s"
    done
    echo ""; ok "LLM 就绪: GPU ${gpu}, port ${port} (${w}s)"
}

# ──────────────────────────────────────────────────────────────────────────────
# 阶段 1: 无 LLM 方法
# ──────────────────────────────────────────────────────────────────────────────

run_agentevolver() {
    local d="${OUT}/agentevolver"
    mkdir -p "${d}/test"

    ts "AgentEvolver: mine..."
    $PY -m terrabox.evolution.agentevolver.runner mine \
        --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" --store-dir "$d" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "AgentEvolver: eval..."
        $PY -m terrabox.evolution.agentevolver.runner eval \
            --eval-data "$EVAL_DATA" --store-dir "$d" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "AgentEvolver done"
}

run_memrl() {
    local d="${OUT}/memrl"
    mkdir -p "${d}/test"

    ts "MemRL: populate..."
    $PY -m terrabox.evolution.memrl.runner populate \
        --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
        --memory-db "${d}/episodic_memory.db" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "MemRL: eval..."
        $PY -m terrabox.evolution.memrl.runner eval \
            --memory-db "${d}/episodic_memory.db" --eval-data "$EVAL_DATA" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "MemRL done"
}

run_causalevo() {
    local d="${OUT}/causalevo"
    mkdir -p "${d}/test"

    ts "CausalEvo: build..."
    $PY -m terrabox.evolution.causalevo.runner build \
        --train-data "$TRAIN_DATA" --store-dir "$d" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "CausalEvo: eval..."
        $PY -m terrabox.evolution.causalevo.runner eval \
            --eval-data "$EVAL_DATA" --store-dir "$d" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "CausalEvo done"
}

run_seqgraphevo() {
    local d="${OUT}/seqgraphevo"
    local store="${d}/store"
    mkdir -p "$store" "${d}/test"

    # SeqGraphEvo 从 AgentEvolver JSONL 数据构建序列图
    # 优先用 agentevolver 输出（若已生成），否则直接用原始 AE_DATA
    local traj_src="${OUT}/agentevolver/experience_pool.jsonl"
    if [ ! -f "$traj_src" ]; then
        traj_src="$AE_DATA"
    fi

    ts "SeqGraphEvo: build seq graph from ${traj_src}..."
    $PY -m terrabox.evolution.seqgraphevo.runner build \
        --traj-file "$traj_src" \
        --store-dir "$store" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "SeqGraphEvo: eval..."
        $PY -m terrabox.evolution.seqgraphevo.runner eval \
            --store-dir "$store" \
            --eval-data "$EVAL_DATA" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "SeqGraphEvo done"
}

run_causaltextevo_build() {
    local d="${OUT}/causaltextevo"
    local store="${d}/store"
    mkdir -p "$store" "${d}/test"

    ts "CausalTextEvo: build (CCA + SeqGraph + keywords)..."
    $PY -m terrabox.evolution.causaltextevo.runner build \
        --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
        --store-dir "$store" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "CausalTextEvo: offline eval (pre-optimize)..."
        $PY -m terrabox.evolution.causaltextevo.runner eval \
            --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
            --store-dir "$store" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "CausalTextEvo build done"
}

run_causalpolicyevo_build() {
    local d="${OUT}/causalpolicyevo"
    local store="${d}/store"
    mkdir -p "$store" "${d}/test"

    ts "CausalPolicyEvo: build policy state..."
    $PY -m terrabox.evolution.causalpolicyevo.runner build \
        --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
        --store-dir "$store" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "CausalPolicyEvo: offline eval (pre-optimize)..."
        $PY -m terrabox.evolution.causalpolicyevo.runner eval \
            --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
            --store-dir "$store" \
            --output "${d}/test/eval_offline_preopt.json"
    fi
    ok "CausalPolicyEvo build done"
}

run_causaltextevo_optimize() {
    local d="${OUT}/causaltextevo"
    local store="${d}/store"

    ts "CausalTextEvo: optimize (TextGrad @ port 9103)..."
    EVOLUTION_LLM_URL="http://localhost:9103" \
    $PY -m terrabox.evolution.causaltextevo.runner optimize \
        --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
        --store-dir "$store" \
        --max-epochs 10 --patience 3 $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "CausalTextEvo: offline eval (post-optimize)..."
        $PY -m terrabox.evolution.causaltextevo.runner eval \
            --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
            --store-dir "$store" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "CausalTextEvo optimize done"
}

run_causalpolicyevo_optimize() {
    local d="${OUT}/causalpolicyevo"
    local store="${d}/store"

    ts "CausalPolicyEvo: optimize (@ port 9103)..."
    EVOLUTION_LLM_URL="http://localhost:9103" \
    $PY -m terrabox.evolution.causalpolicyevo.runner optimize \
        --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
        --store-dir "$store" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "CausalPolicyEvo: offline eval (post-optimize)..."
        $PY -m terrabox.evolution.causalpolicyevo.runner eval \
            --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
            --store-dir "$store" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "CausalPolicyEvo optimize done"
}

# ──────────────────────────────────────────────────────────────────────────────
# 阶段 2: 需要 LLM 的方法
# ──────────────────────────────────────────────────────────────────────────────

run_skillrl() {
    local d="${OUT}/skillrl"
    local store="${d}/store"
    mkdir -p "$store" "${d}/test"

    # 灾害数据：先用 SFT 数据做初始种子，再蒸馏轨迹
    # OpenEarth 数据：直接蒸馏转换好的轨迹（无 SFT augmented 数据）
    if [[ "$DATASET" == "disaster" && -n "$SFT_DATA" && -n "$IMAGE_MAPPING" ]]; then
        ts "SkillRL: seed from SFT data..."
        $PY scripts/convert_sft_to_evolution.py \
            --sft "$SFT_DATA" --mapping "$IMAGE_MAPPING" --skillrl-store "$store"
    fi

    ts "SkillRL: distill trajectories (LLM @ port 9100)..."
    EVOLUTION_LLM_URL="http://localhost:9100" \
    $PY -m terrabox.evolution.skillrl.runner distill \
        --train-data "$TRAIN_DATA" --store-dir "$store" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "SkillRL: eval..."
        $PY -m terrabox.evolution.skillrl.runner eval \
            --train-data "$TRAIN_DATA" --eval-data "$EVAL_DATA" \
            --store-dir "$store" --output "${d}/test/eval_offline.json"
    fi
    ok "SkillRL done"
}

run_rewardevo() {
    local d="${OUT}/rewardevo"
    local store="${d}/store"
    mkdir -p "$store" "${d}/test"

    ts "RewardEvo: LLM-judge labeling (port 9101)..."
    EVOLUTION_LLM_URL="http://localhost:9101" \
    $PY << REWARDEVO_TRAIN
import sys, json, os
sys.path.insert(0, 'src')
os.environ['EVOLUTION_LLM_URL'] = 'http://localhost:9101'

from terrabox.evolution.rewardevo import SelfConsistentLabeler
from terrabox.evolution.shared.trajectory import Trajectory, Turn

with open('${TRAIN_DATA}') as f:
    traj_dicts = json.load(f)

trajectories = []
for td in traj_dicts:
    turns = [Turn(**t) for t in td.get('turns', [])]
    traj = Trajectory(
        task_id=td.get('task_id'), question=td.get('question'),
        images=td.get('images', []), turns=turns,
        tools_called=td.get('tools_called', []),
        expected_tools=td.get('expected_tools', []),
        final_answer=td.get('final_answer', ''),
        success=td.get('success', True),
        source=td.get('source', '${DATASET}'),
        task_type=td.get('task_type', '')
    )
    trajectories.append(traj)

limit = ${LIMIT:-0}
if limit > 0:
    trajectories = trajectories[:limit]
    print(f'Limited to {len(trajectories)} trajectories (--limit {limit})')
else:
    print(f'Loaded {len(trajectories)} trajectories')

try:
    labeler = SelfConsistentLabeler()
    labeled = labeler.label_trajectories(trajectories)
    result = labeler.write_to_memrl(labeled, '${store}/episodic_memory.db')
    print(f'LLM judge: pos={result["positive_written"]}, neg={result["negative_written"]}, unc={result["uncertain_written"]}')
except Exception as e:
    print(f'LLM judge failed ({type(e).__name__}: {str(e)[:200]})')
    print('Falling back to heuristic labeling...')
    labeler = SelfConsistentLabeler()
    labeled = {
        'labeled_positive': [(t, 0.8) for t in trajectories if len(t.tools_called) >= 2],
        'labeled_negative': [(t, 0.2) for t in trajectories if len(t.tools_called) < 2],
        'uncertain': []
    }
    result = labeler.write_to_memrl(labeled, '${store}/episodic_memory.db')
    print(f'Heuristic: pos={result["positive_written"]}, neg={result["negative_written"]}')
REWARDEVO_TRAIN

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "RewardEvo: eval (via MemRL runner)..."
        $PY -m terrabox.evolution.memrl.runner eval \
            --memory-db "${store}/episodic_memory.db" --eval-data "$EVAL_DATA" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "RewardEvo done"
}

run_graphskillevo() {
    local d="${OUT}/graphskillevo"
    local store="${d}/store_v2"
    local skillrl_store="${OUT}/skillrl/store"
    mkdir -p "$store" "${d}/test"

    # GraphSkillEvo v2: 从轨迹数据统计工具共现图（不依赖 SkillRL LLM 输出）
    ts "GraphSkillEvo: build tool co-occurrence graph..."
    $PY -m terrabox.evolution.graphskillevo.runner build \
        --traj-file "$AE_DATA" \
        --store-dir "$store" $LIMIT_ARG

    if [[ "$SKIP_EVAL" == "false" ]]; then
        ts "GraphSkillEvo: eval..."
        $PY -m terrabox.evolution.graphskillevo.runner eval \
            --store-dir "$store" --eval-data "$EVAL_DATA" \
            --output "${d}/test/eval_offline.json"
    fi
    ok "GraphSkillEvo done"
}

# ──────────────────────────────────────────────────────────────────────────────
# 评测汇总
# ──────────────────────────────────────────────────────────────────────────────

run_comparison() {
    local comp="${OUT}/comparison"
    mkdir -p "$comp"

    ts "汇总离线评测..."
    $PY << COMPARE_EOF
import json, os

out = '${OUT}'
exp = '${EXPERIMENT}'
methods = ['agentevolver', 'memrl', 'causalevo', 'seqgraphevo', 'causalpolicyevo', 'causaltextevo', 'skillrl', 'rewardevo', 'graphskillevo']
comp = {}

for m in methods:
    path = os.path.join(out, m, 'test', 'eval_offline.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if 'metrics' in data:
            f1 = data['metrics'].get('f1', 0)
            n = data['metrics'].get('n', 0)
        elif 'summary' in data:
            f1 = data['summary'].get('f1', 0)
            n = data['summary'].get('n_cases', 0)
        else:
            f1 = data.get('avg_f1', 0)
            n = len(data.get('results', []))
        comp[m] = {'f1': round(f1, 4), 'n': n}
        print(f'  {m:20s}: F1={f1:.4f} ({n} samples)')
    else:
        comp[m] = {'f1': None, 'status': 'missing'}
        print(f'  {m:20s}: [not found]')

comp_path = os.path.join(out, 'comparison', 'comparison_offline.json')
with open(comp_path, 'w') as f:
    json.dump({'experiment': exp, 'dataset': '${DATASET}', 'methods': comp}, f, indent=2)
print(f'\nSaved to {comp_path}')
COMPARE_EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# 前置检查：openearth 需要先转换数据
# ══════════════════════════════════════════════════════════════════════════════

prepare_openearth_data() {
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$AE_DATA" ]; then
        ts "OpenEarthAgent 数据尚未转换，自动运行转换器..."
        python3 scripts/convert_openearth_to_evolution.py \
            --train data/openearth/train.json \
            --traj  "$TRAIN_DATA" \
            --ae    "$AE_DATA"
        ok "数据转换完成: ${TRAIN_DATA}, ${AE_DATA}"
    else
        ok "OpenEarthAgent 数据已存在: ${TRAIN_DATA}"
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

section "实验 ${EXPERIMENT} [dataset=${DATASET}] — 自进化$([ "$SKIP_EVAL" == "true" ] && echo '（跳过评测）' || echo ' + 离线评测')"
echo "输出目录: ${OUT}/"
echo "数据集:   ${DATASET}"
echo "方法:     ${METHODS[*]}"
echo "阶段:     ${PHASE}"
echo "跳过评测: ${SKIP_EVAL}"
echo "数据限制: ${LIMIT:-全量}"
echo "Conda:    unsloth"
echo ""

# ── 数据检查 ─────────────────────────────────────────────────────────────────

if [[ "$DATASET" == "openearth" ]]; then
    prepare_openearth_data
    [ -f "$EVAL_DATA" ] || { err "缺失评测集: $EVAL_DATA"; exit 1; }
else
    for f in "$EVAL_DATA" "$TRAIN_DATA" "$AE_DATA"; do
        [ -f "$f" ] || { err "缺失数据文件: $f"; exit 1; }
    done
    [[ -n "$SFT_DATA" ]] && { [ -f "$SFT_DATA" ] || { err "缺失: $SFT_DATA"; exit 1; }; }
    [[ -n "$IMAGE_MAPPING" ]] && { [ -f "$IMAGE_MAPPING" ] || { err "缺失: $IMAGE_MAPPING"; exit 1; }; }
fi
ok "数据文件就绪"

mkdir -p "$OUT"
echo "experiment=${EXPERIMENT} dataset=${DATASET} started=$(date -Iseconds) methods=${METHODS[*]} phase=${PHASE} skip_eval=${SKIP_EVAL}" > "${OUT}/experiment.meta"

# ━━━ 阶段 1 ━━━
if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
    section "阶段 1: 无 LLM 方法并行"
    t1=$(date +%s)
    pids_1=()

    if has_method agentevolver; then
        mkdir -p "${OUT}/agentevolver"
        run_agentevolver > "${OUT}/agentevolver/run.log" 2>&1 &
        pids_1+=("$!:agentevolver")
        echo "  [PID $!] AgentEvolver"
    fi
    if has_method memrl; then
        mkdir -p "${OUT}/memrl"
        run_memrl > "${OUT}/memrl/run.log" 2>&1 &
        pids_1+=("$!:memrl")
        echo "  [PID $!] MemRL"
    fi
    if has_method causalevo; then
        mkdir -p "${OUT}/causalevo"
        run_causalevo > "${OUT}/causalevo/run.log" 2>&1 &
        pids_1+=("$!:causalevo")
        echo "  [PID $!] CausalEvo"
    fi
    if has_method seqgraphevo; then
        mkdir -p "${OUT}/seqgraphevo"
        run_seqgraphevo > "${OUT}/seqgraphevo/run.log" 2>&1 &
        pids_1+=("$!:seqgraphevo")
        echo "  [PID $!] SeqGraphEvo"
    fi
    if has_method causaltextevo; then
        mkdir -p "${OUT}/causaltextevo"
        run_causaltextevo_build > "${OUT}/causaltextevo/run.log" 2>&1 &
        pids_1+=("$!:causaltextevo")
        echo "  [PID $!] CausalTextEvo (build)"
    fi
    if has_method causalpolicyevo; then
        mkdir -p "${OUT}/causalpolicyevo"
        run_causalpolicyevo_build > "${OUT}/causalpolicyevo/run.log" 2>&1 &
        pids_1+=("$!:causalpolicyevo")
        echo "  [PID $!] CausalPolicyEvo (build)"
    fi

    echo ""
    ts "等待阶段 1..."
    for entry in "${pids_1[@]}"; do
        pid="${entry%%:*}"; name="${entry##*:}"
        wait "$pid" && ok "${name} ✓" || err "${name} ✗ (see ${OUT}/${name}/run.log)"
    done

    d1=$(( $(date +%s) - t1 ))
    ok "阶段 1: ${d1}s ($((d1/60))m)"
fi

# ━━━ 阶段 2 ━━━
if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
    section "阶段 2: LLM 方法（GPU 并行）"
    t2=$(date +%s)

    need_skillrl=false; need_rewardevo=false; need_graphskillevo=false; need_causaltextevo=false; need_causalpolicyevo=false
    has_method skillrl && need_skillrl=true
    has_method rewardevo && need_rewardevo=true
    has_method graphskillevo && need_graphskillevo=true
    has_method causaltextevo && need_causaltextevo=true
    has_method causalpolicyevo && need_causalpolicyevo=true

    if [[ "$need_skillrl" == "false" && "$need_rewardevo" == "false" && "$need_graphskillevo" == "false" && "$need_causaltextevo" == "false" && "$need_causalpolicyevo" == "false" ]]; then
        ok "阶段 2 无需运行的方法，跳过"
    else
        ts "启动 Docker LLM 服务..."
        $need_skillrl && start_llm 0 9100
        $need_rewardevo && start_llm 1 9101
        if $need_causaltextevo || $need_causalpolicyevo; then
            start_llm 3 9103
        fi

        # 并行: SkillRL + RewardEvo
        if $need_skillrl; then
            mkdir -p "${OUT}/skillrl"
            run_skillrl > "${OUT}/skillrl/run.log" 2>&1 &
            p_sk=$!; echo "  [PID $p_sk] SkillRL (GPU 0)"
        fi
        if $need_rewardevo; then
            mkdir -p "${OUT}/rewardevo"
            { run_rewardevo > "${OUT}/rewardevo/run.log" 2>&1; docker stop terrabox-agent-llm-9101 >/dev/null 2>&1 && ts "已释放 GPU 1 (port 9101)"; } &
            p_re=$!; echo "  [PID $p_re] RewardEvo (GPU 1)"
        fi

        # GraphSkillEvo 不再等待 SkillRL，直接从轨迹数据构建工具图
        if $need_graphskillevo; then
            start_llm 2 9102
            mkdir -p "${OUT}/graphskillevo"
            { run_graphskillevo > "${OUT}/graphskillevo/run.log" 2>&1; docker stop terrabox-agent-llm-9102 >/dev/null 2>&1 && ts "已释放 GPU 2 (port 9102)"; } &
            p_gs=$!; echo "  [PID $p_gs] GraphSkillEvo (GPU 2)"
        fi

        # GPU 3 queue: CausalTextEvo optimize + CausalPolicyEvo optimize
        if $need_causaltextevo || $need_causalpolicyevo; then
            mkdir -p "${OUT}/causaltextevo" "${OUT}/causalpolicyevo"
            {
                if $need_causaltextevo; then
                    run_causaltextevo_optimize >> "${OUT}/causaltextevo/run.log" 2>&1
                fi
                if $need_causalpolicyevo; then
                    run_causalpolicyevo_optimize >> "${OUT}/causalpolicyevo/run.log" 2>&1
                fi
                docker stop terrabox-agent-llm-9103 >/dev/null 2>&1 && ts "已释放 GPU 3 (port 9103)"
            } &
            p_cp=$!; echo "  [PID $p_cp] GPU 3 optimize queue"
        fi

        # 等待所有任务
        $need_skillrl && { wait $p_sk && ok "SkillRL ✓" || err "SkillRL ✗ (see ${OUT}/skillrl/run.log)"; }
        $need_rewardevo && { wait $p_re && ok "RewardEvo ✓" || err "RewardEvo ✗ (see ${OUT}/rewardevo/run.log)"; }
        $need_graphskillevo && { wait $p_gs && ok "GraphSkillEvo ✓" || err "GraphSkillEvo ✗ (see ${OUT}/graphskillevo/run.log)"; }
        if $need_causaltextevo || $need_causalpolicyevo; then
            if wait $p_cp; then
                $need_causaltextevo && ok "CausalTextEvo ✓"
                $need_causalpolicyevo && ok "CausalPolicyEvo ✓"
            else
                $need_causaltextevo && err "CausalTextEvo ✗ (see ${OUT}/causaltextevo/run.log)"
                $need_causalpolicyevo && err "CausalPolicyEvo ✗ (see ${OUT}/causalpolicyevo/run.log)"
            fi
        fi

        # 释放 GPU 0 (SkillRL 完成后停)
        if $need_skillrl; then
            docker stop terrabox-agent-llm-9100 >/dev/null 2>&1 && ts "已释放 GPU 0 (port 9100)"
        fi

        d2=$(( $(date +%s) - t2 ))
        ok "阶段 2: ${d2}s ($((d2/60))m)"
    fi
fi

# ━━━ 汇总 ━━━
if [[ "$SKIP_EVAL" == "false" ]]; then
    section "离线评测汇总"
    run_comparison
fi

# ━━━ 完成 ━━━
date > "${OUT}/finished_at"
section "实验 ${EXPERIMENT} 完成！"
echo "数据集: ${DATASET}"
echo "结果:   ${OUT}/"
ls -1 "${OUT}"/*/run.log 2>/dev/null | while read f; do
    dir=$(dirname "$f"); method=$(basename "$dir")
    done_mark="${dir}/test/eval_offline.json"
    if [[ "$SKIP_EVAL" == "true" ]]; then
        # Show store files instead
        store_files=$(ls -1 "${dir}/store"* 2>/dev/null | head -3 | tr '\n' ' ' || ls -1 "${dir}"/*.json* 2>/dev/null | head -3 | tr '\n' ' ' || echo "")
        echo "  ${method}: ${store_files:-see run.log}"
    elif [ -f "$done_mark" ]; then
        echo "  ${done_mark}"
    else
        echo "  ${method}: [eval not found — see run.log]"
    fi
done
if [[ "$SKIP_EVAL" == "false" ]]; then
    echo "  ${OUT}/comparison/comparison_offline.json"
fi
echo ""
echo "━━━ 监控命令（实验进行中使用） ━━━"
echo ""
echo "  # 查看各方法日志"
for m in "${METHODS[@]}"; do
    echo "  tail -f ${OUT}/${m}/run.log"
done
echo ""
echo "  # GPU 使用率"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "  # 检查完成状态"
echo "  for m in ${METHODS[*]}; do"
echo "    [ -f ${OUT}/\$m/run.log ] && tail -1 ${OUT}/\$m/run.log && echo \"  [\$m]\""
echo "  done"
