#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# 自进化 + 离线评测管道（支持多数据集）
#
# 用法：bash scripts/run_evolution.sh --experiment <name> [选项]
#
# 选项：
#   --experiment <name>   实验名（必填），结果输出到 evo_res/<name>/
#   --dataset <ds>        数据集：disaster（默认）| openearth
#   --methods <m1 m2 ..>  运行指定方法（默认保留的旧方法）
#   --phase <1|2|all>     仅运行指定阶段（默认 all）
#   --no-eval             仅自进化，跳过离线评测步骤
#
# 支持的方法：
#   agentevolver  memrl                                                    ← 阶段 1（无 LLM，并行）
#   skillrl                                                                 ← 阶段 2（需 LLM）
#
# 执行流程：
#   阶段 1（无 LLM，并行）：AgentEvolver / MemRL  (~5min)
#   阶段 2（需 LLM）：
#     GPU 0 (port 9100): SkillRL 蒸馏  (~2h)
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

# ─── 全局清理（Ctrl-C / set -e 异常退出时兜底关闭容器）─────────────────────
_cleanup_docker() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "[cleanup] 脚本异常退出 (exit=$exit_code)，关闭所有 vLLM 容器..."
    else
        echo "[cleanup] 实验完成，关闭所有 vLLM 容器..."
    fi
    for port in 9100 9101 9102 9103; do
        docker stop "terrabox-agent-llm-${port}" >/dev/null 2>&1 && \
            echo "[cleanup] 已停止 terrabox-agent-llm-${port}" || true
    done
}
trap _cleanup_docker EXIT

# ─── 参数解析 ────────────────────────────────────────────────────────────────

EXPERIMENT=""
DATASET="disaster"
PHASE="all"
SKIP_EVAL=false
LIMIT=""
ALL_METHODS=(agentevolver memrl skillrl)
METHODS=()

print_usage() {
    cat << 'EOF'
用法：bash scripts/run_evolution.sh --experiment <name> [选项]

选项：
  --experiment <name>      实验名（必填），输出到 evo_res/<name>/
  --dataset <ds>           数据集: disaster（默认）| openearth
  --methods <m1 m2 ..>     运行指定方法（默认保留的旧方法）
                           可选: agentevolver memrl skillrl
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

for m in "${METHODS[@]}"; do
    case " $m " in
        " agentevolver "|" memrl "|" skillrl ") ;;
        *) echo "错误：方法 $m 已不在当前保留列表中；旧原创方法已归档。"; exit 1 ;;
    esac
done

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
methods = ['agentevolver', 'memrl', 'skillrl']
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

    need_skillrl=false
    has_method skillrl && need_skillrl=true

    if [[ "$need_skillrl" == "false" ]]; then
        ok "阶段 2 无需运行的方法，跳过"
    else
        ts "启动 Docker LLM 服务..."
        $need_skillrl && start_llm 0 9100

        if $need_skillrl; then
            mkdir -p "${OUT}/skillrl"
            run_skillrl > "${OUT}/skillrl/run.log" 2>&1 &
            p_sk=$!; echo "  [PID $p_sk] SkillRL (GPU 0)"
        fi

        $need_skillrl && { wait $p_sk && ok "SkillRL ✓" || err "SkillRL ✗ (see ${OUT}/skillrl/run.log)"; }

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
