#!/bin/bash
# 并行运行两个实验，完整数据集，使用 Docker LLM 服务

set -e

REPO_ROOT="/data1/yuhongjie2/terrabox"
cd "$REPO_ROOT"

# ============================================================================
# 实验配置
# ============================================================================

TASK_FILE="${TASK_FILE:-data/merged/merged_train_tasks.json}"
MODE="${MODE:-standard}"
EXP_SUFFIX="${EXP_SUFFIX:-20260525_no_vlm}"
TOTAL_TASKS=$(/home/yuhongjie/miniconda3/envs/unsloth/bin/python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("data/merged/merged_train_tasks.json").read_text())
tasks = data.get("tasks", data) if isinstance(data, dict) else data
print(len(tasks))
PY
)
SPLIT_POINT=$(((TOTAL_TASKS + 1) / 2))

# Experiment 1: GPU0 - LLM, GPU1 - 感知工具
EXP1_NAME="${EXP1_NAME:-merged_tokens_part1_${EXP_SUFFIX}}"
EXP1_START=0
EXP1_END=$SPLIT_POINT
EXP1_PORT=9100
EXP1_GPU_DEVICES="0"
EXP1_TOOL_GPU_DEVICES="1"
EXP1_TMUX_SESSION="exp_part1"

# Experiment 2: GPU2 - LLM, GPU3 - 感知工具
EXP2_NAME="${EXP2_NAME:-merged_tokens_part2_${EXP_SUFFIX}}"
EXP2_START=$SPLIT_POINT
EXP2_END=$TOTAL_TASKS
EXP2_PORT=9102
EXP2_GPU_DEVICES="2"
EXP2_TOOL_GPU_DEVICES="3"
EXP2_TMUX_SESSION="exp_part2"
PYTHON_BIN="/home/yuhongjie/miniconda3/envs/unsloth/bin/python"

worker_env() {
  local agent_gpu="$1"
  local tool_gpu="$2"
  cat <<EOF
  export PYTHONPATH=src && \
  export PYTHONUNBUFFERED=1 && \
  export CUDA_VISIBLE_DEVICES=${agent_gpu},${tool_gpu} && \
  export AGENT_LLM_GPU_DEVICES=${agent_gpu} && \
  export AGENT_LLM_MODEL_PATH=/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/ && \
  export TERRABOX_USE_DOCKER=true && \
  export TERRABOX_TOOL_GPU_DEVICES=${tool_gpu} && \
  export TERRABOX_TOOL_MAX_GPUS=1 && \
  export TERRABOX_TOOL_SERVICE_SCOPE=call && \
  export VLM_GPU_DEVICES=${tool_gpu} && \
  export VLM_TENSOR_PARALLEL_SIZE=1 && \
  export VLM_MIN_IMAGE_MODEL_LEN=8192 && \
  export VLM_MAX_MODEL_LEN=8192 && \
  export VLM_GPU_MEMORY_UTILIZATION=0.9 && \
  export VLM_LIMIT_MM_PER_PROMPT='{"image":1,"video":0}' && \
  export VLM_SKIP_MM_PROFILING=true && \
  export VLM_MAX_NUM_SEQS=1 && \
  export SAM2_GPU_DEVICES=${tool_gpu} && \
  export REMOTESAM_GPU_DEVICES=${tool_gpu} && \
  export STRIP_RCNN_GPU_DEVICES=${tool_gpu} && \
  export INSTRUCTSAM_GPU_DEVICES=${tool_gpu} && \
  export REMOTECLIP_GPU_DEVICES=${tool_gpu} && \
  export no_proxy=localhost,127.0.0.1 && \
  export NO_PROXY=localhost,127.0.0.1
EOF
}

# ============================================================================
# 清理旧会话和 Docker 容器
# ============================================================================

echo "Cleaning up old tmux sessions..."
tmux kill-session -t "$EXP1_TMUX_SESSION" 2>/dev/null || true
tmux kill-session -t "$EXP2_TMUX_SESSION" 2>/dev/null || true
tmux kill-session -t "monitor" 2>/dev/null || true

echo "Cleaning up old Docker containers..."
docker rm -f agent_llm_gpu0 agent_llm_gpu2 2>/dev/null || true
docker ps -q --filter "label=terrabox.service=agent-llm" | xargs -r docker rm -f 2>/dev/null || true

# ============================================================================
# 创建日志目录
# ============================================================================

mkdir -p logs

# ============================================================================
# 启动 Experiment 1
# ============================================================================

echo "【Part 1】Starting experiment on GPU0 (LLM) + GPU1 (perception)..."
echo "  Tasks: $EXP1_START-$((EXP1_END-1)) ($((EXP1_END - EXP1_START)) tasks)"
echo "  Port: $EXP1_PORT"
echo ""

tmux new-session -d -s "$EXP1_TMUX_SESSION" -c "$REPO_ROOT"
tmux send-keys -t "$EXP1_TMUX_SESSION" \
  "$(worker_env "$EXP1_GPU_DEVICES" "$EXP1_TOOL_GPU_DEVICES") && \
  export AGENT_LLM_PORT=$EXP1_PORT && \
  $PYTHON_BIN scripts/run_trajectory_experiment.py rollout \
    --task-file $TASK_FILE \
    --experiment $EXP1_NAME \
    --mode $MODE \
    --start-index $EXP1_START \
    --end-index $EXP1_END \
    --port $EXP1_PORT \
    --use-docker \
    --resume \
    2>&1 | tee logs/${EXP1_NAME}.log" C-m

# ============================================================================
# 启动 Experiment 2
# ============================================================================

echo "【Part 2】Starting experiment on GPU2 (LLM) + GPU3 (perception)..."
echo "  Tasks: $EXP2_START-$((EXP2_END-1)) ($((EXP2_END - EXP2_START)) tasks)"
echo "  Port: $EXP2_PORT"
echo ""

tmux new-session -d -s "$EXP2_TMUX_SESSION" -c "$REPO_ROOT"
tmux send-keys -t "$EXP2_TMUX_SESSION" \
  "$(worker_env "$EXP2_GPU_DEVICES" "$EXP2_TOOL_GPU_DEVICES") && \
  export AGENT_LLM_PORT=$EXP2_PORT && \
  $PYTHON_BIN scripts/run_trajectory_experiment.py rollout \
    --task-file $TASK_FILE \
    --experiment $EXP2_NAME \
    --mode $MODE \
    --start-index $EXP2_START \
    --end-index $EXP2_END \
    --port $EXP2_PORT \
    --use-docker \
    --resume \
    2>&1 | tee logs/${EXP2_NAME}.log" C-m

# ============================================================================
# 启动监控
# ============================================================================

echo ""
echo "✅ Both experiments started in tmux sessions!"
echo ""
echo "【Monitor】"
echo "  Attach to Part 1: tmux attach -t $EXP1_TMUX_SESSION"
echo "  Attach to Part 2: tmux attach -t $EXP2_TMUX_SESSION"
echo ""
echo "【Progress tracking】"
echo "  Part 1: watch -n 10 'ls tmp/trajectories/$EXP1_NAME/standard/results 2>/dev/null | wc -l'"
echo "  Part 2: watch -n 10 'ls tmp/trajectories/$EXP2_NAME/standard/results 2>/dev/null | wc -l'"
echo ""
echo "【Resume (if interrupted)】"
echo "  Re-run this script. Existing per-task JSON files are skipped because --resume is set."
echo ""

# ============================================================================
# 可选：启动监控面板（在单独的 tmux 窗口中）
# ============================================================================

if command -v watch &> /dev/null; then
  tmux new-session -d -s "monitor" -c "$REPO_ROOT"
  tmux send-keys -t "monitor" \
    "watch -n 10 'echo \"Part 1: \$(ls tmp/trajectories/$EXP1_NAME/standard/results 2>/dev/null | wc -l)/$((EXP1_END-EXP1_START)) tasks\"; echo \"Part 2: \$(ls tmp/trajectories/$EXP2_NAME/standard/results 2>/dev/null | wc -l)/$((EXP2_END-EXP2_START)) tasks\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,noheader'" C-m
  echo "  Monitor panel: tmux attach -t monitor"
fi

echo ""
echo "⏱️  Experiments are running in the background..."
