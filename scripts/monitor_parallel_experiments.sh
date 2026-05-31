#!/bin/bash
# 监控并行实验进度

REPO_ROOT="/data1/yuhongjie2/terrabox"
EXP_SUFFIX="${EXP_SUFFIX:-20260525_no_vlm}"
EXP1_NAME="${EXP1_NAME:-merged_tokens_part1_${EXP_SUFFIX}}"
EXP2_NAME="${EXP2_NAME:-merged_tokens_part2_${EXP_SUFFIX}}"
TOTAL_TASKS=$(/home/yuhongjie/miniconda3/envs/unsloth/bin/python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/data1/yuhongjie2/terrabox/data/merged/merged_train_tasks.json").read_text())
tasks = data.get("tasks", data) if isinstance(data, dict) else data
print(len(tasks))
PY
)
SPLIT_POINT=$(((TOTAL_TASKS + 1) / 2))
PART1_TOTAL=$SPLIT_POINT
PART2_TOTAL=$((TOTAL_TASKS - SPLIT_POINT))

echo "======================================================================"
echo "  PARALLEL EXPERIMENTS MONITORING"
echo "======================================================================"
echo ""

# Monitor progress
while true; do
    clear
    echo "======================================================================"
    echo "  PARALLEL EXPERIMENTS MONITORING — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
    echo ""

    # Part 1 progress
    echo "【Part 1】GPU0+GPU1 (tasks 0-$((SPLIT_POINT-1))):"
    part1_results=$(ls -1 "$REPO_ROOT/tmp/trajectories/$EXP1_NAME/standard/results" 2>/dev/null | wc -l)
    echo "  ✓ Completed: $part1_results / $PART1_TOTAL"
    if [ -f "$REPO_ROOT/logs/$EXP1_NAME.log" ]; then
        echo "  Recent log:"
        tail -3 "$REPO_ROOT/logs/$EXP1_NAME.log" | sed 's/^/    /'
    fi
    echo ""

    # Part 2 progress
    echo "【Part 2】GPU2+GPU3 (tasks $SPLIT_POINT-$((TOTAL_TASKS-1))):"
    part2_results=$(ls -1 "$REPO_ROOT/tmp/trajectories/$EXP2_NAME/standard/results" 2>/dev/null | wc -l)
    echo "  ✓ Completed: $part2_results / $PART2_TOTAL"
    if [ -f "$REPO_ROOT/logs/$EXP2_NAME.log" ]; then
        echo "  Recent log:"
        tail -3 "$REPO_ROOT/logs/$EXP2_NAME.log" | sed 's/^/    /'
    fi
    echo ""

    # GPU status
    echo "【GPU Status】:"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,noheader | \
        awk '{print "  GPU " $1 ": GPU=" $2 " Memory=" $3}'
    echo ""

    echo "【Tmux Sessions】:"
    echo "  Part 1: tmux attach -t exp_part1"
    echo "  Part 2: tmux attach -t exp_part2"
    echo ""

    sleep 30
done
