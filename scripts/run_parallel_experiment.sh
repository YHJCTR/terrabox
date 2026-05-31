#!/bin/bash
# 并行运行两个experiment的脚本

REPO_ROOT="/data1/yuhongjie2/terrabox"
cd "$REPO_ROOT"

# ============================================================================
# 实验配置
# ============================================================================

TASK_FILE="data/merged/merged_train_tasks.json"
MODE="standard"
TOTAL_TASKS=9767
SPLIT_POINT=4883

# Experiment 1: GPU0 + GPU1, 任务0-4882
EXP1_NAME="merged_parallel_part1"
EXP1_START=0
EXP1_END=4882
EXP1_PORT=9100
EXP1_TMUX_SESSION="exp_part1"

# Experiment 2: GPU2 + GPU3, 任务4883-9766
# Port 9102对应GPU2，Port 9103对应GPU3，但LLM只需要一个端口（对应主GPU）
EXP2_NAME="merged_parallel_part2"
EXP2_START=4883
EXP2_END=9766
EXP2_PORT=9102
EXP2_TMUX_SESSION="exp_part2"

# ============================================================================
# 创建并启动 tmux 窗口
# ============================================================================

echo "【Part 1】Starting experiment on GPU0+GPU1..."
echo "  Tasks: $EXP1_START-$EXP1_END ($((EXP1_END - EXP1_START + 1)) tasks)"
echo "  Port: $EXP1_PORT"
echo ""

# 创建 Part 1 的 tmux 会话
tmux new-session -d -s "$EXP1_TMUX_SESSION" -c "$REPO_ROOT"
tmux send-keys -t "$EXP1_TMUX_SESSION" "conda activate unsloth && cd $REPO_ROOT && export CUDA_VISIBLE_DEVICES=0,1 && python scripts/run_trajectory_experiment.py rollout --task-file $TASK_FILE --experiment $EXP1_NAME --mode $MODE --start-index $EXP1_START --end-index $((EXP1_END+1)) --port $EXP1_PORT --resume 2>&1 | tee logs/${EXP1_NAME}.log" C-m

echo "【Part 2】Starting experiment on GPU2+GPU3..."
echo "  Tasks: $EXP2_START-$EXP2_END ($((EXP2_END - EXP2_START + 1)) tasks)"
echo "  Port: $EXP2_PORT"
echo ""

# 创建 Part 2 的 tmux 会话
tmux new-session -d -s "$EXP2_TMUX_SESSION" -c "$REPO_ROOT"
tmux send-keys -t "$EXP2_TMUX_SESSION" "conda activate unsloth && cd $REPO_ROOT && export CUDA_VISIBLE_DEVICES=2,3 && python scripts/run_trajectory_experiment.py rollout --task-file $TASK_FILE --experiment $EXP2_NAME --mode $MODE --start-index $EXP2_START --end-index $((EXP2_END+1)) --port $EXP2_PORT --resume 2>&1 | tee logs/${EXP2_NAME}.log" C-m

echo "✅ Both experiments started in tmux sessions:"
echo "   Part 1: tmux attach -t $EXP1_TMUX_SESSION"
echo "   Part 2: tmux attach -t $EXP2_TMUX_SESSION"
echo ""
echo "📊 Monitor progress:"
echo "   Part 1: watch -n 10 'ls tmp/trajectories/$EXP1_NAME/standard/results | wc -l'"
echo "   Part 2: watch -n 10 'ls tmp/trajectories/$EXP2_NAME/standard/results | wc -l'"
echo ""
echo "🔄 Resume (if interrupted):"
echo "   tmux send-keys -t $EXP1_TMUX_SESSION 'python scripts/run_trajectory_experiment.py rollout --task-file $TASK_FILE --experiment $EXP1_NAME --mode $MODE --start-index $EXP1_START --end-index $((EXP1_END+1)) --port $EXP1_PORT --resume' C-m"
echo "   tmux send-keys -t $EXP2_TMUX_SESSION 'python scripts/run_trajectory_experiment.py rollout --task-file $TASK_FILE --experiment $EXP2_NAME --mode $MODE --start-index $EXP2_START --end-index $((EXP2_END+1)) --port $EXP2_PORT --resume' C-m"

