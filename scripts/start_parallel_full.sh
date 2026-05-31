#!/bin/bash

cd /data1/yuhongjie2/terrabox

echo "Starting parallel experiments on full merged dataset..."
echo ""

# Part 1: GPU0+GPU1, 任务0-4882
echo "【Part 1】GPU0+GPU1 (tasks 0-4882)..."
tmux new-session -d -s exp_part1 -c /data1/yuhongjie2/terrabox
tmux send-keys -t exp_part1 "conda activate unsloth && export CUDA_VISIBLE_DEVICES=0,1 && python scripts/run_trajectory_experiment.py rollout --task-file data/merged/merged_train_tasks.json --experiment merged_parallel_part1 --mode standard --start-index 0 --end-index 4883 --port 9100 --resume 2>&1 | tee logs/merged_parallel_part1.log" C-m

# Part 2: GPU2+GPU3, 任务4883-9766
echo "【Part 2】GPU2+GPU3 (tasks 4883-9766)..."
tmux new-session -d -s exp_part2 -c /data1/yuhongjie2/terrabox
tmux send-keys -t exp_part2 "conda activate unsloth && export CUDA_VISIBLE_DEVICES=2,3 && python scripts/run_trajectory_experiment.py rollout --task-file data/merged/merged_train_tasks.json --experiment merged_parallel_part2 --mode standard --start-index 4883 --end-index 9767 --port 9102 --resume 2>&1 | tee logs/merged_parallel_part2.log" C-m

echo ""
echo "✅ Both experiments started!"
echo ""
echo "📊 监控进度："
echo "  tmux attach -t exp_part1     # Part 1窗口"
echo "  tmux attach -t exp_part2     # Part 2窗口"
echo ""
echo "🔄 断点续跑（如果中断）："
echo "  tmux send-keys -t exp_part1 'python scripts/run_trajectory_experiment.py rollout --task-file data/merged/merged_train_tasks.json --experiment merged_parallel_part1 --mode standard --start-index 0 --end-index 4883 --port 9100 --resume' C-m"
echo "  tmux send-keys -t exp_part2 'python scripts/run_trajectory_experiment.py rollout --task-file data/merged/merged_train_tasks.json --experiment merged_parallel_part2 --mode standard --start-index 4883 --end-index 9767 --port 9102 --resume' C-m"
echo ""
echo "📁 结果位置："
echo "  Part 1: tmp/trajectories/merged_parallel_part1/standard/"
echo "  Part 2: tmp/trajectories/merged_parallel_part2/standard/"
echo ""
echo "合并结果："
echo "  python scripts/merge_parallel_results.py merged_parallel_part1 merged_parallel_part2 merged_final_9767"

