#!/bin/bash
# 监控两个并行实验的进度

EXP1="merged_parallel_part1"
EXP2="merged_parallel_part2"
REPO="/data1/yuhongjie2/terrabox"

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "PARALLEL EXPERIMENT MONITOR"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Part 1 进度
    count1=$(ls -1 "$REPO/tmp/trajectories/$EXP1/standard/results"/*.json 2>/dev/null | wc -l)
    echo "【Part 1】GPU0+GPU1 ($EXP1)"
    echo "  Progress: $count1 / 4883 tasks"
    if [ -f "$REPO/tmp/trajectories/$EXP1/standard/report.json" ]; then
        echo "  ✅ COMPLETED"
    else
        pct=$((count1 * 100 / 4883))
        echo "  Progress: $pct%"
    fi
    echo ""
    
    # Part 2 进度
    count2=$(ls -1 "$REPO/tmp/trajectories/$EXP2/standard/results"/*.json 2>/dev/null | wc -l)
    echo "【Part 2】GPU2+GPU3 ($EXP2)"
    echo "  Progress: $count2 / 4884 tasks"
    if [ -f "$REPO/tmp/trajectories/$EXP2/standard/report.json" ]; then
        echo "  ✅ COMPLETED"
    else
        pct=$((count2 * 100 / 4884))
        echo "  Progress: $pct%"
    fi
    echo ""
    
    # 总进度
    total=$((count1 + count2))
    echo "【Total】Combined Progress"
    echo "  Total: $total / 9767 tasks"
    pct=$((total * 100 / 9767))
    echo "  Progress: $pct%"
    echo ""
    
    # 显示日志尾部
    if [ -f "$REPO/logs/${EXP1}.log" ]; then
        echo "Last update from Part 1:"
        tail -3 "$REPO/logs/${EXP1}.log" | sed 's/^/  /'
    fi
    echo ""
    
    if [ -f "$REPO/logs/${EXP2}.log" ]; then
        echo "Last update from Part 2:"
        tail -3 "$REPO/logs/${EXP2}.log" | sed 's/^/  /'
    fi
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Press Ctrl+C to stop monitoring"
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    
    sleep 30
done
