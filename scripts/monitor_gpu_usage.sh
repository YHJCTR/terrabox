#!/bin/bash
# 实时监控 GPU 使用情况和进程分配

set -e

INTERVAL=${1:-1}  # 默认每秒更新一次

echo "GPU 监控工具 - 监控间隔: ${INTERVAL}秒"
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""

    # GPU 详细信息
    echo "📊 GPU 使用状态："
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,processes.process_name,processes.pid \
        --format=csv,noheader,nounits | awk -F', ' '
    BEGIN {
        print "GPU | GPU利用率 | 内存利用率 | 内存占用 | 进程名 (PID)"
        print "────────────────────────────────────────────────────────────────"
    }
    {
        gpu=$1
        name=$2
        gpu_util=$3
        mem_util=$4
        mem_used=$5
        mem_total=$6
        proc=$7
        pid=$8

        # 简化 GPU 名称
        gsub(/NVIDIA /, "", name)
        gsub(/NVIDIA/, "", name)

        printf "GPU%s | %6.1f%% | %6.1f%% | %5.0fMB/%5.0fMB | %s (%s)\n",
            gpu, gpu_util, mem_util, mem_used, mem_total, proc, pid
    }
    '

    echo ""
    echo "🔄 运行中的 Python 进程:"
    ps aux | grep -E "python.*evolution" | grep -v grep | awk '{print $2, $11}' | while read pid cmd; do
        if [ -n "$pid" ]; then
            gpu=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep "$pid" | cut -d',' -f1 | head -1)
            if [ -n "$gpu" ]; then
                gpu_idx=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader 2>/dev/null | grep "$gpu" | cut -d',' -f1)
                echo "  PID $pid 运行在 GPU $gpu_idx"
            else
                echo "  PID $pid (未使用 GPU)"
            fi
        fi
    done

    echo ""
    echo "📁 输出文件生成进度:"
    json_count=$(find evo_res/disaster_1 -name "*.json" 2>/dev/null | wc -l)
    echo "  已生成 JSON 文件: $json_count 个"

    echo ""
    echo "─────────────────────────────────────────────────────────────────────────────"
    echo "💡 提示:"
    echo "  • 如果 GPU 利用率始终 < 10%，可能存在串行执行问题"
    echo "  • 理想情况下，阶段 2 应该有 3 个进程分别占用 GPU 0/1/2"
    echo "  • 更新间隔: ${INTERVAL}秒 (按 Ctrl+C 停止)"
    sleep $INTERVAL
done
