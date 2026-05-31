#!/usr/bin/env python3
"""合并两个并行实验的结果"""
import json
import os
from pathlib import Path
from collections import Counter

def merge_experiments(exp1_name, exp2_name, output_name="merged_final"):
    repo_root = Path(__file__).resolve().parent.parent
    traj_dir = repo_root / "tmp" / "trajectories"
    
    # 加载两个实验的结果
    exp1_full = traj_dir / exp1_name / "standard" / "trajectories_full.jsonl"
    exp2_full = traj_dir / exp2_name / "standard" / "trajectories_full.jsonl"
    
    all_results = []
    total_success = 0
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    tool_freq = Counter()
    status_dist = Counter()
    source_dist = Counter()
    f1_scores = []
    
    print("=" * 80)
    print(f"Merging experiments: {exp1_name} + {exp2_name}")
    print("=" * 80)
    
    for exp_file in [exp1_full, exp2_full]:
        if not exp_file.exists():
            print(f"⚠️  File not found: {exp_file}")
            continue
        
        print(f"\nProcessing {exp_file.name}...")
        with open(exp_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    all_results.append(result)
                    
                    # 统计
                    if result.get('success'):
                        total_success += 1
                    
                    for k in total_tokens:
                        total_tokens[k] += result.get('tokens', {}).get(k, 0)
                    
                    for tool in result.get('tool_sequence', []):
                        tool_freq[tool] += 1
                    
                    status_dist[result.get('status', 'unknown')] += 1
                    source_dist[result.get('source', 'unknown')] += 1
                    
                    f1 = result.get('metrics', {}).get('f1', 0)
                    f1_scores.append(f1)
        
        print(f"  ✓ Loaded {len(all_results)} trajectories so far")
    
    # 生成合并的报告
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    report = {
        "merged_from": [exp1_name, exp2_name],
        "output_name": output_name,
        "total_tasks": len(all_results),
        "success_count": total_success,
        "success_rate": total_success / len(all_results) if all_results else 0,
        "total_tokens": total_tokens,
        "avg_f1": avg_f1,
        "status_distribution": dict(status_dist),
        "source_distribution": dict(source_dist),
        "tool_frequency": dict(tool_freq.most_common(20)),
    }
    
    # 保存合并的trajectories_full.jsonl
    output_dir = traj_dir / output_name / "standard"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_traj = output_dir / "trajectories_full.jsonl"
    with open(merged_traj, 'w') as f:
        for result in sorted(all_results, key=lambda x: x['task_id']):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 生成合并报告
    report_path = output_dir / "merged_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\n{'='*80}")
    print(f"MERGE COMPLETE: {output_name}")
    print(f"{'='*80}")
    print(f"Total tasks: {len(all_results)}")
    print(f"Success: {total_success} ({report['success_rate']:.1%})")
    print(f"Avg F1: {avg_f1:.3f}")
    print(f"Total tokens: {total_tokens['total_tokens']}")
    print(f"\nStatus distribution: {dict(status_dist)}")
    print(f"Source distribution: {dict(source_dist)}")
    print(f"\nOutput files:")
    print(f"  {merged_traj}")
    print(f"  {report_path}")
    print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exp1 = sys.argv[1]
        exp2 = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else "merged_final"
    else:
        exp1 = "merged_parallel_part1"
        exp2 = "merged_parallel_part2"
        output = "merged_final"
    
    merge_experiments(exp1, exp2, output)
