
import json
from collections import defaultdict, Counter
from pathlib import Path

data_file = Path("/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl")

sample_count = 0
tool_call_count = 0
tool_counter = Counter()
sample_to_tools = defaultdict(set)
samples_with_ipython = 0
samples_without_ipython = 0

print("Processing", data_file, "...")

with open(data_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        try:
            sample = json.loads(line.strip())
            sample_count += 1
            
            gold_calls = sample.get('gold_tool_calls', [])
            tools_in_sample = []
            
            for call in gold_calls:
                tool_name = call.get('tool')
                if tool_name:
                    tool_counter[tool_name] += 1
                    tools_in_sample.append(tool_name)
                    tool_call_count += 1
            
            # Track which tools are used in each sample
            for t in tools_in_sample:
                sample_to_tools[t].add(line_num)
            
            # Check if sample has ipython
            has_ipython = False
            for t in tools_in_sample:
                if 'ipython' in t.lower():
                    has_ipython = True
                    break
            if has_ipython:
                samples_with_ipython += 1
            else:
                samples_without_ipython += 1
                
        except Exception as e:
            print("Error at line", line_num, ":", e)
            continue

print("\n=== Overall Statistics ===")
print("Total samples:", sample_count)
print("Total tool calls:", tool_call_count)
print("Unique tools:", len(tool_counter))
print("\nSamples with ipython:", samples_with_ipython, "(", samples_with_ipython*100//sample_count, "%)")
print("Samples without ipython:", samples_without_ipython, "(", samples_without_ipython*100//sample_count, "%)")

print("\n=== All Tools by Usage ===")
for tool, count in sorted(tool_counter.items(), key=lambda x: -x[1]):
    percent = count * 100 // tool_call_count
    sample_percent = len(sample_to_tools.get(tool, set())) * 100 // sample_count
    print("%40s | %5d calls (%2d%%) | %5d samples (%2d%%)" % (
        tool, count, percent, len(sample_to_tools.get(tool, set())), sample_percent))

# Show some samples without ipython
print("\n\n=== First 5 samples without ipython ===")
seen = 0
with open(data_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        if seen >= 5:
            break
        sample = json.loads(line.strip())
        gold_calls = sample.get('gold_tool_calls', [])
        tools_in_sample = [call.get('tool') for call in gold_calls if call.get('tool')]
        has_ipython = False
        for t in tools_in_sample:
            if 'ipython' in t.lower():
                has_ipython = True
                break
        if not has_ipython:
            q = sample.get('question', '')
            q_short = q[:80]
            if len(q) > 80:
                q_short += "..."
            print("\nSample", line_num, ":", q_short)
            print("  Tools:", tools_in_sample)
            seen += 1
