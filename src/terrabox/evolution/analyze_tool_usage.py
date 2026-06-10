
import json
from collections import Counter
from pathlib import Path

data_path = Path("/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl")

tool_counter = Counter()
samples_with_ipython = []
samples_without_ipython = []

print("Loading data from", data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        try:
            sample = json.loads(line.strip())
            tool_sequence = sample.get('tool_sequence', [])
            
            if tool_sequence:
                for tool in tool_sequence:
                    tool_counter[tool] += 1
                
                has_ipython = False
                for tool in tool_sequence:
                    if 'ipython' in tool.lower():
                        has_ipython = True
                        break
                
                if has_ipython:
                    samples_with_ipython.append(sample)
                else:
                    samples_without_ipython.append(sample)
        except Exception as e:
            print("Error parsing line", line_num, ":", e)

print("\n=== Tool Usage Statistics ===")
print("Total samples:", line_num + 1)
print("Samples with ipython:", len(samples_with_ipython))
print("Samples without ipython:", len(samples_without_ipython))

print("\n=== Top 20 Tools ===")
for tool, count in tool_counter.most_common(20):
    print(tool, ":", count)

print("\n=== Total unique tools:", len(tool_counter), "===")

# Look at some ipython examples
print("\n=== Example of ipython usage (first 3 samples) ===")
for i in range(min(3, len(samples_with_ipython))):
    sample = samples_with_ipython[i]
    print("\nSample", i+1, ":")
    print("Question:", sample.get('question', 'N/A'))
    print("Tool sequence:", sample.get('tool_sequence', []))
    print("Gold tool calls:", sample.get('gold_tool_calls', []))
    
    # Show part of the conversation
    messages = sample.get('messages', [])
    print("Messages (truncated):")
    for j in range(max(0, len(messages)-4), len(messages)):  # Show last 4 messages
        msg = messages[j]
        role = msg.get('role', '')
        content = msg.get('content', '')
        if len(content) &gt; 500:
            content = content[:500] + "..."
        print("  ", role, ":", content)
