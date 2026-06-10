
import json

data_file = "/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl"

# Find first sample with ipython
print("Looking for ipython samples...")
with open(data_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        tools = sample.get('tool_sequence', [])
        has_ipython = any('ipython' in t.lower() for t in tools)
        if has_ipython:
            print(f"\nFound sample {i} with tools: {tools}")
            print(f"Question: {sample.get('question', 'N/A')}")
            gold_calls = sample.get('gold_tool_calls', [])
            print(f"\nGold tool calls:")
            for j, call in enumerate(gold_calls[:3]):  # Show first 3 calls
                print(f"\n  Call {j}:")
                print(f"    Tool: {call.get('tool')}")
                args = call.get('arguments', {})
                args_str = json.dumps(args, ensure_ascii=False)
                if len(args_str) > 500:
                    args_str = args_str[:500] + "..."
                print(f"    Args: {args_str}")
            break

# Check tool catalog
print("\n\nChecking tool catalog...")
first_sample = json.loads(open(data_file, 'r', encoding='utf-8').readline())
messages = first_sample.get('messages', [])
for msg in messages:
    if msg.get('role') == 'system':
        content = msg.get('content', '')
        if 'Tool catalog' in content:
            print("Found tool catalog in first sample.")
            # Extract part of the catalog
            idx = content.find('Tool catalog:')
            if idx >= 0:
                catalog = content[idx:]
                print(f"Catalog preview ({len(catalog)} chars):")
                print(catalog[:2000])
            break

