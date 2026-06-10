
import json

# 读取原始的OpenEarthAgent数据
with open('/data1/yuhongjie2/OpenEarthAgent/data/train.json', 'r', encoding='utf-8') as f:
    orig_data = json.load(f)

print("=== 原始OpenEarthAgent数据示例 ===")
sample_idx = 0
sample = orig_data[sample_idx]
print(f"问题: {sample['question']}")
print("\n对话历史:")
for turn in sample['conversation']:
    from_user = turn['from']
    value = turn['value']
    print(f"\n[{from_user}]")
    if from_user == 'gpt':
        try:
            parsed = json.loads(value)
            print(f"  思考: {parsed['thought']}")
            for action in parsed.get('actions', []):
                print(f"  动作: {action['name']}")
                print(f"  参数: {json.dumps(action['arguments'], ensure_ascii=False, indent=6)}")
        except:
            print(f"  {value[:200]}")
    else:
        print(f"  {value[:300]}")

print("\n\n=== 读取新的strict数据 ===")
with open('/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        new_sample = json.loads(line)
        print(f"新样本问题: {new_sample.get('question')}")
        print(f"工具序列: {new_sample.get('tool_sequence')}")
        gold_calls = new_sample.get('gold_tool_calls', [])
        print(f"Gold工具调用数: {len(gold_calls)}")
        for call in gold_calls[:3]:  # 只显示前3个
            print(f"\n工具名: {call.get('tool')}")
            args = call.get('arguments', {})
            print(f"参数:")
            for k, v in args.items():
                val_str = str(v)
                if len(val_str) &gt; 200:
                    val_str = val_str[:200] + "..."
                print(f"  {k}: {val_str}")
        break

