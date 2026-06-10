
import json

print("=== 读取 OpenEarthAgent 原始数据中的一个样本 ===\n")
with open('/data1/yuhongjie2/OpenEarthAgent/data/train.json', 'r', encoding='utf-8') as f:
    orig_data = json.load(f)
    sample_orig = orig_data[0]  # 取第一个样本
    print("原始问题:", sample_orig['question'])
    print("\n原始对话中的工具调用:")
    for turn in sample_orig['conversation']:
        if turn['from'] == 'gpt':
            try:
                parsed = json.loads(turn['value'])
                for action in parsed.get('actions', []):
                    print(f"  工具名: {action['name']}")
                    print(f"  参数: {json.dumps(action['arguments'], ensure_ascii=False)}")
            except:
                pass
            break  # 只看第一个gpt回复

print("\n\n=== 读取我们的新数据，找一个对应的样本看看 ===\n")
with open('/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl', 'r', encoding='utf-8') as f:
    # 找一个用ipython的样本
    for line_num, line in enumerate(f):
        sample = json.loads(line)
        gold_calls = sample.get('gold_tool_calls', [])
        has_ipython = False
        for call in gold_calls:
            if 'ipython' in call.get('tool', '').lower():
                has_ipython = True
                break
        
        if has_ipython:
            print("新样本问题:", sample.get('question'))
            print("\n新数据工具调用:")
            for call in gold_calls[:3]:  # 只看前3个
                tool = call.get('tool')
                args = call.get('arguments')
                print(f"  工具名: {tool}")
                if tool == 'ipython.execute':
                    code = args.get('action', '')
                    print(f"  Python 代码:")
                    print("-" * 80)
                    print(code[:1000])  # 打印前1000字符
                    if len(code) > 1000:
                        print("... (截断)")
                    print("-" * 80)
                else:
                    print(f"  参数: {json.dumps(args, ensure_ascii=False)}")
            break

