
import json

print("=== 找一个涉及批量处理或真实遥感数据的样本 ===\n")

with open('/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        sample = json.loads(line)
        gold_calls = sample.get('gold_tool_calls', [])
        
        # 找有较长Python代码的
        has_long_ipython = False
        for call in gold_calls:
            if call.get('tool') == 'ipython.execute':
                code = call.get('arguments', {}).get('action', '')
                if len(code) > 300:
                    has_long_ipython = True
                    break
        
        if has_long_ipython:
            print("样本", line_num, "问题:", sample.get('question', ''))
            print("\n工具调用:")
            for call_idx, call in enumerate(gold_calls):
                tool = call.get('tool')
                args = call.get('arguments')
                print("\n--- 调用", call_idx, ":", tool, "---")
                if tool == 'ipython.execute':
                    code = args.get('action', '')
                    if len(code) > 1000:
                        print("Python 代码 (", len(code), "字符):")
                        print(code[:1000], "...")
                    else:
                        print("Python 代码:")
                        print(code)
                else:
                    print("参数:")
                    print(json.dumps(args, ensure_ascii=False))
            
            # 只看一个完整样本
            break
