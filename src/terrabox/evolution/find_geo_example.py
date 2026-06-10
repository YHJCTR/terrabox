
import json

print("=== 找一个涉及地理/遥感处理的样本 ===\n")

with open('/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl', 'r', encoding='utf-8') as f:
    # 找一个有"NDVI"、"index"、"raster"等关键词的样本
    for line_num, line in enumerate(f):
        sample = json.loads(line)
        question = sample.get('question', '')
        keywords = ['ndvi', 'index', 'raster', 'landsat', 'sentinel', 'change', 'compute', 'calculate']
        has_keyword = False
        for k in keywords:
            if k in question.lower():
                has_keyword = True
                break
        
        if has_keyword:
            gold_calls = sample.get('gold_tool_calls', [])
            has_ipython = False
            for call in gold_calls:
                if 'ipython' in call.get('tool', '').lower():
                    has_ipython = True
                    break
            
            if has_ipython:
                print(f"样本 {line_num} 问题: {question}")
                print("\n工具调用:")
                for call_idx, call in enumerate(gold_calls):
                    tool = call.get('tool')
                    args = call.get('arguments')
                    print(f"\n--- 调用 {call_idx}: {tool} ---")
                    if tool == 'ipython.execute':
                        code = args.get('action', '')
                        print(f"Python 代码:\n{code}")
                    else:
                        print(f"参数: {json.dumps(args, ensure_ascii=False)}")
                break

