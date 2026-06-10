
import json
from collections import Counter

# 1. 读取 OpenEarthAgent 原始数据，提取所有工具名
print("=" * 80)
print("1. OpenEarthAgent 原始工具统计")
print("=" * 80)

with open('/data1/yuhongjie2/OpenEarthAgent/data/train.json', 'r', encoding='utf-8') as f:
    oea_data = json.load(f)

oea_tool_counter = Counter()
oea_sample_count = 0

for sample in oea_data:
    oea_sample_count += 1
    for turn in sample.get('conversation', []):
        if turn.get('from') == 'gpt':
            try:
                parsed = json.loads(turn['value'])
                for action in parsed.get('actions', []):
                    oea_tool_counter[action['name']] += 1
            except:
                pass

print("OpenEarthAgent 样本数:", oea_sample_count)
print("OpenEarthAgent 原始工具数:", len(oea_tool_counter))
print("\n所有原始工具:")
for tool, count in sorted(oea_tool_counter.items(), key=lambda x: -x[1]):
    print("  %-50s %5d calls" % (tool, count))

# 2. 读取 EarthBench 数据
print("\n" + "=" * 80)
print("2. EarthBench 原始工具统计")
print("=" * 80)

eb_tool_counter = Counter()
eb_sample_count = 0

# 尝试读取 EarthBench 数据
import os
eb_paths = [
    '/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl',
]

# 先看 newdata 目录下有没有 EarthBench 原始数据
eb_raw_dir = '/data1/yuhongjie2/terrabox/data'
for root, dirs, files in os.walk(eb_raw_dir):
    for fn in files:
        if 'earthbench' in fn.lower() and fn.endswith('.json'):
            print("Found EarthBench file:", os.path.join(root, fn))

# 3. 读取我们的 strict 数据，按来源分别统计
print("\n" + "=" * 80)
print("3. 新 strict 数据工具统计 (按来源)")
print("=" * 80)

new_oea_tool_counter = Counter()
new_eb_tool_counter = Counter()
new_oea_ipython_count = 0
new_eb_ipython_count = 0
new_oea_total = 0
new_eb_total = 0

with open('/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        source = sample.get('source', 'unknown')
        gold_calls = sample.get('gold_tool_calls', [])
        
        if source == 'openearth':
            new_oea_total += 1
            for call in gold_calls:
                tool = call.get('tool')
                if tool:
                    new_oea_tool_counter[tool] += 1
                    if 'ipython' in tool.lower():
                        new_oea_ipython_count += 1
        elif source == 'earthbench':
            new_eb_total += 1
            for call in gold_calls:
                tool = call.get('tool')
                if tool:
                    new_eb_tool_counter[tool] += 1
                    if 'ipython' in tool.lower():
                        new_eb_ipython_count += 1

print("\n--- OpenEarth 来源 (新数据) ---")
print("样本数:", new_oea_total)
print("ipython 调用数:", new_oea_ipython_count)
print("非ipython工具:")
for tool, count in sorted(new_oea_tool_counter.items(), key=lambda x: -x[1]):
    if 'ipython' not in tool.lower():
        print("  %-50s %5d calls" % (tool, count))

print("\n--- EarthBench 来源 (新数据) ---")
print("样本数:", new_eb_total)
print("ipython 调用数:", new_eb_ipython_count)
print("非ipython工具:")
for tool, count in sorted(new_eb_tool_counter.items(), key=lambda x: -x[1]):
    if 'ipython' not in tool.lower():
        print("  %-50s %5d calls" % (tool, count))

# 4. 对比：OpenEarthAgent 原始工具 vs 新数据工具
print("\n" + "=" * 80)
print("4. 工具映射对比：原始 OpenEarthAgent 工具 -> 新数据")
print("=" * 80)

# 收集新数据中所有非ipython工具
new_non_ipython = set()
for tool in new_oea_tool_counter:
    if 'ipython' not in tool.lower():
        new_non_ipython.add(tool)

# 收集原始工具
orig_tools = set(oea_tool_counter.keys())

# 原始有但新数据中没有的工具（可能被ipython替代了）
replaced_tools = orig_tools - new_non_ipython
# 新数据中有但原始没有的工具（新加的）
new_tools = new_non_ipython - orig_tools
# 两边都有的工具
kept_tools = orig_tools & new_non_ipython

print("\n--- 被保留的原始工具 (原始和新数据都有) ---")
for tool in sorted(kept_tools):
    print("  %-50s (原始: %d calls, 新: %d calls)" % (tool, oea_tool_counter.get(tool, 0), new_oea_tool_counter.get(tool, 0)))

print("\n--- 可能被 ipython 替代的原始工具 (原始有但新数据中无) ---")
for tool in sorted(replaced_tools):
    print("  %-50s (原始: %d calls)" % (tool, oea_tool_counter.get(tool, 0)))

print("\n--- 新增的工具 (新数据有但原始无) ---")
for tool in sorted(new_tools):
    print("  %-50s (新: %d calls)" % (tool, new_oea_tool_counter.get(tool, 0)))
