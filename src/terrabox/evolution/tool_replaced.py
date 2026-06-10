
import json
from collections import Counter, defaultdict

# 读取 OpenEarthAgent 原始数据，建立 question -> 工具列表 的映射
print("读取 OpenEarthAgent 原始数据...")
with open('/data1/yuhongjie2/OpenEarthAgent/data/train.json', 'r', encoding='utf-8') as f:
    oea_data = json.load(f)

# 建立 question -> 原始工具列表 的映射
oea_by_question = {}
for sample in oea_data:
    q = sample.get('question', '')
    if not q:
        continue
    tools = []
    for turn in sample.get('conversation', []):
        if turn.get('from') == 'gpt':
            try:
                parsed = json.loads(turn['value'])
                for action in parsed.get('actions', []):
                    tools.append(action['name'])
            except:
                pass
    oea_by_question[q] = tools

# 读取新 strict 数据，按 question 匹配
print("读取新 strict 数据...")
replaced_by_ipython = Counter()  # 原始工具名 -> 被ipython替代的次数
filtered_out = Counter()         # 原始工具名 -> 被过滤掉的次数（新数据中完全没有）
kept_as_is = Counter()           # 原始工具名 -> 保留在新数据中的次数（映射到新工具）
kept_mapping = defaultdict(Counter)  # 原始工具 -> 新工具 -> 次数

with open('/data1/yuhongjie2/terrabox/data/newdata/sft_train_strict.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        source = sample.get('source', '')
        if source != 'openearth':
            continue
        
        q = sample.get('question', '')
        gold_calls = sample.get('gold_tool_calls', [])
        
        # 新数据中的工具列表
        new_tools = [call.get('tool') for call in gold_calls if call.get('tool')]
        has_ipython = any('ipython' in t.lower() for t in new_tools)
        
        # 找到原始数据中的工具
        orig_tools = oea_by_question.get(q, [])
        
        if not orig_tools:
            continue
        
        # 对每个原始工具，判断它的去向
        for orig_tool in orig_tools:
            # 检查这个原始工具是否被映射到新工具
            # 映射关系：
            # ObjectDetection -> geo_perception.instructsam / strip_rcnn_detect
            # SegmentObjectPixels -> geo_perception.sam2_segment
            # DrawBox -> geo_perception.draw_bboxes
            # AddText -> geo_perception.add_text
            # OCR -> geo_perception.ocr_extract
            
            tool_mapping = {
                'ObjectDetection': ['geo_perception.instructsam', 'geo_perception.strip_rcnn_detect'],
                'SegmentObjectPixels': ['geo_perception.sam2_segment'],
                'DrawBox': ['geo_perception.draw_bboxes'],
                'AddText': ['geo_perception.add_text'],
                'OCR': ['geo_perception.ocr_extract'],
            }
            
            mapped_new_tools = tool_mapping.get(orig_tool, [])
            
            if mapped_new_tools:
                # 这个工具有明确的新工具映射
                found = False
                for mt in mapped_new_tools:
                    if mt in new_tools:
                        kept_as_is[orig_tool] += 1
                        kept_mapping[orig_tool][mt] += 1
                        found = True
                        break
                if not found:
                    # 有映射但在新数据中没找到，可能被过滤或被ipython替代
                    if has_ipython:
                        replaced_by_ipython[orig_tool] += 1
                    else:
                        filtered_out[orig_tool] += 1
            else:
                # 这个工具没有明确的新工具映射
                # 如果新数据中有ipython，说明被ipython替代了
                if has_ipython:
                    replaced_by_ipython[orig_tool] += 1
                else:
                    filtered_out[orig_tool] += 1

print("\n" + "=" * 80)
print("被 ipython 替代的原始工具（新数据中同一问题用 ipython 代替了原工具）")
print("=" * 80)
total_replaced = 0
for tool, count in sorted(replaced_by_ipython.items(), key=lambda x: -x[1]):
    print("  %-40s %5d 次" % (tool, count))
    total_replaced += count
print("\n合计: %d 次原始工具调用被 ipython 替代" % total_replaced)

print("\n" + "=" * 80)
print("被保留/重命名的原始工具（映射到新工具）")
print("=" * 80)
for tool, count in sorted(kept_as_is.items(), key=lambda x: -x[1]):
    mapping_str = ", ".join(["%s(%d)" % (k, v) for k, v in kept_mapping[tool].items()])
    print("  %-40s %5d 次 -> %s" % (tool, count, mapping_str))

print("\n" + "=" * 80)
print("被过滤掉的原始工具（新数据中同一问题既没有映射工具也没有 ipython）")
print("=" * 80)
total_filtered = 0
for tool, count in sorted(filtered_out.items(), key=lambda x: -x[1]):
    print("  %-40s %5d 次" % (tool, count))
    total_filtered += count
print("\n合计: %d 次原始工具调用被过滤掉" % total_filtered)
