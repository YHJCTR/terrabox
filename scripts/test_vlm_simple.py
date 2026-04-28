#!/usr/bin/env python3
"""
简单的 VLM 测试脚本 - 直接测试 API
"""
import requests
import json
import base64

API_BASE = "http://127.0.0.1:9000/v1"

# 测试图像
test_image = "/data1/yuhongjie2/disasterData/DisasterM3/data/DisasterM3/train_images/kalehe_flooding_post_1000.png"

# 编码图像
with open(test_image, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode('utf-8')

# 测试不同的模型名称
model_names = [
    "/model",
    "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model",
    "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model/",
]

prompt = """请识别图中被洪水淹没的区域，并输出每个区域的边界框坐标。

输出格式要求：
1. 先用文字描述淹没区域的位置和严重程度
2. 然后输出边界框列表，格式为 JSON：
   {"bboxes": [{"x1": 左上角x坐标, "y1": 左上角y坐标, "x2": 右下角x坐标, "y2": 右下角y坐标}, ...]}

示例输出：
被洪水淹没的区域主要分布在图像中部，面积较大。
{"bboxes": [{"x1": 120, "y1": 80, "x2": 340, "y2": 210}]}"""

for model_name in model_names:
    print(f"\n{'='*60}")
    print(f"测试模型名称: {model_name}")
    print(f"{'='*60}")
    
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
            proxies={"http": None, "https": None}
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ 成功!")
            print(f"输出:\n{content[:500]}...")
            
            # 尝试提取 bbox
            import re
            json_pattern = r'\{[^{}]*"bboxes"[^{}]*\}'
            match = re.search(json_pattern, content, re.IGNORECASE)
            if match:
                print(f"\n✅ 找到 bboxes!")
                print(f"bbox 数据: {match.group()}")
            break
        else:
            print(f"❌ 失败: {response.status_code}")
            print(f"错误: {response.text[:200]}")
    except Exception as e:
        print(f"❌ 异常: {e}")
