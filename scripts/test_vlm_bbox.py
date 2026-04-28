#!/usr/bin/env python3
"""
测试 VLM 服务是否能输出 bbox 数据
用法: python scripts/test_vlm_bbox.py
"""

import sys
import os
import json
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from terrabox.managers import vllm_manager


def encode_image_to_base64(image_path: str) -> str:
    """将图像转换为 base64"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"


def test_vlm_with_prompt(image_path: str, prompt: str, test_name: str):
    """测试 VLM 是否能输出 bbox"""
    print(f"\n{'='*60}")
    print(f"测试: {test_name}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}\n")
    
    # 启动 VLM 服务
    try:
        vllm_manager.start_service()
    except Exception as e:
        print(f"❌ 启动 VLM 服务失败: {e}")
        return None
    
    # 编码图像
    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return None
    
    b64_image = encode_image_to_base64(image_path)
    
    # 构建请求
    api_url = f"{vllm_manager.API_BASE}/chat/completions"
    payload = {
        "model": getattr(vllm_manager, "MODEL_NAME", vllm_manager.MODEL_PATH),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]
        }],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    # 调用 API
    import requests
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
            proxies={"http": None, "https": None}
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ VLM 输出:\n{content}\n")
            
            # 尝试提取 bbox
            bboxes = extract_bboxes_from_text(content)
            if bboxes:
                print(f"✅ 成功提取 bbox: {bboxes}")
            else:
                print(f"⚠️ 未能从输出中提取 bbox")
            
            return {
                "prompt": prompt,
                "response": content,
                "bboxes": bboxes
            }
        else:
            print(f"❌ API 错误 {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None


def extract_bboxes_from_text(text: str) -> list:
    """从文本中提取 bbox"""
    import re
    
    bboxes = []
    
    # 方法1: 匹配 JSON 格式的 bbox
    # {"x1": 120, "y1": 80, "x2": 340, "y2": 210}
    json_pattern = r'\{[^{}]*"x1"\s*:\s*(\d+)[^{}]*"y1"\s*:\s*(\d+)[^{}]*"x2"\s*:\s*(\d+)[^{}]*"y2"\s*:\s*(\d+)[^{}]*\}'
    for match in re.finditer(json_pattern, text, re.IGNORECASE):
        bboxes.append({
            "x1": int(match.group(1)),
            "y1": int(match.group(2)),
            "x2": int(match.group(3)),
            "y2": int(match.group(4))
        })
    
    # 方法2: 匹配数组格式 [x1, y1, x2, y2]
    if not bboxes:
        array_pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
        for match in re.finditer(array_pattern, text):
            bboxes.append({
                "x1": int(match.group(1)),
                "y1": int(match.group(2)),
                "x2": int(match.group(3)),
                "y2": int(match.group(4))
            })
    
    # 方法3: 匹配括号格式 (x1, y1, x2, y2)
    if not bboxes:
        paren_pattern = r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)'
        for match in re.finditer(paren_pattern, text):
            bboxes.append({
                "x1": int(match.group(1)),
                "y1": int(match.group(2)),
                "x2": int(match.group(3)),
                "y2": int(match.group(4))
            })
    
    return bboxes


def main():
    # 使用一个测试图像
    test_image = "/data1/yuhongjie2/disasterData/DisasterM3/data/DisasterM3/train_images/kalehe_flooding_post_1000.png"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        print("请修改 test_image 路径为实际存在的图像")
        return
    
    print(f"使用测试图像: {test_image}")
    
    # 测试不同的 prompt
    test_cases = [
        {
            "name": "原始 prompt（无 bbox 要求）",
            "prompt": "请识别并标注图中被洪水淹没的区域，描述积水范围和严重程度"
        },
        {
            "name": "改进 prompt（要求输出 bbox）",
            "prompt": """请识别图中被洪水淹没的区域，并输出每个区域的边界框坐标。

输出格式要求：
1. 先用文字描述淹没区域的位置和严重程度
2. 然后输出边界框列表，格式为 JSON：
   {"bboxes": [{"x1": 左上角x坐标, "y1": 左上角y坐标, "x2": 右下角x坐标, "y2": 右下角y坐标}, ...]}

示例输出：
被洪水淹没的区域主要分布在图像中部，面积较大。
{"bboxes": [{"x1": 120, "y1": 80, "x2": 340, "y2": 210}]}"""
        },
        {
            "name": "简洁 prompt（直接要求 bbox）",
            "prompt": """分析这张洪水灾害图像，找出所有被淹没的区域。

请按照以下格式输出：
1. 文字描述：简要描述淹没区域的位置和特征
2. 边界框坐标：用 JSON 格式输出，例如：
   {"bboxes": [{"x1": 100, "y1": 50, "x2": 300, "y2": 200}]}

注意：坐标是像素坐标，(0,0) 是图像左上角，x 向右增加，y 向下增加。"""
        },
        {
            "name": "多区域 prompt",
            "prompt": """识别图像中所有被洪水淹没的区域，为每个区域提供一个边界框。

输出格式：
描述：[文字描述]
边界框：{"bboxes": [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}, ...]}

如果有多个淹没区域，请输出多个边界框。"""
        }
    ]
    
    results = []
    for test_case in test_cases:
        result = test_vlm_with_prompt(test_image, test_case["prompt"], test_case["name"])
        if result:
            results.append(result)
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n测试 {i}: {test_cases[i-1]['name']}")
        if result['bboxes']:
            print(f"  ✅ 成功提取 bbox: {len(result['bboxes'])} 个")
            print(f"  bbox 示例: {result['bboxes'][0]}")
        else:
            print(f"  ❌ 未能提取 bbox")
    
    # 保存结果
    output_file = REPO_ROOT / "data" / "vlm_bbox_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
