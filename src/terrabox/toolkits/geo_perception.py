"""
Geo Perception Toolkit
----------------------
Advanced AI perception tools including Multi-image VLM analysis.
"""

import json
import ast
import logging
# 保留原有的其他 import
import os
import base64
import mimetypes
import requests
from typing import Any, Dict, List
from ..core.registry import ToolSpec
from ..utils.vllm_manager import vllm_manager

# 确保有 logger
logger = logging.getLogger(__name__)
# --- 辅助函数 ---

def _encode_image_to_base64(image_path: str) -> str:
    """Helper: Convert local image file to data URI scheme."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
        
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    return f"data:{mime_type};base64,{encoded_string}"

# --- Handler ---

def vlm_analyze_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for VLM Analysis (终极修复版：处理嵌套列表字符串)
    """
    raw_images = arguments.get("image_paths") or arguments.get("image_path")
    
    if not raw_images:
         return {"status": "error", "message": "Missing image_paths parameter."}

    logger.info(f"DEBUG INPUT: Type={type(raw_images)} Value={raw_images}")
    print(f"DEBUG INPUT: Type={type(raw_images)} Value={raw_images}")

    image_paths = []

    # === 修复核心逻辑 ===
    if isinstance(raw_images, list):
        # 检查是否命中了 "列表包字符串" 的情况
        # 例如: ['["/path/a", "/path/b"]']
        if len(raw_images) == 1 and isinstance(raw_images[0], str):
            s = raw_images[0].strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    # 尝试把里面那个字符串解开
                    image_paths = json.loads(s)
                    logger.info("Unwrapped nested JSON string inside list.")
                except:
                    # 解不开就当做普通路径
                    image_paths = raw_images
            else:
                image_paths = raw_images
        else:
            # 正常的列表 ["/path/a", "/path/b"]
            image_paths = raw_images
            
    elif isinstance(raw_images, str):
        # 处理纯字符串的情况 (之前的逻辑)
        s = raw_images.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                image_paths = json.loads(s)
            except:
                try:
                    image_paths = ast.literal_eval(s)
                except:
                    # 暴力拆解
                    content = s[1:-1]
                    parts = content.split(',')
                    for p in parts:
                        clean_p = p.strip().strip('"').strip("'")
                        if clean_p:
                            image_paths.append(clean_p)
        else:
            image_paths = [s]
    # === 逻辑结束 ===

    # 打印最终解析结果，再次确认是否正确
    logger.info(f"DEBUG PARSED PATHS: {image_paths}")
    print(f"DEBUG PARSED PATHS: {image_paths}")

    # ... (后续代码：start_service, prompt, max_tokens 处理保持不变) ...
    prompt = arguments.get("prompt", "Analyze these images.")
    max_tokens = arguments.get("max_tokens", 512)
    
    try:
        vllm_manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start AI Service: {str(e)}"}

    message_content = [{"type": "text", "text": prompt}]
    
    try:
        for idx, path in enumerate(image_paths):
            clean_path = str(path).strip()
            # 再次检查路径是否存在
            if not os.path.exists(clean_path):
                # 打印出具体是哪个路径错了
                logger.error(f"File not found: {clean_path}")
                return {"status": "error", "message": f"File not found on server: {clean_path}"}
                
            b64_str = _encode_image_to_base64(clean_path)
            message_content.append({
                "type": "image_url", 
                "image_url": {"url": b64_str}
            })
    except Exception as e:
        return {"status": "error", "message": f"Image error: {str(e)}"}

    # 发送请求
    api_url = f"{vllm_manager.API_BASE}/chat/completions"
    payload = {
        "model": vllm_manager.MODEL_PATH,
        "messages": [{"role": "user", "content": message_content}],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=300)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            return {
                "status": "success",
                "analysis": content,
                "processed_images": len(image_paths)
            }
        else:
            return {"status": "error", "message": f"API Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}
# --- 注册 ---

def setup(registrar):
    registrar.toolkit(
        name="geo_perception",
        description="Advanced AI perception tools including Multi-Image VLM.",
        version="0.1.0"
    )

    registrar.tool(
        ToolSpec(
            slug="geo_perception.vlm_analyze",
            name="VLM Image Analysis",
            description="Analyze one or multiple satellite/drone images using a VLM.",
            parameters={
                "type": "object",
                "properties": {
                    "image_paths": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of absolute paths to image files."
                    },
                    # 兼容旧参数名（可选）
                    "image_path": {
                        "type": "string",
                        "description": "Legacy: Single image path (optional)."
                    },
                    "prompt": {
                        "type": "string", 
                        "description": "Question or instruction.", 
                        "default": ""
                    },
                    "max_tokens": {"type": "integer", "default": 4096}
                },
                # 这里要求至少提供一种图片输入
                "required": [] 
            },
            requires_connection=False
        ),
        vlm_analyze_handler
    )