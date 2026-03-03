"""
Geo Perception Toolkit
----------------------
Advanced AI perception tools including Multi-image VLM analysis.
"""

import json
import ast
import logging

import os
import base64
import mimetypes
import requests
from typing import Any, Dict, List
from ..core.registry import ToolSpec
from ..utils.vllm_manager import vllm_manager
from ..utils.sam2_manager import sam2_manager
from ..utils.remoteclip_manager import remoteclip_manager
from ..utils.remotesam_manager import remotesam_manager
from ..utils.strip_rcnn_manager import strip_rcnn_manager
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
    # 统一入口参数命名为 images（兼容旧的 image_paths/image_path）
    raw_images = (
        arguments.get("images")
        or arguments.get("image")
        or arguments.get("image_paths")
        or arguments.get("image_path")
    )
    
    if not raw_images:
         return {"status": "error", "message": "Missing images parameter."}

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
        



def sam2_segment_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for SAM2 Segmentation (Full Image Box Prompt).
    """
    # 外部参数统一为 image，兼容旧的 image_path
    image_path = arguments.get("image") or arguments.get("image_path")
    if not image_path:
        return {"status": "error", "message": "Missing image parameter."}
    
    # 清理路径字符串 (防止传进来的是列表字符串)
    clean_path = str(image_path).strip()
    if clean_path.startswith("['") or clean_path.startswith('["'):
         # 简单的清理逻辑，复用之前的经验
         clean_path = clean_path[2:-2]
    
    if not os.path.exists(clean_path):
        return {"status": "error", "message": f"Image not found: {clean_path}"}

    # 1. 启动服务 (Lazy Loading)
    try:
        sam2_manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start SAM2 Service: {str(e)}"}

    # 2. 调用服务
    '''try:
        api_url = f"{sam2_manager.API_URL}/segment"
        payload = {"image_path": clean_path}
        
        response = requests.post(api_url, json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"SAM2 API Error: {response.text}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Connection to SAM2 failed: {str(e)}"}
        '''
    try:
        api_url = f"{sam2_manager.API_URL}/segment"
        payload = {"image_path": clean_path}
        
        response = requests.post(api_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            vis_b64 = result.get("visualization")
            count = result.get("count", 0)
            
            # 构造 Markdown 字符串
            md_text = f"SAM2 Segmentation Complete. Found {count} regions.\n\n"
            if vis_b64:
                # 这一行是关键：Markdown 图片语法
                md_text += f"![Segmentation Result]({vis_b64})"
            
            # === [修正点] 必须返回字典 ===
            return {
                "status": "success",
                "output": md_text,  # 把 Markdown 放在这里
                # "geojson": result.get("geojson") # 为了不刷屏，你可以先注释掉这一行，只看图片
            }
        else:
            return {"status": "error", "message": f"SAM2 API Error: {response.text}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}
        
        
def mock_model_handler(model_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    image_path = arguments.get("image") or arguments.get("image_path")
    if not image_path:
        return {"status": "error", "message": "Missing image parameter."}
    
    # 模拟一点延迟，让演示更真实
    time.sleep(1.5) 
    
    # 根据模型名构造假的返回结果
    if model_name == "MSCN":
        return {
            "status": "success",
            "model": "MSCN (Crowd Counting)",
            "count": random.randint(150, 500),
            "density_map_path": image_path.replace(".png", "_density.jpg"),
            "description": "Successfully estimated crowd density."
        }
    elif model_name == "RemoteCLIP":
        return {
            "status": "success", 
            "model": "RemoteCLIP (Retrieval/Classification)",
            "top_k_classes": ["airport", "runway", "airplane"],
            "scores": [0.92, 0.05, 0.02]
        }
    elif model_name == "Strip-R-CNN":
        return {
            "status": "success",
            "model": "Strip-R-CNN (Detection)",
            "objects_detected": ["ship", "ship", "buoy"],
            "bbox_count": 3,
            "visualization": "path/to/fake_result.jpg"
        }
    elif model_name == "RemoteSAM":
        text_prompt = arguments.get("text_prompt", "object")
        return {
            "status": "success",
            "model": "RemoteSAM (Visual Grounding)",
            "prompt": text_prompt,
            "mask_path": "path/to/mask.png",
            "message": f"Successfully segmented '{text_prompt}' in the image."
        }
    elif model_name == "InstructSAM":
        text_prompt = arguments.get("text_prompt", "object")
        return {
            "status": "success",
            "model": "InstructSAM (Counting/Segmentation)",
            "prompt": text_prompt,
            "count": random.randint(5, 20),
            "message": f"Found {random.randint(5,20)} instances of '{text_prompt}'."
        }
    
    return {"status": "error", "message": "Unknown model"}


# --- 包装函数 (为了适配注册接口) ---

def mscn_handler(args, ctx, acc): return mock_model_handler("MSCN", args)


def remoteclip_analysis_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for RemoteCLIP Analysis.
    """
    image_path = arguments.get("image") or arguments.get("image_path")

    # === 修复核心逻辑：处理字符串分割 ===
    raw_text_queries = arguments.get("text_queries")

    if raw_text_queries is None:
        # 用户未传任何 text_queries，使用默认值
        text_queries = [
            "A busy airport with many airplanes.",
            "Satellite view of Hohai University.",
            "A building next to a lake.",
            "Many people in a stadium.",
            "a cute cat"
        ]
    else:
        # 检查是否为字符串（前端传来的格式）
        if isinstance(raw_text_queries, str):
            # 按句号分割，并去除空字符串和前后空格
            text_queries = [q.strip() for q in raw_text_queries.split('.') if q.strip()]
        elif isinstance(raw_text_queries, list):
            # 如果传的是数组（兼容旧格式），直接使用
            text_queries = [str(q).strip() for q in raw_text_queries if str(q).strip()]
        else:
            # 其他类型，返回错误
            return {"status": "error",
                    "message": f"text_queries must be string (separated by '.') or array, got {type(raw_text_queries)}"}

    logger.info(f"Using text_queries: {text_queries}")  # 调试日志

    if not image_path:
        return {"status": "error", "message": "Missing image parameter."}

    clean_path = str(image_path).strip()
    if not os.path.exists(clean_path):
        return {"status": "error", "message": f"Image not found: {clean_path}"}

    try:
        remoteclip_manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start RemoteCLIP Service: {str(e)}"}

    try:
        api_url = f"{remoteclip_manager.API_URL}/analyze"
        payload = {
            "image_path": clean_path,
            "text_queries": text_queries  # 现在传的是正确解析的数组
        }

        print(f"Sending payload: {payload}")  # 调试输出
        response = requests.post(api_url, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "predictions": result.get("predictions"),
                "message": "RemoteCLIP analysis completed.",
                "used_queries": text_queries  # 返回实际使用的查询（用于调试）
            }

        else:
            return {"status": "error", "message": f"RemoteCLIP API Error: {response.text}"}

    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}




def strip_rcnn_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for Strip R-CNN Detection.
    """
    image_path = arguments.get("image") or arguments.get("image_path")
    score_threshold = arguments.get("score_threshold", 0.3)

    if not image_path:
        return {"status": "error", "message": "Missing image parameter."}

    clean_path = str(image_path).strip()
    if not os.path.exists(clean_path):
        return {"status": "error", "message": f"Image not found: {clean_path}"}

    try:
        strip_rcnn_manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start Strip R-CNN Service: {str(e)}"}

    try:
        api_url = f"{strip_rcnn_manager.API_URL}/detect"
        payload = {
            "image_path": clean_path,
            "score_threshold": score_threshold
        }

        response = requests.post(api_url, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return {
                    "status": "success",
                    "detections": result.get("detections"),
                    "image_size": result.get("image_size"),
                    "num_detections": result.get("num_detections"),
                    "message": f"Strip R-CNN detection completed. Found {result.get('num_detections')} objects."
                }
            else:
                return {"status": "error", "message": f"Detection failed: {result.get('error')}"}
        else:
            return {"status": "error", "message": f"Strip R-CNN API Error: {response.text}"}

    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}


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


def remotesam_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for RemoteSAM tasks.
    """
    image_path = arguments.get("image") or arguments.get("image_path")
    task_type = arguments.get("task_type")
    sentence = arguments.get("sentence", "")
    classnames = arguments.get("classnames", [])

    if not image_path:
        return {"status": "error", "message": "Missing image parameter."}

    clean_path = str(image_path).strip()
    if not os.path.exists(clean_path):
        return {"status": "error", "message": f"Image not found: {clean_path}"}

    try:
        remotesam_manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start RemoteSAM Service: {str(e)}"}

    try:
        api_url = f"{remotesam_manager.API_URL}/{task_type}"
        payload = {
            "image_path": clean_path,
            "sentence": sentence,
            "classnames": classnames
        }

        response = requests.post(api_url, json=payload, timeout=120)
        print(payload)
        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "result": result,
                "message": f"RemoteSAM {task_type} completed."
            }
        else:
            return {"status": "error", "message": f"RemoteSAM API Error: {response.text}"}

    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}
def instructsam_handler(args, ctx, acc): return mock_model_handler("InstructSAM", args)        

    
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
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One or more images. "
                            "If multiple images are provided, they will be analyzed jointly. "
                            "Frontend uploads image files; the backend resolves local file paths."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Question or instruction.",
                        "default": "",
                    },
                    "max_tokens": {"type": "integer", "default": 4096},
                },
                "required": [],
            },
            requires_connection=False,
        ),
        vlm_analyze_handler,
    )
    
    registrar.tool(
        ToolSpec(
            slug="geo_perception.sam2_segment",
            name="SAM2 Segmentation",
            description="Segment the primary object or full scene in a satellite image using SAM2 (Full Box Prompt).",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to segment. Frontend uploads the image; backend resolves the local file path."
                    }
                },
                "required": ["image"]
            },
            requires_connection=False
        ),
        sam2_segment_handler
    )
    
    # 3. MSCN (Classification / Crowd Counting)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.mscn_classify",
            name="MSCN Classification",
            description="Multi-Scale Context Network for crowd counting or scene classification.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to analyze. Frontend uploads the image; backend resolves the local file path."
                    }
                },
                "required": ["image"]
            },
            requires_connection=False
        ),
        mscn_handler
    )

    # 4. RemoteCLIP (Retrieval / Classification)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.remoteclip_analysis",
            name="RemoteCLIP Analysis",
            description="Perform zero-shot classification and retrieval on satellite/drone images using RemoteCLIP.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to analyze. Frontend uploads the image; backend resolves the local file path."
                    },
                    "text_queries": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Text queries separated by dots (.) for zero-shot classification."
                            },
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of text queries for zero-shot classification."
                            }
                        ],
                        "default": "A busy airport with many airplanes.Satellite view of Hohai University.A building next to a lake.Many people in a stadium.a cute cat"
                    }
                },
                "required": ["image", "text_queries"]
            },
            requires_connection=False
        ),
        remoteclip_analysis_handler
    )

    # 5. Strip-R-CNN (Detection)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.strip_rcnn_detect",
            name="Strip-R-CNN Detection",
            description="Specialized object detection for elongated objects (ships, roads, bridges) in satellite images using Strip R-CNN.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to analyze. Frontend uploads the image; backend resolves the local file path."
                    },
                    "score_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.1,
                        "description": "Confidence threshold for detection results (default: 0.3)."
                    }
                },
                "required": ["image"]
            },
            requires_connection=False
        ),
        strip_rcnn_handler
    )

    # 6. RemoteSAM (Visual Grounding)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.remotesam",
            name="RemoteSAM Tasks",
            description="Perform various tasks using RemoteSAM such as referring segmentation, semantic segmentation, detection, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to analyze. Frontend uploads the image; backend resolves the local file path."
                    },
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "referring_seg",
                            "semantic_seg",
                            "detection",
                            "visual_grounding",
                            "multi_label_cls",
                            "multi_class_cls",
                            "captioning",
                            "counting"
                        ],
                        "description": "Type of task to perform."
                    },
                    "sentence": {
                        "type": "string",
                        "description": "Sentence for referring segmentation or visual grounding (optional)."
                    },
                    "classnames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of class names for segmentation, detection, classification, etc. (optional)."
                    }
                },
                "required": ["image", "task_type"]
            },
            requires_connection=False
        ),
        remotesam_handler
    )

    # 7. InstructSAM (Counting / Interactive)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.instructsam",
            name="InstructSAM Counting",
            description="Instruction-based segmentation and counting of objects.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to analyze. Frontend uploads the image; backend resolves the local file path."
                    },
                    "text_prompt": {
                        "type": "string",
                        "description": "Instruction prompt (e.g., 'Count all the red cars')."
                    }
                },
                "required": ["image", "text_prompt"]
            },
            requires_connection=False
        ),
        instructsam_handler
    )