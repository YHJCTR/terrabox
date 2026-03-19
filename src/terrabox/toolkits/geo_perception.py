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

from ..managers import (
    vllm_manager,
    sam2_manager,
    remoteclip_manager,
    remotesam_manager,
    strip_rcnn_manager,
    instructsam_manager,
)

logger = logging.getLogger(__name__)

# --- Helpers ---

def _get_image_path(arguments: dict) -> str:
    """Extract image path from arguments, trying multiple key names."""
    path = arguments.get("image") or arguments.get("image_path")
    if not path:
        raise ValueError("Missing required parameter: image path (provide 'image' or 'image_path')")
    return str(path).strip()


def _resolve_image_path(arguments: dict) -> str:
    """Extract and validate that the image file exists."""
    path = _get_image_path(arguments)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


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


def _call_service(manager, url: str, payload: dict, timeout: int = 120) -> dict:
    """Start *manager* if not running, POST to *url*, return parsed JSON or error dict."""
    try:
        manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start service: {e}"}
    try:
        resp = requests.post(url, json=payload, timeout=timeout, proxies={"http": None, "https": None})
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"API error {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {e}"}


# --- Handlers ---

def vlm_analyze_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Handler for VLM multi-image analysis."""
    raw_images = (
        arguments.get("images")
        or arguments.get("image")
        or arguments.get("image_paths")
        or arguments.get("image_path")
    )
    
    if not raw_images:
         return {"status": "error", "message": "Missing images parameter."}

    logger.debug(f"raw_images: type={type(raw_images)} value={raw_images}")

    image_paths = []

    if isinstance(raw_images, list):
        # Handle list-wrapping-a-JSON-string: e.g. ['["/path/a", "/path/b"]']
        if len(raw_images) == 1 and isinstance(raw_images[0], str):
            s = raw_images[0].strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    image_paths = json.loads(s)
                    logger.info("Unwrapped nested JSON string inside list.")
                except:
                    image_paths = raw_images
            else:
                image_paths = raw_images
        else:
            image_paths = raw_images

    elif isinstance(raw_images, str):
        s = raw_images.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                image_paths = json.loads(s)
            except:
                try:
                    image_paths = ast.literal_eval(s)
                except:
                    # Fallback: split the bracketed string manually
                    content = s[1:-1]
                    parts = content.split(',')
                    for p in parts:
                        clean_p = p.strip().strip('"').strip("'")
                        if clean_p:
                            image_paths.append(clean_p)
        else:
            image_paths = [s]

    logger.debug(f"parsed image_paths: {image_paths}")

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
            if not os.path.exists(clean_path):
                logger.error(f"File not found: {clean_path}")
                return {"status": "error", "message": f"File not found on server: {clean_path}"}
                
            b64_str = _encode_image_to_base64(clean_path)
            message_content.append({
                "type": "image_url", 
                "image_url": {"url": b64_str}
            })
    except Exception as e:
        return {"status": "error", "message": f"Image error: {str(e)}"}

    api_url = f"{vllm_manager.API_BASE}/chat/completions"
    payload = {
        "model": getattr(vllm_manager, "MODEL_NAME", vllm_manager.MODEL_PATH),
        "messages": [{"role": "user", "content": message_content}],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
            proxies={"http": None, "https": None},
        )
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
    """Handler for SAM2 Segmentation (Full Image Box Prompt)."""
    clean_path = _resolve_image_path(arguments)

    result = _call_service(sam2_manager, f"{sam2_manager.API_URL}/segment", {"image_path": clean_path}, timeout=60)
    if result.get("status") == "error":
        return result

    vis_b64 = result.get("visualization")
    count = result.get("count", 0)
    md_text = f"SAM2 Segmentation Complete. Found {count} regions.\n\n"
    if vis_b64:
        md_text += f"![Segmentation Result]({vis_b64})"
    return {"status": "success", "output": md_text}
        
        
# --- Wrappers ---

def remoteclip_analysis_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for RemoteCLIP Analysis.
    """
    raw_text_queries = arguments.get("text_queries")

    if raw_text_queries is None:
        text_queries = [
            "A busy airport with many airplanes.",
            "Satellite view of Hohai University.",
            "A building next to a lake.",
            "Many people in a stadium.",
            "a cute cat"
        ]
    else:
        if isinstance(raw_text_queries, str):
            # Split on '.' and drop empty segments
            text_queries = [q.strip() for q in raw_text_queries.split('.') if q.strip()]
        elif isinstance(raw_text_queries, list):
            text_queries = [str(q).strip() for q in raw_text_queries if str(q).strip()]
        else:
            return {"status": "error",
                    "message": f"text_queries must be string (separated by '.') or array, got {type(raw_text_queries)}"}

    logger.info(f"Using text_queries: {text_queries}")

    clean_path = _resolve_image_path(arguments)

    result = _call_service(
        remoteclip_manager, f"{remoteclip_manager.API_URL}/analyze",
        {"image_path": clean_path, "text_queries": text_queries},
    )
    if result.get("status") == "error":
        return result
    return {
        "status": "success",
        "predictions": result.get("predictions"),
        "message": "RemoteCLIP analysis completed.",
        "used_queries": text_queries,
    }




def strip_rcnn_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Handler for Strip R-CNN Detection.
    """
    clean_path = _resolve_image_path(arguments)
    score_threshold = arguments.get("score_threshold", 0.3)

    result = _call_service(
        strip_rcnn_manager, f"{strip_rcnn_manager.API_URL}/detect",
        {"image_path": clean_path, "score_threshold": score_threshold},
    )
    if result.get("status") == "error":
        return result
    if result.get("success"):
        return {
            "status": "success",
            "detections": result.get("detections"),
            "image_size": result.get("image_size"),
            "num_detections": result.get("num_detections"),
            "message": f"Strip R-CNN detection completed. Found {result.get('num_detections')} objects.",
        }
    return {"status": "error", "message": f"Detection failed: {result.get('error')}"}


def remotesam_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Handler for RemoteSAM tasks."""
    clean_path = _resolve_image_path(arguments)
    task_type = arguments.get("task_type")

    result = _call_service(
        remotesam_manager, f"{remotesam_manager.API_URL}/{task_type}",
        {"image_path": clean_path, "sentence": arguments.get("sentence", ""), "classnames": arguments.get("classnames", [])},
    )
    if result.get("status") == "error":
        return result
    return {"status": "success", "result": result, "message": f"RemoteSAM {task_type} completed."}
def instructsam_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Handler for InstructSAM — instruction-based segmentation and counting."""
    clean_path = _resolve_image_path(arguments)
    text_prompt = arguments.get("text_prompt", "objects in the image")

    result = _call_service(
        instructsam_manager, f"{instructsam_manager.API_URL}/segment",
        {"image_path": clean_path, "text_prompt": text_prompt},
        timeout=300,
    )
    if result.get("status") == "error":
        return result

    count = result.get("count", 0)
    vis_b64 = result.get("visualization", "")
    md_text = f"InstructSAM found **{count}** object(s) matching '{text_prompt}'.\n\n"
    if vis_b64:
        md_text += f"![InstructSAM Result]({vis_b64})"
    return {
        "status": "success",
        "count": count,
        "objects": result.get("objects", []),
        "detections": result.get("detections", []),
        "output": md_text,
    }


    
# --- Tool Registration ---

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
                    "max_tokens": {"type": "integer", "default": 512},
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
    
    # 3. RemoteCLIP (Retrieval / Classification)
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