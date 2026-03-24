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




# --- Bbox / Annotation / Mock Handlers ---

def draw_bboxes_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Draw labeled bounding boxes on an image using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Missing Pillow. Install: pip install Pillow")

    image_path = _resolve_image_path(arguments)
    bboxes = arguments.get("bboxes", [])
    output_path = arguments["output_path"]
    line_width = int(arguments.get("line_width", 2))

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for bbox in bboxes:
        x1, y1, x2, y2 = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
        color = bbox.get("color", "red")
        label = bbox.get("label", "")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if label:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
            draw.text((x1 + 2, y1 + 2), label, fill=color, font=font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    return {"status": "success", "output_path": output_path, "boxes_drawn": len(bboxes)}


def add_text_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Add text annotations to an image at specified positions."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Missing Pillow. Install: pip install Pillow")

    image_path = _resolve_image_path(arguments)
    annotations = arguments.get("annotations", [])
    output_path = arguments["output_path"]

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ann in annotations:
        text = str(ann["text"])
        x, y = float(ann["x"]), float(ann["y"])
        color = ann.get("color", "white")
        font_size = int(ann.get("font_size", 16))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x, y), text, fill=color, font=font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    return {"status": "success", "output_path": output_path, "annotations_added": len(annotations)}


def ocr_extract_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Extract text from an image using EasyOCR."""
    try:
        import easyocr
    except ImportError:
        raise ImportError("Missing EasyOCR. Install: pip install easyocr")

    image_path = _resolve_image_path(arguments)
    languages = arguments.get("languages", ["en"])
    if isinstance(languages, str):
        languages = [languages]

    reader = easyocr.Reader(languages, gpu=True)
    results = reader.readtext(image_path)

    texts = []
    for (bbox_coords, text, confidence) in results:
        texts.append({
            "text": text,
            "confidence": round(float(confidence), 3),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox_coords],
        })

    return {
        "status": "success",
        "texts": texts,
        "count": len(texts),
        "full_text": " ".join(t["text"] for t in texts),
    }


def bbox_expand_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Expand bounding boxes by a spatial buffer radius."""
    bboxes = arguments.get("bboxes", [])
    radius_px = arguments.get("radius_px")
    radius_m = arguments.get("radius_m")
    gsd_m = arguments.get("gsd_m")

    if radius_m is not None and gsd_m is not None:
        r = float(radius_m) / float(gsd_m)
    elif radius_px is not None:
        r = float(radius_px)
    else:
        raise ValueError("Provide 'radius_px' or both 'radius_m' and 'gsd_m'")

    expanded = []
    for bb in bboxes:
        expanded.append({
            "x1": float(bb["x1"]) - r,
            "y1": float(bb["y1"]) - r,
            "x2": float(bb["x2"]) + r,
            "y2": float(bb["y2"]) + r,
            **{k: v for k, v in bb.items() if k not in ("x1", "y1", "x2", "y2")}
        })

    return {"expanded_bboxes": expanded, "radius_px": r, "count": len(expanded)}


def bbox_to_centroid_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Convert bounding boxes to centroid coordinates."""
    bboxes = arguments.get("bboxes", [])
    centroids = []
    for bb in bboxes:
        cx = (float(bb["x1"]) + float(bb["x2"])) / 2.0
        cy = (float(bb["y1"]) + float(bb["y2"])) / 2.0
        centroids.append({"x": cx, "y": cy})
    return {"centroids": centroids, "count": len(centroids)}


def centroid_distance_extremes_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute minimum and maximum pairwise Euclidean distances between centroids."""
    import math
    centroids = arguments.get("centroids", [])
    if len(centroids) < 2:
        return {"error": "Need at least 2 centroids to compute pairwise distances"}

    min_dist = float("inf")
    max_dist = 0.0
    min_pair = None
    max_pair = None

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = centroids[i]["x"] - centroids[j]["x"]
            dy = centroids[i]["y"] - centroids[j]["y"]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
            if d > max_dist:
                max_dist = d
                max_pair = (i, j)

    return {
        "min_distance": round(min_dist, 3),
        "max_distance": round(max_dist, 3),
        "min_pair_indices": list(min_pair),
        "max_pair_indices": list(max_pair),
        "centroid_count": len(centroids),
    }


def bbox_area_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Calculate the total area of bounding boxes in pixels² or m²."""
    bboxes = arguments.get("bboxes", [])
    gsd_m = arguments.get("gsd_m")

    individual_areas = []
    for bb in bboxes:
        w = abs(float(bb["x2"]) - float(bb["x1"]))
        h = abs(float(bb["y2"]) - float(bb["y1"]))
        area_px2 = w * h
        individual_areas.append(area_px2)

    total_px2 = sum(individual_areas)

    result = {"total_area_px2": total_px2, "per_bbox_area_px2": individual_areas, "count": len(bboxes)}
    if gsd_m is not None:
        pixel_area_m2 = float(gsd_m) ** 2
        result["total_area_m2"] = total_px2 * pixel_area_m2
        result["per_bbox_area_m2"] = [a * pixel_area_m2 for a in individual_areas]
    return result


# --- Mock Handlers (placeholder until models are deployed) ---

def mscn_classify_mock_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """[Mock] MSCN scene classification - returns stub result."""
    image_path = _resolve_image_path(arguments)
    return {
        "status": "mock",
        "note": "MSCN model not yet deployed. This is a placeholder response.",
        "image": image_path,
        "predicted_class": "Unknown",
        "confidence": 0.0,
        "categories": [
            "Airport", "Beach", "Bridge", "Commercial", "Desert",
            "Farmland", "Forest", "Industrial", "Meadow", "Mountain",
            "Park", "Parking", "Port", "Railway", "Residential",
            "River", "Runway", "Stadium", "Storage_tank", "Urban"
        ]
    }


def sm3det_detect_mock_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """[Mock] SM3Det multi-category detection - returns stub result."""
    image_path = _resolve_image_path(arguments)
    return {
        "status": "mock",
        "note": "SM3Det model not yet deployed. This is a placeholder response.",
        "image": image_path,
        "detections": [],
        "num_detections": 0,
        "categories": [
            "airplane", "ship", "storage_tank", "baseball_diamond",
            "tennis_court", "basketball_court", "ground_track_field",
            "harbor", "bridge", "large_vehicle", "small_vehicle",
            "helicopter", "roundabout", "soccer_ball_field", "swimming_pool"
        ]
    }


def change_os_detect_mock_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """[Mock] ChangeOS change detection / building extraction - returns stub result."""
    pre_path = arguments.get("pre_image") or arguments.get("image")
    mode = arguments.get("mode", "change_detection")
    return {
        "status": "mock",
        "note": "ChangeOS model not yet deployed. This is a placeholder response.",
        "mode": mode,
        "pre_image": pre_path,
        "post_image": arguments.get("post_image"),
        "change_mask": None,
        "changed_pixels": 0,
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

    # 8. Draw Bounding Boxes
    registrar.tool(
        ToolSpec(
            slug="geo_perception.draw_bboxes",
            name="Draw Bounding Boxes",
            description="Draw labeled bounding boxes on an image and save the result. Useful for visualizing detection results.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the input image."},
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "number"}, "y1": {"type": "number"},
                                "x2": {"type": "number"}, "y2": {"type": "number"},
                                "label": {"type": "string"},
                                "color": {"type": "string", "default": "red"}
                            },
                            "required": ["x1", "y1", "x2", "y2"]
                        },
                        "description": "List of bounding boxes with coordinates and optional label/color."
                    },
                    "output_path": {"type": "string", "description": "Path to save the annotated image."},
                    "line_width": {"type": "integer", "default": 2, "description": "Width of the bounding box lines."}
                },
                "required": ["image", "bboxes", "output_path"]
            },
            requires_connection=False
        ),
        draw_bboxes_handler
    )

    # 9. Add Text Annotation
    registrar.tool(
        ToolSpec(
            slug="geo_perception.add_text",
            name="Add Text to Image",
            description="Add text annotations to an image at specified positions. Useful for labeling analysis results.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the input image."},
                    "annotations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "x": {"type": "number", "description": "X position (pixels from left)."},
                                "y": {"type": "number", "description": "Y position (pixels from top)."},
                                "color": {"type": "string", "default": "white"},
                                "font_size": {"type": "integer", "default": 16}
                            },
                            "required": ["text", "x", "y"]
                        },
                        "description": "List of text annotations with position and style."
                    },
                    "output_path": {"type": "string", "description": "Path to save the annotated image."}
                },
                "required": ["image", "annotations", "output_path"]
            },
            requires_connection=False
        ),
        add_text_handler
    )

    # 10. OCR
    registrar.tool(
        ToolSpec(
            slug="geo_perception.ocr_extract",
            name="OCR Text Extraction",
            description="Extract text from an image using EasyOCR. Supports Chinese and English text. Useful for reading labels, legends, or text in satellite imagery.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the input image."},
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["en"],
                        "description": "List of language codes to recognize (e.g., ['en', 'ch_sim'] for English + Simplified Chinese)."
                    }
                },
                "required": ["image"]
            },
            requires_connection=False
        ),
        ocr_extract_handler
    )

    # 11. BBox Expand (spatial buffer)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.bbox_expand",
            name="Expand Bounding Boxes",
            description="Expand bounding boxes by a spatial buffer radius. Optionally provide GSD (meters/pixel) to specify radius in meters.",
            parameters={
                "type": "object",
                "properties": {
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "number"}, "y1": {"type": "number"},
                                "x2": {"type": "number"}, "y2": {"type": "number"}
                            },
                            "required": ["x1", "y1", "x2", "y2"]
                        },
                        "description": "List of bounding boxes to expand."
                    },
                    "radius_px": {"type": "number", "description": "Buffer radius in pixels."},
                    "radius_m": {"type": "number", "description": "Buffer radius in meters (requires gsd_m)."},
                    "gsd_m": {"type": "number", "description": "Ground sampling distance in meters/pixel (required when using radius_m)."}
                },
                "required": ["bboxes"]
            },
            requires_connection=False
        ),
        bbox_expand_handler
    )

    # 12. BBox to Centroid
    registrar.tool(
        ToolSpec(
            slug="geo_perception.bbox_to_centroid",
            name="Bounding Boxes to Centroids",
            description="Convert a list of bounding boxes to their centroid coordinates.",
            parameters={
                "type": "object",
                "properties": {
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "number"}, "y1": {"type": "number"},
                                "x2": {"type": "number"}, "y2": {"type": "number"}
                            },
                            "required": ["x1", "y1", "x2", "y2"]
                        },
                        "description": "List of bounding boxes."
                    }
                },
                "required": ["bboxes"]
            },
            requires_connection=False
        ),
        bbox_to_centroid_handler
    )

    # 13. Centroid Distance Extremes
    registrar.tool(
        ToolSpec(
            slug="geo_perception.centroid_distance_extremes",
            name="Centroid Distance Extremes",
            description="Compute minimum and maximum pairwise distances between a set of centroid points.",
            parameters={
                "type": "object",
                "properties": {
                    "centroids": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"}, "y": {"type": "number"}
                            },
                            "required": ["x", "y"]
                        },
                        "description": "List of centroid points with x and y coordinates."
                    }
                },
                "required": ["centroids"]
            },
            requires_connection=False
        ),
        centroid_distance_extremes_handler
    )

    # 14. BBox Area
    registrar.tool(
        ToolSpec(
            slug="geo_perception.bbox_area",
            name="Calculate BBox Area",
            description="Calculate the total area of bounding boxes in pixels² or m² (when GSD is provided).",
            parameters={
                "type": "object",
                "properties": {
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "number"}, "y1": {"type": "number"},
                                "x2": {"type": "number"}, "y2": {"type": "number"}
                            },
                            "required": ["x1", "y1", "x2", "y2"]
                        },
                        "description": "List of bounding boxes."
                    },
                    "gsd_m": {"type": "number", "description": "Ground sampling distance in meters/pixel. If provided, area is returned in m²."}
                },
                "required": ["bboxes"]
            },
            requires_connection=False
        ),
        bbox_area_handler
    )

    # 15. MSCN Classify (Mock)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.mscn_classify",
            name="MSCN Scene Classification",
            description="[Mock] Classify a remote sensing image into one of 30 scene categories (Airport, Forest, Urban, Farmland, etc.) using MSCN scene classifier. NOTE: Returns mock results; model not yet deployed.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the input image."}
                },
                "required": ["image"]
            },
            requires_connection=False
        ),
        mscn_classify_mock_handler
    )

    # 16. SM3Det (Mock)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.sm3det_detect",
            name="SM3Det Multi-Category Detection",
            description="[Mock] Detect objects in 15 categories (aircraft, ships, vehicles, buildings, fields, etc.) using SM3Det. NOTE: Returns mock results; model not yet deployed.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the input image."},
                    "score_threshold": {"type": "number", "default": 0.3, "description": "Confidence threshold."}
                },
                "required": ["image"]
            },
            requires_connection=False
        ),
        sm3det_detect_mock_handler
    )

    # 17. ChangeOS (Mock)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.change_os_detect",
            name="ChangeOS Change Detection",
            description="[Mock] Detect changes between two multi-temporal satellite images or extract building footprints using ChangeOS. NOTE: Returns mock results; model not yet deployed.",
            parameters={
                "type": "object",
                "properties": {
                    "pre_image": {"type": "string", "description": "Path to the pre-event image (or single image for building extraction)."},
                    "post_image": {"type": "string", "description": "Path to the post-event image (optional for building extraction mode)."},
                    "mode": {
                        "type": "string",
                        "enum": ["change_detection", "building_extraction"],
                        "default": "change_detection",
                        "description": "'change_detection': detect changes between two images. 'building_extraction': extract building footprints from a single image."
                    }
                },
                "required": ["pre_image"]
            },
            requires_connection=False
        ),
        change_os_detect_mock_handler
    )