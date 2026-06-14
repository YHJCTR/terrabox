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
import re
import subprocess
import sys
import time
import uuid
import requests
from pathlib import Path
from typing import Any, Dict, List
from ..core.registry import ToolSpec

from ..managers import (
    vllm_manager,
    sam2_manager,
    remoteclip_manager,
    remotesam_manager,
    strip_rcnn_manager,
    instructsam_manager,
    changeos_manager,
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
    return os.path.abspath(path)


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_devices(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _easyocr_gpu_setting() -> bool | str:
    """Pick the OCR device without stealing the agent LLM GPU in experiments."""
    if not _env_bool("TERRABOX_OCR_USE_GPU", True):
        return False

    explicit = os.environ.get("TERRABOX_OCR_GPU_DEVICE", "").strip()
    if explicit:
        return explicit

    tool_devices = _split_devices(os.environ.get("TERRABOX_TOOL_GPU_DEVICES", ""))
    if not tool_devices:
        return True

    target = tool_devices[0]
    visible = _split_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if visible:
        try:
            return f"cuda:{visible.index(target)}"
        except ValueError:
            pass
    return f"cuda:{target}"


def _max_perception_long_side() -> int:
    raw = os.environ.get("TERRABOX_PERCEPTION_MAX_IMAGE_LONG_SIDE", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _prepare_perception_image(image_path: str, context: Any | None = None) -> tuple[str, dict[str, Any]]:
    """Optionally resize large inputs for memory-heavy perception services."""
    max_long_side = _max_perception_long_side()
    metadata = {
        "resized": False,
        "original_path": image_path,
        "input_path": image_path,
        "max_long_side": max_long_side or None,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    if max_long_side <= 0:
        return image_path, metadata

    try:
        from PIL import Image
    except ImportError:
        return image_path, metadata

    with Image.open(image_path) as img:
        width, height = img.size
        long_side = max(width, height)
        metadata["original_size"] = [width, height]
        if long_side <= max_long_side:
            metadata["resized_size"] = [width, height]
            return image_path, metadata

        scale = max_long_side / float(long_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        if isinstance(context, dict) and context.get("artifact_dir"):
            out_dir = Path(context["artifact_dir"]) / "resized_inputs"
        else:
            out_dir = Path("tmp") / "terrabox_runtime" / "resized_inputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image_path).suffix or ".jpg"
        resized_path = out_dir / f"{Path(image_path).stem}_{max_long_side}_{uuid.uuid4().hex[:8]}{suffix}"

        resized = img.convert("RGB") if img.mode not in {"RGB", "L"} else img.copy()
        resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized.save(resized_path)

    metadata.update({
        "resized": True,
        "input_path": str(resized_path),
        "resized_size": [new_width, new_height],
        "scale_x": new_width / float(width),
        "scale_y": new_height / float(height),
    })
    return str(resized_path), metadata


def _scale_bbox_to_original(bbox: Any, preprocessing: dict[str, Any]) -> Any:
    if not preprocessing.get("resized") or not isinstance(bbox, list) or len(bbox) < 4:
        return bbox
    try:
        scale_x = float(preprocessing.get("scale_x") or 1.0)
        scale_y = float(preprocessing.get("scale_y") or 1.0)
        if scale_x <= 0 or scale_y <= 0:
            return bbox
        scaled = list(bbox)
        scaled[0] = round(float(scaled[0]) / scale_x, 3)
        scaled[1] = round(float(scaled[1]) / scale_y, 3)
        scaled[2] = round(float(scaled[2]) / scale_x, 3)
        scaled[3] = round(float(scaled[3]) / scale_y, 3)
        return scaled
    except Exception:
        return bbox


def _scale_detection_bboxes_to_original(items: Any, preprocessing: dict[str, Any]) -> Any:
    if not preprocessing.get("resized") or not isinstance(items, list):
        return items
    scaled_items = []
    for item in items:
        if isinstance(item, dict):
            new_item = dict(item)
            for key in ("bbox", "box"):
                if key in new_item:
                    new_item[key] = _scale_bbox_to_original(new_item[key], preprocessing)
            scaled_items.append(new_item)
        else:
            scaled_items.append(item)
    return scaled_items


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


def _reduced_vlm_output_budget(error_text: str) -> int | None:
    """Parse vLLM context errors and return a safe retry output budget."""
    max_match = re.search(r"maximum context length is\s+(\d+)", error_text)
    input_match = re.search(r"request has\s+(\d+)\s+input tokens", error_text)
    if not max_match or not input_match:
        return None
    max_context = int(max_match.group(1))
    input_tokens = int(input_match.group(1))
    retry_budget = max_context - input_tokens - 128
    if retry_budget <= 0:
        return None
    return max(1, retry_budget)


def _call_service(manager, url: str, payload: dict, timeout: int = 120) -> dict:
    """Start *manager* if not running, POST to *url*, return parsed JSON or error dict."""
    try:
        manager.start_service()
    except Exception as e:
        if _is_cuda_oom_text(str(e)):
            _stop_tool_service_after_call(manager)
            return _tool_oom_response(str(e))
        return {"status": "error", "message": f"Failed to start service: {e}"}
    try:
        resp = requests.post(url, json=payload, timeout=timeout, proxies={"http": None, "https": None})
        if resp.status_code == 200:
            return resp.json()
        if _is_cuda_oom_text(resp.text):
            return _tool_oom_response(f"API error {resp.status_code}: {resp.text}")
        return {"status": "error", "message": f"API error {resp.status_code}: {resp.text}"}
    except Exception as e:
        if _is_cuda_oom_text(str(e)):
            return _tool_oom_response(str(e))
        return {"status": "error", "message": f"Connection failed: {e}"}
    finally:
        _stop_tool_service_after_call(manager)


def _is_cuda_oom_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "outofmemory" in lowered
        or "out of memory" in lowered
        or "cuda oom" in lowered
        or "cuda error" in lowered and "memory" in lowered
    )


def _tool_oom_response(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": "tool_oom",
        "retryable": False,
        "message": message,
        "recovery_suggestions": [
            "Do not repeat the same tool call with the same arguments.",
            "Stop tool use for this task if no smaller image or cheaper tool is available.",
            "Explain that the current task is blocked by a system GPU memory limit, not by the user's question.",
        ],
    }


def _normalize_remotesam_request(arguments: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_task_type = str(arguments.get("task_type") or "").strip().lower()
    raw_task_type = re.sub(r"[\s-]+", "_", raw_task_type)
    sentence = str(arguments.get("sentence") or "").strip()
    classnames = arguments.get("classnames") or []
    if isinstance(classnames, str):
        classnames = [classnames]
    classnames = [str(name).strip() for name in classnames if str(name).strip()]

    if raw_task_type in {"detect", "object_detection", "bbox", "bounding_box", "bounding_boxes"}:
        task_type = "detection"
    elif raw_task_type in {"segment", "segmentation", "referring_segmentation", "referring"}:
        task_type = "referring_seg" if sentence and not classnames else "semantic_seg"
    elif raw_task_type in {"semantic_segmentation", "semantic"}:
        task_type = "semantic_seg"
    elif raw_task_type in {"visual_ground", "grounding"}:
        task_type = "visual_grounding"
    else:
        task_type = raw_task_type

    if task_type == "detection" and not classnames and sentence:
        classnames = [sentence]

    return task_type, {
        "sentence": sentence,
        "classnames": classnames,
    }


def _stop_tool_service_after_call(manager) -> None:
    scope = os.environ.get("TERRABOX_TOOL_SERVICE_SCOPE", "").strip().lower()
    if scope not in {"call", "per-call", "tool-call"}:
        return
    # Keep the heavy VLM (Qwen3-VL) container warm when requested: it is slow to
    # load (~minutes) and reused by many instructsam/vlm_analyze calls. Other,
    # lighter tools still stop after each call so they don't pile up on the
    # shared tool GPU. The VLM has its own dedicated GPU(s), so it does not
    # contend with the cycling perception tools.
    if (
        manager is vllm_manager
        and os.environ.get("TERRABOX_KEEP_VLM_WARM", "").strip().lower() in {"1", "true", "yes", "on"}
    ):
        return
    try:
        manager.stop_service()
    except Exception as exc:
        logger.warning("Failed to stop call-scoped tool service %s: %s", manager, exc)


def _write_tool_artifact(tool_name: str, payload: dict, context: Any | None = None) -> str:
    """Persist large tool payloads and return a repo-local artifact path."""
    root = (
        context.get("artifact_dir")
        if isinstance(context, dict) and context.get("artifact_dir")
        else os.environ.get("TERRABOX_TOOL_ARTIFACT_DIR", "tmp/tool_artifacts")
    )
    tool_slug = tool_name.lower().replace(".", "_").replace("-", "_")
    out_dir = os.path.abspath(os.path.join(root, tool_slug))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    return out_path


def _array_shape(value: Any) -> list[int]:
    shape: list[int] = []
    cursor = value
    while isinstance(cursor, list):
        shape.append(len(cursor))
        cursor = cursor[0] if cursor else None
    return shape


def _count_positive(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_positive(item) for item in value)
    try:
        return int(float(value) > 0)
    except Exception:
        return 0


def _summarize_remotesam_result(result: Any) -> dict:
    if not isinstance(result, dict):
        return {"value": result}

    summary: dict[str, Any] = {}
    payload = result.get("result", result)
    if isinstance(payload, dict):
        for name, value in payload.items():
            if isinstance(value, list):
                summary[str(name)] = {
                    "shape": _array_shape(value),
                    "positive_pixels": _count_positive(value),
                }
            elif isinstance(value, dict):
                summary[str(name)] = {"keys": sorted(value.keys())}
            else:
                summary[str(name)] = value
    return summary


def _extract_remotesam_bboxes(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    if isinstance(result.get("bboxes"), list):
        return result["bboxes"]

    bboxes: list[dict[str, Any]] = []
    for label, boxes in result.items():
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, list) or len(box) < 4:
                continue
            item = {"label": str(label), "bbox": box[:4]}
            if len(box) > 4:
                item["score"] = box[4]
            bboxes.append(item)
    return bboxes


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
    max_tokens = arguments.get("max_tokens", 8192)
    try:
        max_tokens = int(max_tokens)
    except Exception:
        max_tokens = 8192
    
    try:
        vllm_manager.start_service()
    except Exception as e:
        return {"status": "error", "message": f"Failed to start AI Service: {str(e)}"}

    try:
        try:
            message_content = [{"type": "text", "text": prompt}]

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

        response = None
        for attempt in range(2):
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300,
                proxies={"http": None, "https": None},
            )
            if response.status_code == 200:
                break
            if attempt == 0 and response.status_code == 400:
                retry_budget = _reduced_vlm_output_budget(response.text)
                if retry_budget is not None and retry_budget < int(payload["max_tokens"]):
                    logger.warning(
                        "Retrying VLM request with max_tokens=%s after context error: %s",
                        retry_budget,
                        response.text,
                    )
                    payload["max_tokens"] = retry_budget
                    continue
            return {"status": "error", "message": f"API Error {response.status_code}: {response.text}"}

        if response is None or response.status_code != 200:
            return {"status": "error", "message": "VLM request failed before receiving a response"}

        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        # Try to extract structured bboxes from model text output
        import json as _json
        bboxes = []
        m = re.search(r'"bboxes"\s*:\s*(\[.*?\])', content, re.DOTALL)
        if not m:
            m = re.search(r'(\[\s*\{[^[\]]*"x1"[^[\]]*\}\s*\])', content, re.DOTALL)
        if m:
            try:
                bboxes = _json.loads(m.group(1))
            except Exception:
                bboxes = []
        return {
            "status": "success",
            "analysis": content,
            "bboxes": bboxes,
            "processed_images": len(image_paths),
            "max_tokens_used": payload["max_tokens"],
        }
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}
    finally:
        _stop_tool_service_after_call(vllm_manager)




def sam2_segment_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Handler for SAM2 Segmentation (Full Image Box Prompt)."""
    clean_path = _resolve_image_path(arguments)
    service_path, preprocessing = _prepare_perception_image(clean_path, context)

    result = _call_service(sam2_manager, f"{sam2_manager.API_URL}/segment", {"image_path": service_path}, timeout=60)
    if result.get("status") == "error":
        return result

    vis_b64 = result.get("visualization")
    count = result.get("count", 0)
    md_text = f"SAM2 Segmentation Complete. Found {count} regions.\n\n"
    if vis_b64:
        md_text += f"![Segmentation Result]({vis_b64})"

    bboxes = []
    for feature in result.get("geojson", {}).get("features", []):
        coords_list = feature.get("geometry", {}).get("coordinates", [])
        xs, ys = [], []
        for ring in coords_list:
            for pt in ring:
                xs.append(pt[0])
                ys.append(pt[1])
        if xs and ys:
            bbox = _scale_bbox_to_original([min(xs), min(ys), max(xs), max(ys)], preprocessing)
            bboxes.append({"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]})

    return {"status": "success", "output": md_text, "bboxes": bboxes, "image_preprocessing": preprocessing}
        
        
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




def _flatten_strip_rcnn_detections(raw: Any) -> list[dict[str, Any]]:
    """Flatten Strip R-CNN's {class: [[cx,cy,w,h,angle,score],...]} into a list of
    per-object dicts, adding an axis-aligned enclosing bbox [x1,y1,x2,y2] for the
    rotated box so downstream tools (draw_bboxes/bbox_to_centroid/bbox_area) work.
    The original rotated geometry is preserved under cx/cy/w/h/angle."""
    import math
    objects: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return objects
    for label, boxes in raw.items():
        if not isinstance(boxes, (list, tuple)):
            continue
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) < 5:
                continue
            cx, cy, w, h, angle = (float(box[0]), float(box[1]), float(box[2]),
                                   float(box[3]), float(box[4]))
            score = float(box[5]) if len(box) > 5 else None
            # Axis-aligned half-extents of the rotated rectangle.
            c, s = abs(math.cos(angle)), abs(math.sin(angle))
            dx = (w / 2.0) * c + (h / 2.0) * s
            dy = (w / 2.0) * s + (h / 2.0) * c
            x1, y1, x2, y2 = round(cx - dx, 2), round(cy - dy, 2), round(cx + dx, 2), round(cy + dy, 2)
            item: dict[str, Any] = {
                "label": str(label),
                "bbox": [x1, y1, x2, y2],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": round(cx, 2), "cy": round(cy, 2),
                "w": round(w, 2), "h": round(h, 2), "angle": round(angle, 4),
            }
            if score is not None:
                item["score"] = round(score, 4)
            objects.append(item)
    return objects


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
        raw = result.get("detections") or {}
        # Strip R-CNN returns ROTATED boxes grouped by class:
        #   {class_name: [[cx, cy, w, h, angle_rad, score], ...]}.
        # Flatten into a per-object list and add an axis-aligned enclosing bbox
        # so the result is consumable by draw_bboxes / bbox_to_centroid /
        # bbox_area, mirroring the InstructSAM detection schema.
        objects = _flatten_strip_rcnn_detections(raw)
        detections_bboxes = [o["bbox"] for o in objects]
        num = result.get("num_detections")
        if num is None:
            num = len(objects)
        return {
            "status": "success",
            # `count`/`objects` mirror the InstructSAM schema so the agent sees a
            # single, consistent detection contract across both detectors.
            "count": num,
            "objects": objects,
            # Backward-compatible raw rotated output (grouped by class).
            "detections": raw,
            "detections_bboxes": detections_bboxes,
            "image_size": result.get("image_size"),
            "num_detections": num,
            "message": f"Strip R-CNN detection completed. Found {num} objects.",
        }
    return {"status": "error", "message": f"Detection failed: {result.get('error')}"}


def remotesam_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Handler for RemoteSAM tasks."""
    clean_path = _resolve_image_path(arguments)
    task_type, normalized_args = _normalize_remotesam_request(arguments)

    result = _call_service(
        remotesam_manager, f"{remotesam_manager.API_URL}/{task_type}",
        {"image_path": clean_path, **normalized_args},
        timeout=int(os.environ.get("REMOTESAM_TOOL_TIMEOUT", "300")),
    )
    if isinstance(result, dict) and result.get("status") == "error":
        return result
    artifact_path = _write_tool_artifact("geo_perception.remotesam", result, context)
    result_summary = _summarize_remotesam_result(result)
    bboxes = _extract_remotesam_bboxes(result)
    result_keys = sorted(result.keys()) if isinstance(result, dict) else []
    return {
        "status": "success",
        "artifact_path": artifact_path,
        "result": result,
        "result_summary": result_summary,
        "bboxes": bboxes,
        "result_keys": result_keys,
        "message": f"RemoteSAM {task_type} completed.",
    }
# ---------------------------------------------------------------------------
# DEPRECATED 2026-06-05: original InstructSAM-service implementation.
# The InstructSAM detection service returned 0 objects on ~100% of OpenEarth
# rollouts (broken/misconfigured service). It is kept here for reference only.
# The active `instructsam_handler` below preserves the exact same input/output
# contract but performs grounding via the VLM (vllm_manager) backend.
# ---------------------------------------------------------------------------
# def instructsam_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
#     """Handler for InstructSAM — instruction-based segmentation and counting."""
#     clean_path = _resolve_image_path(arguments)
#     service_path, preprocessing = _prepare_perception_image(clean_path, context)
#     text_prompt = arguments.get("text_prompt", "objects in the image")
#
#     result = _call_service(
#         instructsam_manager, f"{instructsam_manager.API_URL}/segment",
#         {"image_path": service_path, "text_prompt": text_prompt},
#         timeout=300,
#     )
#     if result.get("status") == "error":
#         return result
#
#     count = result.get("count", 0)
#     vis_b64 = result.get("visualization", "")
#     md_text = f"InstructSAM found **{count}** object(s) matching '{text_prompt}'.\n\n"
#     if vis_b64:
#         md_text += f"![InstructSAM Result]({vis_b64})"
#     return {
#         "status": "success",
#         "count": count,
#         "objects": _scale_detection_bboxes_to_original(result.get("objects", []), preprocessing),
#         "detections": _scale_detection_bboxes_to_original(result.get("detections", []), preprocessing),
#         "image_preprocessing": preprocessing,
#         "output": md_text,
#     }


def _parse_vlm_grounding_bboxes(content: str) -> List[Dict[str, Any]]:
    """Parse bbox dicts from VLM text output. Robust to fences and minor noise.

    Accepts: {"bboxes": [{"label","x1","y1","x2","y2","score"}, ...]},
    a bare JSON array of such dicts, or items carrying a 4-number "bbox" list.
    Returns a list of normalized dicts with float x1,y1,x2,y2 (+label/score).
    """
    if not content:
        return []
    text = content.strip()
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    raw_items = None
    m = re.search(r'"bboxes"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if m:
        try:
            raw_items = json.loads(m.group(1))
        except Exception:
            raw_items = None
    if raw_items is None:
        m2 = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
        if m2:
            try:
                raw_items = json.loads(m2.group(1))
            except Exception:
                raw_items = None
    if raw_items is None:
        try:
            obj = json.loads(text)
            raw_items = obj.get("bboxes") if isinstance(obj, dict) else obj
        except Exception:
            raw_items = None
    if not isinstance(raw_items, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for it in raw_items:
        coords = None
        label = None
        score = None
        if isinstance(it, dict):
            label = it.get("label") or it.get("name") or it.get("category")
            score = it.get("score") or it.get("confidence")
            if all(k in it for k in ("x1", "y1", "x2", "y2")):
                coords = [it["x1"], it["y1"], it["x2"], it["y2"]]
            else:
                for key in ("bbox", "box", "bbox_2d", "coordinates"):
                    val = it.get(key)
                    if isinstance(val, (list, tuple)) and len(val) >= 4:
                        coords = list(val[:4])
                        break
        elif isinstance(it, (list, tuple)) and len(it) >= 4:
            coords = list(it[:4])
        if coords is None:
            continue
        try:
            x1, y1, x2, y2 = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        parsed.append({
            "label": str(label) if label is not None else None,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return parsed


def _hide_backend(text: str) -> str:
    """Scrub VLM/model-path wording from any text surfaced to the agent so the
    tool appears purely as InstructSAM."""
    if not text:
        return text
    out = re.sub(r"(?i)\bvlm\b", "InstructSAM", text)
    out = out.replace("vllm", "InstructSAM").replace("vLLM", "InstructSAM")
    # Strip the on-disk model path if it ever leaks into an error string.
    out = re.sub(r"/data1/\S*?merged_model/?", "<model>", out)
    return out


def instructsam_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Dispatch to the configured InstructSAM kernel.

    Selected by env ``TERRABOX_INSTRUCTSAM_BACKEND`` (default ``service``):
      - ``service``: original training-free pipeline = SAM2 masks + GeoRSCLIP
        matching + VLM counting (the InstructSAM docker service). Restored.
      - ``vlm``: single Qwen3-VL grounding call (lighter, coarser).
    Switching kernels is an env-var change only — no experiment code edits.
    """
    backend = os.environ.get("TERRABOX_INSTRUCTSAM_BACKEND", "service").strip().lower()
    if backend == "vlm":
        return _instructsam_via_vlm(arguments, context, account)
    return _instructsam_via_service(arguments, context, account)


def _instructsam_via_service(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Original InstructSAM: SAM2 + GeoRSCLIP + VLM-counting docker service.

    Root-cause fix vs the old code: the counting step inside the service calls
    the host VLM (port ~9000); nobody used to start it, so counting silently
    returned 0. We now ensure that backend is up before calling the service.
    """
    # Bring up the counting backend the InstructSAM service depends on.
    try:
        vllm_manager.start_service()
    except Exception as e:
        return {"status": "error",
                "message": f"Failed to start InstructSAM counting backend: {_hide_backend(str(e))}"}

    clean_path = _resolve_image_path(arguments)
    service_path, preprocessing = _prepare_perception_image(clean_path, context)
    # Accept the OpenEarthAgent TextToBbox arg names too (`text`, `top1`).
    text_prompt = (arguments.get("text_prompt") or arguments.get("text")
                   or arguments.get("object") or "objects in the image")
    top1 = bool(arguments.get("top1", False))

    result = _call_service(
        instructsam_manager, f"{instructsam_manager.API_URL}/segment",
        {"image_path": service_path, "text_prompt": text_prompt},
        timeout=int(os.environ.get("INSTRUCTSAM_TOOL_TIMEOUT", "300")),
    )
    if result.get("status") == "error":
        return result

    count = result.get("count", 0)
    # SAM2 returns bbox as [x, y, w, h]; convert to [x1, y1, x2, y2] (the gold /
    # downstream convention) BEFORE scaling, clamp to the original image, and
    # expose x1..y2. Otherwise downstream (draw_bboxes, distance/area) misreads
    # w,h as x2,y2 → y2<y1 errors and wrong geometry.
    try:
        from PIL import Image
        with Image.open(clean_path) as _oim:
            orig_w, orig_h = _oim.size
    except Exception:
        orig_w, orig_h = 0, 0

    def _norm(items):
        out = []
        for it in _scale_detection_bboxes_to_original(items or [], preprocessing):
            if not isinstance(it, dict):
                continue
            bb = it.get("bbox") or it.get("box")
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                x1, y1, x2, y2 = x, y, x + w, y + h
                if orig_w:
                    x1, x2 = max(0.0, min(x1, orig_w)), max(0.0, min(x2, orig_w))
                if orig_h:
                    y1, y2 = max(0.0, min(y1, orig_h)), max(0.0, min(y2, orig_h))
                new = dict(it)
                new["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                new["x1"], new["y1"], new["x2"], new["y2"] = new["bbox"]
                out.append(new)
            else:
                out.append(it)
        return out

    objects = _norm(result.get("objects", []))
    detections = _norm(result.get("detections", []))
    # The docker service populates only `detections` and leaves `objects` empty.
    # Mirror them so the two aliases are always consistent — otherwise the agent
    # may read the empty `objects` field and wrongly conclude "nothing detected"
    # even though `count`>0 and `detections` is full (observed in rollouts).
    if detections and not objects:
        objects = detections
    elif objects and not detections:
        detections = objects
    # Keep `count` consistent with the actual detection list when the service
    # under-reports it.
    if not count and detections:
        count = len(detections)
    if top1:
        # TextToBbox top1: keep only the highest-scoring detection.
        def _best(items):
            return sorted(items, key=lambda d: float(d.get("score", 0) or 0), reverse=True)[:1] if items else items
        objects, detections = _best(objects), _best(detections)
        count = len(detections)
    md_text = f"InstructSAM found **{count}** object(s) matching '{text_prompt}'."
    return {
        "status": "success",
        "count": count,
        "objects": objects,
        "detections": detections,
        "image_preprocessing": preprocessing,
        "output": md_text,
    }


def _instructsam_via_vlm(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """InstructSAM grounding via the VLM backend (alternate kernel).

    Same I/O contract as the original: input {image, text_prompt}; output
    {status, count, objects, detections, image_preprocessing, output}. Each
    detection carries a pixel-space ``bbox`` [x1,y1,x2,y2] (original-image
    coordinates) plus x1/y1/x2/y2/label/score for easy downstream consumption.
    """
    clean_path = _resolve_image_path(arguments)
    service_path, preprocessing = _prepare_perception_image(clean_path, context)
    text_prompt = arguments.get("text_prompt", "objects in the image")

    # Image size of the (possibly resized) service image fed to the VLM.
    try:
        from PIL import Image
        with Image.open(service_path) as _im:
            img_w, img_h = _im.size
    except Exception:
        img_w, img_h = 0, 0

    grounding_prompt = (
        "You are a precise object detector for remote sensing / aerial imagery.\n"
        f"The image is {img_w} pixels wide and {img_h} pixels tall.\n"
        f"Detect EVERY instance of this target: \"{text_prompt}\".\n"
        "Return STRICT JSON only, no prose, in exactly this schema:\n"
        '{"bboxes": [{"label": "<name>", "x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "score": <float 0-1>}]}\n'
        "Coordinates MUST be absolute pixel values within the given image size, "
        "with 0<=x1<x2<=width and 0<=y1<y2<=height.\n"
        'If there are no such objects, return {"bboxes": []}.'
    )

    try:
        vllm_manager.start_service()
    except Exception as e:
        # Keep the backend opaque to the agent: surface as an InstructSAM error.
        return {"status": "error", "message": f"Failed to start InstructSAM service: {_hide_backend(str(e))}"}

    try:
        try:
            b64_str = _encode_image_to_base64(service_path)
        except Exception as e:
            return {"status": "error", "message": f"Image error: {e}"}

        payload = {
            "model": getattr(vllm_manager, "MODEL_NAME", vllm_manager.MODEL_PATH),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": grounding_prompt},
                    {"type": "image_url", "image_url": {"url": b64_str}},
                ],
            }],
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        api_url = f"{vllm_manager.API_BASE}/chat/completions"
        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            return {"status": "error", "message": f"InstructSAM connection failed: {_hide_backend(str(e))}"}
        if response.status_code != 200:
            return {"status": "error", "message": f"InstructSAM detection error {response.status_code}: {_hide_backend(response.text)}"}

        content = response.json()["choices"][0]["message"]["content"]
    finally:
        _stop_tool_service_after_call(vllm_manager)

    parsed = _parse_vlm_grounding_bboxes(content)

    # The grounding model reports coordinates in the (service) image space it was
    # shown (img_w × img_h, whatever the real size is — not hard-coded). VLM
    # grounding can overshoot the canvas, so clamp to the shown bounds and drop
    # degenerate boxes, then scale back to the original image and clamp again.
    def _clip(v: float, hi: float) -> float:
        if hi <= 0:
            return float(v)
        return max(0.0, min(float(v), hi))

    detections: List[Dict[str, Any]] = []
    for b in parsed:
        x1, y1 = _clip(b["x1"], img_w), _clip(b["y1"], img_h)
        x2, y2 = _clip(b["x2"], img_w), _clip(b["y2"], img_h)
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate / empty box after clamping
        detections.append({
            "label": b.get("label") or text_prompt,
            "score": b.get("score") if b.get("score") is not None else 0.99,
            "bbox": [x1, y1, x2, y2],
        })
    detections = _scale_detection_bboxes_to_original(detections, preprocessing)

    # Final clamp to the ORIGINAL image bounds (any resolution) + expose x1..y2.
    try:
        from PIL import Image
        with Image.open(clean_path) as _oim:
            orig_w, orig_h = _oim.size
    except Exception:
        orig_w, orig_h = 0, 0
    for det in detections:
        bb = det.get("bbox") or []
        if len(bb) >= 4:
            bb = [round(_clip(bb[0], orig_w), 2), round(_clip(bb[1], orig_h), 2),
                  round(_clip(bb[2], orig_w), 2), round(_clip(bb[3], orig_h), 2)]
            det["bbox"] = bb
            det["x1"], det["y1"], det["x2"], det["y2"] = bb[0], bb[1], bb[2], bb[3]

    count = len(detections)
    md_text = f"InstructSAM found **{count}** object(s) matching '{text_prompt}'."
    return {
        "status": "success",
        "count": count,
        "objects": detections,
        "detections": detections,
        "image_preprocessing": preprocessing,
        "output": md_text,
    }




# --- Bbox / Annotation / Mock Handlers ---

def _coerce_bbox(b: Any) -> dict | None:
    """Accept a bbox as a dict {x1,y1,x2,y2}, a string '(x1, y1, x2, y2)', or a
    4-element list/tuple, and normalise to a dict."""
    if isinstance(b, dict):
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            return b
        if "bbox" in b:
            return _coerce_bbox(b["bbox"])
        return None
    if isinstance(b, str):
        nums = re.findall(r"-?\d+\.?\d*", b)
        if len(nums) >= 4:
            x1, y1, x2, y2 = (float(n) for n in nums[:4])
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        return None
    if isinstance(b, (list, tuple)) and len(b) >= 4 and all(isinstance(v, (int, float)) for v in b[:4]):
        return {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
    return None


def _auto_output_path(prefix: str, ext: str = "png") -> str:
    out_dir = os.environ.get("TERRABOX_TOOL_ARTIFACT_DIR", "tmp/tool_artifacts")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")


def draw_bboxes_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Draw labeled bounding boxes on an image using Pillow.

    Accepts either the native ``bboxes`` list ([{x1,y1,x2,y2,label?,color?}, ...])
    or a single OpenEarthAgent-style ``bbox`` ('(x1, y1, x2, y2)' string or list)
    with an optional ``annotation`` label. ``output_path`` is optional (auto)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Missing Pillow. Install: pip install Pillow")

    image_path = _resolve_image_path(arguments)
    bboxes = arguments.get("bboxes")
    if not bboxes and arguments.get("bbox") is not None:
        # OpenEarthAgent DrawBox form: single bbox + annotation label.
        one = _coerce_bbox(arguments.get("bbox"))
        if one is None:
            return {"status": "error", "message": f"could not parse bbox: {arguments.get('bbox')!r}"}
        label = arguments.get("annotation") or arguments.get("label") or ""
        if label:
            one = {**one, "label": str(label)}
        bboxes = [one]
    bboxes = bboxes or []
    if not isinstance(bboxes, list):
        return {"status": "error", "message": f"bboxes must be a list, got {type(bboxes).__name__}: {bboxes!r}"}
    bboxes = [_coerce_bbox(b) or b for b in bboxes]
    output_path = arguments.get("output_path") or _auto_output_path("drawbox")
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
    annotations = arguments.get("annotations")
    output_path = arguments.get("output_path") or _auto_output_path("addtext")

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if not annotations and arguments.get("text") is not None:
        # OpenEarthAgent AddText form: single text + position keyword/coords.
        W, H = img.size
        pos = arguments.get("position", "lt")
        pad = 8
        pos_map = {
            "lt": (pad, pad), "rt": (W * 0.7, pad), "lb": (pad, H * 0.9),
            "rb": (W * 0.7, H * 0.9), "center": (W * 0.4, H * 0.5),
            "mm": (W * 0.4, H * 0.5), "mt": (W * 0.4, pad), "mb": (W * 0.4, H * 0.9),
            "lm": (pad, H * 0.5), "rm": (W * 0.7, H * 0.5),
            "top": (W * 0.4, pad), "bottom": (W * 0.4, H * 0.9),
        }
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
        elif isinstance(pos, str) and re.findall(r"-?\d+\.?\d*", pos) and len(re.findall(r"-?\d+\.?\d*", pos)) >= 2:
            n = re.findall(r"-?\d+\.?\d*", pos)
            x, y = float(n[0]), float(n[1])
        else:
            x, y = pos_map.get(str(pos).lower(), (pad, pad))
        annotations = [{"text": str(arguments["text"]), "x": x, "y": y}]
    annotations = annotations or []

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
    image_path = _resolve_image_path(arguments)
    languages = arguments.get("languages", ["en"])
    if isinstance(languages, str):
        languages = [languages]

    gpu_setting = _easyocr_gpu_setting()
    timeout = int(os.environ.get("TERRABOX_OCR_TIMEOUT", "180"))
    payload = {"image_path": image_path, "languages": languages, "gpu": gpu_setting}
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.pathsep.join(sys.path))
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "terrabox.toolkits._ocr_worker"],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error_type": "tool_timeout",
            "message": f"OCR timed out after {timeout} seconds.",
            "retryable": False,
        }

    if completed.returncode != 0:
        error_text = completed.stderr or completed.stdout
        if _is_cuda_oom_text(error_text):
            return _tool_oom_response(error_text)
        return {"status": "error", "message": f"OCR subprocess failed: {error_text}"}

    try:
        worker_result = json.loads(completed.stdout)
    except Exception as exc:
        return {"status": "error", "message": f"OCR subprocess returned invalid JSON: {exc}"}

    texts = worker_result.get("texts", [])

    return {
        "status": "success",
        "texts": texts,
        "count": len(texts),
        "full_text": " ".join(t["text"] for t in texts),
        "ocr_subprocess": True,
        "ocr_gpu": gpu_setting,
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


def change_os_detect_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """ChangeOS building-damage assessment (RSE 2021, Z-Zheng/ChangeOS).

    Runs the real TorchScript model via the docker service: given pre- and
    post-disaster images it returns a building localization mask plus a per-
    building damage assessment (no-damage / minor / major / destroyed),
    pixel-level statistics, and saved colorized masks.
    """
    pre_path = _resolve_image_path({"image": arguments.get("pre_image") or arguments.get("pre_image_path") or arguments.get("image")})
    post_raw = arguments.get("post_image") or arguments.get("post_image_path")
    if not post_raw:
        return {
            "status": "error",
            "message": "ChangeOS requires BOTH a pre-event and a post-event image "
                       "(it performs object-based semantic change detection). "
                       "Provide 'post_image'.",
        }
    post_path = _resolve_image_path({"image": post_raw})

    payload = {"pre_image_path": pre_path, "post_image_path": post_path}
    output_path = arguments.get("output_path")
    if output_path:
        payload["output_path"] = output_path

    result = _call_service(
        changeos_manager, f"{changeos_manager.API_URL}/detect", payload,
        timeout=int(os.environ.get("CHANGEOS_TOOL_TIMEOUT", "300")),
    )
    if isinstance(result, dict) and result.get("status") == "error":
        return result
    if result.get("success"):
        dam = result.get("damage_pixel_counts") or {}
        n_obj = result.get("num_building_objects", 0)
        changed = result.get("changed_pixels", 0)
        return {
            "status": "success",
            "checkpoint": result.get("checkpoint"),
            "num_building_objects": n_obj,
            "building_pixels": result.get("building_pixels", 0),
            "changed_pixels": changed,
            "damage_pixel_counts": dam,
            "damage_classes": result.get("damage_classes"),
            "outputs": result.get("outputs", {}),
            "pre_image": pre_path,
            "post_image": post_path,
            "message": (
                f"ChangeOS detected {n_obj} building object(s); "
                f"{changed} damaged (minor/major/destroyed) pixels. "
                f"Damage pixel breakdown: {dam}."
            ),
        }
    return {"status": "error", "message": f"ChangeOS failed: {result.get('error')}"}


def count_given_object_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Count instances of a named object class in an image (OpenEarthAgent
    `CountGivenObject`). Thin wrapper over InstructSAM detection: detect the
    requested class, return the count + boxes."""
    obj = (
        arguments.get("object")
        or arguments.get("text_prompt")
        or arguments.get("text")
        or arguments.get("category")
        or arguments.get("class")
        or "objects"
    )
    is_args = dict(arguments)
    is_args["text_prompt"] = obj
    result = instructsam_handler(is_args, context, account)
    if not isinstance(result, dict) or result.get("status") == "error":
        return result if isinstance(result, dict) else {"status": "error", "message": str(result)}
    count = result.get("count", 0)
    return {
        "status": "success",
        "object": obj,
        "count": count,
        "objects": result.get("objects", []),
        "detections": result.get("detections", []),
        "output": f"Found {count} instance(s) of '{obj}'.",
    }


def region_attribute_description_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Describe attributes of a region/object in an image (OpenEarthAgent
    `RegionAttributeDescription`). Wraps the VLM with a description prompt;
    an optional bbox/region focuses the description."""
    region = arguments.get("region") or arguments.get("bbox") or arguments.get("box")
    attr = arguments.get("attribute") or arguments.get("query") or "color, type, condition, and notable attributes"
    focus = f" within the region {region}" if region else ""
    prompt = (
        f"Describe the {attr} of the object{focus} in this remote-sensing image. "
        "Be concise and factual; do not invent measurements or counts."
    )
    vlm_args = dict(arguments)
    vlm_args["prompt"] = prompt
    return vlm_analyze_handler(vlm_args, context, account)


# --- Tool Registration ---

def setup(registrar):
    registrar.toolkit(
        name="geo_perception",
        description="AI perception for remote sensing imagery: VLM analysis, object detection (STRIP-RCNN), semantic/instance segmentation (RemoteSAM, InstructSAM, SAM2), change detection, OCR extraction, text/marker annotation, and image compositing.",
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
                    "image": {
                        "type": "string",
                        "description": "A single image to analyze/describe (alias of 'images' with one entry).",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Question or instruction.",
                        "default": "",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 8192,
                        "minimum": 1024,
                        "description": (
                            "Maximum output tokens for the VLM analysis. "
                            "This is the response budget, not the image input context. "
                            "Use 8192 by default for image tasks, especially multi-image comparison, "
                            "bbox extraction, damage/flood/fire analysis, or area/statistics reasoning. "
                            "Do not use small values such as 256 or 512 for image analysis."
                        ),
                    },
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
            description="Segment objects / the full scene in a satellite image with SAM2 and return their pixel regions. Returns {bboxes:[{x1,y1,x2,y2}] of segments, output, image_preprocessing}. An optional 'text' hint names the object of interest.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to segment. Frontend uploads the image; backend resolves the local file path."
                    },
                    "text": {"type": "string", "description": "Optional object name to focus the segmentation on (e.g., 'vehicle')."},
                    "flag": {"type": "boolean", "description": "Optional mode flag."}
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
            description="Oriented object detection (Strip R-CNN) for the DOTA-15 classes only: plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool. Returns 0 for any other object — use instructsam for open-vocabulary targets. Returns {count, objects:[{label, score, bbox:[x1,y1,x2,y2], cx, cy, w, h, angle}], detections_bboxes:[[x1,y1,x2,y2]], num_detections}. `objects`/`detections_bboxes` give axis-aligned boxes ready for draw_bboxes/bbox_to_centroid/bbox_area; original rotated geometry (center,size,radians) is kept per-object in cx/cy/w/h/angle and in the raw `detections` dict.",
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
            description="Perform various tasks using RemoteSAM such as referring segmentation, semantic segmentation, detection, etc. Returns {result, result_summary, bboxes, artifact_path}.",
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
            description="Open-vocabulary, instruction-based detection and counting (and text-to-bbox grounding) of objects described in text. Returns {count, objects, detections:[{label, score, x1, y1, x2, y2}]} — axis-aligned boxes ready for draw_bboxes/bbox_to_centroid. Set top1=true to locate just the single best-matching object (text-to-bbox).",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image to analyze. Frontend uploads the image; backend resolves the local file path."
                    },
                    "text": {
                        "type": "string",
                        "description": "Object/instruction to detect (e.g., 'trainstation', 'all the red cars'). Alias of text_prompt."
                    },
                    "text_prompt": {
                        "type": "string",
                        "description": "Object/instruction to detect (alias of 'text')."
                    },
                    "top1": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, return only the single highest-scoring detection (text-to-bbox)."
                    }
                },
                "required": ["image"]
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
            description="Draw labeled bounding box(es) on an image and save the result. Provide either a list of 'bboxes' (each {x1,y1,x2,y2,label?,color?}) or a single 'bbox' ('(x1, y1, x2, y2)' string or [x1,y1,x2,y2]) with an optional 'annotation' label. output_path is optional (auto-generated). Returns {status, output_path, boxes_drawn}.",
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
                    "bbox": {"type": "string", "description": "A single bbox as '(x1, y1, x2, y2)' or [x1,y1,x2,y2] (alternative to 'bboxes')."},
                    "annotation": {"type": "string", "description": "Label for the single 'bbox'."},
                    "output_path": {"type": "string", "description": "Optional path to save the annotated image (auto-generated if omitted)."},
                    "line_width": {"type": "integer", "default": 2, "description": "Width of the bounding box lines."}
                },
                "required": ["image"]
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
            description="Overlay text on an image and save it. Provide either a single 'text' with a 'position' (keyword 'lt'/'rt'/'lb'/'rb'/'center' or [x,y]), or a list of 'annotations' (each {text,x,y,color?,font_size?}). output_path is optional (auto-generated). Returns {status, output_path, annotations_added}.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the input image."},
                    "text": {"type": "string", "description": "Text to overlay (single-annotation form)."},
                    "position": {"type": "string", "description": "Placement for 'text': keyword 'lt'/'rt'/'lb'/'rb'/'center'/'top'/'bottom', or '[x, y]' pixel coords."},
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
                        "description": "List of text annotations with position and style (multi-annotation form)."
                    },
                    "output_path": {"type": "string", "description": "Optional path to save the annotated image (auto-generated if omitted)."}
                },
                "required": ["image"]
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
            description="Extract text from an image using EasyOCR. Supports Chinese and English text. Useful for reading labels, legends, or text in satellite imagery. Returns {texts:[{text, bbox}], count, full_text}.",
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
            description="Convert a list of bounding boxes to their centroid coordinates. Each input bbox must have x1,y1,x2,y2 keys. Returns {centroids:[{x,y}], count}.",
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

    # 17. ChangeOS building-damage assessment (real TorchScript model via docker)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.change_os_detect",
            name="ChangeOS Change Detection",
            description=(
                "Building-damage assessment via deep object-based semantic change "
                "detection (ChangeOS, RSE 2021). Given a PRE-event and a POST-event "
                "image of the same area, returns a building localization mask and a "
                "per-building damage rating (no-damage / minor / major / destroyed) "
                "with pixel-level statistics. Requires BOTH images."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pre_image": {"type": "string", "description": "Path to the pre-event (before-disaster) image."},
                    "post_image": {"type": "string", "description": "Path to the post-event (after-disaster) image of the same area."},
                    "text": {"type": "string", "description": "Optional free-text hint describing what to assess (e.g. 'buildings damaged by the hurricane')."},
                    "output_path": {"type": "string", "description": "Optional directory to save colorized localization/damage masks."}
                },
                "required": ["pre_image", "post_image"]
            },
            requires_connection=False
        ),
        change_os_detect_handler
    )

    # 18. CountGivenObject (wraps InstructSAM detection)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.count_given_object",
            name="CountGivenObject",
            description=(
                "Count the number of instances of a named object class in an image "
                "(e.g. 'ship', 'airplane', 'storage tank'). Returns {count, objects}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the image."},
                    "object": {"type": "string", "description": "The object class to count."},
                    "text": {"type": "string", "description": "The object class to count (alias of 'object')."},
                },
                "required": ["image"],
            },
        ),
        count_given_object_handler
    )

    # 19. RegionAttributeDescription (wraps VLM)
    registrar.tool(
        ToolSpec(
            slug="geo_perception.region_attribute_description",
            name="RegionAttributeDescription",
            description=(
                "Describe attributes (color, type, condition, etc.) of an object or "
                "region in a remote-sensing image, optionally focused on a bbox region. "
                "Returns a textual description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the image."},
                    "region": {"type": "string", "description": "Optional bbox/region [x1,y1,x2,y2] to focus on."},
                    "bbox": {"type": "string", "description": "Optional bbox region to focus on (alias of 'region')."},
                    "attribute": {"type": "string", "description": "Which attribute(s) to describe (optional)."},
                },
                "required": ["image"],
            },
        ),
        region_attribute_description_handler
    )
