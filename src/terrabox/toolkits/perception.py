#"""
#Geo Perception Toolkit
#----------------------
#AI Perception tools for Object Detection, Semantic Segmentation, and Change Detection.
#Adapts Earth-Agent perception logic to Terrabox standard using ONNX Runtime.
#
#Key Features:
#1. Dynamic Model Loading: Looks for models in TERRA_MODEL_DIR or default cache.
#2. Geospatial Awareness: Preserves georeference info in output masks.
#3. ONNX Support: Lightweight inference without heavy framework dependencies.
#"""
#
#import os
#import json
#from typing import Any, Dict, List, Optional
#from ..core.registry import ToolSpec
#
## ------------------------------------------------------------------------------
## Configuration: Dynamic Model Registry
## ------------------------------------------------------------------------------
#
## 1. Determine Model Directory
## Priority: Environment Variable -> User Cache -> Local 'weights' folder
#DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "terrabox", "models")
#MODEL_DIR = os.environ.get("TERRA_MODEL_DIR", DEFAULT_MODEL_DIR)
#
## 2. Registry: Map easy names to filenames
#MODEL_REGISTRY = {
#    "segmentation": {
#        "building": "building_seg_v1.onnx",
#        "road": "road_extract_v1.onnx",
#        "water": "water_body_v1.onnx"
#    },
#    "detection": {
#        "ship": "ship_detect_v1.onnx",
#        "plane": "plane_detect_v1.onnx",
#        "vehicle": "vehicle_detect_v1.onnx"
#    },
#    "change": {
#        "general": "change_detect_base.onnx"
#    }
#}
#
#def _get_model_path(category: str, task: str, custom_path: Optional[str] = None) -> str:
#    """Resolve model path from registry or custom input."""
#    if custom_path:
#        if not os.path.exists(custom_path):
#            raise FileNotFoundError(f"Custom model not found: {custom_path}")
#        return custom_path
#
#    if category not in MODEL_REGISTRY or task not in MODEL_REGISTRY[category]:
#        raise ValueError(f"No built-in model for {category}/{task}. Please provide a custom_model_path.")
#
#    filename = MODEL_REGISTRY[category][task]
#    full_path = os.path.join(MODEL_DIR, filename)
#
#    if not os.path.exists(full_path):
#        raise FileNotFoundError(
#            f"Model file missing: {full_path}\n"
#            f"Action: Please download '{filename}' to '{MODEL_DIR}' or set TERRA_MODEL_DIR."
#        )
#
#    return full_path
#
## ------------------------------------------------------------------------------
## Inference Engine & Helpers
## ------------------------------------------------------------------------------
#
#def _lazy_deps():
#    try:
#        import numpy as np
#        import cv2
#        import rasterio
#        import onnxruntime as ort
#        return np, cv2, rasterio, ort
#    except ImportError:
#        raise ImportError("Missing dependencies. Install: pip install numpy opencv-python-headless rasterio onnxruntime")
#
#class OnnxInferencer:
#    """Wrapper for ONNX Runtime Inference."""
#    def __init__(self, model_path: str):
#        _, _, _, ort = _lazy_deps()
#        try:
#            # Try using CUDA if available, else CPU
#            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#            self.session = ort.InferenceSession(model_path, providers=providers)
#        except Exception:
#            # Fallback to CPU only
#            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#
#        self.input_name = self.session.get_inputs()[0].name
#        # Note: In a real system, we should read input shape from metadata.
#        # Here we assume dynamic or fixed 512x512 based on the handler logic.
#
#    def preprocess(self, img_path: str, target_size=(512, 512)):
#        np, cv2, _, _ = _lazy_deps()
#
#        # Read image
#        img = cv2.imread(img_path)
#        if img is None:
#            raise ValueError(f"Could not read image: {img_path}")
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#        # Resize
#        if target_size:
#            img = cv2.resize(img, target_size)
#
#        # Normalize (0-1) & CHW
#        img = img.astype(np.float32) / 255.0
#        img = img.transpose(2, 0, 1)
#        img = np.expand_dims(img, 0)
#        return img
#
#    def run(self, input_tensor):
#        return self.session.run(None, {self.input_name: input_tensor})
#
## ------------------------------------------------------------------------------
## Handlers
## ------------------------------------------------------------------------------
#
#def semantic_segmentation_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
#    """
#    Extract features (Building, Road, etc.) and save as GeoTIFF mask.
#    """
#    np, cv2, rasterio, _ = _lazy_deps()
#
#    image_path = arguments["image_path"]
#    target = arguments.get("target_object", "building")
#    output_path = arguments["output_path"]
#    custom_model = arguments.get("custom_model_path")
#
#    # 1. Prepare Model
#    model_path = _get_model_path("segmentation", target, custom_model)
#    engine = OnnxInferencer(model_path)
#
#    # 2. Inference
#    # Assuming standard 512x512 input for sliding window or resize
#    # For simplicity, this demo resizes the whole image
#    input_tensor = engine.preprocess(image_path, target_size=(512, 512))
#    preds = engine.run(input_tensor)[0] # [1, C, H, W] or [1, H, W]
#
#    # 3. Post-process
#    if preds.ndim == 4:
#        # Multiclass: [1, C, H, W] -> argmax
#        mask = np.argmax(preds, axis=1).squeeze()
#    else:
#        # Binary: [1, H, W] -> threshold
#        mask = (preds > 0.5).squeeze()
#
#    # 4. Georeferenced Save
#    with rasterio.open(image_path) as src:
#        orig_h, orig_w = src.height, src.width
#        profile = src.profile
#
#    # Resize mask back to original resolution
#    if mask.shape != (orig_h, orig_w):
#        mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#
#    os.makedirs(os.path.dirname(output_path), exist_ok=True)
#    profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress='lzw')
#
#    with rasterio.open(output_path, 'w', **profile) as dst:
#        dst.write(mask.astype(rasterio.uint8), 1)
#
#    return {
#        "status": "success",
#        "output_path": output_path,
#        "model_used": os.path.basename(model_path)
#    }
#
#def object_detection_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
#    """
#    Detect objects and return bounding boxes.
#    """
#    image_path = arguments["image_path"]
#    target = arguments.get("target_object", "ship")
#    custom_model = arguments.get("custom_model_path")
#    conf_thresh = arguments.get("confidence", 0.5)
#
#    model_path = _get_model_path("detection", target, custom_model)
#    engine = OnnxInferencer(model_path)
#
#    input_tensor = engine.preprocess(image_path, target_size=(640, 640)) # YOLO standard often 640
#    outputs = engine.run(input_tensor)
#
#    # Assume generic YOLO output: [Batch, N, 6] (x1, y1, x2, y2, score, class)
#    raw_dets = outputs[0]
#
#    results = []
#    # Handle batch dim
#    dets = raw_dets[0] if len(raw_dets.shape) == 3 else raw_dets
#
#    for det in dets:
#        score = float(det[4])
#        if score > conf_thresh:
#            results.append({
#                "bbox": [float(x) for x in det[:4]],
#                "score": score,
#                "label": target # Simplified: assume model only detects the target
#            })
#
#    return {
#        "count": len(results),
#        "detections": results,
#        "model_used": os.path.basename(model_path)
#    }
#
#def change_detection_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
#    """
#    Detect changes between pre/post event images.
#    """
#    np, cv2, rasterio, _ = _lazy_deps()
#
#    pre_path = arguments["pre_image_path"]
#    post_path = arguments["post_image_path"]
#    output_path = arguments["output_path"]
#    custom_model = arguments.get("custom_model_path")
#
#    model_path = _get_model_path("change", "general", custom_model)
#    engine = OnnxInferencer(model_path)
#
#    # Preprocess both
#    t1 = engine.preprocess(pre_path, target_size=(512, 512))
#    t2 = engine.preprocess(post_path, target_size=(512, 512))
#
#    # Concatenate (Standard Early Fusion: [1, 6, 512, 512])
#    input_tensor = np.concatenate([t1, t2], axis=1)
#
#    preds = engine.run(input_tensor)[0]
#    mask = (preds > 0.5).squeeze()
#
#    # Save with georeference from Pre-image
#    with rasterio.open(pre_path) as src:
#        h, w = src.height, src.width
#        profile = src.profile
#
#    if mask.shape != (h, w):
#        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
#
#    os.makedirs(os.path.dirname(output_path), exist_ok=True)
#    profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress='lzw')
#
#    with rasterio.open(output_path, 'w', **profile) as dst:
#        dst.write(mask.astype(rasterio.uint8), 1)
#
#    return {"status": "success", "output_path": output_path}
#
## ------------------------------------------------------------------------------
## Registration
## ------------------------------------------------------------------------------
#
#def setup(registrar):
#    registrar.toolkit("geo_perception", "AI Perception tools (Segmentation, Detection, Change).", "0.2.0")
#
#    registrar.tool(
#        ToolSpec(
#            slug="geo_perception.segment_image",
#            name="Extract Features",
#            description="Semantic segmentation (e.g. extract buildings, roads).",
#            parameters={
#                "type": "object",
#                "properties": {
#                    "image_path": {"type": "string"},
#                    "target_object": {"type": "string", "enum": ["building", "road", "water"]},
#                    "output_path": {"type": "string"},
#                    "custom_model_path": {"type": "string", "description": "Optional path to override built-in model."}
#                },
#                "required": ["image_path", "target_object", "output_path"]
#            },
#            requires_connection=False
#        ),
#        semantic_segmentation_handler
#    )
#
#    registrar.tool(
#        ToolSpec(
#            slug="geo_perception.detect_objects",
#            name="Detect Objects",
#            description="Object detection (e.g. ships, planes).",
#            parameters={
#                "type": "object",
#                "properties": {
#                    "image_path": {"type": "string"},
#                    "target_object": {"type": "string", "enum": ["ship", "plane", "vehicle"]},
#                    "confidence": {"type": "number", "default": 0.5},
#                    "custom_model_path": {"type": "string"}
#                },
#                "required": ["image_path", "target_object"]
#            },
#            requires_connection=False
#        ),
#        object_detection_handler
#    )
#
#    registrar.tool(
#        ToolSpec(
#            slug="geo_perception.detect_changes",
#            name="Detect Changes",
#            description="Detect changes between two temporal images.",
#            parameters={
#                "type": "object",
#                "properties": {
#                    "pre_image_path": {"type": "string"},
#                    "post_image_path": {"type": "string"},
#                    "output_path": {"type": "string"},
#                    "custom_model_path": {"type": "string"}
#                },
#                "required": ["pre_image_path", "post_image_path", "output_path"]
#            },
#            requires_connection=False
#        ),
#        change_detection_handler
#    )