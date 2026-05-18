"""
InstructSAM Docker Server
==========================
Training-free three-step inference pipeline:
  1. SAM2 AutomaticMaskGenerator produces mask candidates
  2. Host vLLM service (HTTP) parses the instruction and predicts target class/count
  3. GeoRSCLIP computes image-text similarity to match labels to masks

Runtime environment variables:
  SAM2_CHECKPOINT   SAM2 weights path      (default /models/sam2_hiera_large.pt)
  SAM2_CONFIG       SAM2 hydra config name (default sam2_hiera_l.yaml)
  CLIP_MODEL_NAME   OpenCLIP architecture  (default ViT-L-14)
  CLIP_CHECKPOINT   CLIP weights path      (default /models/GeoRSCLIP-ViT-L-14.pt)
  VLLM_API_URL      vLLM service URL       (default http://host.docker.internal:9000)
  VLLM_MODEL_NAME   model name in vLLM     (default /model)

API:
  GET  /health    health check
  POST /segment   full inference
"""

import os
import sys
import json
import re
import traceback
import base64
import mimetypes

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
import requests as _http

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE          = os.environ.get("DEVICE",          "cuda:0")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/models/sam2_hiera_large.pt")
SAM2_CONFIG     = os.environ.get("SAM2_CONFIG",     "sam2_hiera_l.yaml")
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "ViT-L-14")
CLIP_CHECKPOINT = os.environ.get("CLIP_CHECKPOINT", "/models/GeoRSCLIP-ViT-L-14.pt")
VLLM_API_URL    = os.environ.get("VLLM_API_URL",    "http://host.docker.internal:9000")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "/model")

# ── Load SAM2 ─────────────────────────────────────────────────────────────────
print(f"[InstructSAM] Loading SAM2 from {SAM2_CHECKPOINT} ...")
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=16,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.85,
        min_mask_region_area=200,
    )
    print("[InstructSAM] SAM2 ready.")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ── Load CLIP (GeoRSCLIP / RemoteCLIP both use the OpenCLIP format) ──────────
print(f"[InstructSAM] Loading CLIP ({CLIP_MODEL_NAME}) from {CLIP_CHECKPOINT} ...")
try:
    import open_clip

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_CHECKPOINT
    )
    clip_model = clip_model.to(DEVICE).eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    print("[InstructSAM] CLIP ready.")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print(f"[InstructSAM] Will call vLLM at {VLLM_API_URL} for counting.")
print("[InstructSAM] All models loaded. Server starting...")


# ── Utility functions ─────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    """Encode a local image file as a data URI for the vLLM image_url field."""
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _count_objects_with_vllm(image_path: str, text_prompt: str) -> dict:
    """
    Call the host vLLM service (OpenAI-compatible) to parse the instruction and
    return the predicted object categories and count.
    Returns: {"count": N, "objects": ["label", ...]}
    """
    counting_prompt = (
        f"Look at this remote sensing image.\n"
        f"Task: {text_prompt}\n\n"
        f"Identify all matching objects and count them. "
        f'Respond ONLY with a JSON object:\n'
        f'{{"count": <integer>, "objects": [<list of category names>]}}\n'
        f'Example: {{"count": 3, "objects": ["airplane", "airplane", "airplane"]}}'
    )

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                    {"type": "text", "text": counting_prompt},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    try:
        resp = _http.post(
            f"{VLLM_API_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        generated = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[InstructSAM] vLLM call failed: {e}")
        return {"count": 0, "objects": []}

    # Extract JSON from the generated output
    try:
        start = generated.find("{")
        end   = generated.rfind("}") + 1
        if start >= 0 and end > start:
            result  = json.loads(generated[start:end])
            count   = int(result.get("count", 0))
            objects = list(result.get("objects", []))
            if len(objects) < count:
                last = objects[-1] if objects else text_prompt.split()[-1]
                objects.extend([last] * (count - len(objects)))
            return {"count": count, "objects": objects[:count]}
    except Exception:
        pass

    # Fallback: extract the first number from the output
    nums    = re.findall(r"\b\d+\b", generated)
    count   = int(nums[0]) if nums else 0
    keyword = text_prompt.strip().split()[-1].lower()
    return {"count": count, "objects": [keyword] * count}


def _clip_image_features(image: np.ndarray, masks: list) -> np.ndarray:
    """Compute CLIP image features for each mask's bounding-box crop."""
    features = []
    for m in masks:
        x, y, w, h = [int(v) for v in m["bbox"]]
        region = image[y: y + h, x: x + w]
        if region.size == 0:
            dim = 512
            features.append(np.zeros(dim, dtype=np.float32))
            continue
        tensor = clip_preprocess(Image.fromarray(region)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.squeeze(0).cpu().float().numpy())
    return np.array(features, dtype=np.float32) if features else np.zeros((0, 512), dtype=np.float32)


def _clip_text_features(labels: list) -> np.ndarray:
    """Compute CLIP text features for each label with a remote-sensing context prefix."""
    prompts = [f"a satellite image of a {lb}" for lb in labels]
    tokens  = clip_tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def _visualize(image: np.ndarray, masks: list, labels: list) -> str:
    """Draw masks and labels; return a base64-encoded PNG data URI."""
    vis = image.copy().astype(np.uint8)
    rng = np.random.RandomState(42)
    for m, label in zip(masks, labels):
        color = rng.randint(80, 220, size=3).tolist()
        vis[m["segmentation"]] = (
            vis[m["segmentation"]] * 0.45 + np.array(color) * 0.55
        ).astype(np.uint8)
        x, y, w, h = [int(v) for v in m["bbox"]]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, label[:20], (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/segment", methods=["POST"])
def segment():
    """
    Request body: {"image_path": "...", "text_prompt": "...", "max_masks": 150}
    Response body: {"status", "count", "objects", "detections", "visualization"}
    """
    data        = request.json or {}
    image_path  = data.get("image_path")
    text_prompt = data.get("text_prompt", "objects in the image")
    max_masks   = int(data.get("max_masks", 150))

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 400

    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return jsonify({"error": f"Cannot read image: {image_path}"}), 400
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Step 1: SAM2 generates mask candidates
        masks = mask_generator.generate(image)
        if not masks:
            return jsonify({"status": "success", "count": 0, "objects": [],
                            "detections": [], "message": "No regions found."}), 200
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:max_masks]

        # Step 2: vLLM parses the instruction
        count_result = _count_objects_with_vllm(image_path, text_prompt)
        count        = count_result.get("count", 0)
        objects      = count_result.get("objects", [])

        if count == 0 or not objects:
            return jsonify({"status": "success", "count": 0, "objects": [],
                            "detections": [],
                            "message": f"No objects matching '{text_prompt}' found."}), 200

        # Step 3: CLIP mask-to-label matching
        unique_labels  = list(dict.fromkeys(objects))
        mask_feats     = _clip_image_features(image, masks)
        text_feats     = _clip_text_features(unique_labels)
        sim            = mask_feats @ text_feats.T
        best_label_idx = sim.argmax(axis=1)
        best_score     = sim.max(axis=1)
        top_n          = min(count, len(masks))
        top_idx        = np.argsort(best_score)[::-1][:top_n]

        sel_masks  = [masks[i] for i in top_idx]
        sel_labels = [unique_labels[best_label_idx[i]] for i in top_idx]

        detections = [
            {"label": lbl, "bbox": [float(v) for v in m["bbox"]],
             "score": float(m.get("predicted_iou", 0.0)), "area": int(m["area"])}
            for m, lbl in zip(sel_masks, sel_labels)
        ]

        return jsonify({
            "status":        "success",
            "count":         len(detections),
            "objects":       sel_labels,
            "detections":    detections,
            "visualization": _visualize(image, sel_masks, sel_labels),
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9006)
