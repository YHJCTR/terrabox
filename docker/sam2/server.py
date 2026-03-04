"""
SAM2 Server for Docker deployment.
Derived from sam2_server2.py with the following changes:
  1. CHECKPOINT and MODEL_CFG read from environment variables
  2. uvicorn binds to 0.0.0.0 instead of 127.0.0.1
"""
import uvicorn
from fastapi import FastAPI, Body
import torch
import numpy as np
import os
import sys
import base64
import cv2
from io import BytesIO
from PIL import Image
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.utils import instantiate
from omegaconf import OmegaConf

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/checkpoints/sam2.1_hiera_large.pt")
MODEL_CFG = os.environ.get("SAM2_CONFIG", "/sam2_configs/sam2.1/sam2.1_hiera_l.yaml")


def build_sam2_from_abs_path(config_file, checkpoint_path, device):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = OmegaConf.load(config_file)
    try:
        model = instantiate(cfg.model, _recursive_=True)
    except Exception:
        pass

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


print(f"Loading SAM2 from {CHECKPOINT} ...")
try:
    model = build_sam2_from_abs_path(MODEL_CFG, CHECKPOINT, DEVICE)
    predictor = SAM2ImagePredictor(model)
    print("SAM2 Ready!")
except Exception as e:
    print(f"Error loading SAM2: {e}")
    sys.exit(1)


def draw_masks_on_image(image_np, masks):
    vis_image = image_np.copy()
    overlay = vis_image.copy()
    mask_bool = masks.astype(bool)
    overlay[mask_bool] = [0, 0, 255]
    cv2.addWeighted(overlay, 0.5, vis_image, 0.5, 0, vis_image)
    contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    return vis_image


def image_to_base64(image_np):
    img_pil = Image.fromarray(image_np)
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def mask_to_geojson(mask, transform):
    results = []
    mask_uint8 = (mask.astype(np.uint8) * 255)
    for geom, val in shapes(mask_uint8, mask=mask_uint8, transform=transform):
        if val != 0:
            results.append(geom)
    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/segment")
def segment_image(data: dict = Body(...)):
    image_path = data.get("image_path")
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image not found"}

    try:
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        try:
            with rasterio.open(image_path) as src:
                transform = src.transform
                crs = src.crs.to_string() if src.crs else "Pixel"
        except Exception:
            import rasterio.transform
            transform = rasterio.transform.from_origin(0, H, 1, 1)
            crs = "Pixel"

        full_image_box = [0, 0, W, H]
        box_tensor = torch.tensor([full_image_box], dtype=torch.float32).to(DEVICE)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_np)
            try:
                masks, scores, logits = predictor.predict({"box": box_tensor})
            except Exception:
                masks, scores, logits = predictor.predict(box=box_tensor, multimask_output=False)

        best_mask = masks[0]
        vis_image_np = draw_masks_on_image(image_np, best_mask)
        vis_base64 = image_to_base64(vis_image_np)
        polygons = mask_to_geojson(best_mask, transform)

        return {
            "status": "success",
            "count": len(polygons),
            "visualization": vis_base64,
            "geojson": {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": crs}},
                "features": [
                    {"type": "Feature", "geometry": p, "properties": {"label": "object"}}
                    for p in polygons
                ]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)
