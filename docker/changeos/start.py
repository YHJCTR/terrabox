# ChangeOS Building-Damage Assessment API Server
#
# Faithful wrapper around the official Z-Zheng/ChangeOS SDK: we reuse its
# ChangeOS class (pre/post normalisation + object_based_infer voting) and its
# visualize() palette verbatim, only loading the TorchScript weights from a
# volume-mounted path instead of letting the SDK download them at runtime.
#
# Routes:
#   GET  /health   -> liveness + which checkpoint/model is loaded
#   GET  /classes  -> xBD damage class legend
#   POST /detect   -> {pre_image_path, post_image_path, output_path?} ->
#                     localization + damage masks, per-class pixel stats,
#                     building object count, and saved colorized PNGs.

import os
from argparse import ArgumentParser

import numpy as np
import torch
from flask import Flask, request, jsonify
from skimage.io import imread, imsave
from skimage.transform import resize as sk_resize
from skimage import measure

import changeos  # official SDK: provides ChangeOS wrapper + visualize palette


# xBD damage taxonomy (matches changeos.visualize palette order)
DAMAGE_CLASSES = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}
MODEL_INPUT_SIZE = 1024  # ChangeOS demo/xBD tiles are 1024x1024


def parse_args():
    p = ArgumentParser(description="ChangeOS Building-Damage API Server")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a ChangeOS TorchScript .pt (e.g. /ckpt/changeos_r101.pt)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9007)
    return p.parse_args()


def _load_rgb(path):
    """Read an image as HxWx3 uint8 RGB, resized to the model's 1024x1024 input."""
    img = imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    orig_hw = (int(img.shape[0]), int(img.shape[1]))
    if orig_hw != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        # preserve_range keeps 0-255; anti_aliasing for downscale fidelity
        img = sk_resize(img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
                        order=1, preserve_range=True, anti_aliasing=True)
    return np.ascontiguousarray(img.astype(np.uint8)), orig_hw


def _validate(path):
    if not path:
        return False, "missing path"
    if not os.path.exists(path):
        return False, f"path does not exist: {path}"
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        return False, f"unsupported image format: {ext}"
    return True, ""


def create_app(checkpoint, device):
    app = Flask(__name__)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    jit_model = torch.jit.load(checkpoint, map_location=dev)
    jit_model.eval()
    # Reuse the official ChangeOS wrapper (normalisation + object-based voting).
    model = changeos.ChangeOS(jit_model)
    model.device = dev
    model.model.to(dev)
    model_name = os.path.basename(checkpoint)
    print(f"ChangeOS loaded: {model_name} on {dev}", flush=True)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "checkpoint": model_name,
            "device": str(dev),
            "damage_classes": DAMAGE_CLASSES,
        }), 200

    @app.route("/classes", methods=["GET"])
    def classes():
        return jsonify({"damage_classes": DAMAGE_CLASSES}), 200

    @app.route("/detect", methods=["POST"])
    def detect():
        try:
            data = request.json or {}
            pre = data.get("pre_image_path") or data.get("pre_image") or data.get("image")
            post = data.get("post_image_path") or data.get("post_image")

            for label, path in (("pre", pre), ("post", post)):
                ok, msg = _validate(path)
                if not ok:
                    return jsonify({"success": False, "error": f"{label}_image: {msg}"}), 400

            pre_img, pre_hw = _load_rgb(pre)
            post_img, _ = _load_rgb(post)

            with torch.no_grad():
                loc, dam = model(pre_img, post_img)  # uint8 HxW masks

            # Building objects = connected components in the localization mask.
            _, n_objects = measure.label(loc > 0, connectivity=2,
                                         background=0, return_num=True)

            # Per damage-class pixel counts (over building pixels).
            dam_counts = {}
            for cls_id, cls_name in DAMAGE_CLASSES.items():
                if cls_id == 0:
                    continue
                dam_counts[cls_name] = int(np.sum(dam == cls_id))
            building_pixels = int(np.sum(loc > 0))
            changed_pixels = int(np.sum(dam >= 2))  # minor/major/destroyed

            saved = {}
            output_path = data.get("output_path")
            if output_path:
                # Treat output_path as a directory unless it has a file extension
                # (so a not-yet-existing dir like ".../out" is created, not split).
                if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
                    out_dir = output_path
                else:
                    out_dir = os.path.dirname(output_path) or "."
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(post))[0]
                loc_vis, dam_vis = changeos.visualize(loc, dam)
                loc_p = os.path.join(out_dir, f"{base}_localization.png")
                dam_p = os.path.join(out_dir, f"{base}_damage.png")
                imsave(loc_p, loc_vis)
                imsave(dam_p, dam_vis)
                saved = {"localization_mask": loc_p, "damage_mask": dam_p}

            return jsonify({
                "success": True,
                "checkpoint": model_name,
                "input_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "original_size": list(pre_hw),
                "num_building_objects": int(n_objects),
                "building_pixels": building_pixels,
                "changed_pixels": changed_pixels,
                "damage_pixel_counts": dam_counts,
                "damage_classes": DAMAGE_CLASSES,
                "outputs": saved,
            }), 200

        except Exception as e:  # surface error text like the other services
            return jsonify({"success": False, "error": str(e)}), 500

    return app


def main():
    args = parse_args()
    app = create_app(args.checkpoint, args.device)
    print(f"Starting ChangeOS API Server on http://{args.host}:{args.port}", flush=True)
    print(f"Checkpoint: {args.checkpoint}  Device: {args.device}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
