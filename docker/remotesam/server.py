"""
RemoteSAM Docker Server
========================
Docker-aware Flask server for RemoteSAM, analogous to docker/sam2/server.py.

Environment variables:
  REMOTESAM_CHECKPOINT  Path to checkpoint inside container
                        (default: /checkpoints/swin_base_patch4_window12_384_22k.pth)
  REMOTESAM_BERT_PATH   Local bert-base-uncased directory inside container
                        (default: /checkpoints/bert-base-uncased)
  REMOTESAM_USE_EPOC    Enable EPOC contour refinement (default: "false")

Volume mounts expected at runtime:
  -v /your/pretrained_weights:/checkpoints:ro
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro
  -v /data1:/data1
"""

import os
import sys
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# RemoteSAM source is cloned to /app in the Dockerfile; /app is already on sys.path
# because this script lives there. The tasks/code/model.py also appends /app itself.
from tasks.code.model import RemoteSAM, init_demo_model

app = Flask(__name__)

DEVICE = "cuda:0"  # Docker maps the allocated GPU (--gpus device=N) to cuda:0
CHECKPOINT = os.environ.get(
    "REMOTESAM_CHECKPOINT",
    "/checkpoints/swin_base_patch4_window12_384_22k.pth",
)
BERT_PATH = os.environ.get("REMOTESAM_BERT_PATH", "/checkpoints/bert-base-uncased")
USE_EPOC = os.environ.get("REMOTESAM_USE_EPOC", "false").lower() == "true"


def _patch_local_bert_path():
    """Force RemoteSAM hardcoded bert-base-uncased loads to local checkpoint files."""
    try:
        import args as remotesam_args
        import transformers
    except Exception:
        traceback.print_exc()
        return

    if not os.path.isdir(BERT_PATH):
        print(f"RemoteSAM local BERT path not found: {BERT_PATH}")
        return

    original_get_parser = remotesam_args.get_parser

    def get_parser_with_local_bert():
        parser = original_get_parser()
        parser.set_defaults(ck_bert=BERT_PATH, bert_tokenizer=BERT_PATH)
        return parser

    remotesam_args.get_parser = get_parser_with_local_bert

    original_tokenizer_from_pretrained = transformers.BertTokenizer.from_pretrained

    def local_tokenizer_from_pretrained(name_or_path, *args, **kwargs):
        if name_or_path == "bert-base-uncased":
            name_or_path = BERT_PATH
            kwargs.setdefault("local_files_only", True)
        return original_tokenizer_from_pretrained(name_or_path, *args, **kwargs)

    transformers.BertTokenizer.from_pretrained = local_tokenizer_from_pretrained
    print(f"RemoteSAM BERT path: {BERT_PATH}")

print(f"Loading RemoteSAM from {CHECKPOINT} ...")
print(f"EPOC refinement enabled: {USE_EPOC}")
try:
    _patch_local_bert_path()
    _base_model = init_demo_model(CHECKPOINT, DEVICE)
    model = RemoteSAM(_base_model, DEVICE, use_EPOC=USE_EPOC)
    print("RemoteSAM Ready!")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)


def _read_image(image_path):
    """Read and convert image; return PIL Image or None if path invalid or unreadable."""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _json_safe(value):
    """Convert numpy values returned by RemoteSAM into Flask-jsonifiable data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.get("/health")
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/referring_seg", methods=["POST"])
def referring_seg():
    data = request.json
    image_path = data.get("image_path")
    sentence = data.get("sentence")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        mask = model.referring_seg(image=image, sentence=sentence)
        return jsonify(_json_safe({"mask": mask})), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/semantic_seg", methods=["POST"])
def semantic_seg():
    data = request.json
    image_path = data.get("image_path")
    classnames = data.get("classnames")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        result = model.semantic_seg(image=image, classnames=classnames)
        masks = {cn: result[cn] for cn in classnames}
        return jsonify(_json_safe(masks)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/detection", methods=["POST"])
def detection():
    data = request.json
    image_path = data.get("image_path")
    classnames = data.get("classnames")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        result = model.detection(image=image, classnames=classnames)
        # result[classname] is None (no detections) or a list of [xmin,ymin,xmax,ymax,score]
        boxes = {
            cn: [list(b) for b in result[cn]] if result[cn] else []
            for cn in classnames
        }
        return jsonify(_json_safe(boxes)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/visual_grounding", methods=["POST"])
def visual_grounding():
    data = request.json
    image_path = data.get("image_path")
    sentence = data.get("sentence")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        box = model.visual_grounding(image=image, sentence=sentence)
        # box is None or [xmin, ymin, xmax, ymax] (Python list of numpy floats)
        return jsonify(_json_safe({"box": box if box is not None else None})), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/multi_label_cls", methods=["POST"])
def multi_label_cls():
    data = request.json
    image_path = data.get("image_path")
    classnames = data.get("classnames")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        result = model.multi_label_cls(image=image, classnames=classnames)
        return jsonify(_json_safe(result)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/multi_class_cls", methods=["POST"])
def multi_class_cls():
    data = request.json
    image_path = data.get("image_path")
    classnames = data.get("classnames")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        result = model.multi_class_cls(image=image, classnames=classnames)
        return jsonify(_json_safe(result)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/captioning", methods=["POST"])
def captioning():
    data = request.json
    image_path = data.get("image_path")
    classnames = data.get("classnames")
    region_split = data.get("region_split", 9)

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        result = model.captioning(image=image, classnames=classnames, region_split=region_split)
        return jsonify(_json_safe(result)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/counting", methods=["POST"])
def counting():
    data = request.json
    image_path = data.get("image_path")
    classnames = data.get("classnames")

    image = _read_image(image_path)
    if image is None:
        return jsonify({"error": f"Image not found or unreadable: {image_path}"}), 400

    try:
        result = model.counting(image=image, classnames=classnames)
        counts = {cn: result[cn] for cn in classnames}
        return jsonify(_json_safe(counts)), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9004)
