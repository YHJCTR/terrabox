"""
RemoteCLIP Server for Docker deployment.
Derived from RemoteCLIP/start2.py with the following changes:
  1. Checkpoint path read from environment variable REMOTECLIP_CKPT_DIR
  2. Flask app binds to 0.0.0.0 instead of 127.0.0.1
"""
import os
import torch
import open_clip
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

model_name = 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

CKPT_DIR = os.environ.get("REMOTECLIP_CKPT_DIR", "/checkpoints")
ckpt_path = os.path.join(
    CKPT_DIR,
    "models--chendelong--RemoteCLIP",
    "snapshots",
    "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38",
    f"RemoteCLIP-{model_name}.pt"
)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
model = model.cuda().eval()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("=== Received Request ===")
    print(f"Full request JSON: {request.json}")
    print(f"Raw text_queries: {request.json.get('text_queries')}")

    data = request.json
    image_path = data.get("image_path")
    text_queries = data.get("text_queries")

    print(f"Final text_queries used: {text_queries}")
    print("========================")

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid image path"}), 400

    text = tokenizer(text_queries)
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image.cuda())
        text_features = model.encode_text(text.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

    predictions = []
    for query, prob in zip(text_queries, text_probs):
        predictions.append({
            "query": query,
            "probability": float(prob * 100)
        })

    return jsonify({"predictions": predictions}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9003)
