# strip_api_server.py
# Strip R-CNN Detection API Server
# Usage: python strip_api_server.py

import os
import sys
from argparse import ArgumentParser

from flask import Flask, request, jsonify
import cv2
import numpy as np

# Import MMRotate first to register StripRCNN model type
import mmrotate  # Must import before mmdet APIs to register models
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = ArgumentParser(description='Strip R-CNN API Server')
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference (e.g., cuda:0 or cpu)')
    parser.add_argument('--host', default='0.0.0.0', help='Host address to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=9005, help='Port to listen on (default: 9005)')
    args = parser.parse_args()
    return args


def init_strip_model(config_path, checkpoint_path, device):
    """
    Initialize Strip R-CNN model
    """
    # Now MMDet can recognize StripRCNN because mmrotate was imported first
    model = init_detector(config_path, checkpoint_path, device=device)
    
    # Manually set DOTA classes to override checkpoint's default COCO classes
    # Checkpoint doesn't save class names, so we must set them manually
    model.CLASSES = (
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    )
    
    print(f"Model loaded successfully on {device}")
    print(f"Classes: {model.CLASSES}")
    return model


def process_detection_result(result, class_names):
    """
    Convert MMRotate detection result to JSON format

    Args:
        result: MMRotate output format [(bbox, score, label), ...]
        class_names: Class name list

    Returns:
        dict: {'class_name': [[cx, cy, w, h, angle, score], ...]}
    """
    detections = {}
    
    print(f"[DEBUG] Number of classes in result: {len(result)}")
    
    for cls_idx, cls_dets in enumerate(result):
        if len(cls_dets) == 0:
            continue
        
        cls_name = class_names[cls_idx]
        detections[cls_name] = []
        
        print(f"[DEBUG] Class {cls_idx} ({cls_name}): len={len(cls_dets)}, shape={cls_dets.shape if hasattr(cls_dets, 'shape') else 'N/A'}")
        if len(cls_dets) > 0:
            print(f"[DEBUG]   First det: {cls_dets[0]}")
            print(f"[DEBUG]   Last det: {cls_dets[-1]}")
        
        for det in cls_dets:
            # det format: [cx, cy, w, h, angle, score] (for oriented R-CNN)
            cx, cy, w, h, angle, score = det
            detections[cls_name].append([float(cx), float(cy), float(w), float(h), float(angle), float(score)])
    
    return detections


def validate_image_path(image_path):
    """
    Validate image path exists and is supported format
    """
    if not os.path.exists(image_path):
        return False, f"Image path does not exist: {image_path}"

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        return False, f"Unsupported image format: {ext}"

    return True, ""


def create_app(config_path, checkpoint_path, device):
    app = Flask(__name__)

    # Initialize model using MMRotate registered types
    global model
    model = init_strip_model(config_path, checkpoint_path, device)

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "device": device,
            "classes_count": len(model.CLASSES),
            "classes": list(model.CLASSES)
        }), 200

    @app.route('/detect', methods=['POST'])
    def detect_objects():
        """
        Receive image path, return oriented bounding box detection results

        Request JSON:
        {
            "image_path": "/path/to/image.jpg",
            "score_threshold": 0.3  // optional, default 0.3
        }

        Response JSON:
        {
            "success": true,
            "detections": {
                "class_name": [
                    [cx, cy, w, h, angle, score],
                    ...
                ]
            },
            "image_size": [height, width],
            "num_detections": total_count
        }
        """
        try:
            data = request.json
            image_path = data.get("image_path")

            if not image_path:
                return jsonify({"error": "Missing 'image_path' in request"}), 400

            # Validate image path
            is_valid, msg = validate_image_path(image_path)
            if not is_valid:
                return jsonify({"error": msg}), 400

            # Get confidence threshold
            score_threshold = float(data.get("score_threshold", 0.3))

            # Perform inference with MMRotate model through MMDet API
            result = inference_detector(model, image_path)

            # Process detection results
            detections = process_detection_result(result, model.CLASSES)

            # Filter low-confidence detections
            filtered_detections = {}
            total_count = 0

            for cls_name, dets in detections.items():
                high_conf_dets = [det for det in dets if det[5] >= score_threshold]
                if high_conf_dets:
                    filtered_detections[cls_name] = high_conf_dets
                    total_count += len(high_conf_dets)

            # Get image size
            img = cv2.imread(image_path)
            height, width = img.shape[:2]

            response = {
                "success": True,
                "detections": filtered_detections,
                "image_size": [height, width],
                "num_detections": total_count,
                "score_threshold": score_threshold
            }

            return jsonify(response), 200

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route('/classes', methods=['GET'])
    def get_classes():
        """
        Get all supported classes by the model
        """
        return jsonify({
            "classes": list(model.CLASSES),
            "count": len(model.CLASSES)
        }), 200

    return app


def main():
    args = parse_args()

    # Create Flask app
    app = create_app(args.config, args.checkpoint, args.device)

    print(f"Starting Strip R-CNN API Server...")
    print(f"Model Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Server: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
