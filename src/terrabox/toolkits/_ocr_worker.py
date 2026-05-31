"""Subprocess worker for EasyOCR so GPU memory is released on process exit."""
from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    image_path = payload["image_path"]
    languages = payload["languages"]
    gpu = payload["gpu"]

    import easyocr

    reader = easyocr.Reader(languages, gpu=gpu)
    results = reader.readtext(image_path)
    texts = []
    for bbox_coords, text, confidence in results:
        texts.append({
            "text": text,
            "confidence": round(float(confidence), 3),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox_coords],
        })

    print(json.dumps({"texts": texts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
