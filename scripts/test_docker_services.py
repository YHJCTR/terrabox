#!/usr/bin/env python3
"""
Terrabox Docker Services Integration Test
==========================================

Tests each AI perception service by:
  1. Starting the Docker container (skips if already running)
  2. Waiting for the /health endpoint to become ready
  3. Sending a real inference request with a test image
  4. Printing the result and reporting PASS / FAIL
  5. Optionally stopping the container (default: stop after test)

Usage:
    # Test all services:
    python scripts/test_docker_services.py

    # Test a specific service:
    python scripts/test_docker_services.py --service sam2
    python scripts/test_docker_services.py --service remoteclip
    python scripts/test_docker_services.py --service remotesam
    python scripts/test_docker_services.py --service strip-rcnn

    # Keep containers running after test (useful for debugging):
    python scripts/test_docker_services.py --service sam2 --no-stop

    # Use a custom test image:
    python scripts/test_docker_services.py --image /path/to/image.png
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests


# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------

DEFAULT_TEST_IMAGE = "/data1/yuhongjie2/test.png"

SERVICES = {
    "sam2": {
        "container_name": "terrabox-sam2",
        "docker_image":   "terrabox/sam2:latest",
        "port":           9002,
        "api_url":        "http://127.0.0.1:9002",
        "gpu_env_var":    "SAM2_GPU_DEVICES",
        "default_gpu":    "0",
        "docker_run_extra": [
            "-v", "/data1/yuhongjie2/sam2/checkpoints:/checkpoints:ro",
            "-v", "/data1/yuhongjie2/sam2/sam2/configs:/sam2_configs:ro",
            "-v", "/data1:/data1",
            "-e", "SAM2_CHECKPOINT=/checkpoints/sam2.1_hiera_large.pt",
            "-e", "SAM2_CONFIG=/sam2_configs/sam2.1/sam2.1_hiera_l.yaml",
        ],
        "ready_timeout":  120,
        "poll_interval":  2,
    },
    "vllm": {
        "container_name": "terrabox-vllm",
        "docker_image":   "terrabox/vllm:latest",
        "port":           9000,
        "container_port": 8000,
        "api_url":        "http://127.0.0.1:9000",
        "gpu_env_var":    "VLM_GPU_DEVICES",
        "default_gpu":    "2,3",
        "docker_run_extra": [
            "-v", "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model:/model:ro",
            "-v", "/data1:/data1",
            "--shm-size=16g",
        ],
        "container_args": [
            "--model", "/model",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "2",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.8",
            "--enforce-eager",
            "--allowed-local-media-path", "/data1",
        ],
        "ready_timeout":  300,
        "poll_interval":  5,
    },
    "remoteclip": {
        "container_name": "terrabox-remoteclip",
        "docker_image":   "terrabox/remoteclip:latest",
        "port":           9003,
        "api_url":        "http://127.0.0.1:9003",
        "gpu_env_var":    "REMOTECLIP_GPU_DEVICES",
        "default_gpu":    "1",
        "docker_run_extra": [
            "-v", "/data1/yuhongjie2/RemoteCLIP/checkpoints:/checkpoints:ro",
            "-v", "/data1:/data1",
            "-e", "REMOTECLIP_CKPT_DIR=/checkpoints",
        ],
        "ready_timeout":  90,
        "poll_interval":  2,
    },
    "remotesam": {
        "container_name": "terrabox-remotesam",
        "docker_image":   "terrabox/remotesam:latest",
        "port":           9004,
        "api_url":        "http://127.0.0.1:9004",
        "gpu_env_var":    "REMOTESAM_GPU_DEVICES",
        "default_gpu":    "2",
        "docker_run_extra": [
            "-v", "/data1/yuhongjie2/RemoteSAM/pretrained_weights:/checkpoints:ro",
            "-v", "/data1:/data1",
            # Mount HuggingFace cache so bert-base-uncased resolves without network
            "-v", f"{os.path.expanduser('~/.cache/huggingface')}:/root/.cache/huggingface:ro",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_DATASETS_OFFLINE=1",
        ],
        "ready_timeout":  100,   # Swin-B model loading can take ~3 min
        "poll_interval":  2,
    },
    "strip-rcnn": {
        "container_name": "terrabox-strip-rcnn",
        "docker_image":   "terrabox/strip-rcnn:latest",
        "port":           9005,
        "api_url":        "http://127.0.0.1:9005",
        "gpu_env_var":    "STRIP_RCNN_GPU_DEVICES",
        "default_gpu":    "3",
        "docker_run_extra": [
            "-v", "/data1/yuhongjie2/Strip-RCNN/ckpt:/ckpt:ro",
            "-v", "/data1/yuhongjie2/Strip-RCNN/configs:/configs:ro",
            "-v", "/data1:/data1",
            # Strip-RCNN ships a custom mmrotate that registers StripRCNN.
            # Adding the repo root to PYTHONPATH makes Python prefer the local
            # mmrotate over the globally installed one in the Docker image.
            "-e", "PYTHONPATH=/data1/yuhongjie2/Strip-RCNN",
        ],
        # These args are appended after the image name and forwarded to start.py
        "container_args": [
            "--config",     "/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py",
            "--checkpoint", "/ckpt/stripnet_s.pth",
            "--device",     "cuda:0",
            "--host",       "0.0.0.0",
            "--port",       "9005",
        ],
        "ready_timeout":  180,   # mmrotate model init can take ~2 min
        "poll_interval":  2,
    },
    "instructsam": {
        # InstructSAM (NeurIPS 2025): Training-Free instruction-based segmentation.
        # Pipeline: SAM2 mask proposals → vLLM counting (HTTP) → GeoRSCLIP matching.
        # 镜像与 sam2 使用相同基础: nvidia/cuda:11.8.0-devel-ubuntu22.04
        # Models: /data1/yuhongjie2/terra_model/instructsam/
        #   sam2_hiera_large.pt  |  GeoRSCLIP-ViT-L-14.pt
        # Download first:  python scripts/download_instructsam_models.py --skip-qwen
        # Build image:     cd docker/instructsam && bash build.sh
        "container_name": "terrabox-instructsam",
        "docker_image":   "terrabox/instructsam:latest",
        "port":           9006,
        "api_url":        "http://127.0.0.1:9006",
        "gpu_env_var":    "INSTRUCTSAM_GPU_DEVICES",
        "default_gpu":    "1",
        "docker_run_extra": [
            "--add-host", "host.docker.internal:host-gateway",
            "-v", "/data1/yuhongjie2/terra_model/instructsam:/models:ro",
            "-v", "/data1:/data1",
        ],
        "ready_timeout":  120,   # SAM2 + CLIP loading ~30s
        "poll_interval":  2,
    },
}


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def container_is_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def service_is_healthy(api_url: str) -> bool:
    try:
        resp = requests.get(
            f"{api_url}/health",
            timeout=2,
            proxies={"http": None, "https": None},
        )
        return resp.status_code == 200
    except Exception:
        return False


def start_container(service_name: str, cfg: dict, gpu: str) -> bool:
    """Start the Docker container. Returns True on success."""
    container_name = cfg["container_name"]

    if container_is_running(container_name):
        print(f"  [INFO] Container '{container_name}' already running, skipping start.")
        return True

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
    ]
    
    # Handle GPU specification: use --gpus all + NVIDIA_VISIBLE_DEVICES for multi-GPU
    if "," in gpu:
        cmd.extend(["--gpus", "all", "-e", f"NVIDIA_VISIBLE_DEVICES={gpu}"])
    else:
        cmd.extend(["--gpus", f"device={gpu}"])
    
    # Port mapping: use container_port if specified, otherwise use same port
    container_port = cfg.get("container_port", cfg["port"])
    cmd.extend(["-p", f"{cfg['port']}:{container_port}"])
    cmd.extend(cfg.get("docker_run_extra", []))
    cmd.append(cfg["docker_image"])
    cmd.extend(cfg.get("container_args", []))

    print(f"  [CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] Failed to start container:\n{result.stderr.strip()}")
        return False

    print(f"  [OK] Container started (ID: {result.stdout.strip()[:12]})")
    return True


def wait_for_service(api_url: str, timeout: int, poll_interval: int) -> bool:
    """Poll /health until ready or timeout. Returns True if healthy."""
    elapsed = 0
    while elapsed < timeout:
        if service_is_healthy(api_url):
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed % 20 == 0:
            print(f"  [WAIT] Still waiting... ({elapsed}s / {timeout}s)")
    return False


def stop_container(container_name: str):
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm",   container_name], capture_output=True)
    print(f"  [OK] Container '{container_name}' stopped and removed.")


def check_image_exists(docker_image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", docker_image],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def print_container_logs(container_name: str, tail: int = 50):
    """Print the last N log lines from a container to aid failure diagnosis."""
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_name],
        capture_output=True, text=True,
    )
    logs = (result.stdout + result.stderr).strip()
    if logs:
        print(f"  [LOGS] Last {tail} lines from '{container_name}':")
        for line in logs.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# Per-service inference tests
# ---------------------------------------------------------------------------

def test_sam2(api_url: str, image_path: str) -> dict:
    """POST /segment — full-image box prompt."""
    resp = requests.post(
        f"{api_url}/segment",
        json={"image_path": image_path},
        timeout=120,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    result = resp.json()
    passed = result.get("status") == "success" and result.get("count", -1) >= 0
    return {
        "passed":  passed,
        "status":  result.get("status"),
        "count":   result.get("count"),
        "message": result.get("message", ""),
    }


def test_remoteclip(api_url: str, image_path: str) -> dict:
    """POST /analyze — zero-shot classification with text queries."""
    resp = requests.post(
        f"{api_url}/analyze",
        json={
            "image_path":   image_path,
            "text_queries": [
                "A satellite image of a city.",
                "A satellite image of a road.",
                "A satellite image of farmland.",
            ],
        },
        timeout=60,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    result = resp.json()
    predictions = result.get("predictions", [])
    top = max(predictions, key=lambda p: p["probability"]) if predictions else None
    return {
        "passed":      len(predictions) > 0,
        "predictions": predictions,
        "top_match":   top["query"] if top else None,
        "top_prob":    round(top["probability"], 2) if top else None,
    }


def test_remotesam(api_url: str, image_path: str) -> dict:
    """POST /referring_seg — referring segmentation with a text sentence."""
    resp = requests.post(
        f"{api_url}/referring_seg",
        json={
            "image_path": image_path,
            "sentence":   "road",
        },
        timeout=120,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    result = resp.json()
    mask = result.get("mask", [])
    mask_h = len(mask)
    mask_w = len(mask[0]) if mask_h > 0 else 0
    passed = mask_h > 0 and mask_w > 0
    return {
        "passed":     passed,
        "mask_shape": [mask_h, mask_w],
    }


def test_strip_rcnn(api_url: str, image_path: str) -> dict:
    """POST /detect — oriented object detection."""
    resp = requests.post(
        f"{api_url}/detect",
        json={"image_path": image_path, "score_threshold": 0.01},
        timeout=60,
        proxies={"http": None, "https": None},
    )
    result = resp.json()
    
    # Print debug info
    print(f"[DEBUG] num_detections: {result.get('num_detections', 0)}")
    
    return {
        "passed":           result.get("success", False),
        "num_detections":   result.get("num_detections", 0),
        "classes_detected": list(result.get("detections", {}).keys()),
        "error":            result.get("error"),
    }


def test_vllm(api_url: str, image_path: str) -> dict:
    """POST /v1/chat/completions — VLM inference via OpenAI-compatible API."""
    resp = requests.post(
        f"{api_url}/v1/chat/completions",
        json={
            "model": "/model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                        {"type": "text", "text": "What is in this image?"},
                    ],
                }
            ],
            "max_tokens": 256,
        },
        timeout=120,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    result = resp.json()
    passed = "choices" in result and len(result.get("choices", [])) > 0
    return {
        "passed":    passed,
        "model":     result.get("model"),
        "content":   result.get("choices", [{}])[0].get("message", {}).get("content", "")[:100],
    }


def test_instructsam(api_url: str, image_path: str) -> dict:
    """POST /segment — InstructSAM instruction-based segmentation and counting."""
    resp = requests.post(
        f"{api_url}/segment",
        json={
            "image_path":  image_path,
            "text_prompt": "Count all the objects in this image.",
        },
        timeout=300,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    result = resp.json()
    count  = result.get("count", -1)
    passed = result.get("status") == "success" and count >= 0
    return {
        "passed":     passed,
        "status":     result.get("status"),
        "count":      count,
        "objects":    result.get("objects", [])[:5],   # 最多显示 5 个
        "error":      result.get("error"),
    }


INFERENCE_FUNCS = {
    "sam2":         test_sam2,
    "remoteclip":   test_remoteclip,
    "remotesam":    test_remotesam,
    "strip-rcnn":   test_strip_rcnn,
    "vllm":         test_vllm,
    "instructsam":  test_instructsam,
}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_service_test(service_name: str, image_path: str, stop_after: bool) -> bool:
    """Full test lifecycle for one service. Returns True if the test passed."""
    cfg = SERVICES[service_name]
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Testing: {service_name.upper()}  (image: {cfg['docker_image']}, port: {cfg['port']})")
    print(sep)

    # Check image exists
    if not check_image_exists(cfg["docker_image"]):
        print(f"  [SKIP] Docker image '{cfg['docker_image']}' not found.")
        print(f"         Run './build_models.sh' to build it first.")
        return False

    # Resolve GPU
    gpu = os.environ.get(cfg["gpu_env_var"], cfg["default_gpu"])
    print(f"  GPU: {gpu} (override with ${cfg['gpu_env_var']})")

    # Step 1: Start container (if not already running / healthy)
    already_healthy = service_is_healthy(cfg["api_url"])
    if already_healthy:
        print("  [INFO] Service already healthy, skipping container start.")
    else:
        print("  [STEP 1] Starting container...")
        ok = start_container(service_name, cfg, gpu)
        if not ok:
            print(f"  [FAIL] Could not start container for '{service_name}'.")
            return False

        # Step 2: Wait for health
        print(f"  [STEP 2] Waiting for service (timeout={cfg['ready_timeout']}s)...")
        healthy = wait_for_service(cfg["api_url"], cfg["ready_timeout"], cfg["poll_interval"])
        if not healthy:
            print(f"  [FAIL] Service did not become healthy within {cfg['ready_timeout']}s.")
            print_container_logs(cfg["container_name"])
            if stop_after:
                stop_container(cfg["container_name"])
            return False

        print("  [OK] Service is healthy.")

    # Step 3: Inference
    print(f"  [STEP 3] Running inference on: {image_path}")
    test_fn = INFERENCE_FUNCS[service_name]
    try:
        result = test_fn(cfg["api_url"], image_path)
    except Exception as exc:
        print(f"  [FAIL] Inference error: {exc}")
        print_container_logs(cfg["container_name"])
        if stop_after and not already_healthy:
            stop_container(cfg["container_name"])
        return False

    passed = result.pop("passed", False)
    print(f"  [RESULT]\n{json.dumps(result, indent=6, ensure_ascii=False)}")

    status_tag = "[PASS]" if passed else "[FAIL]"
    print(f"  {status_tag} {service_name.upper()}")

    # Step 4: Optionally stop
    if stop_after and not already_healthy:
        print("  [STEP 4] Stopping container...")
        stop_container(cfg["container_name"])

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test Terrabox Docker AI perception services."
    )
    parser.add_argument(
        "--service",
        choices=list(SERVICES.keys()) + ["all"],
        default="all",
        help="Which service to test (default: all). "
             "Services: " + ", ".join(SERVICES.keys()),
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_TEST_IMAGE,
        help=f"Path to test image (default: {DEFAULT_TEST_IMAGE}).",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Keep containers running after the test.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Test image not found: {args.image}")
        sys.exit(1)

    stop_after = not args.no_stop
    services_to_test = list(SERVICES.keys()) if args.service == "all" else [args.service]

    results = {}
    for svc in services_to_test:
        results[svc] = run_service_test(svc, args.image, stop_after)

    # Summary
    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")
    all_passed = True
    for svc, passed in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  {svc:<15} {tag}")
        if not passed:
            all_passed = False
    print(f"{'=' * 64}")

    if all_passed:
        print("  All tests PASSED.")
        sys.exit(0)
    else:
        print("  Some tests FAILED. See output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
