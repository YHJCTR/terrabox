"""
vLLM Service Manager - Docker Mode
===================================
Uses 'docker run' to start vllm/vllm-openai:latest instead of subprocess.
Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception.py imports.

Volume mounts:
  -v ${VLM_MODEL_PATH}:/model:ro    model weights (read-only)
  --shm-size=16g                    shared memory for tensor parallelism

Port mapping:
  -p 9000:8000  (host:container)
  The official vllm-openai image serves on port 8000 internally.

GPU:
  --gpus "device=2,3"  remapped inside container as cuda:0, cuda:1
"""

import subprocess
import time
import os
import requests
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docker.vllm_manager")


class VLLMDockerManager:
    _instance = None

    HOST = "127.0.0.1"
    PORT = 9000
    API_BASE = f"http://{HOST}:{PORT}/v1"

    # Host path to model directory, mounted as /model inside the container
    MODEL_PATH = os.environ.get(
        "VLM_MODEL_PATH",
        "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model/"
    )
    # In-container path used as the model identifier in vLLM API requests
    MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "/model")

    CONTAINER_NAME = "terrabox-vllm"
    DOCKER_IMAGE = "vllm/vllm-openai:latest"
    GPU_DEVICES = os.environ.get("VLM_GPU_DEVICES", "2,3")
    TENSOR_PARALLEL_SIZE = os.environ.get("VLM_TENSOR_PARALLEL_SIZE", "2")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        try:
            resp = requests.get(f"http://{cls.HOST}:{cls.PORT}/health", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def _container_is_running(cls):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", cls.CONTAINER_NAME],
            capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} already running.")
            return

        # Remove any stopped container with the same name to avoid "name already in use"
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "--gpus", f"device={cls.GPU_DEVICES}",
            "-p", f"{cls.PORT}:8000",   # vllm-openai image listens on 8000 internally
            "-v", f"{cls.MODEL_PATH}:/model:ro",
            "--shm-size=16g",
            cls.DOCKER_IMAGE,
            "--model", "/model",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", cls.TENSOR_PARALLEL_SIZE,
            "--max-model-len", "4096",
            "--limit-mm-per-prompt", '{"image": 8}',
            "--gpu-memory-utilization", "0.9",
            "--enforce-eager",
        ]

        logger.info(f"Starting vLLM container (GPU: {cls.GPU_DEVICES}, model: {cls.MODEL_PATH})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start vLLM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for vLLM to load model (this may take 5-10 minutes)...")
        max_retries = 120   # poll every 5s, up to 600s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("vLLM service is READY!")
                return

            if i % 6 == 0:
                logger.info(f"Still loading... ({i * 5}s elapsed)")

            time.sleep(5)

        cls.stop_service()
        raise TimeoutError("vLLM service failed to start within timeout. Check: docker logs terrabox-vllm")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", cls.CONTAINER_NAME], capture_output=True)


vllm_manager = VLLMDockerManager()


@atexit.register
def _auto_cleanup():
    # Containers intentionally keep running after the host process exits (allows reuse).
    # Uncomment below to stop automatically:
    # vllm_manager.stop_service()
    pass
