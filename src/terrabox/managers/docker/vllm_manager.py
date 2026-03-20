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
from ..gpu_allocator import allocate_gpus
from ..base_manager import BaseServiceManager

logger = logging.getLogger("docker.vllm_manager")


class VLLMDockerManager(BaseServiceManager):
    _instance = None

    HOST = "127.0.0.1"
    PORT = 9000
    API_BASE = f"http://{HOST}:{PORT}/v1"

    # Host path to model directory, mounted as /model inside the container
    MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "")
    # In-container path used as the model identifier in vLLM API requests
    MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "/model")

    CONTAINER_NAME = "terrabox-vllm"
    DOCKER_IMAGE = os.environ.get("VLM_DOCKER_IMAGE", "terrabox/vllm:latest")
    GPU_DEVICES = os.environ.get("VLM_GPU_DEVICES", "0")
    TENSOR_PARALLEL_SIZE = os.environ.get("VLM_TENSOR_PARALLEL_SIZE", "1")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")
    MAX_MODEL_LEN = 4096
    GPU_MEMORY_UTILIZATION = 0.8

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        try:
            resp = requests.get(
                f"http://{cls.HOST}:{cls.PORT}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            )
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

        gpu = allocate_gpus(
            count=int(cls.TENSOR_PARALLEL_SIZE),
            min_free_mib=16384,
            fallback=cls.GPU_DEVICES,
            env_var="VLM_GPU_DEVICES",
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            # --gpus all exposes all GPUs to the container; CUDA_VISIBLE_DEVICES then
            # restricts which ones CUDA actually uses (works even with --gpus all,
            # unlike NVIDIA_VISIBLE_DEVICES which --gpus all overrides).
            "--gpus", "all",
            "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
            "-p", f"{cls.PORT}:8000",
            "-v", f"{cls.MODEL_PATH}:/model:ro",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "--shm-size=16g",
            cls.DOCKER_IMAGE,
            "--model", "/model",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", cls.TENSOR_PARALLEL_SIZE,
            "--max-model-len", str(cls.MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(cls.GPU_MEMORY_UTILIZATION),
            "--enforce-eager",
            "--allowed-local-media-path", cls.DATA_MOUNT_HOST,
        ]

        logger.info(f"Starting vLLM container (GPU: {gpu}, model: {cls.MODEL_PATH})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start vLLM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def _get_container_logs(cls, tail: int = 50) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), cls.CONTAINER_NAME],
            capture_output=True, text=True
        )
        return (result.stdout + result.stderr).strip()

    @classmethod
    def start_service(cls):
        try:
            from ...agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "vlm_model_path"             in d: cls.MODEL_PATH             = str(d["vlm_model_path"])
            if "vlm_port"                   in d: cls.PORT                   = int(d["vlm_port"]); cls.API_BASE = f"http://{cls.HOST}:{cls.PORT}/v1"
            if "vlm_gpu_devices"            in d: cls.GPU_DEVICES            = str(d["vlm_gpu_devices"])
            if "vlm_tensor_parallel"        in d: cls.TENSOR_PARALLEL_SIZE   = str(int(d["vlm_tensor_parallel"]))
            if "vlm_max_model_len"          in d: cls.MAX_MODEL_LEN          = int(d["vlm_max_model_len"])
            if "vlm_gpu_memory_utilization" in d: cls.GPU_MEMORY_UTILIZATION = float(d["vlm_gpu_memory_utilization"])
            if "docker_data_mount_host"     in d: cls.DATA_MOUNT_HOST        = str(d["docker_data_mount_host"])
        except Exception:
            pass

        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for vLLM to load model (this may take 5-10 minutes)...")
        max_retries = 120   # poll every 5s, up to 600s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("vLLM service is READY!")
                return

            # Mirror the non-Docker manager's process.poll() check:
            # if the container has already exited, stop waiting immediately.
            if not cls._container_is_running():
                logs = cls._get_container_logs()
                raise RuntimeError(
                    f"vLLM container exited unexpectedly.\n"
                    f"--- docker logs (last 50 lines) ---\n{logs}"
                )

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
