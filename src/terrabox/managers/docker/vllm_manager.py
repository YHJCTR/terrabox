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
from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    find_reusable_managed_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

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
    CONTAINER_BASE = "terrabox-vllm"
    DOCKER_IMAGE = os.environ.get("VLM_DOCKER_IMAGE", "terrabox/vllm:latest")
    GPU_DEVICES = os.environ.get("VLM_GPU_DEVICES", "0")
    TENSOR_PARALLEL_SIZE = os.environ.get("VLM_TENSOR_PARALLEL_SIZE", "1")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")
    MIN_IMAGE_MODEL_LEN = int(os.environ.get("VLM_MIN_IMAGE_MODEL_LEN", "16384"))
    MAX_MODEL_LEN = MIN_IMAGE_MODEL_LEN
    GPU_MEMORY_UTILIZATION = 0.8
    _lease = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        return cls._is_healthy_on_port(cls.PORT)

    @classmethod
    def _is_healthy_on_port(cls, port: int):
        try:
            resp = requests.get(
                f"http://{cls.HOST}:{port}/health",
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
    def _ensure_image_context_len(cls):
        if int(cls.MAX_MODEL_LEN) < int(cls.MIN_IMAGE_MODEL_LEN):
            logger.warning(
                "vlm_max_model_len=%s is too small for image tasks; using %s.",
                cls.MAX_MODEL_LEN,
                cls.MIN_IMAGE_MODEL_LEN,
            )
            cls.MAX_MODEL_LEN = int(cls.MIN_IMAGE_MODEL_LEN)

    @classmethod
    def _running_container_model_len(cls):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Cmd}}", cls.CONTAINER_NAME],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        try:
            import json
            cmd = json.loads(result.stdout)
            idx = cmd.index("--max-model-len")
            return int(cmd[idx + 1])
        except Exception:
            return None

    @classmethod
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} is running but service is not healthy; rebuilding it.")
            cls.stop_service()

        lease = acquire_docker_lease(
            service="vlm",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host=cls.HOST,
            base_port=cls.PORT,
            internal_port=8000,
            gpu_count=int(cls.TENSOR_PARALLEL_SIZE),
            min_free_mib=16384,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="VLM_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.PORT = lease.port
        cls.API_BASE = f"http://{cls.HOST}:{cls.PORT}/v1"
        remove_container_if_exists(cls.CONTAINER_NAME, service="vlm", reason="before_docker_run")

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            # --gpus all exposes all GPUs to the container; CUDA_VISIBLE_DEVICES then
            # restricts which ones CUDA actually uses (works even with --gpus all,
            # unlike NVIDIA_VISIBLE_DEVICES which --gpus all overrides).
            "--gpus", "all",
            "-e", f"CUDA_VISIBLE_DEVICES={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
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

        logger.info(f"Starting vLLM container (GPU: {lease.gpu_devices}, port: {lease.port}, model: {cls.MODEL_PATH})...")
        record_service_event({"event": "start_requested", "service": "vlm", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "vlm", "stderr": result.stderr, "lease": lease.__dict__})
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

        cls._ensure_image_context_len()

        if cls.is_running():
            running_len = cls._running_container_model_len()
            if running_len is None or running_len >= cls.MIN_IMAGE_MODEL_LEN:
                return
            logger.warning(
                "Running %s uses --max-model-len %s, below image-safe minimum %s; recreating.",
                cls.CONTAINER_NAME,
                running_len,
                cls.MIN_IMAGE_MODEL_LEN,
            )
            cls.stop_service()

        adopted = find_reusable_managed_lease(
            service="vlm",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host=cls.HOST,
            internal_port=8000,
            required_model_path=cls.MODEL_PATH,
            required_cmd_args={
                "--tensor-parallel-size": cls.TENSOR_PARALLEL_SIZE,
                "--max-model-len": str(cls.MAX_MODEL_LEN),
            },
            health_check=cls._is_healthy_on_port,
        )
        if adopted is not None:
            cls._lease = adopted
            cls.CONTAINER_NAME = adopted.container_name
            cls.PORT = adopted.port
            cls.API_BASE = f"http://{cls.HOST}:{cls.PORT}/v1"
            logger.info("Adopted existing vLLM container %s on port %s", adopted.container_name, adopted.port)
            return

        cls._start_docker()

        logger.info("Waiting for vLLM to load model (this may take 5-10 minutes)...")
        max_retries = 120   # poll every 5s, up to 600s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("vLLM service is READY!")
                record_service_event({"event": "ready", "service": "vlm", "lease": cls._lease.__dict__ if cls._lease else None})
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
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        record_service_event({"event": "stopped", "service": "vlm", "container": cls.CONTAINER_NAME})


vllm_manager = VLLMDockerManager()
