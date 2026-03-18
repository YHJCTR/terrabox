"""
InstructSAM Service Manager - Docker Mode
==========================================
Launches terrabox/instructsam:latest via 'docker run'.
Imported by geo_perception.py when TERRABOX_USE_DOCKER=true.

Volume mounts:
  -v /data1:/data1                          image files (same path inside container)
  -v ${INSTRUCTSAM_MODELS_HOST}:/models:ro  model directory (SAM2 + CLIP, no Qwen)

Model directory layout (host side, default /data1/yuhongjie2/terra_model/instructsam):
  sam2_hiera_large.pt       SAM2 Hiera Large weights
  GeoRSCLIP-ViT-L-14.pt     GeoRSCLIP CLIP weights

Note: the counting step calls the host vLLM service (port 9000) via HTTP; Qwen is not
downloaded inside the container.

Build the image first:
  cd docker/instructsam && docker build -t terrabox/instructsam:latest .

Download models first:
  python scripts/download_instructsam_models.py --skip-qwen
"""

import subprocess
import time
import os
import requests
import logging
from ..gpu_allocator import allocate_gpu
from ..base_manager import BaseServiceManager

logger = logging.getLogger("docker.instructsam_manager")


class InstructSAMDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:9006"

    CONTAINER_NAME = "terrabox-instructsam"
    DOCKER_IMAGE   = "terrabox/instructsam:latest"
    GPU_DEVICES    = os.environ.get("INSTRUCTSAM_GPU_DEVICES", "1")

    # host-side model directory, mounted as /models inside the container
    MODELS_HOST = os.environ.get(
        "INSTRUCTSAM_MODELS_HOST",
        "/data1/yuhongjie2/terra_model/instructsam",
    )
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        try:
            return requests.get(
                f"{cls.API_URL}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            ).status_code == 200
        except Exception:
            return False

    @classmethod
    def _container_is_running(cls):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", cls.CONTAINER_NAME],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} already running.")
            return

        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

        # InstructSAM loads SAM2 + CLIP only, requiring ~4 GB VRAM
        gpu = allocate_gpu(
            min_free_mib=4096,
            fallback=cls.GPU_DEVICES,
            env_var="INSTRUCTSAM_GPU_DEVICES",
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "--gpus", f"device={gpu}",
            "-p", "9006:9006",
            # Allow the container to reach the host vLLM service (port 9000)
            "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.MODELS_HOST}:/models:ro",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting InstructSAM container (GPU: {gpu})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start InstructSAM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        cls._start_docker()

        # SAM2 (~15s) + CLIP (~5s), no Qwen — ready in ~30s
        logger.info("Waiting for InstructSAM service (SAM2 + CLIP loading ~30s)...")
        max_retries = 60    # poll every 2s, wait up to 120s total
        for i in range(max_retries):
            if cls.is_running():
                logger.info("InstructSAM service is READY.")
                return
            if i % 15 == 0 and i > 0:
                logger.info(f"Still loading... ({i * 2}s elapsed)")
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError(
            "InstructSAM service failed to start. "
            "Check: docker logs terrabox-instructsam"
        )

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm",   cls.CONTAINER_NAME], capture_output=True)


instructsam_manager = InstructSAMDockerManager()
