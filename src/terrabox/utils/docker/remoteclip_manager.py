"""
RemoteCLIP Service Manager - Docker Mode
==========================================
Uses 'docker run' to start terrabox/remoteclip:latest instead of subprocess.
Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception.py imports.

Volume mounts:
  -v /data1:/data1                              image files (same path inside container)
  -v ${REMOTECLIP_CKPT_HOST}:/checkpoints:ro    RemoteCLIP checkpoint

Build image first:
  cd docker/remoteclip && docker build -t terrabox/remoteclip:latest .
"""

import subprocess
import time
import os
import requests
import logging
import atexit
from ..gpu_allocator import allocate_gpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docker.remoteclip_manager")


class RemoteCLIPDockerManager:
    _instance = None

    API_URL = "http://127.0.0.1:9003"

    CONTAINER_NAME = "terrabox-remoteclip"
    DOCKER_IMAGE = "terrabox/remoteclip:latest"
    GPU_DEVICES = os.environ.get("REMOTECLIP_GPU_DEVICES", "3")

    CKPT_HOST = os.environ.get(
        "REMOTECLIP_CKPT_HOST",
        "/data1/yuhongjie2/RemoteCLIP/checkpoints"
    )
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        try:
            return requests.get(f"{cls.API_URL}/health", timeout=1).status_code == 200
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

        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

        gpu = allocate_gpu(
            min_free_mib=4096,
            fallback=cls.GPU_DEVICES,
            env_var="REMOTECLIP_GPU_DEVICES",
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "--gpus", f"device={gpu}",
            "-p", "9003:9003",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.CKPT_HOST}:/checkpoints:ro",
            "-e", "REMOTECLIP_CKPT_DIR=/checkpoints",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting RemoteCLIP container (GPU: {gpu})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start RemoteCLIP container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for RemoteCLIP service to start...")
        max_retries = 60
        for i in range(max_retries):
            if cls.is_running():
                logger.info("RemoteCLIP service is READY.")
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("RemoteCLIP service failed to start. Check: docker logs terrabox-remoteclip")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", cls.CONTAINER_NAME], capture_output=True)


remoteclip_manager = RemoteCLIPDockerManager()


@atexit.register
def _cleanup():
    pass
