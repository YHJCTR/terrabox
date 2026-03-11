"""
Strip-RCNN Service Manager - Docker Mode
==========================================
Uses 'docker run' to start terrabox/strip-rcnn:latest instead of subprocess.
Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception.py imports.

Volume mounts:
  -v /data1:/data1                                image files (same path inside container)
  -v ${STRIP_RCNN_CKPT_HOST}:/ckpt:ro             model weights
  -v ${STRIP_RCNN_CONFIG_HOST}:/configs:ro        mmdet config files

Container args (passed to start.py argparse):
  --config /configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py
  --checkpoint /ckpt/stripnet_s.pth
  --device cuda:0
  --host 0.0.0.0
  --port 9005

Build image first:
  cd docker/strip_rcnn && docker build -t terrabox/strip-rcnn:latest .
"""

import subprocess
import time
import os
import requests
import logging
import atexit
from ..gpu_allocator import allocate_gpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docker.strip_rcnn_manager")


class StripRCNNDockerManager:
    _instance = None

    API_URL = "http://127.0.0.1:9005"

    CONTAINER_NAME = "terrabox-strip-rcnn"
    DOCKER_IMAGE = "terrabox/strip-rcnn:latest"
    GPU_DEVICES = os.environ.get("STRIP_RCNN_GPU_DEVICES", "0")

    CKPT_HOST = os.environ.get(
        "STRIP_RCNN_CKPT_HOST",
        "/data1/yuhongjie2/Strip-RCNN/ckpt"
    )
    CONFIG_HOST = os.environ.get(
        "STRIP_RCNN_CONFIG_HOST",
        "/data1/yuhongjie2/Strip-RCNN/configs"
    )
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")

    # In-container paths passed as argparse arguments to start.py
    CONFIG_IN_CONTAINER = os.environ.get(
        "STRIP_RCNN_CONFIG_IN_CONTAINER",
        "/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py"
    )
    CHECKPOINT_IN_CONTAINER = os.environ.get(
        "STRIP_RCNN_CHECKPOINT_IN_CONTAINER",
        "/ckpt/stripnet_s.pth"
    )

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
            env_var="STRIP_RCNN_GPU_DEVICES",
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "--gpus", f"device={gpu}",
            "-p", "9005:9005",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.CKPT_HOST}:/ckpt:ro",
            "-v", f"{cls.CONFIG_HOST}:/configs:ro",
            cls.DOCKER_IMAGE,
            "--config", cls.CONFIG_IN_CONTAINER,
            "--checkpoint", cls.CHECKPOINT_IN_CONTAINER,
            "--device", "cuda:0",
            "--host", "0.0.0.0",
            "--port", "9005",
        ]

        logger.info(f"Starting Strip-RCNN container (GPU: {gpu})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start Strip-RCNN container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for Strip-RCNN service to start...")
        max_retries = 60
        for i in range(max_retries):
            if cls.is_running():
                logger.info("Strip-RCNN service is READY.")
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("Strip-RCNN service failed to start. Check: docker logs terrabox-strip-rcnn")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", cls.CONTAINER_NAME], capture_output=True)


strip_rcnn_manager = StripRCNNDockerManager()


@atexit.register
def _cleanup():
    pass
