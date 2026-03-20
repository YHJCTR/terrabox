"""
RemoteSAM Service Manager - Docker Mode
==========================================
Uses 'docker run' to start terrabox/remotesam:latest instead of subprocess.
Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception.py imports.

Volume mounts:
  -v /data1:/data1                             image files (same path inside container)
  -v ${REMOTESAM_CHECKPOINT_HOST}:/checkpoints:ro  pretrained_weights directory
  -v ${HF_CACHE_HOST}:/root/.cache/huggingface:ro  HuggingFace model cache

Note: The image clones RemoteSAM source from GitHub at build time (no host source mount needed).
      bert-base-uncased and the EPOC model must be present in the HF cache; TRANSFORMERS_OFFLINE=1
      prevents network download attempts inside the container.

Build image first:
  cd docker/remotesam && docker build -t terrabox/remotesam:latest .
"""

import subprocess
import time
import os
import requests
import logging
from ..base_manager import BaseServiceManager

logger = logging.getLogger("docker.remotesam_manager")


class RemoteSAMDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("REMOTESAM_PORT", "9004")

    CONTAINER_NAME  = "terrabox-remotesam"
    DOCKER_IMAGE    = "terrabox/remotesam:latest"
    GPU_DEVICES     = os.environ.get("REMOTESAM_GPU_DEVICES", "0")
    CHECKPOINT_HOST = os.environ.get("REMOTESAM_CHECKPOINT_HOST", "")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")
    HF_CACHE_HOST   = os.environ.get("HF_CACHE_HOST", os.path.expanduser("~/.cache/huggingface"))

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
            capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} already running.")
            return

        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

        gpu = cls._ensure_gpu(
            min_free_mib=6144,
            env_var="REMOTESAM_GPU_DEVICES",
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "--gpus", f"device={gpu}",
            "-p", f"{cls.API_URL.rsplit(':', 1)[-1]}:{cls.API_URL.rsplit(':', 1)[-1]}",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            # Mount pretrained_weights as /checkpoints (source code is baked into the image)
            "-v", f"{cls.CHECKPOINT_HOST}:/checkpoints:ro",
            # Mount HuggingFace cache so bert-base-uncased and EPOC model are available offline
            "-v", f"{cls.HF_CACHE_HOST}:/root/.cache/huggingface:ro",
            # Force offline mode — prevents transformers from trying to reach HuggingFace Hub
            "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_DATASETS_OFFLINE=1",
            "-e", "REMOTESAM_CHECKPOINT=/checkpoints/swin_base_patch4_window12_384_22k.pth",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting RemoteSAM container (GPU: {gpu})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start RemoteSAM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ...agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "remotesam_checkpoint_host" in d: cls.CHECKPOINT_HOST = str(d["remotesam_checkpoint_host"])
            if "remotesam_gpu_devices"     in d: cls.GPU_DEVICES     = str(d["remotesam_gpu_devices"])
            if "docker_data_mount_host"    in d: cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "remotesam_port"            in d: cls.API_URL         = f"http://127.0.0.1:{int(d['remotesam_port'])}"
        except Exception:
            pass

        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for RemoteSAM service to start (model loading may take ~3 min)...")
        max_retries = 150  # poll every 2s, up to 300s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("RemoteSAM service is READY.")
                return
            if i % 15 == 0 and i > 0:
                logger.info(f"Still loading... ({i * 2}s elapsed)")
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("RemoteSAM service failed to start. Check: docker logs terrabox-remotesam")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", cls.CONTAINER_NAME], capture_output=True)


remotesam_manager = RemoteSAMDockerManager()
