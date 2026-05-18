"""
InstructSAM Service Manager - Docker Mode
==========================================
Launches terrabox/instructsam:latest via 'docker run'.
Exported directly by terrabox.managers in this tools-only branch.

Volume mounts:
  -v /data1:/data1                          image files (same path inside container)
  -v ${INSTRUCTSAM_MODELS_HOST}:/models:ro  model directory (SAM2 + CLIP, no Qwen)

Model directory layout (host side, default /data1/yuhongjie2/terra_model/instructsam):
  sam2_hiera_large.pt       SAM2 Hiera Large weights
  GeoRSCLIP-ViT-L-14.pt     GeoRSCLIP CLIP weights

Note: the counting step calls the host vLLM service via HTTP; Qwen is not downloaded
inside the container. If the VLM manager has selected a dynamic host port in this
process, that port is passed through VLLM_API_URL.

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
from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

logger = logging.getLogger("docker.instructsam_manager")


class InstructSAMDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("INSTRUCTSAM_PORT", "9006")

    CONTAINER_NAME = "terrabox-instructsam"
    CONTAINER_BASE = "terrabox-instructsam"
    DOCKER_IMAGE   = "terrabox/instructsam:latest"
    GPU_DEVICES    = os.environ.get("INSTRUCTSAM_GPU_DEVICES", "0")

    # host-side model directory, mounted as /models inside the container
    MODELS_HOST     = os.environ.get("INSTRUCTSAM_MODELS_HOST", "")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")
    _lease = None

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
            logger.info(f"Container {cls.CONTAINER_NAME} is running but service is not healthy; rebuilding it.")
            cls.stop_service()

        base_port = int(cls.API_URL.rsplit(":", 1)[-1])
        lease = acquire_docker_lease(
            service="instructsam",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            base_port=base_port,
            internal_port=9006,
            gpu_count=1,
            min_free_mib=4096,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="INSTRUCTSAM_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.API_URL = lease.api_url
        remove_container_if_exists(cls.CONTAINER_NAME, service="instructsam", reason="before_docker_run")
        vllm_api_url = os.environ.get("VLLM_API_URL")
        if not vllm_api_url:
            try:
                from .vllm_manager import VLLMDockerManager
                vllm_api_url = f"http://host.docker.internal:{VLLMDockerManager.PORT}"
            except Exception:
                vllm_api_url = "http://host.docker.internal:9000"

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", f"device={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            # Allow the container to reach the host vLLM service (port 9000)
            "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.MODELS_HOST}:/models:ro",
            "-e", "DEVICE=cuda:0",  # Use first GPU visible to container
            "-e", f"VLLM_API_URL={vllm_api_url}",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting InstructSAM container (GPU: {lease.gpu_devices}, port: {lease.port})...")
        record_service_event({"event": "start_requested", "service": "instructsam", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "instructsam", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start InstructSAM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ..config import load_raw_yaml
            d = load_raw_yaml()
            if "instructsam_models_host" in d: cls.MODELS_HOST     = str(d["instructsam_models_host"])
            if "instructsam_gpu_devices" in d: cls.GPU_DEVICES     = str(d["instructsam_gpu_devices"])
            if "docker_data_mount_host"  in d: cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "instructsam_port"        in d: cls.API_URL         = f"http://127.0.0.1:{int(d['instructsam_port'])}"
        except Exception:
            pass

        if cls.is_running():
            return

        cls._start_docker()

        # SAM2 (~15s) + CLIP (~5s), no Qwen — ready in ~30s
        logger.info("Waiting for InstructSAM service (SAM2 + CLIP loading ~30s)...")
        max_retries = 60    # poll every 2s, wait up to 120s total
        for i in range(max_retries):
            if cls.is_running():
                logger.info("InstructSAM service is READY.")
                record_service_event({"event": "ready", "service": "instructsam", "lease": cls._lease.__dict__ if cls._lease else None})
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
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        record_service_event({"event": "stopped", "service": "instructsam", "container": cls.CONTAINER_NAME})


instructsam_manager = InstructSAMDockerManager()
