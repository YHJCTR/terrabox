"""
SAM2 Service Manager - Docker Mode
====================================
Uses 'docker run' to start terrabox/sam2:latest instead of subprocess.
Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception.py imports.

Volume mounts:
  -v /data1:/data1                            image files (same path inside container)
  -v ${SAM2_CHECKPOINT_HOST}:/checkpoints:ro  SAM2 checkpoint
  -v ${SAM2_CONFIG_HOST}:/sam2_configs:ro     SAM2 YAML config directory

Build image first:
  cd docker/sam2 && docker build -t terrabox/sam2:latest .
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

logger = logging.getLogger("docker.sam2_manager")


class SAM2DockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("SAM2_PORT", "9002")

    CONTAINER_NAME  = "terrabox-sam2"
    CONTAINER_BASE  = "terrabox-sam2"
    DOCKER_IMAGE    = "terrabox/sam2:latest"
    GPU_DEVICES     = os.environ.get("SAM2_GPU_DEVICES", "0")
    CHECKPOINT_HOST = os.environ.get("SAM2_CHECKPOINT_HOST", "")
    CONFIG_HOST     = os.environ.get("SAM2_CONFIG_HOST", "")
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
            capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def _apply_env_overrides(cls):
        if os.environ.get("SAM2_PORT"):
            cls.API_URL = f"http://127.0.0.1:{int(os.environ['SAM2_PORT'])}"
        if os.environ.get("SAM2_GPU_DEVICES"):
            cls.GPU_DEVICES = os.environ["SAM2_GPU_DEVICES"]
        if os.environ.get("SAM2_CHECKPOINT_HOST"):
            cls.CHECKPOINT_HOST = os.environ["SAM2_CHECKPOINT_HOST"]
        if os.environ.get("SAM2_CONFIG_HOST"):
            cls.CONFIG_HOST = os.environ["SAM2_CONFIG_HOST"]
        if os.environ.get("DATA_MOUNT_HOST"):
            cls.DATA_MOUNT_HOST = os.environ["DATA_MOUNT_HOST"]

    @classmethod
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} is running but service is not healthy; rebuilding it.")
            cls.stop_service()

        base_port = int(cls.API_URL.rsplit(":", 1)[-1])
        lease = acquire_docker_lease(
            service="sam2",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            base_port=base_port,
            internal_port=9002,
            gpu_count=1,
            min_free_mib=8192,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="SAM2_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.API_URL = lease.api_url
        remove_container_if_exists(cls.CONTAINER_NAME, service="sam2", reason="before_docker_run")

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", f"device={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            # Mount data directory at the same path so image paths are identical inside the container
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.CHECKPOINT_HOST}:/checkpoints:ro",
            "-v", f"{cls.CONFIG_HOST}:/sam2_configs:ro",
            # Tell the server script where to find the checkpoint and config via env vars
            "-e", "SAM2_CHECKPOINT=/checkpoints/sam2.1_hiera_large.pt",
            "-e", "SAM2_CONFIG=/sam2_configs/sam2.1/sam2.1_hiera_l.yaml",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting SAM2 container (GPU: {lease.gpu_devices}, port: {lease.port})...")
        record_service_event({"event": "start_requested", "service": "sam2", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "sam2", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start SAM2 container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ...agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "sam2_checkpoint_host" in d: cls.CHECKPOINT_HOST = str(d["sam2_checkpoint_host"])
            if "sam2_config_host"     in d: cls.CONFIG_HOST     = str(d["sam2_config_host"])
            if "sam2_gpu_devices"     in d: cls.GPU_DEVICES     = str(d["sam2_gpu_devices"])
            if "docker_data_mount_host" in d: cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "sam2_port"            in d: cls.API_URL         = f"http://127.0.0.1:{int(d['sam2_port'])}"
        except Exception:
            pass
        cls._apply_env_overrides()

        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for SAM2 service to start...")
        max_retries = 60    # poll every 2s, up to 120s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("SAM2 service is READY.")
                record_service_event({"event": "ready", "service": "sam2", "lease": cls._lease.__dict__ if cls._lease else None})
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("SAM2 service failed to start. Check: docker logs terrabox-sam2")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        record_service_event({"event": "stopped", "service": "sam2", "container": cls.CONTAINER_NAME})


sam2_manager = SAM2DockerManager()
