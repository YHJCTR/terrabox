"""
RemoteCLIP Service Manager - Docker Mode
==========================================
Uses 'docker run' to start terrabox/remoteclip:latest instead of subprocess.
Exported directly by terrabox.managers in this tools-only branch.

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
from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

logger = logging.getLogger("docker.remoteclip_manager")


class RemoteCLIPDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("REMOTECLIP_PORT", "9003")

    CONTAINER_NAME  = "terrabox-remoteclip"
    CONTAINER_BASE  = "terrabox-remoteclip"
    DOCKER_IMAGE    = "terrabox/remoteclip:latest"
    GPU_DEVICES     = os.environ.get("REMOTECLIP_GPU_DEVICES", "0")
    CKPT_HOST       = os.environ.get("REMOTECLIP_CKPT_HOST", "")
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
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} is running but service is not healthy; rebuilding it.")
            cls.stop_service()

        base_port = int(cls.API_URL.rsplit(":", 1)[-1])
        lease = acquire_docker_lease(
            service="remoteclip",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            base_port=base_port,
            internal_port=9003,
            gpu_count=1,
            min_free_mib=4096,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="REMOTECLIP_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.API_URL = lease.api_url
        remove_container_if_exists(cls.CONTAINER_NAME, service="remoteclip", reason="before_docker_run")

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", f"device={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.CKPT_HOST}:/checkpoints:ro",
            "-e", "REMOTECLIP_CKPT_DIR=/checkpoints",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting RemoteCLIP container (GPU: {lease.gpu_devices}, port: {lease.port})...")
        record_service_event({"event": "start_requested", "service": "remoteclip", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "remoteclip", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start RemoteCLIP container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ..config import load_raw_yaml
            d = load_raw_yaml()
            if "remoteclip_ckpt_host"     in d: cls.CKPT_HOST       = str(d["remoteclip_ckpt_host"])
            if "remoteclip_gpu_devices"   in d: cls.GPU_DEVICES     = str(d["remoteclip_gpu_devices"])
            if "docker_data_mount_host"   in d: cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "remoteclip_port"          in d: cls.API_URL         = f"http://127.0.0.1:{int(d['remoteclip_port'])}"
        except Exception:
            pass

        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for RemoteCLIP service to start...")
        max_retries = 60
        for i in range(max_retries):
            if cls.is_running():
                logger.info("RemoteCLIP service is READY.")
                record_service_event({"event": "ready", "service": "remoteclip", "lease": cls._lease.__dict__ if cls._lease else None})
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("RemoteCLIP service failed to start. Check: docker logs terrabox-remoteclip")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        record_service_event({"event": "stopped", "service": "remoteclip", "container": cls.CONTAINER_NAME})


remoteclip_manager = RemoteCLIPDockerManager()
