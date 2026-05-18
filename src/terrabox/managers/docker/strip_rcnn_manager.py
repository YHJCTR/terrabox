"""
Strip-RCNN Service Manager - Docker Mode
==========================================
Uses 'docker run' to start terrabox/strip-rcnn:latest instead of subprocess.
Exported directly by terrabox.managers in this tools-only branch.

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
from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

logger = logging.getLogger("docker.strip_rcnn_manager")


class StripRCNNDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("STRIP_RCNN_PORT", "9005")

    CONTAINER_NAME  = "terrabox-strip-rcnn"
    CONTAINER_BASE  = "terrabox-strip-rcnn"
    DOCKER_IMAGE    = "terrabox/strip-rcnn:latest"
    GPU_DEVICES     = os.environ.get("STRIP_RCNN_GPU_DEVICES", "0")
    CKPT_HOST       = os.environ.get("STRIP_RCNN_CKPT_HOST", "")
    CONFIG_HOST     = os.environ.get("STRIP_RCNN_CONFIG_HOST", "")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")

    # In-container paths passed as argparse arguments to start.py
    CONFIG_IN_CONTAINER     = os.environ.get("STRIP_RCNN_CONFIG_IN_CONTAINER", "/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py")
    CHECKPOINT_IN_CONTAINER = os.environ.get("STRIP_RCNN_CHECKPOINT_IN_CONTAINER", "/ckpt/stripnet_s.pth")
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
            service="strip-rcnn",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            base_port=base_port,
            internal_port=9005,
            gpu_count=1,
            min_free_mib=4096,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="STRIP_RCNN_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.API_URL = lease.api_url
        remove_container_if_exists(cls.CONTAINER_NAME, service="strip-rcnn", reason="before_docker_run")

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", f"device={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.CKPT_HOST}:/ckpt:ro",
            "-v", f"{cls.CONFIG_HOST}:/configs:ro",
            cls.DOCKER_IMAGE,
            "--config", cls.CONFIG_IN_CONTAINER,
            "--checkpoint", cls.CHECKPOINT_IN_CONTAINER,
            "--device", "cuda:0",
            "--host", "0.0.0.0",
            "--port", str(lease.internal_port),
        ]

        logger.info(f"Starting Strip-RCNN container (GPU: {lease.gpu_devices}, port: {lease.port})...")
        record_service_event({"event": "start_requested", "service": "strip-rcnn", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "strip-rcnn", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start Strip-RCNN container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ..config import load_raw_yaml
            d = load_raw_yaml()
            if "strip_rcnn_ckpt_host"     in d: cls.CKPT_HOST       = str(d["strip_rcnn_ckpt_host"])
            if "strip_rcnn_config_host"   in d: cls.CONFIG_HOST     = str(d["strip_rcnn_config_host"])
            if "strip_rcnn_gpu_devices"   in d: cls.GPU_DEVICES     = str(d["strip_rcnn_gpu_devices"])
            if "docker_data_mount_host"   in d: cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "strip_rcnn_port"          in d: cls.API_URL         = f"http://127.0.0.1:{int(d['strip_rcnn_port'])}"
        except Exception:
            pass

        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for Strip-RCNN service to start...")
        max_retries = 60
        for i in range(max_retries):
            if cls.is_running():
                logger.info("Strip-RCNN service is READY.")
                record_service_event({"event": "ready", "service": "strip-rcnn", "lease": cls._lease.__dict__ if cls._lease else None})
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("Strip-RCNN service failed to start. Check: docker logs terrabox-strip-rcnn")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        record_service_event({"event": "stopped", "service": "strip-rcnn", "container": cls.CONTAINER_NAME})


strip_rcnn_manager = StripRCNNDockerManager()
