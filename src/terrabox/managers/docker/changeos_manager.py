"""
ChangeOS Service Manager - Docker Mode
======================================
Runs terrabox/changeos:latest (faithful torch==1.10.0 ChangeOS building-damage
assessment service). Mirrors StripRCNNDockerManager.

Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception imports.

Volume mounts:
  -v /data1:/data1                      image files (same path inside container)
  -v ${CHANGEOS_CKPT_HOST}:/ckpt:ro     ChangeOS .pt weights

Container args (passed to start.py argparse):
  --checkpoint /ckpt/changeos_r101.pt
  --device cuda:0
  --host 0.0.0.0
  --port 9007

Build image first:
  cd docker/changeos && bash build.sh
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

logger = logging.getLogger("docker.changeos_manager")


class ChangeOSDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("CHANGEOS_PORT", "9007")

    CONTAINER_NAME  = "terrabox-changeos"
    CONTAINER_BASE  = "terrabox-changeos"
    DOCKER_IMAGE    = "terrabox/changeos:latest"
    GPU_DEVICES     = os.environ.get("CHANGEOS_GPU_DEVICES", "0")
    CKPT_HOST       = os.environ.get("CHANGEOS_CKPT_HOST", "/data1/yuhongjie2/ChangeOS/checkpoints")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")

    # In-container checkpoint path passed as an argparse argument to start.py.
    CHECKPOINT_IN_CONTAINER = os.environ.get(
        "CHANGEOS_CHECKPOINT_IN_CONTAINER", "/ckpt/changeos_r101.pt"
    )
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
            service="changeos",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            base_port=base_port,
            internal_port=9007,
            gpu_count=1,
            min_free_mib=4096,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="CHANGEOS_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.API_URL = lease.api_url
        remove_container_if_exists(cls.CONTAINER_NAME, service="changeos", reason="before_docker_run")

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", f"device={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.CKPT_HOST}:/ckpt:ro",
            cls.DOCKER_IMAGE,
            "--checkpoint", cls.CHECKPOINT_IN_CONTAINER,
            "--device", "cuda:0",
            "--host", "0.0.0.0",
            "--port", str(lease.internal_port),
        ]

        logger.info(f"Starting ChangeOS container (GPU: {lease.gpu_devices}, port: {lease.port})...")
        record_service_event({"event": "start_requested", "service": "changeos", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "changeos", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start ChangeOS container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ...agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "changeos_ckpt_host" in d:
                cls.CKPT_HOST = str(d["changeos_ckpt_host"])
            if "changeos_checkpoint_path" in d:
                cls.CHECKPOINT_IN_CONTAINER = "/ckpt/" + os.path.basename(str(d["changeos_checkpoint_path"]))
            if "changeos_gpu_devices" in d:
                cls.GPU_DEVICES = str(d["changeos_gpu_devices"])
            if "docker_data_mount_host" in d:
                cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "changeos_port" in d:
                cls.API_URL = f"http://127.0.0.1:{int(d['changeos_port'])}"
        except Exception:
            pass

        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for ChangeOS service to start...")
        max_retries = 60
        for _ in range(max_retries):
            if cls.is_running():
                logger.info("ChangeOS service is READY.")
                record_service_event({"event": "ready", "service": "changeos", "lease": cls._lease.__dict__ if cls._lease else None})
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("ChangeOS service failed to start. Check: docker logs terrabox-changeos")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        record_service_event({"event": "stopped", "service": "changeos", "container": cls.CONTAINER_NAME})


changeos_manager = ChangeOSDockerManager()
