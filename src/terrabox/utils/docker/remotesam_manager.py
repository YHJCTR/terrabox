"""
RemoteSAM Service Manager - Docker Mode
==========================================
Uses 'docker run' to start terrabox/remotesam:latest instead of subprocess.
Set TERRABOX_USE_DOCKER=true to activate this manager via geo_perception.py imports.

Volume mounts:
  -v /data1:/data1                image files (same path inside container)
  -v ${REMOTESAM_SRC_HOST}:/app:ro  RemoteSAM source tree (includes pretrained_weights/)
  -w /app                           working directory so relative paths resolve correctly

Note: RemoteSAM/start.py uses relative paths like "./pretrained_weights/...".
      Mounting the entire source directory as /app preserves those relative paths.

Build image first:
  cd docker/remotesam && docker build -t terrabox/remotesam:latest .
"""

import subprocess
import time
import os
import requests
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docker.remotesam_manager")


class RemoteSAMDockerManager:
    _instance = None

    API_URL = "http://127.0.0.1:9004"

    CONTAINER_NAME = "terrabox-remotesam"
    DOCKER_IMAGE = "terrabox/remotesam:latest"
    GPU_DEVICES = os.environ.get("REMOTESAM_GPU_DEVICES", "3")

    # Entire RemoteSAM source directory (includes pretrained_weights/) mounted as /app
    SRC_HOST = os.environ.get(
        "REMOTESAM_SRC_HOST",
        "/data1/yuhongjie2/RemoteSAM"
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

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "--gpus", f"device={cls.GPU_DEVICES}",
            "-p", "9004:9004",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            # Mount entire RemoteSAM source tree (preserves local imports and pretrained_weights/)
            "-v", f"{cls.SRC_HOST}:/app:ro",
            "-w", "/app",   # keep cwd so ./pretrained_weights/ relative paths work
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting RemoteSAM container (GPU: {cls.GPU_DEVICES})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start RemoteSAM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        cls._start_docker()

        logger.info("Waiting for RemoteSAM service to start...")
        max_retries = 60
        for i in range(max_retries):
            if cls.is_running():
                logger.info("RemoteSAM service is READY.")
                return
            time.sleep(2)

        cls.stop_service()
        raise RuntimeError("RemoteSAM service failed to start. Check: docker logs terrabox-remotesam")

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", cls.CONTAINER_NAME], capture_output=True)


remotesam_manager = RemoteSAMDockerManager()


@atexit.register
def _cleanup():
    pass
