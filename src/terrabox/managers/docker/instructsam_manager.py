"""
InstructSAM Service Manager - Docker Mode
==========================================
使用 'docker run' 启动 terrabox/instructsam:latest。
设置 TERRABOX_USE_DOCKER=true 后，geo_perception.py 会导入此文件。

Volume 挂载:
  -v /data1:/data1                          图像文件（容器内路径相同）
  -v ${INSTRUCTSAM_MODELS_HOST}:/models:ro  模型目录（SAM2 + CLIP，不含 Qwen）

模型目录结构（host 侧，默认 /data1/yuhongjie2/terra_model/instructsam）:
  sam2_hiera_large.pt       SAM2 Hiera Large 权重
  GeoRSCLIP-ViT-L-14.pt     GeoRSCLIP CLIP 权重

注：计数步骤通过 HTTP 调用宿主机 vLLM 服务（port 9000）完成，无需下载 Qwen。

先构建镜像:
  cd docker/instructsam && docker build -t terrabox/instructsam:latest .

先下载模型:
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

    # host 侧模型目录，挂载为容器内的 /models
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

        # InstructSAM 只加载 SAM2 + CLIP，需要约 4 GB 显存
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
            # 让容器通过 host.docker.internal 访问宿主机 vLLM（port 9000）
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

        # SAM2(~15s) + CLIP(~5s)，无 Qwen，约 30s 即可就绪
        logger.info("Waiting for InstructSAM service (SAM2 + CLIP loading ~30s)...")
        max_retries = 60    # 每 2s 轮询一次，最长等 120s
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
