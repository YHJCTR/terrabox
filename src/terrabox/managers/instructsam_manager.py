# instructsam_manager.py — 子进程模式（非 Docker）
# InstructSAM 依赖较重（SAM2 + Qwen + CLIP），推荐使用 Docker 模式。
# 本文件仅供非 Docker 环境下导入时不报错；实际启动逻辑留空，
# 请在宿主机上手动启动 InstructSAM 服务后通过此 manager 对接。
import os
import time
import requests
import logging
from .base_manager import BaseServiceManager

logger = logging.getLogger("instructsam_manager")


class InstructSAMServiceManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:9006"

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
    def start_service(cls):
        if cls.is_running():
            return
        raise RuntimeError(
            "InstructSAM service is not running. "
            "Please start it manually or use TERRABOX_USE_DOCKER=true."
        )

    @classmethod
    def stop_service(cls):
        logger.info("InstructSAM subprocess mode: stop_service is a no-op.")


instructsam_manager = InstructSAMServiceManager()
