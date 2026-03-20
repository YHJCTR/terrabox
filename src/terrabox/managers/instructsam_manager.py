# instructsam_manager.py — subprocess mode (non-Docker)
# InstructSAM has heavy dependencies (SAM2 + Qwen + CLIP); Docker mode is recommended.
# This file exists only so that non-Docker imports succeed; startup logic is intentionally
# left as a no-op. Start the InstructSAM service manually and connect via this manager.
import os
import time
import requests
import logging
from .base_manager import BaseServiceManager

logger = logging.getLogger("instructsam_manager")


class InstructSAMServiceManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("INSTRUCTSAM_PORT", "9006")

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
        try:
            from ..agent.config import load_raw_yaml
            d = load_raw_yaml()
            host = str(d.get("instructsam_host", "127.0.0.1"))
            port = int(d.get("instructsam_port", cls.API_URL.rsplit(":", 1)[-1]))
            cls.API_URL = f"http://{host}:{port}"
        except Exception:
            pass
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
