# strip_rcnn_manager.py
import subprocess
import time
import os
import requests
import logging
from .base_manager import BaseServiceManager

logger = logging.getLogger("strip_rcnn_manager")


class StripRCNNServiceManager(BaseServiceManager):
    _instance = None
    _process = None

    STRIP_RCNN_PYTHON_EXEC = "/home/yuhongjie/miniconda3/envs/strip/bin/python"
    SERVER_SCRIPT = "/data1/yuhongjie2/Strip-RCNN/start.py"
    WORK_DIR = "/data1/yuhongjie2/Strip-RCNN"
    API_URL = "http://127.0.0.1:9005"
    GPU_DEVICES = "0"

    CONFIG_PATH = "/data1/yuhongjie2/Strip-RCNN/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py"
    CHECKPOINT_PATH = "/data1/yuhongjie2/Strip-RCNN/ckpt/stripnet_s.pth"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StripRCNNServiceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        try:
            return requests.get(f"{cls.API_URL}/health", timeout=1).status_code == 200
        except:
            return False

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        if not os.path.exists(cls.SERVER_SCRIPT):
            raise FileNotFoundError(f"Server script not found at: {cls.SERVER_SCRIPT}")

        if not os.path.exists(cls.CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found at: {cls.CONFIG_PATH}")

        if not os.path.exists(cls.CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found at: {cls.CHECKPOINT_PATH}")

        logger.info(f"Starting Strip R-CNN Service from {cls.WORK_DIR}...")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env["PYTHONPATH"] = f"{cls.WORK_DIR}:{env.get('PYTHONPATH', '')}"

        cls._process = subprocess.Popen(
            [
                cls.STRIP_RCNN_PYTHON_EXEC,
                cls.SERVER_SCRIPT,
                "--config", cls.CONFIG_PATH,
                "--checkpoint", cls.CHECKPOINT_PATH,
                "--device", f"cuda:{cls.GPU_DEVICES}",
                "--host", "127.0.0.1",
                "--port", "9005"
            ],
            cwd=cls.WORK_DIR,
            env=env,
            stdout=open(os.path.join(cls.WORK_DIR, "strip_rcnn_stdout.log"), "w"),
            stderr=open(os.path.join(cls.WORK_DIR, "strip_rcnn_stderr.log"), "w")
        )

        max_retries = 30
        for i in range(max_retries):
            if cls.is_running():
                logger.info("Strip R-CNN Service is READY.")
                return
            time.sleep(1)

        cls.stop_service()
        raise RuntimeError("Strip R-CNN service failed to start. Check logs in /data1/yuhongjie2/StripR-CNN/")

    @classmethod
    def stop_service(cls):
        if cls._process:
            logger.info("Stopping Strip R-CNN Service...")
            cls._process.terminate()
            cls._process = None


strip_rcnn_manager = StripRCNNServiceManager()
