# remoteclip_manager.py
import subprocess
import time
import os
import requests
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remoteclip_manager")


class RemoteCLIPServiceManager:
    _instance = None
    _process = None

    REMOTECLIP_PYTHON_EXEC = "/home/yuhongjie/miniconda3/envs/RemoteCLIP/bin/python"
    SERVER_SCRIPT = "/data1/yuhongjie2/RemoteCLIP/start2.py"
    WORK_DIR = "/data1/yuhongjie2/RemoteCLIP"
    API_URL = "http://127.0.0.1:9003"
    GPU_DEVICES = "3"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RemoteCLIPServiceManager, cls).__new__(cls)
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

        logger.info(f"Starting RemoteCLIP Service from {cls.WORK_DIR}...")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env["PYTHONPATH"] = f"{cls.WORK_DIR}:{env.get('PYTHONPATH', '')}"

        cls._process = subprocess.Popen(
            [cls.REMOTECLIP_PYTHON_EXEC, cls.SERVER_SCRIPT],
            cwd=cls.WORK_DIR,
            env=env,
            stdout=open(os.path.join(cls.WORK_DIR, "remoteclip_stdout.log"), "w"),
            stderr=open(os.path.join(cls.WORK_DIR, "remoteclip_stderr.log"), "w")
        )

        max_retries = 30
        for i in range(max_retries):
            if cls.is_running():
                logger.info("RemoteCLIP Service is READY.")
                return
            time.sleep(1)

        cls.stop_service()
        raise RuntimeError("RemoteCLIP service failed to start. Check logs in /data1/yuhongjie2/RemoteCLIP/")

    @classmethod
    def stop_service(cls):
        if cls._process:
            logger.info("Stopping RemoteCLIP Service...")
            cls._process.terminate()
            cls._process = None


remoteclip_manager = RemoteCLIPServiceManager()


@atexit.register
def _cleanup():
    remoteclip_manager.stop_service()
