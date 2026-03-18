import subprocess
import time
import os
import requests
import logging
from .base_manager import BaseServiceManager

logger = logging.getLogger("sam2_manager")

class SAM2ServiceManager(BaseServiceManager):
    _instance = None
    _process = None
    
    # Path to the isolated Python interpreter for the SAM2 conda environment
    SAM2_PYTHON_EXEC = os.environ.get("SAM2_PYTHON_EXEC", "/home/yuhongjie/miniconda3/envs/sam2/bin/python")
    SERVER_SCRIPT = os.environ.get("SAM2_SERVER_SCRIPT", "/data1/yuhongjie2/sam2/sam2_server2.py")
    # Run the subprocess from the SAM2 directory so relative imports resolve correctly
    WORK_DIR = os.environ.get("SAM2_WORK_DIR", "/data1/yuhongjie2/sam2")

    API_URL = "http://127.0.0.1:" + os.environ.get("SAM2_PORT", "9002")
    GPU_DEVICES = os.environ.get("SAM2_GPU_DEVICES", "0")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAM2ServiceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        try:
            return requests.get(
                f"{cls.API_URL}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            ).status_code == 200
        except:
            return False

    @classmethod
    def start_service(cls):
        if cls.is_running():
            return

        if not os.path.exists(cls.SERVER_SCRIPT):
            raise FileNotFoundError(f"Server script not found at: {cls.SERVER_SCRIPT}")

        logger.info(f"Starting SAM2 Service from {cls.WORK_DIR}...")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env["PYTHONPATH"] = f"{cls.WORK_DIR}:{env.get('PYTHONPATH', '')}"

        cls._process = subprocess.Popen(
            [cls.SAM2_PYTHON_EXEC, cls.SERVER_SCRIPT],
            cwd=cls.WORK_DIR,
            env=env,
            stdout=open(os.path.join(cls.WORK_DIR, "sam2_stdout.log"), "w"),
            stderr=open(os.path.join(cls.WORK_DIR, "sam2_stderr.log"), "w")
        )

        max_retries = 30
        for i in range(max_retries):
            if cls.is_running():
                logger.info("SAM2 Service is READY.")
                return
            time.sleep(1)
        
        cls.stop_service()
        raise RuntimeError("SAM2 service failed to start. Check logs in /data1/yuhongjie2/sam2/")

    @classmethod
    def stop_service(cls):
        if cls._process:
            logger.info("Stopping SAM2 Service...")
            cls._process.terminate()
            cls._process = None

sam2_manager = SAM2ServiceManager()