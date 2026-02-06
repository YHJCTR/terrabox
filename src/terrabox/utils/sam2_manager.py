import subprocess
import time
import os
import requests
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam2_manager")

class SAM2ServiceManager:
    _instance = None
    _process = None
    
    # === 配置 ===
    # 1. Python 解释器 (您的 SAM2 环境)
    SAM2_PYTHON_EXEC = "/home/yuhongjie/miniconda3/envs/sam2/bin/python"
    
    # 2. Server 脚本的绝对路径 (修改为您指定的目录)
    # 请确保您把 sam2_server.py 文件保存到了这个路径下！
    SERVER_SCRIPT = "/data1/yuhongjie2/sam2/sam2_server2.py"
    
    # 3. 工作目录 (Working Directory)
    # [关键] 建议让子进程在这个目录下运行，这样能更好加载 sam2 的相对路径依赖
    WORK_DIR = "/data1/yuhongjie2/sam2"

    API_URL = "http://127.0.0.1:9002"
    GPU_DEVICES = "0"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAM2ServiceManager, cls).__new__(cls)
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

        # 检查脚本是否存在
        if not os.path.exists(cls.SERVER_SCRIPT):
            raise FileNotFoundError(f"Server script not found at: {cls.SERVER_SCRIPT}")

        logger.info(f"Starting SAM2 Service from {cls.WORK_DIR}...")
        
        # 准备环境
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        # 有时候需要把当前目录加到 PYTHONPATH，防止导包错误
        env["PYTHONPATH"] = f"{cls.WORK_DIR}:{env.get('PYTHONPATH', '')}"
        
        # 启动
        cls._process = subprocess.Popen(
            [cls.SAM2_PYTHON_EXEC, cls.SERVER_SCRIPT],
            cwd=cls.WORK_DIR, # [关键] 在 SAM2 目录下运行
            env=env,
            stdout=open(os.path.join(cls.WORK_DIR, "sam2_stdout.log"), "w"), # 日志也保存在那边
            stderr=open(os.path.join(cls.WORK_DIR, "sam2_stderr.log"), "w")
        )

        # 等待启动
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

@atexit.register
def _cleanup():
    sam2_manager.stop_service()