import os
from .base_manager import SubprocessServiceManager


class SAM2ServiceManager(SubprocessServiceManager):
    SERVICE_NAME = "SAM2"
    PYTHON_EXEC = os.environ.get("SAM2_PYTHON_EXEC", "/home/yuhongjie/miniconda3/envs/sam2/bin/python")
    SERVER_SCRIPT = os.environ.get("SAM2_SERVER_SCRIPT", "/data1/yuhongjie2/sam2/sam2_server2.py")
    WORK_DIR = os.environ.get("SAM2_WORK_DIR", "/data1/yuhongjie2/sam2")
    API_URL = "http://127.0.0.1:" + os.environ.get("SAM2_PORT", "9002")
    GPU_DEVICES = os.environ.get("SAM2_GPU_DEVICES", "0")


sam2_manager = SAM2ServiceManager()
