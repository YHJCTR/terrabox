import os
from .base_manager import SubprocessServiceManager


class RemoteCLIPServiceManager(SubprocessServiceManager):
    SERVICE_NAME = "RemoteCLIP"
    PYTHON_EXEC = os.environ.get("REMOTECLIP_PYTHON_EXEC", "/home/yuhongjie/miniconda3/envs/RemoteCLIP/bin/python")
    SERVER_SCRIPT = os.environ.get("REMOTECLIP_SERVER_SCRIPT", "/data1/yuhongjie2/RemoteCLIP/start2.py")
    WORK_DIR = os.environ.get("REMOTECLIP_WORK_DIR", "/data1/yuhongjie2/RemoteCLIP")
    API_URL = "http://127.0.0.1:" + os.environ.get("REMOTECLIP_PORT", "9003")
    GPU_DEVICES = os.environ.get("REMOTECLIP_GPU_DEVICES", "3")


remoteclip_manager = RemoteCLIPServiceManager()
