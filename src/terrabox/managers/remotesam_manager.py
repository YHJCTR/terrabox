import os
from .base_manager import SubprocessServiceManager


class RemoteSAMServiceManager(SubprocessServiceManager):
    SERVICE_NAME = "RemoteSAM"
    PYTHON_EXEC = os.environ.get("REMOTESAM_PYTHON_EXEC", "/home/yuhongjie/miniconda3/envs/RemoteSAM/bin/python")
    SERVER_SCRIPT = os.environ.get("REMOTESAM_SERVER_SCRIPT", "/data1/yuhongjie2/RemoteSAM/start.py")
    WORK_DIR = os.environ.get("REMOTESAM_WORK_DIR", "/data1/yuhongjie2/RemoteSAM")
    API_URL = "http://127.0.0.1:" + os.environ.get("REMOTESAM_PORT", "9004")
    GPU_DEVICES = os.environ.get("REMOTESAM_GPU_DEVICES", "3")


remotesam_manager = RemoteSAMServiceManager()
