import os
from .base_manager import SubprocessServiceManager


class RemoteCLIPServiceManager(SubprocessServiceManager):
    SERVICE_NAME  = "RemoteCLIP"
    PYTHON_EXEC   = os.environ.get("REMOTECLIP_PYTHON_EXEC", "")
    SERVER_SCRIPT = os.environ.get("REMOTECLIP_SERVER_SCRIPT", "")
    WORK_DIR      = os.environ.get("REMOTECLIP_WORK_DIR", "")
    API_URL       = "http://127.0.0.1:" + os.environ.get("REMOTECLIP_PORT", "9003")
    GPU_DEVICES   = os.environ.get("REMOTECLIP_GPU_DEVICES", "0")

    @classmethod
    def start_service(cls):
        try:
            from ..agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "remoteclip_python_exec"   in d: cls.PYTHON_EXEC   = str(d["remoteclip_python_exec"])
            if "remoteclip_server_script" in d: cls.SERVER_SCRIPT = str(d["remoteclip_server_script"])
            if "remoteclip_work_dir"      in d: cls.WORK_DIR      = str(d["remoteclip_work_dir"])
            if "remoteclip_gpu_devices"   in d: cls.GPU_DEVICES   = str(d["remoteclip_gpu_devices"])
            if "remoteclip_port"          in d: cls.API_URL       = f"http://127.0.0.1:{int(d['remoteclip_port'])}"
        except Exception:
            pass
        super().start_service()


remoteclip_manager = RemoteCLIPServiceManager()
