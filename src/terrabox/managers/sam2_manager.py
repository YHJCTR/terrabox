import os
from .base_manager import SubprocessServiceManager


class SAM2ServiceManager(SubprocessServiceManager):
    SERVICE_NAME = "SAM2"
    PYTHON_EXEC   = os.environ.get("SAM2_PYTHON_EXEC", "")
    SERVER_SCRIPT = os.environ.get("SAM2_SERVER_SCRIPT", "")
    WORK_DIR      = os.environ.get("SAM2_WORK_DIR", "")
    API_URL       = "http://127.0.0.1:" + os.environ.get("SAM2_PORT", "9002")
    GPU_DEVICES   = os.environ.get("SAM2_GPU_DEVICES", "0")

    @classmethod
    def start_service(cls):
        try:
            from ..agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "sam2_python_exec"   in d: cls.PYTHON_EXEC   = str(d["sam2_python_exec"])
            if "sam2_server_script" in d: cls.SERVER_SCRIPT = str(d["sam2_server_script"])
            if "sam2_work_dir"      in d: cls.WORK_DIR      = str(d["sam2_work_dir"])
            if "sam2_gpu_devices"   in d: cls.GPU_DEVICES   = str(d["sam2_gpu_devices"])
            if "sam2_port"          in d: cls.API_URL       = f"http://127.0.0.1:{int(d['sam2_port'])}"
        except Exception:
            pass
        super().start_service()


sam2_manager = SAM2ServiceManager()
