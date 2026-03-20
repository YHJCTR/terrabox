import os
from .base_manager import SubprocessServiceManager


class RemoteSAMServiceManager(SubprocessServiceManager):
    SERVICE_NAME  = "RemoteSAM"
    PYTHON_EXEC   = os.environ.get("REMOTESAM_PYTHON_EXEC", "")
    SERVER_SCRIPT = os.environ.get("REMOTESAM_SERVER_SCRIPT", "")
    WORK_DIR      = os.environ.get("REMOTESAM_WORK_DIR", "")
    API_URL       = "http://127.0.0.1:" + os.environ.get("REMOTESAM_PORT", "9004")
    GPU_DEVICES   = os.environ.get("REMOTESAM_GPU_DEVICES", "0")

    @classmethod
    def start_service(cls):
        try:
            from ..agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "remotesam_python_exec"   in d: cls.PYTHON_EXEC   = str(d["remotesam_python_exec"])
            if "remotesam_server_script" in d: cls.SERVER_SCRIPT = str(d["remotesam_server_script"])
            if "remotesam_work_dir"      in d: cls.WORK_DIR      = str(d["remotesam_work_dir"])
            if "remotesam_gpu_devices"   in d: cls.GPU_DEVICES   = str(d["remotesam_gpu_devices"])
            if "remotesam_port"          in d: cls.API_URL       = f"http://127.0.0.1:{int(d['remotesam_port'])}"
        except Exception:
            pass
        super().start_service()


remotesam_manager = RemoteSAMServiceManager()
