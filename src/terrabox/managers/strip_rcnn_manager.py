import os
from .base_manager import SubprocessServiceManager


class StripRCNNServiceManager(SubprocessServiceManager):
    SERVICE_NAME    = "Strip R-CNN"
    PYTHON_EXEC     = os.environ.get("STRIP_RCNN_PYTHON_EXEC", "")
    SERVER_SCRIPT   = os.environ.get("STRIP_RCNN_SERVER_SCRIPT", "")
    WORK_DIR        = os.environ.get("STRIP_RCNN_WORK_DIR", "")
    API_URL         = "http://127.0.0.1:" + os.environ.get("STRIP_RCNN_PORT", "9005")
    GPU_DEVICES     = os.environ.get("STRIP_RCNN_GPU_DEVICES", "0")
    CONFIG_PATH     = os.environ.get("STRIP_RCNN_CONFIG_PATH", "")
    CHECKPOINT_PATH = os.environ.get("STRIP_RCNN_CHECKPOINT_PATH", "")

    @classmethod
    def start_service(cls):
        try:
            from ..agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "strip_rcnn_python_exec"     in d: cls.PYTHON_EXEC     = str(d["strip_rcnn_python_exec"])
            if "strip_rcnn_server_script"   in d: cls.SERVER_SCRIPT   = str(d["strip_rcnn_server_script"])
            if "strip_rcnn_work_dir"        in d: cls.WORK_DIR        = str(d["strip_rcnn_work_dir"])
            if "strip_rcnn_gpu_devices"     in d: cls.GPU_DEVICES     = str(d["strip_rcnn_gpu_devices"])
            if "strip_rcnn_config_path"     in d: cls.CONFIG_PATH     = str(d["strip_rcnn_config_path"])
            if "strip_rcnn_checkpoint_path" in d: cls.CHECKPOINT_PATH = str(d["strip_rcnn_checkpoint_path"])
            if "strip_rcnn_port"            in d: cls.API_URL         = f"http://127.0.0.1:{int(d['strip_rcnn_port'])}"
        except Exception:
            pass
        super().start_service()

    @classmethod
    def _check_paths(cls) -> None:
        super()._check_paths()
        if not os.path.exists(cls.CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found at: {cls.CONFIG_PATH}")
        if not os.path.exists(cls.CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found at: {cls.CHECKPOINT_PATH}")

    @classmethod
    def _build_cmd(cls) -> list:
        port = cls.API_URL.rsplit(":", 1)[-1]
        return [
            cls.PYTHON_EXEC, cls.SERVER_SCRIPT,
            "--config", cls.CONFIG_PATH,
            "--checkpoint", cls.CHECKPOINT_PATH,
            "--device", f"cuda:{cls.GPU_DEVICES}",
            "--host", "127.0.0.1",
            "--port", port,
        ]


strip_rcnn_manager = StripRCNNServiceManager()
