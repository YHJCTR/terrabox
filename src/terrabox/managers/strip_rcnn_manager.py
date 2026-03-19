import os
from .base_manager import SubprocessServiceManager


class StripRCNNServiceManager(SubprocessServiceManager):
    SERVICE_NAME = "Strip R-CNN"
    PYTHON_EXEC = os.environ.get("STRIP_RCNN_PYTHON_EXEC", "/home/yuhongjie/miniconda3/envs/strip/bin/python")
    SERVER_SCRIPT = os.environ.get("STRIP_RCNN_SERVER_SCRIPT", "/data1/yuhongjie2/Strip-RCNN/start.py")
    WORK_DIR = os.environ.get("STRIP_RCNN_WORK_DIR", "/data1/yuhongjie2/Strip-RCNN")
    API_URL = "http://127.0.0.1:" + os.environ.get("STRIP_RCNN_PORT", "9005")
    GPU_DEVICES = os.environ.get("STRIP_RCNN_GPU_DEVICES", "0")

    CONFIG_PATH = os.environ.get(
        "STRIP_RCNN_CONFIG_PATH",
        "/data1/yuhongjie2/Strip-RCNN/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py",
    )
    CHECKPOINT_PATH = os.environ.get(
        "STRIP_RCNN_CHECKPOINT_PATH",
        "/data1/yuhongjie2/Strip-RCNN/ckpt/stripnet_s.pth",
    )

    @classmethod
    def _check_paths(cls) -> None:
        super()._check_paths()
        if not os.path.exists(cls.CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found at: {cls.CONFIG_PATH}")
        if not os.path.exists(cls.CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found at: {cls.CHECKPOINT_PATH}")

    @classmethod
    def _build_cmd(cls) -> list:
        return [
            cls.PYTHON_EXEC, cls.SERVER_SCRIPT,
            "--config", cls.CONFIG_PATH,
            "--checkpoint", cls.CHECKPOINT_PATH,
            "--device", f"cuda:{cls.GPU_DEVICES}",
            "--host", "127.0.0.1",
            "--port", "9005",
        ]


strip_rcnn_manager = StripRCNNServiceManager()
