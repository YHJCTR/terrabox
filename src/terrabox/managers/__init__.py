"""Docker-only AI service managers for the tools-only branch."""

from .docker.vllm_manager import vllm_manager
from .docker.sam2_manager import sam2_manager
from .docker.remoteclip_manager import remoteclip_manager
from .docker.remotesam_manager import remotesam_manager
from .docker.strip_rcnn_manager import strip_rcnn_manager
from .docker.instructsam_manager import instructsam_manager

__all__ = [
    "vllm_manager",
    "sam2_manager",
    "remoteclip_manager",
    "remotesam_manager",
    "strip_rcnn_manager",
    "instructsam_manager",
]
