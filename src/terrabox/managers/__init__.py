"""Manager package — selects subprocess or Docker variant based on TERRABOX_USE_DOCKER."""
import os

_use_docker = os.environ.get("TERRABOX_USE_DOCKER", "false").lower() == "true"

if _use_docker:
    from .docker.vllm_manager import vllm_manager
    from .docker.sam2_manager import sam2_manager
    from .docker.remoteclip_manager import remoteclip_manager
    from .docker.remotesam_manager import remotesam_manager
    from .docker.strip_rcnn_manager import strip_rcnn_manager
    from .docker.instructsam_manager import instructsam_manager
else:
    from .vllm_manager import vllm_manager
    from .sam2_manager import sam2_manager
    from .remoteclip_manager import remoteclip_manager
    from .remotesam_manager import remotesam_manager
    from .strip_rcnn_manager import strip_rcnn_manager
    from .instructsam_manager import instructsam_manager

# agent_llm_manager only has a subprocess variant
from .agent_llm_manager import agent_llm_manager

__all__ = [
    "vllm_manager",
    "sam2_manager",
    "remoteclip_manager",
    "remotesam_manager",
    "strip_rcnn_manager",
    "instructsam_manager",
    "agent_llm_manager",
]
