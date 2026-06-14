"""Manager package — selects subprocess or Docker variant based on TERRABOX_USE_DOCKER."""
import os


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_use_docker_from_yaml() -> bool:
    try:
        import yaml

        path = os.environ.get("AGENT_CONFIG_PATH", "agent_config.yaml")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return _boolish(data.get("use_docker", False))
    except Exception:
        return False


_docker_env = os.environ.get("TERRABOX_USE_DOCKER")
_use_docker = _boolish(_docker_env) if _docker_env is not None else _load_use_docker_from_yaml()

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

# ChangeOS is docker-only (faithful torch==1.10.0 image); no subprocess variant.
from .docker.changeos_manager import changeos_manager

# agent_llm_manager only has a subprocess variant
from .agent_llm_manager import agent_llm_manager

__all__ = [
    "vllm_manager",
    "sam2_manager",
    "remoteclip_manager",
    "remotesam_manager",
    "strip_rcnn_manager",
    "instructsam_manager",
    "changeos_manager",
    "agent_llm_manager",
]
