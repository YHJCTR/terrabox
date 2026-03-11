"""Agent configuration loaded from agent_config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass
class AgentConfig:
    # LLM mode
    use_local_llm: bool = True
    # true  = local vLLM process/container; false = remote API
    use_docker: bool = False
    # false = subprocess via local conda env (default); true = Docker container

    # Local LLM (subprocess or Docker)
    local_llm_model_path: str = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
    local_llm_host: str = "127.0.0.1"
    local_llm_port: int = 9100
    local_llm_gpu_devices: str = "0"
    local_llm_tensor_parallel: int = 1
    # Non-Docker: path to the Python interpreter with vLLM installed
    local_llm_python_exec: str = "/home/yuhongjie/miniconda3/envs/unsloth/bin/python"
    # Docker only: image name
    local_llm_docker_image: str = "terrabox/agent-llm:latest"

    # Remote LLM API
    remote_llm_api_base: str = "https://api.openai.com/v1"
    remote_llm_api_key: str = ""
    remote_llm_model: str = "gpt-4o"

    # Local LLM runtime parameters
    local_llm_max_model_len: int = 24576   # token context window (GPU-limited; max ~24960)

    # Agent behavior
    max_iterations: int = 15


def load_config() -> AgentConfig:
    """Load AgentConfig from agent_config.yaml (project root) or return defaults."""
    path = os.environ.get("AGENT_CONFIG_PATH", "agent_config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return AgentConfig(**{k: v for k, v in data.items() if hasattr(AgentConfig, k)})
    return AgentConfig()
