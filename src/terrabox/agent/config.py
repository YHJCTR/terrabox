"""Agent configuration loaded from agent_config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

from .modes import list_agent_modes


@dataclass
class AgentConfig:
    # LLM mode
    # true  = local vLLM process/container; false = remote API
    use_local_llm: bool = True
    # false = subprocess via local conda env (default); true = Docker container
    use_docker: bool = False

    # Local LLM (subprocess or Docker)
    local_llm_model_path: str = os.environ.get("AGENT_LLM_MODEL_PATH", "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/")
    local_llm_host: str = "127.0.0.1"
    local_llm_port: int = 9100
    local_llm_gpu_devices: str = os.environ.get("AGENT_LLM_GPU_DEVICES", "0")
    local_llm_tensor_parallel: int = 1
    # Non-Docker: path to the Python interpreter with vLLM installed
    local_llm_python_exec: str = os.environ.get("AGENT_LLM_PYTHON_EXEC", "/home/yuhongjie/miniconda3/envs/unsloth/bin/python")
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
    max_tool_calls_per_run: int = 6
    max_failed_tool_calls_per_run: int = 3
    max_repeated_tool_failures: int = 1

    # Tool loading strategy: "standard" | "progressive" | "category_scoped" | "artifact_progressive"
    #   standard        — load all tools at once (full ReAct)
    #   progressive     — 3-level discovery, single-tool execution with retry
    #   category_scoped — LLM picks categories first, then full ReAct on subset
    #   artifact_progressive — disclose tools from runtime artifacts and tool IO contracts
    agent_mode: str = "standard"
    max_retries_on_error: int = 3       # retries for progressive mode
    max_progressive_steps: int = 10     # max discovery→execute cycles for progressive mode
    max_category_expansions: int = 2    # max expansion rounds for category_scoped mode

    # Harness runtime guards
    max_concurrent_agent_runs: int = 64
    default_tool_timeout_seconds: int = 120
    default_tool_bulkhead: int = 8
    perception_tool_bulkhead: int = 2
    compute_tool_bulkhead: int = 4
    network_tool_bulkhead: int = 6
    risky_tool_bulkhead: int = 1
    breaker_failure_threshold: int = 5
    breaker_recovery_timeout_s: float = 30.0

    # Safety / approval
    require_approval_for_risky_tools: bool = False
    record_risky_tool_approvals: bool = True
    human_tool_approval: dict = None  # type: ignore[assignment]
    human_tool_approval_timeout_seconds: int = 300
    human_tool_approval_poll_seconds: float = 1.0

    # Agent execution timeout (seconds); 0 = no timeout
    agent_timeout_seconds: int = 0

    # Session history limits (moved from session.py module-level constants)
    max_history_messages: int = 20
    summary_threshold: int = 15
    summary_keep_recent: int = 5

    # Per-bucket circuit breaker overrides.
    # Key = bucket name ("risky", "perception", "compute", "network", "default").
    # Value = dict with optional keys: threshold (int), recovery_s (float).
    # Example: {"risky": {"threshold": 3, "recovery_s": 20.0}}
    circuit_breaker_overrides: dict = None  # type: ignore[assignment]

    # Memory / rate limit are opt-in to avoid changing existing behavior
    enable_memory_writeback: bool = False
    enable_agent_rate_limit: bool = False


def load_config() -> AgentConfig:
    """Load AgentConfig from agent_config.yaml (project root) or return defaults."""
    import logging as _logging
    path = os.environ.get("AGENT_CONFIG_PATH", "agent_config.yaml")
    data: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                f"Failed to load agent config from {path}: {exc}. Using defaults."
            )

    config = AgentConfig(**{k: v for k, v in data.items() if hasattr(AgentConfig, k)})
    if config.circuit_breaker_overrides is None:
        config.circuit_breaker_overrides = {}
    if config.human_tool_approval is None:
        config.human_tool_approval = {}

    valid_modes = list_agent_modes()
    if config.agent_mode not in valid_modes:
        raise ValueError(
            f"Invalid agent_mode {config.agent_mode!r}. "
            f"Must be one of: {list(valid_modes)}"
        )
    if config.max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {config.max_iterations}")
    if config.max_tool_calls_per_run < 0:
        raise ValueError(f"max_tool_calls_per_run must be >= 0, got {config.max_tool_calls_per_run}")
    if config.max_failed_tool_calls_per_run < 0:
        raise ValueError(
            f"max_failed_tool_calls_per_run must be >= 0, got {config.max_failed_tool_calls_per_run}"
        )
    if config.max_repeated_tool_failures < 0:
        raise ValueError(f"max_repeated_tool_failures must be >= 0, got {config.max_repeated_tool_failures}")
    if config.max_retries_on_error < 0:
        raise ValueError(f"max_retries_on_error must be >= 0, got {config.max_retries_on_error}")
    if config.max_category_expansions < 0:
        raise ValueError(f"max_category_expansions must be >= 0, got {config.max_category_expansions}")
    if config.human_tool_approval_timeout_seconds < 0:
        raise ValueError(
            "human_tool_approval_timeout_seconds must be >= 0, "
            f"got {config.human_tool_approval_timeout_seconds}"
        )
    if config.human_tool_approval_poll_seconds <= 0:
        raise ValueError(
            "human_tool_approval_poll_seconds must be > 0, "
            f"got {config.human_tool_approval_poll_seconds}"
        )

    # Allow per-process port override for parallel GPU execution
    if port_env := os.environ.get("AGENT_LLM_PORT"):
        config.local_llm_port = int(port_env)

    return config


_yaml_cache: "dict | None" = None


def load_raw_yaml() -> dict:
    """Return the full agent_config.yaml as a dict (cached, no schema filtering).
    Used by service managers to read their own config keys without going through AgentConfig."""
    global _yaml_cache
    if _yaml_cache is not None:
        return _yaml_cache
    path = os.environ.get("AGENT_CONFIG_PATH", "agent_config.yaml")
    _yaml_cache = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                _yaml_cache = yaml.safe_load(f) or {}
        except Exception:
            pass
    return _yaml_cache
