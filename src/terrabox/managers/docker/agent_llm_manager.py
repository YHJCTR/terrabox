"""
Agent LLM Docker Manager
========================
Manages a vLLM Docker container serving the agent's reasoning LLM
(pure text model, separate from the VLM container at port 9000).

Default port: 9100  (avoids conflict with the VLM container on 9000)
Container name: terrabox-agent-llm
"""
from __future__ import annotations

import logging
import os
import subprocess
import time

import requests

from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

logger = logging.getLogger("docker.agent_llm_manager")


class AgentLLMDockerManager(BaseServiceManager):
    _instance = None

    HOST = "127.0.0.1"
    PORT = 9100
    CONTAINER_NAME = "terrabox-agent-llm"
    CONTAINER_BASE = "terrabox-agent-llm"

    # Runtime values — overwritten by start_service(config)
    MODEL_PATH = os.environ.get("AGENT_LLM_MODEL_PATH", "")
    GPU_DEVICES = "0"
    TENSOR_PARALLEL_SIZE = "1"
    DOCKER_IMAGE = "terrabox/agent-llm:latest"
    MAX_MODEL_LEN = "24576"
    _lease = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _api_base(cls) -> str:
        return f"http://{cls.HOST}:{cls.PORT}/v1"

    @classmethod
    def is_running(cls) -> bool:
        try:
            resp = requests.get(
                f"http://{cls.HOST}:{cls.PORT}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            )
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def _container_is_running(cls) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", cls.CONTAINER_NAME],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def _get_container_logs(cls, tail: int = 50) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), cls.CONTAINER_NAME],
            capture_output=True, text=True,
        )
        return (result.stdout + result.stderr).strip()

    @classmethod
    def start_service(cls, config=None) -> None:
        """Start the agent LLM container.  Pass an AgentConfig to override defaults."""
        if config is not None:
            cls.MODEL_PATH = config.local_llm_model_path
            cls.GPU_DEVICES = config.local_llm_gpu_devices
            cls.TENSOR_PARALLEL_SIZE = str(config.local_llm_tensor_parallel)
            cls.PORT = config.local_llm_port
            cls.DOCKER_IMAGE = config.local_llm_docker_image
            cls.MAX_MODEL_LEN = str(config.local_llm_max_model_len)

        if cls.is_running():
            return

        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} is running but service is not healthy; rebuilding it.")
            cls.stop_service()

        lease = acquire_docker_lease(
            service="agent-llm",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host=cls.HOST,
            base_port=cls.PORT,
            internal_port=8000,
            gpu_count=int(cls.TENSOR_PARALLEL_SIZE),
            min_free_mib=int(os.environ.get("AGENT_LLM_MIN_FREE_MIB", "16000")),
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="AGENT_LLM_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.PORT = lease.port
        remove_container_if_exists(cls.CONTAINER_NAME, service="agent-llm", reason="before_docker_run")

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", "all",
            "-e", f"CUDA_VISIBLE_DEVICES={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            "-v", f"{cls.MODEL_PATH}:/model:ro",
            "--shm-size=8g",
            cls.DOCKER_IMAGE,
            "--model", "/model",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", cls.TENSOR_PARALLEL_SIZE,
            "--max-model-len", cls.MAX_MODEL_LEN,
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",
            # Required for LangChain/LangGraph tool calling (tool_choice="auto")
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        logger.info(f"Starting agent LLM container (GPU: {lease.gpu_devices}, port: {lease.port}, model: {cls.MODEL_PATH})...")
        record_service_event({"event": "start_requested", "service": "agent-llm", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
            record_service_event({"event": "start_failed", "service": "agent-llm", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start agent LLM container.\nstderr: {result.stderr}"
            )

        logger.info("Waiting for agent LLM to load model (this may take a few minutes)...")
        max_retries = 120  # poll every 5 s, up to 600 s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("Agent LLM service is READY!")
                record_service_event({"event": "ready", "service": "agent-llm", "lease": cls._lease.__dict__ if cls._lease else None})
                return

            if not cls._container_is_running():
                logs = cls._get_container_logs()
                raise RuntimeError(
                    f"Agent LLM container exited unexpectedly.\n"
                    f"--- docker logs (last 50 lines) ---\n{logs}"
                )

            if i % 6 == 0:
                logger.info(f"Still loading... ({i * 5}s elapsed)")

            time.sleep(5)

        cls.stop_service()
        raise TimeoutError(
            f"Agent LLM service failed to start within timeout. "
            f"Check: docker logs {cls.CONTAINER_NAME}"
        )

    @classmethod
    def stop_service(cls) -> None:
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        # -t 2: send SIGKILL after 2s instead of default 10s
        subprocess.run(
            ["docker", "stop", "-t", "2", cls.CONTAINER_NAME],
            capture_output=True, timeout=6,
        )
        subprocess.run(
            ["docker", "rm", "-f", cls.CONTAINER_NAME],
            capture_output=True, timeout=5,
        )
        record_service_event({"event": "stopped", "service": "agent-llm", "container": cls.CONTAINER_NAME})


agent_llm_manager = AgentLLMDockerManager()
