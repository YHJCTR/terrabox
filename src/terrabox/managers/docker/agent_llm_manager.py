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
import subprocess
import time

import requests

from ..gpu_allocator import allocate_gpus
from ..base_manager import BaseServiceManager

logger = logging.getLogger("docker.agent_llm_manager")


class AgentLLMDockerManager(BaseServiceManager):
    _instance = None

    HOST = "127.0.0.1"
    PORT = 9100
    CONTAINER_NAME = "terrabox-agent-llm"

    # Runtime values — overwritten by start_service(config)
    MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
    GPU_DEVICES = "0"
    TENSOR_PARALLEL_SIZE = "1"
    DOCKER_IMAGE = "terrabox/agent-llm:latest"

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

        if cls.is_running():
            return

        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} is running but not yet healthy — waiting...")
        else:
            # Remove any stopped container with the same name
            subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

            gpu = allocate_gpus(
                count=int(cls.TENSOR_PARALLEL_SIZE),
                min_free_mib=8192,
                fallback=cls.GPU_DEVICES,
                env_var="AGENT_LLM_GPU_DEVICES",
            )

            cmd = [
                "docker", "run", "-d",
                "--name", cls.CONTAINER_NAME,
                "--gpus", "all",
                "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
                "-p", f"{cls.PORT}:8000",
                "-v", f"{cls.MODEL_PATH}:/model:ro",
                "--shm-size=8g",
                cls.DOCKER_IMAGE,
                "--model", "/model",
                "--trust-remote-code",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--tensor-parallel-size", cls.TENSOR_PARALLEL_SIZE,
                "--max-model-len", "8192",
                "--gpu-memory-utilization", "0.85",
                "--enforce-eager",
                # Required for LangChain/LangGraph tool calling (tool_choice="auto")
                "--enable-auto-tool-choice",
                "--tool-call-parser", "hermes",
            ]

            logger.info(f"Starting agent LLM container (GPU: {gpu}, model: {cls.MODEL_PATH})...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to start agent LLM container.\nstderr: {result.stderr}"
                )

        logger.info("Waiting for agent LLM to load model (this may take a few minutes)...")
        max_retries = 120  # poll every 5 s, up to 600 s
        for i in range(max_retries):
            if cls.is_running():
                logger.info("Agent LLM service is READY!")
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
        subprocess.run(["docker", "stop", cls.CONTAINER_NAME], capture_output=True)
        subprocess.run(["docker", "rm", cls.CONTAINER_NAME], capture_output=True)


agent_llm_manager = AgentLLMDockerManager()
