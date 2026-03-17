"""
Agent LLM Service Manager (non-Docker)
=======================================
Spawns a vLLM OpenAI-compatible server as a subprocess using the local
unsloth conda environment, mirroring the pattern of utils/vllm_manager.py.

Default port: 9100  (avoids conflict with the VLM manager on 9000)
Log files:    agent_llm_stdout.log / agent_llm_stderr.log
"""
import logging
import os
import subprocess
import time

import requests
from .base_manager import BaseServiceManager

logger = logging.getLogger("agent_llm_manager")


class AgentLLMServiceManager(BaseServiceManager):
    _instance = None
    _process = None

    HOST = "127.0.0.1"
    PORT = 9100
    API_BASE = f"http://{HOST}:{PORT}/v1"

    MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"
    GPU_DEVICES = "0"
    TENSOR_PARALLEL_SIZE = 1
    MAX_MODEL_LEN = 24576

    # Python interpreter of the vLLM-capable conda environment
    PYTHON_EXEC = os.environ.get(
        "AGENT_LLM_PYTHON_EXEC",
        "/home/yuhongjie/miniconda3/envs/unsloth/bin/python",
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls) -> bool:
        try:
            resp = requests.get(f"http://{cls.HOST}:{cls.PORT}/health", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def start_service(cls, config=None) -> None:
        """Start the agent LLM subprocess.  Pass an AgentConfig to override defaults."""
        if config is not None:
            cls.MODEL_PATH = config.local_llm_model_path
            cls.GPU_DEVICES = config.local_llm_gpu_devices
            cls.TENSOR_PARALLEL_SIZE = config.local_llm_tensor_parallel
            cls.PORT = config.local_llm_port
            cls.API_BASE = f"http://{cls.HOST}:{cls.PORT}/v1"
            cls.PYTHON_EXEC = config.local_llm_python_exec
            cls.MAX_MODEL_LEN = config.local_llm_max_model_len

        if cls.is_running():
            return

        if not os.path.exists(cls.PYTHON_EXEC):
            raise FileNotFoundError(
                f"Python interpreter not found: {cls.PYTHON_EXEC}\n"
                "Set local_llm_python_exec in agent_config.yaml."
            )

        logger.info(f"Starting Agent LLM using env: {cls.PYTHON_EXEC}")
        logger.info(f"GPUs: {cls.GPU_DEVICES} | Model: {cls.MODEL_PATH}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env.pop("PYTHONPATH", None)

        cmd = [
            cls.PYTHON_EXEC,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", cls.MODEL_PATH,
            "--trust-remote-code",
            "--host", cls.HOST,
            "--port", str(cls.PORT),
            "--tensor-parallel-size", str(cls.TENSOR_PARALLEL_SIZE),
            "--max-model-len", str(cls.MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",
            # Required for LangChain/LangGraph tool calling (tool_choice="auto")
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        cls._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=open("agent_llm_stdout.log", "w"),
            stderr=open("agent_llm_stderr.log", "w"),
        )

        logger.info("Waiting for Agent LLM to load model...")
        max_retries = 120  # poll every 5 s, up to 600 s

        for i in range(max_retries):
            if cls.is_running():
                logger.info("Agent LLM service is READY!")
                return

            if i % 6 == 0:
                logger.info(f"Still loading... ({i * 5}s elapsed)")

            if cls._process.poll() is not None:
                raise RuntimeError(
                    "Agent LLM process exited unexpectedly! Check agent_llm_stderr.log."
                )

            time.sleep(5)

        cls.stop_service()
        raise TimeoutError("Agent LLM service failed to start within timeout.")

    @classmethod
    def stop_service(cls) -> None:
        if cls._process:
            logger.warning("Auto-cleanup: Stopping Agent LLM service...")
            try:
                cls._process.terminate()
                try:
                    cls._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Agent LLM didn't exit in time, force killing...")
                    cls._process.kill()
                    cls._process.wait()
                logger.info("Agent LLM service stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping Agent LLM: {e}")
            finally:
                cls._process = None


agent_llm_manager = AgentLLMServiceManager()
