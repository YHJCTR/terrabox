import subprocess
import time
import os
import requests
import logging
import sys
from .base_manager import BaseServiceManager

logger = logging.getLogger("vllm_manager")


class VLLMServiceManager(BaseServiceManager):
    _instance = None
    _process = None

    MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model/")
    HOST = "127.0.0.1"
    PORT = 9000
    API_BASE = f"http://{HOST}:{PORT}/v1"

    # Path to the isolated Python interpreter for the vLLM conda environment
    VLLM_PYTHON_EXEC = os.environ.get(
        "VLLM_PYTHON_EXEC",
        "/home/yuhongjie/miniconda3/envs/unsloth/bin/python"
    )

    GPU_DEVICES = os.environ.get("VLM_GPU_DEVICES", "2,3")
    TENSOR_PARALLEL_SIZE = int(os.environ.get("VLM_TENSOR_PARALLEL_SIZE", "2"))

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLLMServiceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        """Check if the vLLM service is reachable via its health endpoint."""
        try:
            resp = requests.get(
                f"http://{cls.HOST}:{cls.PORT}/ping",
                timeout=1,
                proxies={"http": None, "https": None},
            )
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def start_service(cls):
        """Start the vLLM subprocess using the isolated conda environment."""
        if cls.is_running():
            return

        if not os.path.exists(cls.VLLM_PYTHON_EXEC):
            logger.error(f"Isolated Python environment not found at: {cls.VLLM_PYTHON_EXEC}")
            raise FileNotFoundError(f"Python interpreter not found: {cls.VLLM_PYTHON_EXEC}")

        logger.info(f"Starting vLLM service using env: {cls.VLLM_PYTHON_EXEC}")
        logger.info(f"GPUs: {cls.GPU_DEVICES} | Model: {cls.MODEL_PATH}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env.pop("PYTHONPATH", None)

        cmd = [
            cls.VLLM_PYTHON_EXEC,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", cls.MODEL_PATH,
            "--trust-remote-code",
            "--host", cls.HOST,
            "--port", str(cls.PORT),
            "--tensor-parallel-size", str(cls.TENSOR_PARALLEL_SIZE),
            "--max-model-len", "4096",
            "--limit-mm-per-prompt", "image=8",
            "--gpu-memory-utilization", "0.9",
            "--enforce-eager",
            "--timeout-keep-alive", "3600",
        ]

        cls._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=open("vllm_stdout.log", "w"),
            stderr=open("vllm_stderr.log", "w")
        )

        logger.info("Waiting for vLLM to load model...")
        max_retries = 480  # poll every 5 s, up to 2400 s (40 min)

        for i in range(max_retries):
            if cls.is_running():
                logger.info("vLLM service is READY!")
                return

            if i % 6 == 0:
                logger.info(f"Still loading... ({i * 5}s elapsed)")

            if cls._process.poll() is not None:
                raise RuntimeError("vLLM process exited unexpectedly! Check vllm_stderr.log.")

            time.sleep(5)

        cls.stop_service()
        raise TimeoutError("vLLM service failed to start within timeout.")

    @classmethod
    def stop_service(cls):
        """Gracefully stop the vLLM subprocess."""
        if cls._process:
            logger.warning("Auto-cleanup: Stopping vLLM service...")
            try:
                cls._process.terminate()
                try:
                    cls._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM didn't exit in time, force killing...")
                    cls._process.kill()
                    cls._process.wait()
                logger.info("vLLM service stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping vLLM: {e}")
            finally:
                cls._process = None


vllm_manager = VLLMServiceManager()
