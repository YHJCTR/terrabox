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

    MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "")
    HOST = "127.0.0.1"
    PORT = 9000
    API_BASE = f"http://{HOST}:{PORT}/v1"

    # Path to the isolated Python interpreter for the vLLM conda environment
    VLLM_PYTHON_EXEC = os.environ.get("VLLM_PYTHON_EXEC", "")

    GPU_DEVICES = os.environ.get("VLM_GPU_DEVICES", "0")
    TENSOR_PARALLEL_SIZE = int(os.environ.get("VLM_TENSOR_PARALLEL_SIZE", "1"))
    MAX_MODEL_LEN = 4096
    GPU_MEMORY_UTILIZATION = 0.9

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
        try:
            from ..agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "vlm_model_path"             in d: cls.MODEL_PATH             = str(d["vlm_model_path"])
            if "vlm_python_exec"            in d: cls.VLLM_PYTHON_EXEC       = str(d["vlm_python_exec"])
            if "vlm_host"                   in d: cls.HOST                   = str(d["vlm_host"])
            if "vlm_port"                   in d: cls.PORT                   = int(d["vlm_port"])
            if "vlm_gpu_devices"            in d: cls.GPU_DEVICES            = str(d["vlm_gpu_devices"])
            if "vlm_tensor_parallel"        in d: cls.TENSOR_PARALLEL_SIZE   = int(d["vlm_tensor_parallel"])
            if "vlm_max_model_len"          in d: cls.MAX_MODEL_LEN          = int(d["vlm_max_model_len"])
            if "vlm_gpu_memory_utilization" in d: cls.GPU_MEMORY_UTILIZATION = float(d["vlm_gpu_memory_utilization"])
            cls.API_BASE = f"http://{cls.HOST}:{cls.PORT}/v1"
        except Exception:
            pass

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
            "--max-model-len", str(cls.MAX_MODEL_LEN),
            "--limit-mm-per-prompt", "image=8",
            "--gpu-memory-utilization", str(cls.GPU_MEMORY_UTILIZATION),
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
