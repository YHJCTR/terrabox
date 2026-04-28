"""
docker_utils.py — GPU 检测 & LLM 服务管理
==========================================
检测空闲 GPU，按需启动 vLLM LLM 子进程（参考 AgentLLMServiceManager）。
同时提供工具 Docker 服务的状态检查。
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "AGENT_LLM_MODEL_PATH",
    "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/",
)
PYTHON_EXEC = os.environ.get(
    "AGENT_LLM_PYTHON_EXEC",
    "/home/yuhongjie/miniconda3/envs/unsloth/bin/python",
)
LLM_BASE_PORT = 9100  # GPU0 → 9100, GPU1 → 9101, ...
MAX_MODEL_LEN = 24576

# Docker 工具服务映射：slug 前缀 → (容器名, 端口)
TOOL_DOCKER_MAP: dict[str, tuple[str, int]] = {
    "geo_perception.remotesam": ("terrabox-remotesam", 9004),
    "geo_perception.sam2_segment": ("terrabox-sam2", 9005),
    "geo_perception.strip_rcnn_detect": ("terrabox-strip-rcnn", 9006),
    "geo_perception.remoteclip_analysis": ("terrabox-remoteclip", 9007),
    "geo_perception.instructsam": ("terrabox-instructsam", 9008),
    # vlm_analyze 使用 LLM docker，不单独列出
}


# ---------------------------------------------------------------------------
# GPU 检测
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    gpu_id: int
    free_mb: int
    total_mb: int
    port: int

    @property
    def free_pct(self) -> float:
        return self.free_mb / self.total_mb * 100 if self.total_mb > 0 else 0


def get_gpu_info() -> list[GPUInfo]:
    """使用 nvidia-smi 获取所有 GPU 的显存信息。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpu_id = int(parts[0])
                free_mb = int(parts[1])
                total_mb = int(parts[2])
                gpus.append(GPUInfo(
                    gpu_id=gpu_id,
                    free_mb=free_mb,
                    total_mb=total_mb,
                    port=LLM_BASE_PORT + gpu_id,
                ))
        return gpus
    except Exception as e:
        logger.warning(f"nvidia-smi failed: {e}")
        return []


def get_free_gpu() -> GPUInfo | None:
    """返回空闲显存最多的 GPU 信息，若无 GPU 则返回 None。"""
    gpus = get_gpu_info()
    if not gpus:
        return None
    return max(gpus, key=lambda g: g.free_mb)


# ---------------------------------------------------------------------------
# LLM 服务管理
# ---------------------------------------------------------------------------

_llm_process: subprocess.Popen | None = None
_llm_port: int | None = None


def is_llm_running(port: int) -> bool:
    """检查指定端口的 vLLM 服务是否响应。"""
    try:
        resp = requests.get(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=2,
            proxies={"http": None, "https": None},
        )
        return resp.status_code == 200
    except Exception:
        return False


def find_running_llm_port() -> int | None:
    """扫描 9100-9103 寻找已运行的 LLM 服务，返回第一个可用端口。"""
    for port in range(LLM_BASE_PORT, LLM_BASE_PORT + 4):
        if is_llm_running(port):
            return port
    return None


def start_llm_service(gpu_id: int | None = None) -> tuple[bool, int, str]:
    """
    启动 vLLM LLM 子进程。

    Returns
    -------
    (success, port, message)
    """
    global _llm_process, _llm_port

    # 先检查是否已有在运行的服务
    existing_port = find_running_llm_port()
    if existing_port is not None:
        _llm_port = existing_port
        return True, existing_port, f"LLM service already running on port {existing_port}"

    # 选择 GPU
    if gpu_id is None:
        gpu = get_free_gpu()
        if gpu is None:
            return False, 0, "No GPU available"
        gpu_id = gpu.gpu_id

    port = LLM_BASE_PORT + gpu_id

    if not Path(PYTHON_EXEC).exists():
        return False, 0, f"Python exec not found: {PYTHON_EXEC}"

    if not Path(MODEL_PATH).exists():
        return False, 0, f"Model path not found: {MODEL_PATH}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["no_proxy"] = "localhost,127.0.0.1"
    env.pop("PYTHONPATH", None)

    log_dir = Path(__file__).parent
    cmd = [
        PYTHON_EXEC,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]

    logger.info(f"Starting LLM on GPU {gpu_id} port {port}...")
    _llm_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_dir / "llm_stdout.log", "w"),
        stderr=open(log_dir / "llm_stderr.log", "w"),
    )
    _llm_port = port

    # 等待启动（最多 10 分钟，每 5 秒检查一次）
    for i in range(120):
        if is_llm_running(port):
            return True, port, f"LLM started on GPU {gpu_id} port {port}"
        if _llm_process.poll() is not None:
            return False, 0, "LLM process exited unexpectedly. Check llm_stderr.log"
        if i % 6 == 0:
            logger.info(f"Waiting for LLM... ({i * 5}s)")
        time.sleep(5)

    return False, 0, "LLM service timed out (>10 min)"


def stop_llm_service():
    """停止 LLM 子进程（若由本进程启动）。"""
    global _llm_process
    if _llm_process and _llm_process.poll() is None:
        _llm_process.terminate()
        try:
            _llm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llm_process.kill()
            _llm_process.wait()
        _llm_process = None
        logger.info("LLM service stopped.")


def get_llm_model_name(port: int) -> str:
    """从 /v1/models 获取模型名称。"""
    try:
        resp = requests.get(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=5,
            proxies={"http": None, "https": None},
        )
        data = resp.json()
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# 工具 Docker 服务检查
# ---------------------------------------------------------------------------

def check_tool_service(slug: str) -> tuple[bool, str]:
    """
    检查 slug 所需的 Docker/子进程服务是否运行。

    Returns
    -------
    (available, message)
    """
    # vlm_analyze 依赖独立的 VLM Docker 容器（terrabox-vllm，port 9000）
    # 与驱动 showcase 的 LLM（9100-9103）不同，不能混用
    if slug == "geo_perception.vlm_analyze":
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format={{.State.Running}}", "terrabox-vllm"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "true":
                return True, "Docker container terrabox-vllm is running (port 9000)"
            return False, "Docker container terrabox-vllm is not running (needed for vlm_analyze)"
        except Exception as e:
            return False, f"Cannot check Docker for terrabox-vllm: {e}"

    info = TOOL_DOCKER_MAP.get(slug)
    if info is None:
        return True, "No external service required"

    container_name, port = info
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container_name],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip() == "true":
            return True, f"Docker container {container_name} is running"
        return False, f"Docker container {container_name} is not running"
    except Exception as e:
        return False, f"Cannot check Docker: {e}"


def get_system_status() -> dict:
    """返回系统状态摘要，供 UI 显示。"""
    gpus = get_gpu_info()
    llm_port = find_running_llm_port()

    gpu_info_list = []
    for g in gpus:
        gpu_info_list.append({
            "gpu_id": g.gpu_id,
            "free_mb": g.free_mb,
            "total_mb": g.total_mb,
            "free_pct": round(g.free_pct, 1),
            "llm_running": is_llm_running(g.port),
            "port": g.port,
        })

    return {
        "gpus": gpu_info_list,
        "llm_port": llm_port,
        "llm_running": llm_port is not None,
    }
