"""
docker_utils.py — GPU 检测 & LLM/Docker 服务管理
================================================
检测 GPU 状态，按需启动 Docker agent LLM，并管理感知工具 Docker 服务。
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
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
AGENT_LLM_DOCKER_IMAGE = os.environ.get("AGENT_LLM_DOCKER_IMAGE", "terrabox/agent-llm:latest")
LLM_BASE_PORT = 9100  # GPU0 → 9100, GPU1 → 9101, ...
MAX_MODEL_LEN = 24576

# Docker 工具服务映射：slug 前缀 → (容器名, 端口)
TOOL_DOCKER_MAP: dict[str, tuple[str, int]] = {
    "geo_perception.remotesam": ("terrabox-remotesam", 9004),
    "geo_perception.sam2_segment": ("terrabox-sam2", 9002),
    "geo_perception.strip_rcnn_detect": ("terrabox-strip-rcnn", 9005),
    "geo_perception.remoteclip_analysis": ("terrabox-remoteclip", 9003),
    "geo_perception.instructsam": ("terrabox-instructsam", 9006),
}

# 完整映射（含 vlm_analyze）
TOOL_DOCKER_FULL_MAP: dict[str, tuple[str, int]] = {
    "geo_perception.vlm_analyze": ("terrabox-vllm", 9000),
    **TOOL_DOCKER_MAP,
}

# 容器名 → slug 静态反查表（TOOL_DOCKER_FULL_MAP 构建后立即生成）
_CONTAINER_TO_SLUG: dict[str, str] = {
    cname: slug for slug, (cname, _) in TOOL_DOCKER_FULL_MAP.items()
}


def ensure_local_no_proxy() -> None:
    """确保 localhost/127.0.0.1 在 no_proxy/NO_PROXY 中，防止被系统代理截获。"""
    local = "localhost,127.0.0.1"
    existing = os.environ.get("no_proxy", "")
    if local not in existing:
        os.environ["no_proxy"] = f"{existing},{local}".lstrip(",")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if var in os.environ:
            existing_no = os.environ.get("NO_PROXY", "")
            if local not in existing_no:
                os.environ["NO_PROXY"] = f"{existing_no},{local}".lstrip(",")
            break


ensure_local_no_proxy()


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

_llm_port: int | None = None
_llm_owned_ports: set[int] = set()


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


def find_running_llm_ports() -> list[int]:
    """Return all healthy showcase agent LLM ports in the 9100-9103 range."""
    return [
        port
        for port in range(LLM_BASE_PORT, LLM_BASE_PORT + 4)
        if is_llm_running(port)
    ]


def _agent_llm_container_name(port: int) -> str:
    return f"terrabox-agent-llm-{port}"


def _container_is_running(container_name: str) -> bool:
    """检查 Docker 容器是否正在运行。"""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container_name],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "true"
    except Exception:
        return False


def _container_exists(container_name: str) -> bool:
    """Return whether a Docker container exists, regardless of running state."""
    try:
        result = subprocess.run(
            ["docker", "inspect", container_name],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _container_logs(container_name: str, tail: int = 50) -> str:
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_name],
            capture_output=True, text=True, timeout=10,
        )
        return (result.stdout + result.stderr).strip()
    except Exception as e:
        return f"Cannot read docker logs: {e}"


def _validate_agent_llm_port(port: int) -> int:
    min_port = LLM_BASE_PORT
    max_port = LLM_BASE_PORT + 3
    if port < min_port or port > max_port:
        raise ValueError(f"LLM port must be between {min_port} and {max_port}, got {port}")
    return port


def _select_gpu_for_port(port: int, gpu_id: int | None) -> int:
    if gpu_id is not None:
        return gpu_id
    inferred = port - LLM_BASE_PORT
    gpus = get_gpu_info()
    if any(g.gpu_id == inferred for g in gpus):
        return inferred
    gpu = get_free_gpu()
    if gpu is None:
        raise RuntimeError("No GPU available")
    return gpu.gpu_id


def get_agent_llm_containers() -> list[dict]:
    """Return known agent LLM Docker containers and health by port."""
    result: list[dict] = []
    for port in range(LLM_BASE_PORT, LLM_BASE_PORT + 4):
        name = _agent_llm_container_name(port)
        running = _container_is_running(name)
        healthy = is_llm_running(port)
        result.append({
            "port": port,
            "container": name,
            "running": running,
            "healthy": healthy,
        })
    return result


def start_llm_service(
    gpu_id: int | None = None,
    port: int | None = None,
) -> tuple[bool, int, str]:
    """
    启动指定端口/GPU 的 Docker agent LLM。

    Returns
    -------
    (success, port, message)
    """
    global _llm_port

    try:
        if port is None:
            if gpu_id is None:
                gpu = get_free_gpu()
                if gpu is None:
                    return False, 0, "No GPU available"
                gpu_id = gpu.gpu_id
            port = LLM_BASE_PORT + gpu_id
        port = _validate_agent_llm_port(int(port))
        gpu_id = _select_gpu_for_port(port, gpu_id)
    except Exception as e:
        return False, 0, str(e)

    if is_llm_running(port):
        _llm_port = port
        return True, port, f"LLM service already running on port {port}"

    if not Path(MODEL_PATH).exists():
        return False, 0, f"Model path not found: {MODEL_PATH}"

    container_name = _agent_llm_container_name(port)
    if _container_is_running(container_name):
        logger.info("Container %s is running but not healthy yet; waiting.", container_name)
    else:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--label", "terrabox.showcase=true",
            "--gpus", "all",
            "-e", f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "-e", "no_proxy=localhost,127.0.0.1",
            "-e", "NO_PROXY=localhost,127.0.0.1",
            "-p", f"{port}:8000",
            "-v", f"{MODEL_PATH}:/model:ro",
            "--shm-size=8g",
            AGENT_LLM_DOCKER_IMAGE,
            "--model", "/model",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "1",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        logger.info("Starting Docker LLM %s on GPU %s port %s...", container_name, gpu_id, port)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return False, 0, f"docker run failed: {result.stderr.strip()}"
        _llm_owned_ports.add(port)

    _llm_port = port

    # 等待启动（最多 10 分钟，每 5 秒检查一次）
    for i in range(120):
        if is_llm_running(port):
            return True, port, f"Docker LLM started on GPU {gpu_id} port {port} ({container_name})"
        if not _container_is_running(container_name):
            logs = _container_logs(container_name)
            return False, 0, f"LLM container exited unexpectedly. Logs:\n{logs}"
        if i % 6 == 0:
            logger.info("Waiting for Docker LLM %s... (%ss)", container_name, i * 5)
        time.sleep(5)

    return False, 0, f"LLM service timed out (>10 min). Check: docker logs {container_name}"


def start_llm_subprocess_service(gpu_id: int | None = None) -> tuple[bool, int, str]:
    """Legacy local subprocess launcher kept for manual fallback."""
    python_exec = os.environ.get(
        "AGENT_LLM_PYTHON_EXEC",
        "/home/yuhongjie/miniconda3/envs/unsloth/bin/python",
    )
    if gpu_id is None:
        gpu = get_free_gpu()
        if gpu is None:
            return False, 0, "No GPU available"
        gpu_id = gpu.gpu_id
    port = LLM_BASE_PORT + gpu_id
    if not Path(python_exec).exists():
        return False, 0, f"Python exec not found: {python_exec}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["no_proxy"] = "localhost,127.0.0.1"
    env.pop("PYTHONPATH", None)
    log_dir = Path(__file__).parent
    cmd = [
        python_exec,
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

    logger.info(f"Starting subprocess LLM on GPU {gpu_id} port {port}...")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_dir / "llm_stdout.log", "w"),
        stderr=open(log_dir / "llm_stderr.log", "w"),
    )

    for i in range(120):
        if is_llm_running(port):
            return True, port, f"Subprocess LLM started on GPU {gpu_id} port {port}"
        if process.poll() is not None:
            return False, 0, "LLM process exited unexpectedly. Check llm_stderr.log"
        if i % 6 == 0:
            logger.info(f"Waiting for LLM... ({i * 5}s)")
        time.sleep(5)

    return False, 0, "LLM service timed out (>10 min)"


def stop_llm_service():
    """停止本进程启动的 Docker agent LLM 容器。"""
    global _llm_port
    ports = sorted(_llm_owned_ports)
    for port in ports:
        container_name = _agent_llm_container_name(int(port))
        subprocess.run(["docker", "stop", "-t", "2", container_name], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
        logger.info("Stopped showcase-owned LLM container %s", container_name)
    _llm_owned_ports.clear()
    _llm_port = None


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
# 工具 Docker 服务生命周期管理
# ---------------------------------------------------------------------------

def _is_container_running(container_name: str) -> bool:
    """检查 Docker 容器是否正在运行。"""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container_name],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "true"
    except Exception:
        return False


def _start_tool_with_docker_manager(slug: str) -> None:
    """Create/start a perception container through Terrabox's Docker managers."""
    if slug == "geo_perception.vlm_analyze":
        from terrabox.managers.docker.vllm_manager import VLLMDockerManager
        VLLMDockerManager.start_service()
    elif slug == "geo_perception.sam2_segment":
        from terrabox.managers.docker.sam2_manager import SAM2DockerManager
        SAM2DockerManager.start_service()
    elif slug == "geo_perception.remotesam":
        from terrabox.managers.docker.remotesam_manager import RemoteSAMDockerManager
        RemoteSAMDockerManager.start_service()
    elif slug == "geo_perception.strip_rcnn_detect":
        from terrabox.managers.docker.strip_rcnn_manager import StripRCNNDockerManager
        StripRCNNDockerManager.start_service()
    elif slug == "geo_perception.remoteclip_analysis":
        from terrabox.managers.docker.remoteclip_manager import RemoteCLIPDockerManager
        RemoteCLIPDockerManager.start_service()
    elif slug == "geo_perception.instructsam":
        from terrabox.managers.docker.instructsam_manager import InstructSAMDockerManager
        InstructSAMDockerManager.start_service()
    else:
        raise ValueError(f"No Docker manager registered for {slug}")


def start_tool_container(slug: str, timeout_s: int = 120) -> tuple[bool, str]:
    """
    启动 slug 对应的 Docker 容器，并轮询直至端口可达。

    Returns
    -------
    (success, message)
    """
    info = TOOL_DOCKER_FULL_MAP.get(slug)
    if info is None:
        return True, "No container needed"

    container_name, port = info

    if _is_container_running(container_name):
        return True, f"Container {container_name} already running"

    # VLM startup args include model length/GPU config. Recreate stopped
    # containers so changes in agent_config.yaml take effect instead of
    # reusing stale `docker start` arguments.
    recreate_on_start = slug == "geo_perception.vlm_analyze"

    if _container_exists(container_name) and not recreate_on_start:
        try:
            result = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return False, f"docker start {container_name} failed: {result.stderr.strip()}"
        except Exception as e:
            return False, f"Cannot start container {container_name}: {e}"
    else:
        try:
            if _container_exists(container_name):
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
            _start_tool_with_docker_manager(slug)
        except Exception as e:
            return False, f"Cannot create/start {container_name}: {e}"

    # 轮询端口可达性（每 3 秒检查一次）
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(3)
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            return True, f"Container {container_name} started and listening on port {port}"
        except OSError:
            pass
        # 容器可能已退出
        if not _is_container_running(container_name):
            return False, f"Container {container_name} exited unexpectedly after start"

    # 超时后若容器仍在运行则视为成功（部分服务无固定健康接口）
    if _is_container_running(container_name):
        return True, f"Container {container_name} started (port {port} not yet reachable, proceeding)"
    return False, f"Container {container_name} failed to respond within {timeout_s}s"


def stop_tool_container(slug: str) -> tuple[bool, str]:
    """
    停止 slug 对应的 Docker 容器。

    Returns
    -------
    (success, message)
    """
    info = TOOL_DOCKER_FULL_MAP.get(slug)
    if info is None:
        return True, "No container to stop"

    container_name, _ = info
    try:
        result = subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, f"Container {container_name} stopped"
        return False, f"docker stop {container_name} failed: {result.stderr.strip()}"
    except Exception as e:
        return False, f"Cannot stop container {container_name}: {e}"


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


# ---------------------------------------------------------------------------
# 感知服务 LRU 管理器
# ---------------------------------------------------------------------------

#: 启动一个感知模型容器所需的最低空闲显存（MB）
_PERCEPTION_MIN_FREE_MB = 4000
#: 空闲超时（秒），超过后后台线程自动关停容器
_PERCEPTION_IDLE_TIMEOUT = 300  # 5 minutes


class PerceptionServiceManager:
    """
    感知类 Docker 服务的 LRU 管理器。

    策略：
    - 调用感知工具前 acquire(slug)：按需启动容器，显存不足时淘汰 LRU 容器
    - 工具调用完成后 touch(slug)：刷新最后使用时间（容器保持运行）
    - 后台线程：每 60 s 扫描一次，停止空闲超过 5 min 的容器
    - shutdown()：程序退出时停止所有托管容器
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # container_name → 最后使用时间戳
        self._last_used: dict[str, float] = {}
        # Only containers started by this showcase process are stopped on exit.
        self._owned_containers: set[str] = set()
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="perception-cleanup"
        )
        self._cleanup_thread.start()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def acquire(self, slug: str) -> tuple[bool, str, bool]:
        """
        确保 slug 对应的感知容器正在运行。

        Returns
        -------
        (success, message, was_newly_started)
            was_newly_started=True 表示本次调用触发了容器启动
        """
        info = TOOL_DOCKER_FULL_MAP.get(slug)
        if info is None:
            return True, "No container needed", False

        container_name, _ = info

        # 先在锁外检查容器状态（避免 docker inspect 持锁）
        if _is_container_running(container_name):
            with self._lock:
                self._last_used[container_name] = time.time()
            return True, f"Container {container_name} already running", False

        # 需要启动：在锁内选出 LRU 候选（仅内存操作），锁外再执行 stop/start
        lru_to_evict: str | None = None
        with self._lock:
            if not self._has_free_gpu():
                lru_to_evict = self._evict_lru(exclude=container_name)
                if lru_to_evict is None:
                    return False, "No free GPU VRAM and no evictable container", False

        # 锁外停止被淘汰的容器（耗时 subprocess，不应持锁）
        if lru_to_evict:
            slug_to_stop = _CONTAINER_TO_SLUG.get(lru_to_evict)
            if slug_to_stop:
                stop_tool_container(slug_to_stop)
            logger.info(f"[PerceptionMgr] Evicted LRU container: {lru_to_evict}")

        # 锁外启动新容器（可能需要数十秒）
        success, msg = start_tool_container(slug)
        if success:
            with self._lock:
                self._last_used[container_name] = time.time()
                self._owned_containers.add(container_name)
        return success, msg, True

    def touch(self, slug: str) -> None:
        """工具调用完成后刷新最后使用时间，重置 5 min 倒计时。"""
        info = TOOL_DOCKER_FULL_MAP.get(slug)
        if info:
            with self._lock:
                self._last_used[info[0]] = time.time()

    def running_containers(self) -> list[dict]:
        """返回当前托管的运行中容器列表（供 UI 状态显示）。"""
        with self._lock:
            snapshot = list(self._last_used.items())
        now = time.time()
        result = []
        for name, ts in snapshot:
            if _is_container_running(name):
                result.append({"container": name, "idle_s": int(now - ts)})
        return result

    def shutdown(self) -> None:
        """停止本进程启动的感知容器并终止后台线程。"""
        self._stop_event.set()
        with self._lock:
            names = list(self._owned_containers)
            self._last_used.clear()
            self._owned_containers.clear()
        for name in names:
            slug = _CONTAINER_TO_SLUG.get(name)
            if slug:
                stop_tool_container(slug)
                logger.info(f"[PerceptionMgr] shutdown: stopped {name}")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _has_free_gpu(self) -> bool:
        """是否存在至少一块有足够空闲显存的 GPU。"""
        gpus = get_gpu_info()
        return any(g.free_mb >= _PERCEPTION_MIN_FREE_MB for g in gpus)

    def _evict_lru(self, exclude: str = "") -> str | None:
        """
        从 _last_used 中选出最久未使用的容器名（不含 exclude），将其从 dict 中移除并返回。
        调用方需持有 self._lock；实际 stop 操作由调用方在锁外执行。
        """
        candidates = [
            (name, ts) for name, ts in self._last_used.items() if name != exclude
            and name in self._owned_containers
        ]
        if not candidates:
            return None
        lru_name = min(candidates, key=lambda x: x[1])[0]
        self._last_used.pop(lru_name, None)
        self._owned_containers.discard(lru_name)
        return lru_name

    def _cleanup_loop(self) -> None:
        """后台线程：每 60 s 检查一次，停止空闲超过 5 min 的容器。"""
        while not self._stop_event.wait(60):
            now = time.time()
            with self._lock:
                to_stop = [
                    name for name, ts in list(self._last_used.items())
                    if now - ts > _PERCEPTION_IDLE_TIMEOUT
                    and name in self._owned_containers
                ]
                for name in to_stop:
                    self._last_used.pop(name, None)
                    self._owned_containers.discard(name)
            for name in to_stop:
                slug = _CONTAINER_TO_SLUG.get(name)
                if slug:
                    stop_tool_container(slug)
                    logger.info(f"[PerceptionMgr] Idle timeout: stopped {name}")

# 全局单例，模块加载时创建
perception_manager = PerceptionServiceManager()


def cleanup_showcase_services() -> None:
    """Stop Docker services started by this showcase process."""
    stop_llm_service()
    perception_manager.shutdown()


def get_system_status() -> dict:
    """返回系统状态摘要，供 UI 显示。"""
    gpus = get_gpu_info()
    running_ports = find_running_llm_ports()
    llm_containers = {c["port"]: c for c in get_agent_llm_containers()}

    gpu_info_list = []
    for g in gpus:
        container = llm_containers.get(g.port, {})
        gpu_info_list.append({
            "gpu_id": g.gpu_id,
            "free_mb": g.free_mb,
            "total_mb": g.total_mb,
            "free_pct": round(g.free_pct, 1),
            "llm_running": g.port in running_ports,
            "llm_container_running": bool(container.get("running")),
            "llm_container": container.get("container", _agent_llm_container_name(g.port)),
            "port": g.port,
        })

    return {
        "gpus": gpu_info_list,
        "llm_port": running_ports[0] if running_ports else None,
        "llm_ports": running_ports,
        "llm_running": bool(running_ports),
        "llm_containers": list(llm_containers.values()),
    }
