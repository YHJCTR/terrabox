"""
base_manager.py
===============
AOP-compliant base class for all GPU service managers.

The cross-cutting concern — "stop all managed services on process exit" — is
defined here exactly once.  Any class that subclasses BaseServiceManager
automatically gets:

  1. Auto-registration in ServiceRegistry the first time start_service() succeeds.
  2. A single atexit handler (not one per manager) that calls cleanup_all().
  3. SIGINT / SIGTERM handlers that chain to uvicorn's existing handlers.
  4. Idempotent cleanup (tracks which managers have already been stopped).
  5. Thread safety via a module-level lock.

Adding a new manager requires only:
    class MyNewManager(BaseServiceManager):
        @classmethod
        def start_service(cls): ...
        @classmethod
        def stop_service(cls): ...
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
import time
from typing import List, Set, Type

import requests

logger = logging.getLogger("base_manager")


class ServiceRegistry:
    """
    Central registry for all BaseServiceManager subclasses that have been
    started at least once.  Provides idempotent cleanup_all() that can be
    called from the FastAPI lifespan, atexit, or signal handlers.
    """

    _registry: List[Type["BaseServiceManager"]] = []
    _stopped: Set[Type["BaseServiceManager"]] = set()
    _last_used: dict = {}           # manager_cls -> float (time.time() of last start)
    _lock: threading.Lock = threading.Lock()
    _handlers_installed: bool = False

    @classmethod
    def register(cls, manager_cls: Type["BaseServiceManager"]) -> None:
        """Add a manager class to the registry (idempotent)."""
        with cls._lock:
            if manager_cls not in cls._registry:
                cls._registry.append(manager_cls)
                logger.debug(f"Registered manager: {manager_cls.__name__}")

    @classmethod
    def cleanup_all(cls) -> None:
        """
        Stop all registered managers that have not already been stopped.
        Idempotent: safe to call multiple times (lifespan + atexit both call it).
        """
        with cls._lock:
            pending = [m for m in cls._registry if m not in cls._stopped]

        if not pending:
            return

        logger.info(
            f"ServiceRegistry: stopping {len(pending)} service(s): "
            + ", ".join(m.__name__ for m in pending)
        )

        for manager_cls in pending:
            try:
                manager_cls.stop_service()
                logger.info(f"Stopped: {manager_cls.__name__}")
            except Exception as exc:
                logger.error(f"Error stopping {manager_cls.__name__}: {exc}", exc_info=True)
            finally:
                with cls._lock:
                    cls._stopped.add(manager_cls)

    @classmethod
    def mark_used(cls, manager_cls: Type["BaseServiceManager"]) -> None:
        """Record that a service was successfully started/reused right now."""
        with cls._lock:
            cls._last_used[manager_cls] = time.time()

    @classmethod
    def evict_lru_for_gpu(
        cls,
        min_free_mib: int,
        exclude_cls: Type["BaseServiceManager"],
        count: int = 1,
    ) -> None:
        """Stop LRU-registered services (excluding exclude_cls) until GPU has enough free memory.

        After each eviction waits 3 s for the container to release GPU memory before re-checking.
        Evicted managers are removed from the registry so they can be re-registered on next start.
        """
        from .gpu_allocator import has_free_gpu, has_free_gpus

        def _enough() -> bool:
            return has_free_gpus(count, min_free_mib) if count > 1 else has_free_gpu(min_free_mib)

        if _enough():
            return

        with cls._lock:
            candidates = sorted(
                [m for m in cls._registry if m is not exclude_cls and m not in cls._stopped],
                key=lambda m: cls._last_used.get(m, 0.0),
            )

        for manager_cls in candidates:
            if _enough():
                return
            try:
                if not manager_cls.is_running():
                    continue
                logger.info(
                    f"GPU eviction: stopping {manager_cls.__name__} "
                    f"(last used: {cls._last_used.get(manager_cls, 0.0):.0f})"
                )
                manager_cls.stop_service()
                with cls._lock:
                    try:
                        cls._registry.remove(manager_cls)
                    except ValueError:
                        pass
                    cls._stopped.discard(manager_cls)
                    cls._last_used.pop(manager_cls, None)
                logger.info(f"Evicted {manager_cls.__name__}. Waiting for GPU memory release...")
                time.sleep(3)
            except Exception as exc:
                logger.warning(f"Error evicting {manager_cls.__name__}: {exc}")

    @classmethod
    def _install_handlers(cls) -> None:
        """
        Install a single atexit handler and SIGINT/SIGTERM signal handlers.
        Called once when the first BaseServiceManager subclass is defined.
        Signal handlers chain to any previously-installed handler so we do
        not break uvicorn's shutdown flow.
        """
        if cls._handlers_installed:
            return
        cls._handlers_installed = True

        atexit.register(cls.cleanup_all)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)

                def handler(signum, frame, _prev=prev, _sig=sig):
                    cls.cleanup_all()
                    if callable(_prev):
                        _prev(signum, frame)
                    elif _prev == signal.SIG_DFL:
                        signal.signal(_sig, signal.SIG_DFL)
                        signal.raise_signal(signum)

                signal.signal(sig, handler)
            except (OSError, ValueError) as exc:
                # signal.signal() can fail if called from a non-main thread
                logger.warning(f"Could not install handler for {sig.name}: {exc}")


class BaseServiceManager:
    """
    Base class for all GPU service managers.

    Subclasses implement start_service(), stop_service(), and optionally
    is_running().  The cleanup registration is woven in automatically via
    __init_subclass__ — no boilerplate needed in the subclass.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        """
        AOP join point: fires at class definition time for every subclass.
        Wraps start_service() to auto-register the class in ServiceRegistry
        after a successful call.
        """
        super().__init_subclass__(**kwargs)

        # Install global handlers once (idempotent internally)
        ServiceRegistry._install_handlers()

        original_start = cls.__dict__.get("start_service")
        if original_start is None:
            # Subclass inherits start_service from a parent whose wrapper
            # already handles registration — no double-wrap needed.
            return

        raw_fn = original_start.__func__ if isinstance(original_start, classmethod) else original_start

        def _wrapped_start(inner_cls, *args, **kwargs):
            raw_fn(inner_cls, *args, **kwargs)
            # Register only on successful return (no exception)
            ServiceRegistry.register(inner_cls)
            ServiceRegistry.mark_used(inner_cls)

        cls.start_service = classmethod(_wrapped_start)

    @classmethod
    def _ensure_gpu(cls, min_free_mib: int, env_var: str | None = None) -> str:
        """Return a GPU ID, evicting LRU services first if GPU memory is insufficient.

        If env_var is set and non-empty, returns its value immediately (user-pinned GPU).
        Falls back to cls.GPU_DEVICES if dynamic allocation fails after eviction.
        """
        from .gpu_allocator import allocate_gpu, has_free_gpu

        if env_var:
            val = os.environ.get(env_var, "").strip()
            if val:
                return val

        if not has_free_gpu(min_free_mib):
            ServiceRegistry.evict_lru_for_gpu(min_free_mib, exclude_cls=cls, count=1)

        return allocate_gpu(
            min_free_mib=min_free_mib,
            fallback=getattr(cls, "GPU_DEVICES", "0"),
            env_var=None,
        )

    @classmethod
    def _ensure_gpus(cls, count: int, min_free_mib: int, env_var: str | None = None) -> str:
        """Multi-GPU variant of _ensure_gpu. Returns comma-separated GPU IDs."""
        from .gpu_allocator import allocate_gpus, has_free_gpus

        if env_var:
            val = os.environ.get(env_var, "").strip()
            if val:
                return val

        if not has_free_gpus(count, min_free_mib):
            ServiceRegistry.evict_lru_for_gpu(min_free_mib, exclude_cls=cls, count=count)

        return allocate_gpus(
            count=count,
            min_free_mib=min_free_mib,
            fallback=getattr(cls, "GPU_DEVICES", "0"),
            env_var=None,
        )


_subprocess_logger = logging.getLogger("subprocess_manager")


class SubprocessServiceManager(BaseServiceManager):
    """Base class for subprocess-based GPU service managers.

    Subclasses declare class variables only; all lifecycle logic lives here.
    Override _check_paths() to add extra pre-start file checks, or
    _build_cmd() to customize the subprocess command.
    """

    _instance = None
    _process = None

    SERVICE_NAME: str = "Service"
    PYTHON_EXEC: str = ""
    SERVER_SCRIPT: str = ""
    WORK_DIR: str = ""
    API_URL: str = "http://127.0.0.1:9000"
    GPU_DEVICES: str = "0"
    MAX_RETRIES: int = 30

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls) -> bool:
        try:
            return requests.get(
                f"{cls.API_URL}/health", timeout=1,
                proxies={"http": None, "https": None},
            ).status_code == 200
        except Exception:
            return False

    @classmethod
    def _check_paths(cls) -> None:
        """Raise FileNotFoundError if required files are missing."""
        if not os.path.exists(cls.SERVER_SCRIPT):
            raise FileNotFoundError(f"Server script not found at: {cls.SERVER_SCRIPT}")

    @classmethod
    def _build_cmd(cls) -> list:
        """Return the subprocess command list."""
        return [cls.PYTHON_EXEC, cls.SERVER_SCRIPT]

    @classmethod
    def start_service(cls) -> None:
        if cls.is_running():
            return
        cls._check_paths()
        _subprocess_logger.info(f"Starting {cls.SERVICE_NAME} Service from {cls.WORK_DIR}...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env["PYTHONPATH"] = f"{cls.WORK_DIR}:{env.get('PYTHONPATH', '')}"
        log_name = cls.SERVICE_NAME.lower().replace(" ", "_").replace("-", "_")
        cls._process = subprocess.Popen(
            cls._build_cmd(),
            cwd=cls.WORK_DIR,
            env=env,
            stdout=open(os.path.join(cls.WORK_DIR, f"{log_name}_stdout.log"), "w"),
            stderr=open(os.path.join(cls.WORK_DIR, f"{log_name}_stderr.log"), "w"),
        )
        for _ in range(cls.MAX_RETRIES):
            if cls.is_running():
                _subprocess_logger.info(f"{cls.SERVICE_NAME} Service is READY.")
                return
            time.sleep(1)
        cls.stop_service()
        raise RuntimeError(
            f"{cls.SERVICE_NAME} service failed to start. Check logs in {cls.WORK_DIR}/"
        )

    @classmethod
    def stop_service(cls) -> None:
        if cls._process:
            _subprocess_logger.info(f"Stopping {cls.SERVICE_NAME} Service...")
            cls._process.terminate()
            cls._process = None
