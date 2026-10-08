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
import re
import subprocess
import time

import requests

from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    find_reusable_managed_lease,
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
    GPU_MEMORY_UTILIZATION = os.environ.get("AGENT_LLM_GPU_MEMORY_UTILIZATION", "0.90")
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
        return cls._is_healthy_on_port(cls.PORT)

    @classmethod
    def _is_healthy_on_port(cls, port: int) -> bool:
        try:
            resp = requests.get(
                f"http://{cls.HOST}:{port}/health",
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
    def _container_cmd_arg(cls, key: str) -> str | None:
        """Return a docker command argument value from the managed container.

        This is deliberately narrow: it prevents reusing a healthy old vLLM
        container whose port matches but whose model/context settings do not.
        """
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Cmd}}", cls.CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        try:
            import json

            cmd = json.loads(result.stdout.strip() or "[]")
            idx = cmd.index(key)
            return str(cmd[idx + 1])
        except Exception:
            return None

    @classmethod
    def _running_container_matches_config(cls) -> bool:
        if not cls._container_is_running():
            return False
        checks = {
            "--tensor-parallel-size": cls.TENSOR_PARALLEL_SIZE,
            "--max-model-len": cls.MAX_MODEL_LEN,
            "--gpu-memory-utilization": cls.GPU_MEMORY_UTILIZATION,
        }
        for key, expected in checks.items():
            actual = cls._container_cmd_arg(key)
            if actual != str(expected):
                logger.info(
                    "Existing %s has %s=%s, expected %s; rebuilding.",
                    cls.CONTAINER_NAME,
                    key,
                    actual,
                    expected,
                )
                return False
        return True

    @classmethod
    def _get_container_logs(cls, tail: int = 50) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), cls.CONTAINER_NAME],
            capture_output=True, text=True,
        )
        return (result.stdout + result.stderr).strip()

    @classmethod
    def _wait_container_absent(cls, container_name: str, *, timeout_s: float = 20.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            result = subprocess.run(
                ["docker", "inspect", container_name],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return True
            time.sleep(0.5)
        return False

    @classmethod
    def _wait_existing_container_ready(cls, *, timeout_s: float = 900.0) -> bool:
        """Wait for an already-created compatible container to become healthy.

        Multiple rollout workers can call ``start_service`` at the same time.
        The first worker may have created the container while vLLM is still
        loading weights, so a second worker must wait rather than attempting a
        second ``docker run`` with the same name.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if cls.is_running():
                logger.info(
                    "Reusing existing agent LLM container %s on port %s after startup wait.",
                    cls.CONTAINER_NAME,
                    cls.PORT,
                )
                return True
            if not cls._container_is_running():
                return False
            time.sleep(2)
        return False

    @classmethod
    def _force_remove_container(cls, container_name: str, *, reason: str) -> None:
        if not container_name:
            return
        result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to force-remove %s during %s: %s",
                container_name,
                reason,
                (result.stderr or result.stdout).strip(),
            )
        cls._wait_container_absent(container_name)

    @staticmethod
    def _conflict_container_id(stderr: str) -> str | None:
        match = re.search(r'container "([0-9a-f]{12,64})"', stderr or "")
        return match.group(1) if match else None

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
            cls.CONTAINER_NAME = f"{cls.CONTAINER_BASE}-{cls.PORT}"

        if cls.is_running():
            if cls._running_container_matches_config():
                return
            cls.stop_service()

        if cls._container_is_running():
            # A concurrent worker may have created the same compatible
            # container and still be waiting for vLLM readiness. Never stop it
            # just because /health is not ready yet.
            if cls._running_container_matches_config() and cls._wait_existing_container_ready():
                return
            logger.info(
                "Container %s is running but incompatible or failed health wait; rebuilding it.",
                cls.CONTAINER_NAME,
            )
            cls.stop_service()

        adopted = None
        if os.environ.get("AGENT_LLM_DISABLE_REUSE", "0") != "1":
            adopted = find_reusable_managed_lease(
                service="agent-llm",
                image=cls.DOCKER_IMAGE,
                container_base=cls.CONTAINER_BASE,
                host=cls.HOST,
                internal_port=8000,
                required_model_path=cls.MODEL_PATH,
                required_cmd_args={
                    "--tensor-parallel-size": cls.TENSOR_PARALLEL_SIZE,
                    "--max-model-len": cls.MAX_MODEL_LEN,
                },
                health_check=cls._is_healthy_on_port,
            )
            if adopted is not None:
                cls._lease = adopted
                cls.CONTAINER_NAME = adopted.container_name
                cls.PORT = adopted.port
                logger.info("Adopted existing agent LLM container %s on port %s", adopted.container_name, adopted.port)
                return
        else:
            logger.info("Agent LLM container reuse disabled by AGENT_LLM_DISABLE_REUSE=1")

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
        cls._wait_container_absent(cls.CONTAINER_NAME)

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
            "--gpu-memory-utilization", cls.GPU_MEMORY_UTILIZATION,
            "--enforce-eager",
            # Required for LangChain/LangGraph tool calling (tool_choice="auto")
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        logger.info(f"Starting agent LLM container (GPU: {lease.gpu_devices}, port: {lease.port}, model: {cls.MODEL_PATH})...")
        record_service_event({"event": "start_requested", "service": "agent-llm", "lease": lease.__dict__})
        result = subprocess.run(cmd, capture_output=True, text=True)
        for retry_idx in range(2):
            if result.returncode == 0 or "Conflict" not in result.stderr or "container name" not in result.stderr:
                break
            # Another worker may have won the startup race. If the existing
            # container matches this request, adopt it and wait for readiness;
            # do not remove a live container owned by the sibling worker.
            if cls._running_container_matches_config():
                if cls._wait_existing_container_ready():
                    return
                logger.warning(
                    "Conflicting agent LLM container %s never became healthy; rebuilding.",
                    cls.CONTAINER_NAME,
                )
            logger.warning(
                "Container name conflict for %s; forcing removal and retrying (%s/2).",
                cls.CONTAINER_NAME,
                retry_idx + 1,
            )
            conflict_id = cls._conflict_container_id(result.stderr)
            cls._force_remove_container(cls.CONTAINER_NAME, reason="docker_run_name_conflict")
            if conflict_id and conflict_id != cls.CONTAINER_NAME:
                cls._force_remove_container(conflict_id, reason="docker_run_conflict_id")
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if cls._container_is_running():
                logger.warning(
                    "docker run for %s returned non-zero, but the target container is running; adopting it and waiting for health. stderr=%s",
                    cls.CONTAINER_NAME,
                    result.stderr.strip(),
                )
            else:
                subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
                record_service_event({"event": "start_failed", "service": "agent-llm", "stderr": result.stderr, "lease": lease.__dict__})
                raise RuntimeError(
                    f"Failed to start agent LLM container.\nstderr: {result.stderr}"
                )

        if result.returncode != 0:
            record_service_event({"event": "start_adopted_after_nonzero", "service": "agent-llm", "stderr": result.stderr, "lease": lease.__dict__})
        else:
            record_service_event({"event": "started", "service": "agent-llm", "lease": lease.__dict__})

        if result.returncode != 0 and not cls._container_is_running():
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
        try:
            subprocess.run(
                ["docker", "stop", "-t", "2", cls.CONTAINER_NAME],
                capture_output=True, timeout=6,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timed out stopping %s; forcing removal.", cls.CONTAINER_NAME)
        subprocess.run(
            ["docker", "rm", "-f", cls.CONTAINER_NAME],
            capture_output=True, timeout=5,
        )
        record_service_event({"event": "stopped", "service": "agent-llm", "container": cls.CONTAINER_NAME})


agent_llm_manager = AgentLLMDockerManager()
