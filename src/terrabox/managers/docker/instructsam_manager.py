"""
InstructSAM Service Manager - Docker Mode
==========================================
Launches terrabox/instructsam:latest via 'docker run'.
Imported by geo_perception.py when TERRABOX_USE_DOCKER=true.

Volume mounts:
  -v /data1:/data1                          image files (same path inside container)
  -v ${INSTRUCTSAM_MODELS_HOST}:/models:ro  model directory (SAM2 + CLIP, no Qwen)

Model directory layout (host side, default /data1/yuhongjie2/terra_model/instructsam):
  sam2_hiera_large.pt       SAM2 Hiera Large weights
  GeoRSCLIP-ViT-L-14.pt     GeoRSCLIP CLIP weights

Note: the counting step calls the host vLLM service via HTTP; Qwen is not downloaded
inside the container. If the VLM manager has selected a dynamic host port in this
process, that port is passed through VLLM_API_URL.

Build the image first:
  cd docker/instructsam && docker build -t terrabox/instructsam:latest .

Download models first:
  python scripts/download_instructsam_models.py --skip-qwen
"""

import subprocess
import time
import os
import requests
import logging
from ..base_manager import BaseServiceManager
from ..resource_allocator import (
    acquire_docker_lease,
    find_reusable_managed_lease,
    labels_for_lease,
    record_service_event,
    remove_container_if_exists,
)

logger = logging.getLogger("docker.instructsam_manager")


class InstructSAMDockerManager(BaseServiceManager):
    _instance = None

    API_URL = "http://127.0.0.1:" + os.environ.get("INSTRUCTSAM_PORT", "9006")

    CONTAINER_NAME = "terrabox-instructsam"
    CONTAINER_BASE = "terrabox-instructsam"
    DOCKER_IMAGE   = "terrabox/instructsam:latest"
    GPU_DEVICES    = os.environ.get("INSTRUCTSAM_GPU_DEVICES", "0")

    # host-side model directory, mounted as /models inside the container
    MODELS_HOST     = os.environ.get("INSTRUCTSAM_MODELS_HOST", "")
    DATA_MOUNT_HOST = os.environ.get("DATA_MOUNT_HOST", "/data1")
    _lease = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        return cls._is_healthy_on_port(int(cls.API_URL.rsplit(":", 1)[-1]))

    @classmethod
    def _is_healthy_on_port(cls, port: int):
        try:
            return requests.get(
                f"http://127.0.0.1:{int(port)}/health",
                timeout=1,
                proxies={"http": None, "https": None},
            ).status_code == 200
        except Exception:
            return False

    @classmethod
    def _container_is_running(cls):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", cls.CONTAINER_NAME],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def _apply_env_overrides(cls):
        if os.environ.get("INSTRUCTSAM_PORT"):
            cls.API_URL = f"http://127.0.0.1:{int(os.environ['INSTRUCTSAM_PORT'])}"
        if os.environ.get("INSTRUCTSAM_GPU_DEVICES"):
            cls.GPU_DEVICES = os.environ["INSTRUCTSAM_GPU_DEVICES"]
        if os.environ.get("INSTRUCTSAM_MODELS_HOST"):
            cls.MODELS_HOST = os.environ["INSTRUCTSAM_MODELS_HOST"]
        if os.environ.get("DATA_MOUNT_HOST"):
            cls.DATA_MOUNT_HOST = os.environ["DATA_MOUNT_HOST"]

    @classmethod
    def _docker_timeout_seconds(cls) -> int:
        try:
            return max(5, int(os.environ.get("TERRABOX_DOCKER_START_TIMEOUT_SECONDS", "90")))
        except ValueError:
            return 90

    @classmethod
    def _startup_timeout_seconds(cls) -> int:
        """Return the health-check budget for loading SAM2 and GeoRSCLIP."""
        try:
            return max(30, int(os.environ.get("TERRABOX_INSTRUCTSAM_STARTUP_TIMEOUT_SECONDS", "420")))
        except ValueError:
            return 420

    @classmethod
    def _record_startup_diagnostics(cls, *, reason: str) -> None:
        """Keep bounded Docker diagnostics so startup failures are actionable."""
        try:
            inspect = subprocess.run(
                ["docker", "inspect", cls.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=10,
            )
            inspect_text = inspect.stdout if inspect.returncode == 0 else inspect.stderr
            inspect_rc = inspect.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            inspect_text = f"diagnostic inspect failed: {exc}"
            inspect_rc = -1
        try:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "120", cls.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=20,
            )
            logs_text = logs.stdout + logs.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            logs_text = f"diagnostic logs failed: {exc}"
        record_service_event({
            "event": "startup_diagnostics",
            "service": "instructsam",
            "container": cls.CONTAINER_NAME,
            "reason": reason,
            "inspect": inspect_text[-12000:],
            "logs": logs_text[-12000:],
        })
        logger.error(
            "InstructSAM startup diagnostics (%s): inspect_rc=%s logs_tail=%s",
            reason,
            inspect_rc,
            logs_text[-2000:].replace("\n", " | "),
        )

    @classmethod
    def _remove_container_bounded(cls, *, reason: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", cls.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timed out removing InstructSAM container %s (%s).", cls.CONTAINER_NAME, reason)
            record_service_event({
                "event": "cleanup_timeout",
                "service": "instructsam",
                "container": cls.CONTAINER_NAME,
                "reason": reason,
            })

    @classmethod
    def _recover_created_container(cls, *, timeout_seconds: int) -> bool:
        """Start a container that Docker created after its client timed out."""
        try:
            inspected = subprocess.run(
                ["docker", "inspect", cls.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if inspected.returncode != 0:
                return False
            started = subprocess.run(
                ["docker", "start", cls.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if started.returncode != 0:
                return False
        except subprocess.TimeoutExpired:
            return False
        record_service_event({
            "event": "start_recovered_after_client_timeout",
            "service": "instructsam",
            "container": cls.CONTAINER_NAME,
        })
        logger.warning("Recovered InstructSAM container %s after Docker client timeout.", cls.CONTAINER_NAME)
        return True

    @classmethod
    def _start_docker(cls):
        if cls._container_is_running():
            logger.info(f"Container {cls.CONTAINER_NAME} is running but service is not healthy; rebuilding it.")
            cls.stop_service()

        base_port = int(cls.API_URL.rsplit(":", 1)[-1])
        lease = acquire_docker_lease(
            service="instructsam",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            base_port=base_port,
            internal_port=9006,
            gpu_count=1,
            min_free_mib=4096,
            fallback_gpu_devices=cls.GPU_DEVICES,
            gpu_env_var="INSTRUCTSAM_GPU_DEVICES",
            exclude_manager_cls=cls,
        )
        cls._lease = lease
        cls.CONTAINER_NAME = lease.container_name
        cls.API_URL = lease.api_url
        remove_container_if_exists(cls.CONTAINER_NAME, service="instructsam", reason="before_docker_run")
        vllm_api_url = os.environ.get("VLLM_API_URL")
        if not vllm_api_url:
            try:
                from .vllm_manager import VLLMDockerManager
                vllm_api_url = f"http://host.docker.internal:{VLLMDockerManager.PORT}"
            except Exception:
                vllm_api_url = "http://host.docker.internal:9000"

        pytorch_cuda_alloc_conf = os.environ.get(
            "INSTRUCTSAM_PYTORCH_CUDA_ALLOC_CONF",
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            *labels_for_lease(lease),
            "--gpus", f"device={lease.gpu_devices}",
            "-p", f"{lease.port}:{lease.internal_port}",
            # Allow the container to reach the host vLLM service (port 9000)
            "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{cls.DATA_MOUNT_HOST}:{cls.DATA_MOUNT_HOST}",
            "-v", f"{cls.MODELS_HOST}:/models:ro",
            "-e", "DEVICE=cuda:0",  # Use first GPU visible to container
            "-e", f"VLLM_API_URL={vllm_api_url}",
            "-e", f"PYTORCH_CUDA_ALLOC_CONF={pytorch_cuda_alloc_conf}",
            cls.DOCKER_IMAGE,
        ]

        logger.info(f"Starting InstructSAM container (GPU: {lease.gpu_devices}, port: {lease.port})...")
        record_service_event({"event": "start_requested", "service": "instructsam", "lease": lease.__dict__})
        timeout_seconds = cls._docker_timeout_seconds()
        # On this host `docker run -d` can create the container yet leave the
        # client blocked indefinitely. Split the equivalent lifecycle so the
        # creation and start boundaries are independently bounded.
        create_cmd = ["docker", "create", *cmd[3:]]
        stage = "create"
        try:
            result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if result.returncode == 0:
                stage = "start"
                result = subprocess.run(
                    ["docker", "start", cls.CONTAINER_NAME],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
        except subprocess.TimeoutExpired as exc:
            # Container creation/start should complete before model loading.
            # A blocked Docker client is infrastructure failure, not a reason
            # to consume the tool-call timeout for the whole task.
            if cls._recover_created_container(timeout_seconds=timeout_seconds):
                return
            cls._remove_container_bounded(reason="start_timeout")
            record_service_event({
                "event": "start_timeout",
                "service": "instructsam",
                "container": cls.CONTAINER_NAME,
                "stage": stage,
                "timeout_seconds": timeout_seconds,
                "lease": lease.__dict__,
            })
            raise TimeoutError(
                f"docker {stage} for {cls.CONTAINER_NAME} did not return within {timeout_seconds}s"
            ) from exc
        if result.returncode != 0:
            cls._remove_container_bounded(reason="start_failed")
            record_service_event({"event": "start_failed", "service": "instructsam", "stderr": result.stderr, "lease": lease.__dict__})
            raise RuntimeError(
                f"Failed to start InstructSAM container.\nstderr: {result.stderr}"
            )

    @classmethod
    def start_service(cls):
        try:
            from ...agent.config import load_raw_yaml
            d = load_raw_yaml()
            if "instructsam_models_host" in d: cls.MODELS_HOST     = str(d["instructsam_models_host"])
            if "instructsam_gpu_devices" in d: cls.GPU_DEVICES     = str(d["instructsam_gpu_devices"])
            if "docker_data_mount_host"  in d: cls.DATA_MOUNT_HOST = str(d["docker_data_mount_host"])
            if "instructsam_port"        in d: cls.API_URL         = f"http://127.0.0.1:{int(d['instructsam_port'])}"
        except Exception:
            pass
        cls._apply_env_overrides()

        if cls.is_running():
            return

        adopted = find_reusable_managed_lease(
            service="instructsam",
            image=cls.DOCKER_IMAGE,
            container_base=cls.CONTAINER_BASE,
            host="127.0.0.1",
            internal_port=9006,
            health_check=cls._is_healthy_on_port,
        )
        if adopted is not None:
            cls._lease = adopted
            cls.CONTAINER_NAME = adopted.container_name
            cls.API_URL = adopted.api_url
            logger.info("Adopted healthy InstructSAM container %s.", adopted.container_name)
            return

        cls._start_docker()

        startup_timeout_seconds = cls._startup_timeout_seconds()
        # The first SAM2/GeoRSCLIP load can take several minutes on a cold host.
        logger.info(
            "Waiting for InstructSAM service (SAM2 + CLIP cold start; timeout=%ss)...",
            startup_timeout_seconds,
        )
        deadline = time.monotonic() + startup_timeout_seconds
        poll_count = 0
        while time.monotonic() < deadline:
            if cls.is_running():
                logger.info("InstructSAM service is READY.")
                record_service_event({"event": "ready", "service": "instructsam", "lease": cls._lease.__dict__ if cls._lease else None})
                return
            poll_count += 1
            if poll_count % 15 == 0:
                elapsed = startup_timeout_seconds - max(0, int(deadline - time.monotonic()))
                logger.info("Still loading... (%ss elapsed)", elapsed)
            time.sleep(2)

        cls._record_startup_diagnostics(reason="health_timeout")
        cls.stop_service()
        raise RuntimeError(
            "InstructSAM service failed to start. "
            f"Health did not become ready within {startup_timeout_seconds}s; "
            "startup diagnostics were recorded."
        )

    @classmethod
    def stop_service(cls):
        logger.info(f"Stopping container {cls.CONTAINER_NAME}...")
        try:
            subprocess.run(
                ["docker", "stop", cls.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timed out stopping InstructSAM container %s; forcing removal.", cls.CONTAINER_NAME)
        cls._remove_container_bounded(reason="stop_service")
        record_service_event({"event": "stopped", "service": "instructsam", "container": cls.CONTAINER_NAME})


instructsam_manager = InstructSAMDockerManager()
