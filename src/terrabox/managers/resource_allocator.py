"""Runtime resource allocation for Docker-backed model services."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
from collections.abc import Callable

from .gpu_allocator import allocate_gpu, allocate_gpus

logger = logging.getLogger(__name__)

AGENT_SERVICE_NAMES = {"agent-llm"}


@dataclass(frozen=True)
class ResourceLease:
    service: str
    container_name: str
    image: str
    host: str
    port: int
    internal_port: int
    gpu_devices: str
    created_at: str
    reused: bool = False

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _boolish(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_gpu_devices(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def tool_gpu_override_for_service(service: str, gpu_count: int) -> str | None:
    """Return experiment-scoped tool GPUs, or None when normal allocation applies.

    The override is intentionally opt-in via TERRABOX_TOOL_GPU_DEVICES so normal
    application/tool behavior is unchanged outside long-running experiments.
    """
    if service in AGENT_SERVICE_NAMES:
        return None

    pool = _split_gpu_devices(os.environ.get("TERRABOX_TOOL_GPU_DEVICES", ""))
    if not pool:
        return None

    max_gpus_raw = os.environ.get("TERRABOX_TOOL_MAX_GPUS", "").strip()
    if max_gpus_raw:
        max_gpus = int(max_gpus_raw)
        if int(gpu_count) > max_gpus:
            raise RuntimeError(
                f"{service} requires {gpu_count} GPU(s), but TERRABOX_TOOL_MAX_GPUS={max_gpus}."
            )

    if int(gpu_count) > len(pool):
        raise RuntimeError(
            f"{service} requires {gpu_count} GPU(s), but TERRABOX_TOOL_GPU_DEVICES "
            f"only provides {len(pool)} device(s): {','.join(pool)}."
        )

    return ",".join(pool[: int(gpu_count)])


def current_run_id() -> str:
    return os.environ.get("TERRABOX_RUN_ID", f"pid-{os.getpid()}")


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) != 0


def allocate_port(
    base_port: int,
    *,
    host: str = "127.0.0.1",
    max_tries: int = 100,
    pin: bool | None = None,
) -> int:
    """Return a free host port, starting at base_port."""
    strict_pin = _boolish(os.environ.get("TERRABOX_PIN_SERVICE_PORTS"), False) if pin is None else pin
    if strict_pin:
        if not is_port_available(base_port, host):
            raise RuntimeError(f"Requested pinned port {base_port} is already in use.")
        return int(base_port)

    for port in range(int(base_port), int(base_port) + int(max_tries)):
        if is_port_available(port, host):
            if port != int(base_port):
                logger.info("Port %s is busy; using %s instead.", base_port, port)
            return port
    raise RuntimeError(f"No free port found from {base_port} to {int(base_port) + int(max_tries) - 1}.")


def docker_image_exists(image: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    return result.returncode == 0


def container_exists(container_name: str) -> bool:
    result = subprocess.run(["docker", "inspect", container_name], capture_output=True, text=True)
    return result.returncode == 0


def container_is_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def remove_container_if_exists(container_name: str, *, service: str | None = None, reason: str = "name_reuse") -> bool:
    """Remove a stale container before docker run reuses the same dynamic name."""
    if not container_exists(container_name):
        return False
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
    record_service_event({
        "event": "stale_container_removed",
        "service": service,
        "container": container_name,
        "reason": reason,
    })
    return True


def _cmd_arg_value(cmd: list[str], key: str) -> str | None:
    try:
        idx = cmd.index(key)
        return str(cmd[idx + 1])
    except Exception:
        return None


def _container_mounts_path(container: dict, *, source: str, destination: str) -> bool:
    for mount in container.get("Mounts") or []:
        if mount.get("Destination") == destination and os.path.abspath(str(mount.get("Source", ""))) == os.path.abspath(source):
            return True
    return False


def _candidate_container_names(*, service: str, container_base: str) -> list[str]:
    names: list[str] = []
    queries = [
        [
            "docker", "ps",
            "--filter", "label=terrabox.managed=true",
            "--filter", f"label=terrabox.service={service}",
            "--format", "{{.Names}}",
        ],
        [
            "docker", "ps",
            "--filter", f"name={container_base}",
            "--format", "{{.Names}}",
        ],
    ]
    for query in queries:
        result = subprocess.run(query, capture_output=True, text=True)
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            name = line.strip()
            if name and name not in names:
                names.append(name)
    return names


def _container_host_port(container: dict, internal_port: int, labels: dict) -> int | None:
    try:
        return int(labels.get("terrabox.port"))
    except Exception:
        pass
    bindings = (
        container.get("NetworkSettings", {})
        .get("Ports", {})
        .get(f"{int(internal_port)}/tcp")
    )
    if bindings:
        try:
            return int(bindings[0]["HostPort"])
        except Exception:
            return None
    return None


def _container_gpu_devices(container: dict, labels: dict) -> str:
    if labels.get("terrabox.gpu_devices"):
        return str(labels["terrabox.gpu_devices"])
    for item in container.get("Config", {}).get("Env") or []:
        if str(item).startswith("CUDA_VISIBLE_DEVICES="):
            return str(item).split("=", 1)[1]
    return ""


def find_reusable_managed_lease(
    *,
    service: str,
    image: str,
    container_base: str,
    host: str,
    internal_port: int,
    required_model_path: str | None = None,
    required_cmd_args: dict[str, str] | None = None,
    health_check: Callable[[int], bool] | None = None,
) -> ResourceLease | None:
    """Return a lease for an already-running compatible managed container."""
    names = _candidate_container_names(service=service, container_base=container_base)
    for name in names:
        if container_base and not name.startswith(container_base):
            continue
        inspected = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
        if inspected.returncode != 0:
            continue
        try:
            containers = json.loads(inspected.stdout)
            container = containers[0] if isinstance(containers, list) and containers else {}
        except Exception:
            continue
        if not container.get("State", {}).get("Running"):
            continue

        config = container.get("Config") or {}
        if image and config.get("Image") != image:
            continue
        if required_model_path and not _container_mounts_path(container, source=required_model_path, destination="/model"):
            continue

        cmd = config.get("Cmd") or []
        if required_cmd_args:
            if any(_cmd_arg_value(cmd, key) != str(value) for key, value in required_cmd_args.items()):
                continue

        labels = config.get("Labels") or {}
        port = _container_host_port(container, internal_port, labels)
        if port is None:
            continue
        if health_check and not health_check(port):
            continue

        gpu_devices = _container_gpu_devices(container, labels)
        lease = ResourceLease(
            service=service,
            container_name=name,
            image=str(config.get("Image") or image),
            host=host,
            port=port,
            internal_port=int(internal_port),
            gpu_devices=gpu_devices,
            created_at=datetime.now().isoformat(timespec="seconds"),
            reused=True,
        )
        record_service_event({"event": "service_adopted", "service": service, "lease": lease.__dict__})
        return lease
    return None


def labels_for_lease(lease: ResourceLease) -> list[str]:
    role = "agent" if lease.service in AGENT_SERVICE_NAMES else "tool"
    labels = [
        "--label", "terrabox.managed=true",
        "--label", f"terrabox.service={lease.service}",
        "--label", f"terrabox.role={role}",
        "--label", f"terrabox.run_id={current_run_id()}",
        "--label", f"terrabox.port={lease.port}",
        "--label", f"terrabox.gpu_devices={lease.gpu_devices}",
    ]
    if role == "tool":
        scope = os.environ.get("TERRABOX_TOOL_SERVICE_SCOPE", "").strip()
        pool = os.environ.get("TERRABOX_TOOL_GPU_DEVICES", "").strip()
        if scope:
            labels.extend(["--label", f"terrabox.service_scope={scope}"])
        if pool:
            labels.extend(["--label", f"terrabox.tool_gpu_pool={pool}"])
    return labels


def lease_to_dict(lease: ResourceLease) -> dict:
    return asdict(lease)


def record_service_event(event: dict) -> None:
    payload = {"time": datetime.now().isoformat(timespec="seconds"), **event}
    events_path = os.environ.get("TERRABOX_SERVICE_EVENTS_PATH")
    if events_path:
        out = Path(events_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    manifest_path = os.environ.get("TERRABOX_RUN_MANIFEST_PATH")
    if manifest_path:
        _update_run_manifest(Path(manifest_path), payload)


def _update_run_manifest(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("run_id", current_run_id())
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("service_events", []).append(event)
    services = manifest.setdefault("services", {})

    lease = event.get("lease")
    if isinstance(lease, dict) and lease.get("container_name"):
        key = lease["container_name"]
        service_state = services.setdefault(key, {})
        service_state.update({
            "service": lease.get("service"),
            "container_name": lease.get("container_name"),
            "image": lease.get("image"),
            "host": lease.get("host"),
            "port": lease.get("port"),
            "internal_port": lease.get("internal_port"),
            "gpu_devices": lease.get("gpu_devices"),
            "reused": lease.get("reused", False),
        })
        service_state.setdefault("events", []).append(event.get("event"))
        service_state["last_event"] = event.get("event")
        service_state["last_seen_at"] = event["time"]
    elif event.get("container"):
        key = str(event["container"])
        service_state = services.setdefault(key, {"container_name": key})
        service_state.setdefault("events", []).append(event.get("event"))
        service_state["last_event"] = event.get("event")
        service_state["last_seen_at"] = event["time"]

    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def cleanup_managed_containers(
    *,
    run_id: str | None = None,
    service: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Stop and remove managed Docker containers selected by label."""
    cmd = ["docker", "ps", "-a", "--filter", "label=terrabox.managed=true"]
    if run_id:
        cmd += ["--filter", f"label=terrabox.run_id={run_id}"]
    if service:
        cmd += ["--filter", f"label=terrabox.service={service}"]
    cmd += ["--format", "{{.Names}}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docker ps failed")
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if dry_run:
        return names
    for name in names:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        record_service_event({"event": "cleanup_removed", "container": name})
    return names


def acquire_docker_lease(
    *,
    service: str,
    image: str,
    container_base: str,
    host: str,
    base_port: int,
    internal_port: int,
    gpu_count: int,
    min_free_mib: int,
    fallback_gpu_devices: str,
    gpu_env_var: str | None = None,
    port_max_tries: int = 100,
    exclude_manager_cls: type | None = None,
) -> ResourceLease:
    """Allocate a host port, GPU(s), and container name for a Docker service."""
    if not docker_image_exists(image):
        raise RuntimeError(f"Docker image {image!r} was not found. Build or pull it before starting {service}.")

    port = allocate_port(base_port, host=host, max_tries=port_max_tries)
    gpu_devices = _allocate_gpu_devices(
        service=service,
        gpu_count=gpu_count,
        min_free_mib=min_free_mib,
        fallback_gpu_devices=fallback_gpu_devices,
        gpu_env_var=gpu_env_var,
    )
    if gpu_devices is None:
        try:
            from .base_manager import ServiceRegistry
            ServiceRegistry.evict_lru_for_gpu(
                min_free_mib=min_free_mib,
                exclude_cls=exclude_manager_cls,
                count=max(1, gpu_count),
            )
        except Exception as exc:
            logger.info("LRU service eviction did not free GPU for %s: %s", service, exc)
        gpu_devices = _allocate_gpu_devices(
            service=service,
            gpu_count=gpu_count,
            min_free_mib=min_free_mib,
            fallback_gpu_devices=fallback_gpu_devices,
            gpu_env_var=gpu_env_var,
            raise_on_failure=True,
        )

    return ResourceLease(
        service=service,
        container_name=f"{container_base}-{port}",
        image=image,
        host=host,
        port=int(port),
        internal_port=int(internal_port),
        gpu_devices=gpu_devices,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def _allocate_gpu_devices(
    *,
    service: str,
    gpu_count: int,
    min_free_mib: int,
    fallback_gpu_devices: str,
    gpu_env_var: str | None,
    raise_on_failure: bool = False,
) -> str | None:
    tool_override = tool_gpu_override_for_service(service, gpu_count)
    if tool_override:
        return tool_override

    if gpu_count <= 1:
        allocator = allocate_gpu
        kwargs = {}
    else:
        allocator = allocate_gpus
        kwargs = {"count": gpu_count}
    try:
        return allocator(
            **kwargs,
            min_free_mib=min_free_mib,
            fallback=fallback_gpu_devices,
            env_var=gpu_env_var,
            strict=True,
        )
    except RuntimeError:
        if raise_on_failure:
            raise
        return None
