"""Concise GPU status snapshots for human approval UI."""
from __future__ import annotations

import subprocess
from typing import Any


def _parse_nvidia_smi_gpu_status(output: str) -> list[dict[str, int | str]]:
    gpus: list[dict[str, int | str]] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(
                {
                    "id": parts[0],
                    "free_mib": int(parts[1]),
                    "utilization_pct": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return gpus


def get_gpu_status_snapshot() -> dict[str, Any]:
    """Return a compact per-GPU free-memory/utilization snapshot."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip(), "gpus": []}
    return {"available": True, "gpus": _parse_nvidia_smi_gpu_status(result.stdout)}
