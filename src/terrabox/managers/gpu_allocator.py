"""
GPU Dynamic Allocator
=====================
Queries available GPU memory via nvidia-smi and selects the GPU(s) with the
most free memory at container start time.

Usage:
    from terrabox.managers.gpu_allocator import allocate_gpu, allocate_gpus

    gpu_id  = allocate_gpu()           # Single GPU → e.g. "2"
    gpu_ids = allocate_gpus(count=2)   # Multi-GPU  → e.g. "0,3"

Environment variable override:
    If the caller passes env_var="SAM2_GPU_DEVICES" and that variable is set,
    the function returns it directly without querying nvidia-smi.
    This lets operators pin GPUs explicitly without touching code.
"""

import subprocess
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_FREE_MIB = 4096


def _query_gpu_free_memory() -> List[dict]:
    """
    Query all GPUs via nvidia-smi.

    Returns list of {"id": "0", "free_mib": 20000} dicts,
    sorted by free_mib descending (most free memory first).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Is the NVIDIA driver installed?")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out after 10 seconds.")

    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed (returncode={result.returncode}): {result.stderr.strip()}"
        )

    gpus = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpus.append({"id": parts[0], "free_mib": int(parts[1])})
        except ValueError:
            logger.warning(f"Unexpected nvidia-smi output line: {line!r}")

    gpus.sort(key=lambda g: g["free_mib"], reverse=True)
    return gpus


def allocate_gpu(
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
    fallback: str = "0",
    env_var: Optional[str] = None,
    strict: bool = False,
) -> str:
    """
    Return the ID of the GPU with the most free memory.

    Args:
        min_free_mib: Minimum free memory (MiB) required. Default: 4096.
        fallback:     GPU ID to use if dynamic allocation fails. Default: "0".
        env_var:      Environment variable name to check first (e.g. "SAM2_GPU_DEVICES").
                      If the variable is set and non-empty, its value is returned as-is.

    Returns:
        A single GPU ID string, e.g. "2".
    """
    if env_var:
        val = os.environ.get(env_var)
        if val is not None and val.strip():
            logger.info(f"GPU override via ${env_var}={val!r}, skipping dynamic allocation.")
            return val.strip()

    try:
        gpus = _query_gpu_free_memory()
    except RuntimeError as e:
        if strict:
            raise
        logger.warning(f"GPU query failed: {e}. Falling back to GPU {fallback!r}.")
        return fallback

    if not gpus:
        if strict:
            raise RuntimeError("No GPUs found.")
        logger.warning(f"No GPUs found. Falling back to GPU {fallback!r}.")
        return fallback

    eligible = [g for g in gpus if g["free_mib"] >= min_free_mib]
    if not eligible:
        if strict:
            raise RuntimeError(
                f"No GPU has >= {min_free_mib} MiB free. "
                f"Best: GPU {gpus[0]['id']} with {gpus[0]['free_mib']} MiB free."
            )
        logger.warning(
            f"No GPU meets min_free_mib={min_free_mib} MiB. "
            f"Best: GPU {gpus[0]['id']} with {gpus[0]['free_mib']} MiB free. "
            f"Falling back to GPU {fallback!r}."
        )
        return fallback

    chosen = eligible[0]
    logger.info(
        f"Dynamic GPU allocation: GPU {chosen['id']} ({chosen['free_mib']} MiB free)."
    )
    return chosen["id"]


def allocate_gpus(
    count: int = 2,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
    fallback: str = "0,1",
    env_var: Optional[str] = None,
    strict: bool = False,
) -> str:
    """
    Return comma-separated IDs of the `count` GPUs with the most free memory.

    Args:
        count:        Number of GPUs to allocate. Default: 2.
        min_free_mib: Minimum free memory (MiB) per GPU. Default: 4096.
        fallback:     Comma-separated GPU IDs to use if allocation fails.
        env_var:      Environment variable name to check first (e.g. "VLM_GPU_DEVICES").

    Returns:
        Comma-separated GPU ID string, e.g. "0,3".
    """
    if env_var:
        val = os.environ.get(env_var)
        if val is not None and val.strip():
            logger.info(f"GPU override via ${env_var}={val!r}, skipping dynamic allocation.")
            return val.strip()

    try:
        gpus = _query_gpu_free_memory()
    except RuntimeError as e:
        if strict:
            raise
        logger.warning(f"GPU query failed: {e}. Falling back to GPUs {fallback!r}.")
        return fallback

    eligible = [g for g in gpus if g["free_mib"] >= min_free_mib]

    if len(eligible) < count:
        if strict:
            best = gpus[0]["free_mib"] if gpus else "none"
            raise RuntimeError(
                f"Not enough GPUs with >= {min_free_mib} MiB free "
                f"(need {count}, eligible {len(eligible)} / {len(gpus)} total, best={best})."
            )
        logger.warning(
            f"Not enough GPUs with >= {min_free_mib} MiB free "
            f"(need {count}, eligible {len(eligible)} / {len(gpus)} total). "
            f"Falling back to GPUs {fallback!r}."
        )
        return fallback

    chosen = eligible[:count]
    ids = ",".join(g["id"] for g in chosen)
    logger.info(
        f"Dynamic GPU allocation: GPUs {ids} "
        f"(free: {[g['free_mib'] for g in chosen]} MiB)."
    )
    return ids


def has_free_gpu(min_free_mib: int) -> bool:
    """Return True if at least one GPU has >= min_free_mib MiB of free memory."""
    try:
        return any(g["free_mib"] >= min_free_mib for g in _query_gpu_free_memory())
    except Exception:
        return True  # Cannot query — assume OK; allocate_gpu fallback will handle it


def has_free_gpus(count: int, min_free_mib: int) -> bool:
    """Return True if at least `count` GPUs each have >= min_free_mib MiB of free memory."""
    try:
        eligible = [g for g in _query_gpu_free_memory() if g["free_mib"] >= min_free_mib]
        return len(eligible) >= count
    except Exception:
        return True
