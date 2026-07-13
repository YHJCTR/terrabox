"""File helpers for the tau2-bench adapter."""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


TAU2_ADAPTER_DIR = os.path.dirname(__file__)
DEFAULT_TAU2_ROOT = "/data1/yuhongjie2/tau2-bench"
DEFAULT_TAU2_EXPERIMENTS_DIR = os.path.join(TAU2_ADAPTER_DIR, "experiments")
DEFAULT_TAU2_VERSIONS_DIR = "evolution_store/promptevo/tau2_bench/versions"
DEFAULT_TAU2_RUNTIME_DIR = str(Path(__file__).resolve().parents[6] / "tmp" / "tau2_runtime")


def _read_json(path: str | os.PathLike[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _write_json(path: str | os.PathLike[str], obj: dict) -> str:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def _iter_result_paths(path: str) -> Iterable[str]:
    if os.path.isfile(path) and path.endswith(".json"):
        yield path
        return
    if os.path.isdir(path):
        direct = os.path.join(path, "results.json")
        if os.path.exists(direct):
            yield direct
            return
        for result_path in glob.glob(os.path.join(path, "**", "results.json"), recursive=True):
            if os.path.isfile(result_path):
                yield result_path
        for result_path in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
            if os.path.isfile(result_path) and os.path.basename(result_path) != "results.json":
                yield result_path


def _load_results_like(path: str) -> tuple[dict, list[dict]]:
    """Return (metadata, simulations) for tau2 Results or submission JSON."""
    data = _read_json(path)
    if "simulations" in data or "simulation_index" in data:
        simulations = data.get("simulations") or []
        if not simulations:
            sim_dir = os.path.join(os.path.dirname(path), "simulations")
            for sim_path in sorted(glob.glob(os.path.join(sim_dir, "*.json"))):
                simulations.append(_read_json(sim_path))
        return data, simulations
    return data, []


def tau2_adapter_experiment_dir(name: str, output_dir: str = DEFAULT_TAU2_EXPERIMENTS_DIR) -> str:
    return os.path.abspath(os.path.join(output_dir, name))


def tau2_adapter_results_path(name_or_path: str, output_dir: str = DEFAULT_TAU2_EXPERIMENTS_DIR) -> str:
    """Resolve an experiment name to the tau2 results copied by the adapter runner.

    Existing absolute/relative paths are returned as-is. Otherwise, the
    convention is:

    adapters/tau2_bench/experiments/<experiment>/tau2_results/
    """

    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    adapter_dir = tau2_adapter_experiment_dir(name_or_path, output_dir)
    candidates = [
        os.path.join(adapter_dir, "tau2_results"),
        os.path.join(adapter_dir, "results.json"),
        adapter_dir,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def tau2_external_simulation_dir(data_dir: str, save_to: str) -> str:
    return os.path.abspath(os.path.join(data_dir, "simulations", save_to))


def copy_results_tree(src: str, dst: str) -> str:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return os.path.abspath(dst)


def relative_or_abs(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path
