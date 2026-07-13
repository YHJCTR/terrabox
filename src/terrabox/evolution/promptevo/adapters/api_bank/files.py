"""File and JSONL helpers for API-Bank adapter components."""
from __future__ import annotations

import glob
import json
import os
from typing import Iterable


API_BANK_ADAPTER_DIR = os.path.dirname(__file__)
DEFAULT_API_BANK_EXPERIMENTS_DIR = os.path.join(API_BANK_ADAPTER_DIR, "experiments")


def _default_data_dir(experiment: str) -> str:
    return experiment


def _default_prediction_path(experiment: str) -> str:
    return experiment


def api_bank_prediction_path(output_dir: str, experiment: str) -> str:
    return os.path.join(output_dir, experiment, "predictions.jsonl")


def api_bank_rollout_path(output_dir: str, experiment: str) -> str:
    return os.path.join(output_dir, experiment, "rollout.jsonl")


def _iter_jsonl_files(data_dir: str) -> Iterable[str]:
    for path in glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True):
        if os.path.isfile(path):
            yield path


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _write_jsonl(path: str, rows: Iterable[dict]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return os.path.abspath(path)


def _sample_file_id(path: str) -> str:
    return os.path.basename(path)


def _sample_task_id(path: str, sample_id: int) -> str:
    return f"{_sample_file_id(path)}::{sample_id}"

def _prediction_key(file_name: str, sample_id: int) -> tuple[str, int]:
    return (file_name, int(sample_id))


def load_predictions(path: str) -> dict[tuple[str, int], dict]:
    if not path:
        return {}
    predictions: dict[tuple[str, int], dict] = {}
    files = [path]
    if os.path.isdir(path):
        files = list(_iter_jsonl_files(path))
    for pred_path in files:
        for obj in _read_jsonl(pred_path):
            if "file" not in obj or "id" not in obj:
                continue
            predictions[_prediction_key(str(obj["file"]), int(obj["id"]))] = obj
    return predictions


def _iter_rollout_rows(path: str) -> Iterable[dict]:
    if not path:
        return []
    files = [path]
    if os.path.isdir(path):
        files = [
            os.path.join(path, name)
            for name in ("rollout.jsonl", "trajectories.jsonl", "trajectories_full.jsonl")
            if os.path.exists(os.path.join(path, name))
        ]
        if not files:
            files = list(_iter_jsonl_files(path))
    rows = []
    for rollout_path in files:
        for obj in _read_jsonl(rollout_path):
            if obj.get("kind") in {"api_call", "response"} or "conversation_history" in obj:
                rows.append(obj)
    return rows
