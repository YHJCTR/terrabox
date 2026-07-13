"""Trajectory-source implementation for API-Bank."""
from __future__ import annotations

import os
from typing import Iterable, Optional

from ...interfaces import Step, Trace
from .api_utils import _score_api_prediction, _steps_from_history, samples_from_history
from .files import (
    _default_data_dir,
    _iter_jsonl_files,
    _iter_rollout_rows,
    _prediction_key,
    _read_jsonl,
    load_predictions,
)

class APIBankTrajectorySource:
    """Read API-Bank gold conversations or prediction files as traces."""

    def __init__(
        self,
        data_dir_fn=_default_data_dir,
        prediction_path_fn: Optional[callable] = None,
        rollout_path_fn: Optional[callable] = None,
    ):
        self._data_dir = data_dir_fn
        self._predictions = prediction_path_fn
        self._rollouts = rollout_path_fn

    def traces(self, experiment: str) -> Iterable[Trace]:
        if self._rollouts:
            rollout_path = self._rollouts(experiment)
            if rollout_path and os.path.exists(rollout_path):
                for row in _iter_rollout_rows(rollout_path):
                    if row.get("kind") != "api_call":
                        continue
                    yield _trace_from_rollout_row(row)
                return

        data_dir = self._data_dir(experiment)
        predictions = load_predictions(self._predictions(experiment)) if self._predictions else {}
        for path in _iter_jsonl_files(data_dir):
            history = _read_jsonl(path)
            for sample in samples_from_history(path, history):
                pred = predictions.get(_prediction_key(sample["file"], sample["id"]), {})
                pred_text = str(pred.get("pred") or "")
                success = False
                if sample["kind"] == "api_call" and pred_text:
                    success, _, _ = _score_api_prediction(pred_text, sample["ground_truth"])
                elif sample["kind"] == "response":
                    success = bool(pred_text)
                query = ""
                for item in sample["chat_history"]:
                    if item.get("role") == "User":
                        query = str(item.get("text") or "")
                        break
                yield Trace(
                    task_id=sample["task_id"],
                    query=query,
                    steps=_steps_from_history(sample["chat_history"], pred_text),
                    success=success,
                    final_answer=pred_text,
                    raw=sample,
                )


def _trace_from_rollout_row(row: dict) -> Trace:
    steps = _steps_from_history(row.get("chat_history") or [], str(row.get("pred") or ""))
    extra_context = row.get("api_descriptions") or ""
    if extra_context:
        steps.insert(0, Step(role="system", text=f"API descriptions: {str(extra_context)[:1200]}"))
    query = ""
    for item in row.get("chat_history") or []:
        if item.get("role") == "User":
            query = str(item.get("text") or "")
            break
    return Trace(
        task_id=str(row.get("task_id") or f"{row.get('file')}::{row.get('id')}"),
        query=query,
        steps=steps,
        success=bool(row.get("success")),
        final_answer=str(row.get("pred") or ""),
        raw=row,
    )
