"""Adapters from real token-experiment runs to evolution Trajectory records.

The token experiment keeps run-oriented JSON: per-mode token usage, tool traces,
raw observations, and sidecar manifests.  Evolution methods consume a smaller
native Trajectory format.  This module is the boundary between the two so the
experiment format can evolve without changing every self-evolution method.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .trajectory import Trajectory, Turn


SUCCESS_STATUSES = {"completed", "completed_with_recovery"}
SOURCE_INVALID_STATUSES = {"source_gold_invalid", "gold_replay_failed", "needs_review"}


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_observation_file(path: str | None, *, max_chars: int) -> str | None:
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars; full file: {path}>"
    return text


def _turn_from_dict(item: dict[str, Any], *, max_tool_result_chars: int) -> Turn:
    role = item.get("role", "")
    tool_result = item.get("tool_result")
    if tool_result is None:
        tool_result = _read_observation_file(item.get("tool_result_path"), max_chars=max_tool_result_chars)
    if tool_result is None and role == "tool":
        tool_result = item.get("observation_preview") or item.get("content", "")
    return Turn(
        role=role,
        content=item.get("content", ""),
        tool_name=item.get("tool_name"),
        tool_args=item.get("tool_args"),
        tool_result=tool_result,
        is_error=bool(item.get("is_error", False)),
    )


def _turns_from_result(result: dict[str, Any], payload: dict[str, Any], *, max_tool_result_chars: int) -> list[Turn]:
    standardized = result.get("standardized_turns")
    if isinstance(standardized, list) and standardized:
        return [
            _turn_from_dict(item, max_tool_result_chars=max_tool_result_chars)
            for item in standardized
            if isinstance(item, dict)
        ]

    turns = [Turn(role="human", content=payload.get("question", ""))]
    for step in result.get("tool_trace", []) or []:
        if not isinstance(step, dict):
            continue
        tool_result = step.get("observation")
        if tool_result is None:
            tool_result = _read_observation_file(step.get("observation_path"), max_chars=max_tool_result_chars)
        if tool_result is None:
            tool_result = step.get("observation_preview", "")
        turns.append(Turn(
            role="tool",
            content=step.get("observation_preview", ""),
            tool_name=step.get("tool"),
            tool_args=step.get("args"),
            tool_result=tool_result,
            is_error=step.get("status") == "error",
        ))
    return turns


def token_result_to_trajectory(
    payload: dict[str, Any],
    *,
    mode: str | None = None,
    max_tool_result_chars: int = 50000,
) -> Trajectory:
    """Convert one token-experiment result payload to a Trajectory."""
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("token result payload has no results")
    if mode is None:
        if len(results) != 1:
            raise ValueError(f"payload has multiple modes; pass mode explicitly: {sorted(results)}")
        mode = next(iter(results))
    result = results.get(mode)
    if not isinstance(result, dict):
        raise ValueError(f"mode {mode!r} not found in token result payload")

    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    legacy_status = str(result.get("status") or result.get("evolution_status", {}).get("legacy_tool_status") or "")
    status = str(validation.get("trajectory_label") or result.get("evolution_status", {}).get("status") or legacy_status)
    success = bool(validation.get("success")) if validation else status in SUCCESS_STATUSES
    task_id = str(payload.get("task_id") or payload.get("id") or payload.get("task_index") or "")
    metadata = {
        "experiment_name": payload.get("experiment_name"),
        "experiment_dir": payload.get("experiment_dir"),
        "mode": mode,
        "dataset": payload.get("dataset"),
        "task_index": payload.get("task_index"),
        "raw_index": payload.get("raw_index"),
        "filtered_index": payload.get("filtered_index"),
        "legacy_tool_status": legacy_status,
        "missing_expected_tools": result.get("missing_expected_tools", []),
        "has_tool_error": result.get("has_tool_error", False),
        "llm_calls": result.get("llm_calls", 0),
        "time": result.get("time", 0),
        "evolution_status": result.get("evolution_status", {}),
        "validation": validation,
        "source_sample": payload.get("source_sample"),
        "image_mapping": payload.get("image_mapping"),
        "tool_trace": result.get("tool_trace", []) or [],
    }
    return Trajectory(
        task_id=task_id,
        question=payload.get("question", ""),
        images=payload.get("images", []) or [],
        turns=_turns_from_result(result, payload, max_tool_result_chars=max_tool_result_chars),
        tools_called=result.get("tool_calls", []) or [],
        expected_tools=payload.get("canonical_expected_tools") or payload.get("expected_tools", []) or [],
        final_answer=result.get("final", ""),
        success=success,
        source=payload.get("dataset", "token_experiment"),
        task_type=payload.get("task_type") or (payload.get("source_sample") or {}).get("task_type") or "unknown",
        status=status,
        tokens=result.get("tokens", {}) or {},
        artifacts=result.get("artifact_manifest", []) or [],
        metadata=metadata,
    )


def trajectory_to_record(trajectory: Trajectory) -> dict[str, Any]:
    """Serialize Trajectory to the native JSON shape consumed by DisasterLoader."""
    data = asdict(trajectory)
    data["turns"] = [asdict(turn) for turn in trajectory.turns]
    return data


def iter_token_result_files(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    if root_path.is_file():
        yield root_path
        return
    seen: set[Path] = set()
    for pattern in ("**/json/*.json", "*.json"):
        for path in sorted(root_path.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            if path.name.endswith((".run_manifest.json", ".services.json")):
                continue
            yield path


def load_token_trajectories(
    root: str | Path,
    *,
    modes: Iterable[str] | None = None,
    max_tool_result_chars: int = 50000,
) -> list[Trajectory]:
    selected_modes = set(modes or [])
    trajectories: list[Trajectory] = []
    for path in iter_token_result_files(root):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            continue
        mode_names = selected_modes or set(results)
        for mode in sorted(mode_names):
            if mode not in results:
                continue
            traj = token_result_to_trajectory(
                payload,
                mode=mode,
                max_tool_result_chars=max_tool_result_chars,
            )
            traj.metadata["token_result_path"] = str(path)
            trajectories.append(traj)
    return trajectories


def _package(name: str, trajectories: list[Trajectory], *, source: str | Path) -> dict[str, Any]:
    return {
        "version": "terrabox_evolution_trajectories_v1",
        "name": name,
        "source": str(source),
        "total": len(trajectories),
        "trajectories": [trajectory_to_record(traj) for traj in trajectories],
    }


def trajectory_to_source_like_sample(trajectory: Trajectory) -> dict[str, Any]:
    """Serialize a real rollout in a Disaster-v2-like sample shape."""
    source_sample = trajectory.metadata.get("source_sample")
    base = {
        key: value
        for key, value in (source_sample or {}).items()
        if key not in {"tool_calls"}
    }
    mode = trajectory.metadata.get("mode") or "unknown_mode"
    base.update({
        "id": f"{trajectory.task_id}__{mode}__real_rollout",
        "source_id": trajectory.task_id,
        "prompt": trajectory.question,
        "images": trajectory.images,
        "tool_calls": [],
        "final_answer": trajectory.final_answer,
        "success": trajectory.success,
        "validation_meta": {
            "status": trajectory.status,
            "mode": mode,
            "legacy_tool_status": trajectory.metadata.get("legacy_tool_status"),
            "expected_tools_reference_only": trajectory.expected_tools,
            "used_gold_tools_as_success": False,
            **(trajectory.metadata.get("validation") or {}),
        },
        "rollout_meta": {
            "source": trajectory.source,
            "tokens": trajectory.tokens,
            "artifacts": trajectory.artifacts,
            "token_result_path": trajectory.metadata.get("token_result_path"),
            "image_mapping": trajectory.metadata.get("image_mapping"),
        },
    })
    for idx, step in enumerate(trajectory.metadata.get("tool_trace", []) or [], start=1):
        if not isinstance(step, dict) or not step.get("tool"):
            continue
        observation = step.get("observation")
        if observation is None:
            observation = step.get("observation_preview")
        base["tool_calls"].append({
            "step": idx,
            "tool": step.get("tool"),
            "args": step.get("args", {}),
            "status": step.get("status"),
            "observation": observation,
            "observation_path": step.get("observation_path"),
            "artifacts": step.get("artifacts", []),
        })
    return base


def _source_like_package(name: str, trajectories: list[Trajectory], *, source: str | Path) -> dict[str, Any]:
    return {
        "version": "terrabox_disaster_real_rollout_v1",
        "name": name,
        "source": str(source),
        "total": len(trajectories),
        "samples": [trajectory_to_source_like_sample(traj) for traj in trajectories],
    }


def export_token_results_to_evolution(
    results_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    modes: Iterable[str] | None = None,
    max_tool_result_chars: int = 50000,
) -> dict[str, Any]:
    """Export token-experiment JSON files into success/failure trajectory packages."""
    root = Path(results_root)
    out_dir = Path(output_dir) if output_dir else root / "evolution"
    trajectories = load_token_trajectories(
        root,
        modes=modes,
        max_tool_result_chars=max_tool_result_chars,
    )
    successes = [traj for traj in trajectories if traj.success]
    source_invalid = [traj for traj in trajectories if traj.status in SOURCE_INVALID_STATUSES]
    failures = [traj for traj in trajectories if not traj.success and traj.status not in SOURCE_INVALID_STATUSES]

    _write_json(out_dir / "trajectories_success.json", _package("success", successes, source=root))
    _write_json(out_dir / "trajectories_failure.json", _package("failure", failures, source=root))
    _write_json(out_dir / "source_invalid.json", _package("source_invalid", source_invalid, source=root))
    _write_json(out_dir / "source_like_success.json", _source_like_package("success", successes, source=root))
    _write_json(out_dir / "source_like_failure.json", _source_like_package("failure", failures, source=root))
    _write_json(out_dir / "source_like_invalid_or_uncertain.json", _source_like_package("source_invalid", source_invalid, source=root))
    summary = {
        "source": str(root),
        "output_dir": str(out_dir),
        "total": len(trajectories),
        "success": len(successes),
        "failure": len(failures),
        "source_invalid": len(source_invalid),
        "modes": sorted(set(traj.metadata.get("mode", "") for traj in trajectories if traj.metadata.get("mode"))),
        "status_counts": {},
    }
    for traj in trajectories:
        summary["status_counts"][traj.status] = summary["status_counts"].get(traj.status, 0) + 1
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export token experiment results to evolution Trajectory JSON.")
    parser.add_argument("results_root", help="Token experiment result directory or one JSON file.")
    parser.add_argument("--output-dir", default=None, help="Default: <results_root>/evolution")
    parser.add_argument("--modes", nargs="*", default=None, help="Optional mode names to export.")
    parser.add_argument("--max-tool-result-chars", type=int, default=50000)
    args = parser.parse_args()
    summary = export_token_results_to_evolution(
        args.results_root,
        args.output_dir,
        modes=args.modes,
        max_tool_result_chars=args.max_tool_result_chars,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
