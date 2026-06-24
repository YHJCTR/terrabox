import json
from pathlib import Path


def _write_result(path: Path, task_id: str, expected: list[str], called: list[str], *, success: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = []
    for name in called:
        messages.append({"type": "AIMessage", "tool_calls": [{"name": name.replace(".", "__"), "args": {}}]})
        messages.append({"type": "ToolMessage", "content": '{"status": "ok"}'})
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "expected_tools": expected,
                "tool_calls": called,
                "conversation_history": messages,
                "status": "completed",
                "success": success,
                "metrics": {"f1": 1.0 if success else 0.0, "multiset_f1": 1.0 if success else 0.0},
            }
        ),
        encoding="utf-8",
    )


def test_resolve_results_dir_supports_tmp_and_reflection_layouts(tmp_path):
    from terrabox.evolution.shared.rollout_report import resolve_results_dir

    tmp_results = tmp_path / "tmp" / "trajectories" / "react_exp" / "standard" / "results"
    refl_results = tmp_path / "src" / "terrabox" / "evolution" / "reflection" / "exp" / "refl_exp" / "eval" / "results"
    refl_train_results = tmp_path / "src" / "terrabox" / "evolution" / "reflection" / "exp" / "refl_exp" / "train" / "results"
    refl_backup_results = tmp_path / "src" / "terrabox" / "evolution" / "reflection" / "exp" / "refl_exp" / "eval_old" / "results"
    tmp_results.mkdir(parents=True)
    refl_results.mkdir(parents=True)
    refl_train_results.mkdir(parents=True)
    refl_backup_results.mkdir(parents=True)

    assert resolve_results_dir("react_exp", repo_root=tmp_path) == tmp_results
    assert resolve_results_dir("refl_exp", repo_root=tmp_path) == refl_results
    assert resolve_results_dir(str(refl_results), repo_root=tmp_path) == refl_results


def test_compare_experiments_uses_only_common_task_ids(tmp_path):
    from terrabox.evolution.shared.rollout_report import compare_experiments, resolve_results_dir

    base_dir = tmp_path / "tmp" / "trajectories" / "base" / "standard" / "results"
    cur_dir = tmp_path / "src" / "terrabox" / "evolution" / "reflection" / "exp" / "cur" / "eval" / "results"
    _write_result(base_dir / "a.json", "a", ["compute.calculator"], ["compute.calculator"])
    _write_result(base_dir / "base_only.json", "base_only", ["compute.solver"], ["compute.solver"])
    _write_result(cur_dir / "a.json", "a", ["compute.calculator"], ["compute.calculator"])
    _write_result(cur_dir / "cur_only.json", "cur_only", ["compute.plot"], ["compute.plot"])

    report = compare_experiments(
        resolve_results_dir("cur", repo_root=tmp_path),
        resolve_results_dir("base", repo_root=tmp_path),
    )

    assert report["n_common"] == 1
    assert report["cur_done"] == 2
    assert report["base_done"] == 2
    assert report["cur"]["success_rate"] == 100.0
    assert report["delta"]["success_rate"] == 0.0
