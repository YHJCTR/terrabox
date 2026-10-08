from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .adapters import DEFAULT_ADAPTER, DEFAULT_SCENE


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Terrabox Metric Viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/scenes")
def scenes() -> dict:
    return {"scenes": DEFAULT_ADAPTER.scenes()}


@app.get("/api/experiments")
def experiments(scene: str = Query(DEFAULT_SCENE)) -> dict:
    refs = DEFAULT_ADAPTER.discover(scene)
    return {
        "experiments": [
            {
                "name": ref.name,
                "scene": ref.scene,
                "id": ref.id,
                "kind": ref.kind,
                # ExperimentRef exposes the discovered result location as
                # source_path; keep the API field name stable for the UI.
                "results_dir": str(ref.source_path),
                "prompt_path": str(ref.prompt_path) if ref.prompt_path else None,
            }
            for ref in refs
        ]
    }


@app.get("/api/status")
def status(
    experiment: str = Query(..., description="Experiment name or results directory"),
    scene: str = Query(DEFAULT_SCENE, description="Evaluation scene"),
    scope: str = Query("all", pattern="^(all|online|offline)$"),
) -> dict:
    try:
        return DEFAULT_ADAPTER.status(scene, experiment, scope=scope)
    except Exception as exc:  # noqa: BLE001 - return readable UI errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/compare")
def compare(
    current: str = Query(..., description="Current experiment name or results directory"),
    baseline: str = Query(..., description="Baseline experiment name or results directory"),
    scene: str = Query(DEFAULT_SCENE, description="Evaluation scene"),
) -> dict:
    try:
        return DEFAULT_ADAPTER.compare(scene, current, baseline)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/prompt")
def prompt(
    experiment: str = Query(..., description="Experiment name or results directory"),
    scene: str = Query(DEFAULT_SCENE, description="Evaluation scene"),
) -> dict:
    try:
        return DEFAULT_ADAPTER.prompt(scene, experiment)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
