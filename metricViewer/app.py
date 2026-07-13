from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .adapters import DEFAULT_ADAPTER, DEFAULT_TOTAL


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Terrabox Metric Viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/experiments")
def experiments() -> dict:
    refs = DEFAULT_ADAPTER.discover()
    return {
        "experiments": [
            {
                "name": ref.name,
                "kind": ref.kind,
                "results_dir": str(ref.results_dir),
                "prompt_path": str(ref.prompt_path) if ref.prompt_path else None,
            }
            for ref in refs
        ]
    }


@app.get("/api/status")
def status(
    experiment: str = Query(..., description="Experiment name or results directory"),
    scope: str = Query("all", pattern="^(all|online|offline)$"),
    total: int = Query(DEFAULT_TOTAL, ge=0),
) -> dict:
    try:
        return DEFAULT_ADAPTER.status(experiment, scope=scope, total=total)
    except Exception as exc:  # noqa: BLE001 - return readable UI errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/compare")
def compare(
    current: str = Query(..., description="Current experiment name or results directory"),
    baseline: str = Query(..., description="Baseline experiment name or results directory"),
) -> dict:
    try:
        return DEFAULT_ADAPTER.compare(current, baseline)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/prompt")
def prompt(experiment: str = Query(..., description="Experiment name or results directory")) -> dict:
    try:
        return DEFAULT_ADAPTER.prompt(experiment)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

