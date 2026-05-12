"""FastAPI router for the Agent chat endpoints."""
from __future__ import annotations

from typing import List, Optional

import asyncio
import functools

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db
from ..core.services.agent_run_service import AgentRunService
from ..core.schemas import (
    AgentApprovalResponse,
    AgentArtifactResponse,
    AgentBtwRequest,
    AgentBtwResponse,
    AgentContextStatusResponse,
    AgentReplayEventResponse,
    AgentRunDetailResponse,
    AgentRunResponse,
    AgentRunStepResponse,
    AgentRuntimeMetricsResponse,
)
from ..routers.deps import current_user_from_jwt
from ..core.utils.uploads import save_upload_files
from .graph import clear_session, new_session_id, run_agent_with_meta, stream_agent
from .config import load_config
from .runtime import get_runtime
from .session import BtwSessionNotFound, compact_session, get_context_status, run_btw_query
from .llm import get_llm

router = APIRouter(prefix="/v1/gui/agent", tags=["agent"])


@router.post("/chat")
async def agent_chat(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """
    Send a message to the Agent. The Agent autonomously selects and chains tools.

    - **message**: User's text prompt
    - **session_id**: Conversation session ID (omit to start a new session)
    - **files**: Optional image uploads (paths will be passed to the Agent)
    """
    if not session_id:
        session_id = new_session_id()

    image_paths = await save_upload_files(files)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(run_agent_with_meta, session_id, message, image_paths, current_user, db)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {e}",
        )

    return {"response": result["response"], "session_id": session_id, "run_id": result["run_id"]}


@router.post("/chat/stream")
async def agent_chat_stream(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """
    Streaming variant of /chat. Returns an SSE stream with events:
      {"type":"thinking","token":"..."}  — content inside <think> tags
      {"type":"response","token":"..."}  — regular response content
      {"type":"done","session_id":"..."}  — stream complete
      {"type":"error","message":"..."}   — on failure
    """
    if not session_id:
        session_id = new_session_id()

    image_paths = await save_upload_files(files)

    return StreamingResponse(
        stream_agent(session_id, message, image_paths, current_user, db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """Clear the conversation history for a session."""
    clear_session(session_id, db, current_user.id)
    return {"status": "ok", "session_id": session_id}


@router.get("/sessions/{session_id}/context", response_model=AgentContextStatusResponse)
def get_session_context_status(
    session_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """Return context budget and compression state for the current user's session."""
    config = load_config()
    return get_context_status(
        session_id,
        db,
        current_user.id,
        max_model_len=getattr(config, "local_llm_max_model_len", 0),
    )


@router.post("/sessions/{session_id}/compact", response_model=AgentContextStatusResponse)
def compact_session_context(
    session_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """Manually compact old raw messages into the session rolling summary."""
    config = load_config()
    return compact_session(
        session_id,
        db,
        current_user.id,
        llm=None,
        max_model_len=getattr(config, "local_llm_max_model_len", 0),
    )


@router.post("/sessions/{session_id}/btw", response_model=AgentBtwResponse)
async def answer_btw_aside(
    session_id: str,
    request: AgentBtwRequest,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """Answer a quick aside from session context without persisting it."""
    config = load_config()
    try:
        loop = asyncio.get_running_loop()
        def _run_btw():
            return run_btw_query(session_id, request.message, current_user, db, get_llm(config))

        return await loop.run_in_executor(
            None,
            _run_btw,
        )
    except BtwSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"/btw error: {exc}") from exc


@router.get("/runs", response_model=list[AgentRunResponse])
def list_agent_runs(
    limit: int = 50,
    status_filter: Optional[str] = None,
    mode: Optional[str] = None,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    runs = AgentRunService.list_runs(db, current_user.id, limit=limit, status=status_filter, mode=mode)
    return [AgentRunResponse(**AgentRunService.parse_run(run)) for run in runs]


@router.get("/runs/{run_id}", response_model=AgentRunDetailResponse)
def get_agent_run(
    run_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    run = AgentRunService.get_run(db, run_id, current_user.id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    steps = AgentRunService.get_steps(db, run_id)
    artifacts = AgentRunService.get_artifacts(db, run_id)
    approvals = AgentRunService.get_approvals(db, run_id)
    return AgentRunDetailResponse(
        run=AgentRunResponse(**AgentRunService.parse_run(run)),
        steps=[AgentRunStepResponse(**AgentRunService.parse_step(step)) for step in steps],
        artifacts=[AgentArtifactResponse(**AgentRunService.parse_artifact(artifact)) for artifact in artifacts],
        approvals=[AgentApprovalResponse(**AgentRunService.parse_approval(approval)) for approval in approvals],
    )


@router.get("/runs/{run_id}/replay", response_model=list[AgentReplayEventResponse])
def replay_agent_run(
    run_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    run = AgentRunService.get_run(db, run_id, current_user.id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    replay = AgentRunService.build_replay(db, run_id)
    return [AgentReplayEventResponse(**item) for item in replay]


@router.post("/runs/{run_id}/approve")
def approve_agent_run(
    run_id: str,
    approval_id: str = Form(...),
    note: Optional[str] = Form(None),
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    run = AgentRunService.get_run(db, run_id, current_user.id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    approval = AgentRunService.update_approval(db, approval_id, "approved", note)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return {"status": "ok", "run_id": run_id, "approval_id": approval_id}


@router.post("/runs/{run_id}/deny")
def deny_agent_run(
    run_id: str,
    approval_id: str = Form(...),
    note: Optional[str] = Form(None),
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    run = AgentRunService.get_run(db, run_id, current_user.id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    approval = AgentRunService.update_approval(db, approval_id, "denied", note)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return {"status": "ok", "run_id": run_id, "approval_id": approval_id}


@router.post("/runs/{run_id}/resume")
def resume_agent_run(
    run_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    run = AgentRunService.get_run(db, run_id, current_user.id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "unsupported", "message": "Resume is only available for future interactive approval mode", "run_id": run_id}


@router.get("/runtime/metrics", response_model=AgentRuntimeMetricsResponse)
def agent_runtime_metrics(
    current_user: models.User = Depends(current_user_from_jwt),
):
    metrics = get_runtime().metrics()
    return AgentRuntimeMetricsResponse(**metrics)


@router.get("/runtime/health")
def agent_runtime_health(
    current_user: models.User = Depends(current_user_from_jwt),
):
    metrics = get_runtime().metrics()
    return {"status": "ok", "active_runs": metrics.get("active_runs", 0), "rejected_runs": metrics.get("rejected_runs", 0)}
