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
from ..routers.deps import current_user_from_jwt
from ..core.utils.uploads import save_upload_files
from .graph import clear_session, new_session_id, run_agent, stream_agent

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
        response = await loop.run_in_executor(
            None, functools.partial(run_agent, session_id, message, image_paths, current_user, db)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {e}",
        )

    return {"response": response, "session_id": session_id}


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
