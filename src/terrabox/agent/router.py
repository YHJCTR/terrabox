"""FastAPI router for the Agent chat endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db
from ..routers.deps import current_user_from_jwt
from ..core.utils.uploads import save_upload_files
from .graph import clear_session, new_session_id, run_agent

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
        response = run_agent(session_id, message, image_paths, current_user, db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {e}",
        )

    return {"response": response, "session_id": session_id}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    """Clear the conversation history for a session."""
    clear_session(session_id, db, current_user.id)
    return {"status": "ok", "session_id": session_id}
