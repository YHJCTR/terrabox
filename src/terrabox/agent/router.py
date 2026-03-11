"""FastAPI router for the Agent chat endpoints."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db
from ..routers.deps import current_user_from_jwt
from .graph import clear_session, new_session_id, run_agent

import os

UPLOAD_DIR = Path(os.getenv("TERRABOX_UPLOAD_DIR", "/data1/terrabox_uploads"))

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

    # Save uploaded files to the shared upload directory
    image_paths: list[str] = []
    for file in files:
        suffix = Path(file.filename or "upload").suffix or ".bin"
        dest = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await file.read())
        image_paths.append(str(dest))

    try:
        response = run_agent(session_id, message, image_paths, current_user)
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
):
    """Clear the conversation history for a session."""
    clear_session(session_id)
    return {"status": "ok", "session_id": session_id}
