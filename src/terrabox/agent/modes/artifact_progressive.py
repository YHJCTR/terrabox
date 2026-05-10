"""Artifact-progressive agent mode."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from sqlalchemy.orm import Session

from ..artifact_progressive_graph import run_artifact_progressive_loop
from ..harness import fail_run, finish_run, record_step
from ..llm import get_llm
from ..session import (
    finalize_session,
    get_io_logger,
    log_messages,
    log_session_start,
    prepare_history,
    stream_sync_agent,
)
from .runtime import ensure_run_context

logger = logging.getLogger(__name__)


def _format_question(user_message: str, image_paths: list[str]) -> str:
    if image_paths:
        return f"{user_message}\n\n[Image files: {', '.join(image_paths)}]"
    return user_message


def run_artifact_progressive_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
    config,
) -> str:
    ensure_run_context(session_id, user_message, image_paths, user, db, config)
    record_step(
        "decision",
        title="Artifact-progressive route selected",
        content="artifact_progressive",
        metadata={"mode": "artifact_progressive", "max_steps": config.max_progressive_steps},
    )
    llm = get_llm(config)
    record, _history = prepare_history(session_id, user_message, image_paths, user, db, llm=llm)
    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths, mode="artifact_progressive")

    result = run_artifact_progressive_loop(
        llm=llm,
        question=_format_question(user_message, image_paths),
        image_paths=image_paths,
        config=config,
        user=user,
        allowed_slugs=None,
        verbose=False,
    )
    log_messages(io, result["messages"])
    final = finalize_session(record, {"messages": result["messages"]}, db, io)
    finish_run(
        "completed",
        final_response=final,
        metadata_update={"mode": "artifact_progressive", "artifact_state": result.get("artifact_state")},
    )
    return final


async def stream_artifact_progressive_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
    config,
) -> AsyncIterator[str]:
    async for sse in stream_sync_agent(
        run_artifact_progressive_agent,
        session_id,
        user_message,
        image_paths,
        user,
        db,
        config,
    ):
        yield sse
