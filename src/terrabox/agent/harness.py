"""Context and event helpers for backend agent harness features."""
from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any

from ..agent.events import build_event
from ..agent.runtime import get_runtime
from ..core.services.agent_run_service import AgentRunService


@dataclass
class HarnessContext:
    run_id: str
    session_id: str | None
    user: Any
    db: Any
    mode: str
    config: Any
    rewritten_message: str = ""
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    released_run_slot: bool = False


_CTX: contextvars.ContextVar[HarnessContext | None] = contextvars.ContextVar("agent_harness_ctx", default=None)
_LAST_EVENTS: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar("agent_harness_last_events", default=[])


def current_context() -> HarnessContext | None:
    return _CTX.get()


def start_run(session_id: str | None, user_message: str, image_paths: list[str], user, db, config) -> HarnessContext:
    from ..core.utils.uploads import inspect_uploaded_files

    existing = current_context()
    if existing is not None:
        return existing
    runtime = get_runtime()
    runtime.configure(config)
    if not runtime.acquire_run():
        raise RuntimeError("Agent runtime is saturated. Please retry shortly.")
    run = AgentRunService.create_run(
        db,
        user_id_fk=user.id,
        session_id=session_id,
        mode=config.agent_mode,
        original_message=user_message,
        image_paths=image_paths,
        metadata={
            "agent_mode": config.agent_mode,
            "uploaded_files": inspect_uploaded_files(image_paths) if image_paths else [],
        },
    )
    ctx = HarnessContext(
        run_id=run.id,
        session_id=session_id,
        user=user,
        db=db,
        mode=config.agent_mode,
        config=config,
    )
    _CTX.set(ctx)
    emit_event("run_start", session_id=session_id, mode=config.agent_mode)
    return ctx


def emit_event(event_type: str, **payload: Any) -> None:
    ctx = current_context()
    if ctx is None:
        return
    ctx.pending_events.append(build_event(event_type, run_id=ctx.run_id, **payload))


def drain_events() -> list[dict[str, Any]]:
    ctx = current_context()
    if ctx is not None:
        if not ctx.pending_events:
            return []
        items = list(ctx.pending_events)
        ctx.pending_events.clear()
        return items
    items = list(_LAST_EVENTS.get())
    _LAST_EVENTS.set([])
    return items


def record_step(step_type: str, **kwargs: Any):
    ctx = current_context()
    if ctx is None:
        return None
    step = AgentRunService.add_step(ctx.db, ctx.run_id, step_type=step_type, **kwargs)
    return step


def update_step(step_id: str, **kwargs: Any):
    ctx = current_context()
    if ctx is None:
        return None
    return AgentRunService.update_step(ctx.db, step_id, **kwargs)


def record_artifact(step_id: str | None, path: str, kind: str = "file", preview_path: str | None = None, metadata: dict[str, Any] | None = None):
    ctx = current_context()
    if ctx is None:
        return None
    artifact = AgentRunService.add_artifact(
        ctx.db,
        run_id=ctx.run_id,
        step_id=step_id,
        path=path,
        kind=kind,
        preview_path=preview_path,
        metadata=metadata,
    )
    emit_event(
        "artifact",
        artifact_id=artifact.id,
        step_id=step_id,
        path=artifact.path,
        preview_path=artifact.preview_path,
        kind=artifact.kind,
        size_bytes=artifact.size_bytes,
    )
    return artifact


def create_approval(tool_slug: str, reason: str, step_id: str | None = None, status: str = "pending"):
    ctx = current_context()
    if ctx is None:
        return None
    approval = AgentRunService.create_approval(
        ctx.db,
        run_id=ctx.run_id,
        tool_slug=tool_slug,
        reason=reason,
        step_id=step_id,
        status=status,
    )
    emit_event(
        "approval_required" if status == "pending" else "approval_recorded",
        approval_id=approval.id,
        step_id=step_id,
        tool_slug=tool_slug,
        status=status,
        reason=reason,
    )
    return approval


def finish_run(status: str, *, final_response: str = "", error_message: str | None = None, metadata_update: dict[str, Any] | None = None) -> None:
    ctx = current_context()
    if ctx is None:
        return
    AgentRunService.finish_run(
        ctx.db,
        ctx.run_id,
        status=status,
        final_response=final_response,
        error_message=error_message,
        rewritten_message=ctx.rewritten_message,
        metadata_update=metadata_update,
    )
    emit_event("run_metrics", status=status, metrics=get_runtime().metrics())
    emit_event("done", session_id=ctx.session_id, status=status)
    _LAST_EVENTS.set(list(ctx.pending_events))
    close_context()


def fail_run(error_message: str, *, metadata_update: dict[str, Any] | None = None) -> None:
    ctx = current_context()
    if ctx is None:
        return
    AgentRunService.finish_run(
        ctx.db,
        ctx.run_id,
        status="failed",
        final_response="",
        error_message=error_message,
        rewritten_message=ctx.rewritten_message,
        metadata_update=metadata_update,
    )
    emit_event("error", message=error_message)
    emit_event("done", session_id=ctx.session_id, status="failed")
    _LAST_EVENTS.set(list(ctx.pending_events))
    close_context()


def close_context() -> None:
    ctx = current_context()
    if ctx is None:
        return
    if not ctx.released_run_slot:
        get_runtime().release_run()
        ctx.released_run_slot = True
    _CTX.set(None)


def log_structured(io, event_type: str, **payload: Any) -> None:
    try:
        io.info(json.dumps({"event_type": event_type, **payload}, ensure_ascii=False))
    except Exception:
        io.info(f"{event_type} {payload}")
