"""Persistence and query helpers for agent harness runs."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ...db.models import AgentApproval, AgentArtifact, AgentRun, AgentRunStep


def _dump(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _load(value: Any, default: Any):
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


class AgentRunService:
    @staticmethod
    def create_run(
        db: Session,
        user_id_fk: str,
        session_id: str | None,
        mode: str,
        original_message: str,
        image_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRun:
        run = AgentRun(
            id=str(uuid.uuid4()),
            user_id_fk=user_id_fk,
            session_id=session_id,
            mode=mode,
            status="running",
            original_message=original_message,
            image_paths_json=_dump(image_paths or []),
            metadata_json=_dump(metadata or {}),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def add_step(
        db: Session,
        run_id: str,
        step_type: str,
        title: str = "",
        content: str = "",
        status: str = "completed",
        tool_slug: str | None = None,
        trace_id: str | None = None,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> AgentRunStep:
        run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
        step_index = (run.step_count if run else 0) + 1
        step = AgentRunStep(
            id=str(uuid.uuid4()),
            run_id=run_id,
            step_index=step_index,
            step_type=step_type,
            status=status,
            title=title,
            content=content,
            tool_slug=tool_slug,
            trace_id=trace_id,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            input_json=_dump(input_data or {}),
            output_json=_dump(output_data or {}),
            metadata_json=_dump(metadata or {}),
        )
        db.add(step)
        if run:
            run.step_count = step_index
            if tool_slug:
                run.tool_call_count = (run.tool_call_count or 0) + 1
        db.commit()
        db.refresh(step)
        return step

    @staticmethod
    def update_step(
        db: Session,
        step_id: str,
        *,
        status: str | None = None,
        content: str | None = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> AgentRunStep | None:
        step = db.query(AgentRunStep).filter(AgentRunStep.id == step_id).first()
        if step is None:
            return None
        if status is not None:
            step.status = status
        if content is not None:
            step.content = content
        if output_data is not None:
            step.output_json = _dump(output_data)
        if metadata is not None:
            step.metadata_json = _dump(metadata)
        if duration_ms is not None:
            step.duration_ms = duration_ms
        if error_type is not None:
            step.error_type = error_type
        if error_message is not None:
            step.error_message = error_message
        step.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(step)
        return step

    @staticmethod
    def add_artifact(
        db: Session,
        run_id: str,
        step_id: str | None,
        path: str,
        kind: str = "file",
        preview_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        artifact = AgentArtifact(
            id=str(uuid.uuid4()),
            run_id=run_id,
            step_id=step_id,
            kind=kind,
            path=path,
            preview_path=preview_path,
            exists=os.path.exists(path),
            size_bytes=os.path.getsize(path) if os.path.exists(path) else None,
            metadata_json=_dump(metadata or {}),
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact

    @staticmethod
    def create_approval(
        db: Session,
        run_id: str,
        tool_slug: str,
        reason: str,
        step_id: str | None = None,
        status: str = "pending",
    ) -> AgentApproval:
        approval = AgentApproval(
            id=str(uuid.uuid4()),
            run_id=run_id,
            step_id=step_id,
            tool_slug=tool_slug,
            reason=reason,
            status=status,
            decided_at=datetime.utcnow() if status != "pending" else None,
        )
        db.add(approval)
        db.commit()
        db.refresh(approval)
        return approval

    @staticmethod
    def update_approval(db: Session, approval_id: str, status: str, note: str | None = None) -> AgentApproval | None:
        approval = db.query(AgentApproval).filter(AgentApproval.id == approval_id).first()
        if approval is None:
            return None
        approval.status = status
        approval.decision_note = note
        approval.decided_at = datetime.utcnow()
        approval.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(approval)
        return approval

    @staticmethod
    def finish_run(
        db: Session,
        run_id: str,
        *,
        status: str,
        final_response: str = "",
        error_message: str | None = None,
        rewritten_message: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> AgentRun | None:
        run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
        if run is None:
            return None
        run.status = status
        run.final_response = final_response
        run.error_message = error_message
        if rewritten_message is not None:
            run.rewritten_message = rewritten_message
        run.finished_at = datetime.utcnow()
        if run.started_at:
            run.latency_ms = int((run.finished_at - run.started_at).total_seconds() * 1000)
        if metadata_update:
            current = _load(run.metadata_json, {})
            current.update(metadata_update)
            run.metadata_json = _dump(current)
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def list_runs(db: Session, user_id_fk: str, limit: int = 50, status: str | None = None, mode: str | None = None) -> list[AgentRun]:
        query = db.query(AgentRun).filter(AgentRun.user_id_fk == user_id_fk)
        if status:
            query = query.filter(AgentRun.status == status)
        if mode:
            query = query.filter(AgentRun.mode == mode)
        return query.order_by(AgentRun.started_at.desc()).limit(limit).all()

    @staticmethod
    def get_run(db: Session, run_id: str, user_id_fk: str) -> AgentRun | None:
        return db.query(AgentRun).filter(AgentRun.id == run_id, AgentRun.user_id_fk == user_id_fk).first()

    @staticmethod
    def get_steps(db: Session, run_id: str) -> list[AgentRunStep]:
        return db.query(AgentRunStep).filter(AgentRunStep.run_id == run_id).order_by(AgentRunStep.step_index.asc()).all()

    @staticmethod
    def get_artifacts(db: Session, run_id: str) -> list[AgentArtifact]:
        return db.query(AgentArtifact).filter(AgentArtifact.run_id == run_id).order_by(AgentArtifact.created_at.asc()).all()

    @staticmethod
    def get_approvals(db: Session, run_id: str) -> list[AgentApproval]:
        return db.query(AgentApproval).filter(AgentApproval.run_id == run_id).order_by(AgentApproval.created_at.asc()).all()

    @staticmethod
    def build_replay(db: Session, run_id: str) -> list[dict[str, Any]]:
        replay: list[dict[str, Any]] = []

        # Index approvals by step_id for O(1) attachment to tool steps
        approvals_by_step: dict[str, list[dict]] = {}
        for approval in AgentRunService.get_approvals(db, run_id):
            entry = {
                "approval_id": approval.id,
                "tool_slug": approval.tool_slug,
                "status": approval.status,
                "reason": approval.reason,
                "note": approval.note,
                "created_at": str(approval.created_at) if approval.created_at else None,
                "decided_at": str(approval.decided_at) if approval.decided_at else None,
            }
            approvals_by_step.setdefault(approval.step_id or "", []).append(entry)

        for step in AgentRunService.get_steps(db, run_id):
            item: dict[str, Any] = {
                "type": step.step_type,
                "step_id": step.id,
                "step_index": step.step_index,
                "status": step.status,
                "title": step.title,
                "content": step.content,
                "tool_slug": step.tool_slug,
                "trace_id": step.trace_id,
                "duration_ms": step.duration_ms,
                "error_type": step.error_type,
                "error_message": step.error_message,
                "input": _load(step.input_json, {}),
                "output": _load(step.output_json, {}),
                "metadata": _load(step.metadata_json, {}),
            }
            step_approvals = approvals_by_step.get(step.id, [])
            if step_approvals:
                item["approvals"] = step_approvals
            replay.append(item)

        for artifact in AgentRunService.get_artifacts(db, run_id):
            replay.append({
                "type": "artifact",
                "artifact_id": artifact.id,
                "path": artifact.path,
                "preview_path": artifact.preview_path,
                "kind": artifact.kind,
                "exists": artifact.exists,
                "size_bytes": artifact.size_bytes,
                "metadata": _load(artifact.metadata_json, {}),
            })
        replay.sort(key=lambda item: (item.get("step_index", 10**9), 0 if item["type"] != "artifact" else 1))
        return replay

    @staticmethod
    def parse_run(run: AgentRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "session_id": run.session_id,
            "user_id": str(run.user_id_fk),
            "mode": run.mode,
            "status": run.status,
            "original_message": run.original_message,
            "rewritten_message": run.rewritten_message,
            "final_response": run.final_response,
            "error_message": run.error_message,
            "image_paths": _load(run.image_paths_json, []),
            "metadata": _load(run.metadata_json, {}),
            "tool_call_count": run.tool_call_count or 0,
            "step_count": run.step_count or 0,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "latency_ms": run.latency_ms,
        }

    @staticmethod
    def parse_step(step: AgentRunStep) -> dict[str, Any]:
        return {
            "id": step.id,
            "run_id": step.run_id,
            "step_index": step.step_index,
            "step_type": step.step_type,
            "status": step.status,
            "title": step.title,
            "content": step.content,
            "tool_slug": step.tool_slug,
            "trace_id": step.trace_id,
            "duration_ms": step.duration_ms,
            "error_type": step.error_type,
            "error_message": step.error_message,
            "input": _load(step.input_json, {}),
            "output": _load(step.output_json, {}),
            "metadata": _load(step.metadata_json, {}),
            "created_at": step.created_at,
            "updated_at": step.updated_at,
        }

    @staticmethod
    def parse_artifact(artifact: AgentArtifact) -> dict[str, Any]:
        return {
            "id": artifact.id,
            "run_id": artifact.run_id,
            "step_id": artifact.step_id,
            "kind": artifact.kind,
            "path": artifact.path,
            "preview_path": artifact.preview_path,
            "exists": artifact.exists,
            "size_bytes": artifact.size_bytes,
            "metadata": _load(artifact.metadata_json, {}),
            "created_at": artifact.created_at,
        }

    @staticmethod
    def parse_approval(approval: AgentApproval) -> dict[str, Any]:
        return {
            "id": approval.id,
            "run_id": approval.run_id,
            "step_id": approval.step_id,
            "tool_slug": approval.tool_slug,
            "reason": approval.reason,
            "status": approval.status,
            "decision_note": approval.decision_note,
            "created_at": approval.created_at,
            "updated_at": approval.updated_at,
            "decided_at": approval.decided_at,
        }
