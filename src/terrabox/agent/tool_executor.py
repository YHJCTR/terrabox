"""Unified tool executor for agent harness mode."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Bounded thread pool for running async tool handlers synchronously.
# Prevents unbounded daemon thread creation under concurrent load.
_EXECUTOR = ThreadPoolExecutor(max_workers=32, thread_name_prefix="tool_async")

from ..core.registry import get_handler, get_tool, list_toolkits, list_tools
from ..core.services.agent_run_service import AgentRunService
from ..core.services.connection_service import ConnectionService
from ..core.services.tool_override_service import ToolOverrideService
from ..core.services.tool_service import ToolService
from .harness import create_approval, current_context, emit_event, record_artifact, record_step, update_step
from .runtime import get_runtime

logger = logging.getLogger(__name__)

_PATH_KEYS = {"output_path", "result_path", "path", "image_path", "preview_path"}


def _toolkit_for_slug(slug: str) -> str | None:
    for toolkit in list_toolkits():
        if any(t.slug == slug for t in list_tools(toolkit.name)):
            return toolkit.name
    return None


def _is_risky(slug: str) -> bool:
    return slug.startswith("bash.") or slug.startswith("ipython_code.") or slug.startswith("github.")


def _bucket_for(slug: str) -> str:
    if slug.startswith("geo_perception."):
        return "perception"
    if slug.startswith("bash.") or slug.startswith("ipython_code.") or slug.startswith("github."):
        return "risky"
    if slug.startswith("bing_search."):
        return "network"
    if slug.startswith("geo_raster.") or slug.startswith("georaster.") or slug.startswith("disaster_response."):
        return "compute"
    return "default"


def _display_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        text = json.dumps(result, ensure_ascii=False)
    except Exception:
        text = str(result)
    return text[:4000]


def _artifact_paths(result: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(result, str) and os.path.exists(result):
        return [result]
    if isinstance(result, dict):
        for key, value in result.items():
            if key in _PATH_KEYS and isinstance(value, str) and value:
                paths.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and os.path.exists(item):
                        paths.append(item)
    return [p for p in dict.fromkeys(paths) if p]


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}
    err: dict[str, BaseException] = {}

    def _runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - passthrough
            err["error"] = exc

    future = _EXECUTOR.submit(_runner)
    future.result()
    if "error" in err:
        raise err["error"]
    return box.get("value")


class AgentToolExecutor:
    @staticmethod
    def execute(slug: str, arguments: dict[str, Any], user) -> str:
        ctx = current_context()
        if ctx is None:
            return AgentToolExecutor._execute_legacy(slug, arguments, user)
        return AgentToolExecutor._execute_harness(slug, arguments, user)

    @staticmethod
    def _execute_legacy(slug: str, arguments: dict[str, Any], user) -> str:
        handler = get_handler(slug)
        if handler is None:
            return f"Tool execution error: Tool not found: {slug}"
        try:
            sig = inspect.signature(handler)
            accepts_connection = len(sig.parameters) >= 3
            if inspect.iscoroutinefunction(handler):
                if accepts_connection:
                    result = _run_coro_sync(handler(arguments, None, user))
                else:
                    result = _run_coro_sync(handler(arguments, None))
            else:
                if accepts_connection:
                    result = handler(arguments, None, user)
                else:
                    result = handler(arguments, None)
            return str(result)
        except Exception as exc:
            logger.warning("Tool execution error: %s", exc)
            return f"Tool execution error: {exc}"

    @staticmethod
    def _execute_harness(slug: str, arguments: dict[str, Any], user) -> str:
        ctx = current_context()
        if ctx is None:
            return AgentToolExecutor._execute_legacy(slug, arguments, user)

        config = ctx.config
        db = ctx.db
        trace_id = str(uuid.uuid4())
        bucket = _bucket_for(slug)
        toolkit_name = _toolkit_for_slug(slug)
        runtime = get_runtime()

        ok, runtime_reason = runtime.start_tool(bucket)
        if not ok:
            step = record_step(
                "tool",
                title=f"Tool blocked: {slug}",
                content=runtime_reason or "runtime_guard",
                status="failed",
                tool_slug=slug,
                trace_id=trace_id,
                input_data=arguments,
                error_type=runtime_reason,
                error_message=f"Tool blocked by runtime guard: {runtime_reason}",
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=False,
                trace_id=trace_id,
                error_type=runtime_reason,
                message=f"Tool blocked by runtime guard: {runtime_reason}",
            )
            return f"Tool execution error: Tool blocked by runtime guard: {runtime_reason}"

        step = record_step(
            "tool",
            title=f"Execute tool {slug}",
            content="tool_start",
            status="running",
            tool_slug=slug,
            trace_id=trace_id,
            input_data=arguments,
            metadata={"bucket": bucket},
        )
        emit_event(
            "tool_start",
            step_id=step.id if step else None,
            tool_slug=slug,
            trace_id=trace_id,
            args=arguments,
        )

        started = time.time()
        connection = None
        approval = None
        try:
            spec = get_tool(slug)
            if spec is None:
                raise RuntimeError(f"Tool not found: {slug}")

            if getattr(config, "record_risky_tool_approvals", True) and _is_risky(slug):
                approval_status = "pending" if config.require_approval_for_risky_tools else "auto_approved"
                approval = create_approval(
                    slug,
                    reason="Manual approval required for risky tool" if approval_status == "pending" else "Risky tool category logged for audit",
                    step_id=step.id if step else None,
                    status=approval_status,
                )
                if approval_status == "pending":
                    raise PermissionError(f"Approval required for risky tool: {slug}")

            if spec.requires_connection and toolkit_name:
                connection = ConnectionService.select_connection(db, user.id, toolkit_name)
                if not connection:
                    raise RuntimeError("No valid connection found for this tool")

            if connection:
                effective = ToolOverrideService.get_effective_tools(db, str(connection.id), include_disabled=False)
                enabled = {item["tool_key"] for item in effective["tools"]}
                if slug not in enabled:
                    raise RuntimeError("Tool is disabled for this connection")

            handler = get_handler(slug)
            if handler is None:
                raise RuntimeError(f"No handler registered for tool: {slug}")

            context = {
                "user_id": getattr(user, "user_id", user.id),
                "connection_id": str(connection.id) if connection else None,
                "trace_id": trace_id,
                "run_id": ctx.run_id,
                "timeout": config.default_tool_timeout_seconds,
            }
            sig = inspect.signature(handler)
            accepts_connection = len(sig.parameters) >= 3

            if inspect.iscoroutinefunction(handler):
                if accepts_connection:
                    result = _run_coro_sync(handler(arguments, context, connection))
                else:
                    result = _run_coro_sync(handler(arguments, context))
            else:
                if accepts_connection:
                    result = handler(arguments, context, connection)
                else:
                    result = handler(arguments, context)

            duration_ms = int((time.time() - started) * 1000)
            display_text = _display_text(result)
            artifact_paths = _artifact_paths(result)

            update_step(
                step.id if step else "",
                status="completed",
                content=display_text,
                output_data=result,
                duration_ms=duration_ms,
                metadata={"bucket": bucket, "artifact_count": len(artifact_paths)},
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=True,
                trace_id=trace_id,
                duration_ms=duration_ms,
                display_text=display_text,
            )
            for path in artifact_paths:
                if os.path.exists(path):
                    record_artifact(step.id if step else None, path=path, metadata={"tool_slug": slug})

            ToolService._log_tool_execution(
                db,
                user.id,
                slug,
                connection,
                trace_id,
                arguments,
                result,
                True,
                None,
                toolkit_name,
            )
            runtime.finish_tool(bucket, True)
            return display_text

        except Exception as exc:
            duration_ms = int((time.time() - started) * 1000)
            error_type = type(exc).__name__
            error_message = traceback.format_exc()
            update_step(
                step.id if step else "",
                status="failed",
                content=f"Tool execution error: {error_message}",
                output_data={},
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=error_message,
                metadata={"bucket": bucket, "approval_id": getattr(approval, "id", None)},
            )
            emit_event(
                "tool_result",
                step_id=step.id if step else None,
                tool_slug=slug,
                ok=False,
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=error_type,
                message=error_message,
            )
            ToolService._log_tool_execution(
                db,
                user.id,
                slug,
                connection,
                trace_id,
                arguments,
                None,
                False,
                error_message,
                toolkit_name,
            )
            runtime.finish_tool(bucket, False, error_type=error_type)
            return f"Tool execution error: {error_message}"
