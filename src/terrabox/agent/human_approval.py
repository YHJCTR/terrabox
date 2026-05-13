"""Configurable human approval gate for selected agent tool calls."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.services.agent_run_service import AgentRunService
from ..managers.gpu_status import get_gpu_status_snapshot
from .harness import create_approval, current_context, emit_event


@dataclass(frozen=True)
class ApprovalRequirement:
    policy_name: str
    reason: str
    include_gpu_snapshot: bool = True


class ConfigurableToolApprovalPolicy:
    """Rule-based matcher loaded from AgentConfig.human_tool_approval."""

    def __init__(self, *, enabled: bool, rules: list[dict[str, Any]], include_gpu_snapshot: bool = True):
        self.enabled = bool(enabled)
        self.rules = list(rules or [])
        self.include_gpu_snapshot = bool(include_gpu_snapshot)

    @classmethod
    def from_config(cls, config: Any) -> "ConfigurableToolApprovalPolicy":
        raw = getattr(config, "human_tool_approval", None) or {}
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            rules=list(raw.get("rules") or []),
            include_gpu_snapshot=bool(raw.get("include_gpu_snapshot", True)),
        )

    def requirement_for(self, tool_slug: str, arguments: dict[str, Any], *, bucket: str) -> ApprovalRequirement | None:
        if not self.enabled:
            return None
        for rule in self.rules:
            if str(rule.get("action", "require_approval")) != "require_approval":
                continue
            if _rule_matches(rule.get("match") or {}, tool_slug, bucket):
                name = str(rule.get("name") or "human_tool_approval")
                reason = str(rule.get("reason") or f"Human approval required before executing {tool_slug}")
                include_gpu_snapshot = bool(rule.get("include_gpu_snapshot", self.include_gpu_snapshot))
                return ApprovalRequirement(
                    policy_name=name,
                    reason=reason,
                    include_gpu_snapshot=include_gpu_snapshot,
                )
        return None


def _rule_matches(match: dict[str, Any], tool_slug: str, bucket: str) -> bool:
    slugs = {str(item) for item in match.get("slugs") or []}
    if tool_slug in slugs:
        return True
    prefixes = [str(item) for item in match.get("prefixes") or []]
    if any(tool_slug.startswith(prefix) for prefix in prefixes):
        return True
    buckets = {str(item) for item in match.get("buckets") or []}
    return bucket in buckets


def _args_preview(arguments: dict[str, Any], limit: int = 1200) -> str:
    try:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    except Exception:
        text = str(arguments)
    return text if len(text) <= limit else text[:limit] + "..."


def require_human_approval_before_tool(
    *,
    tool_slug: str,
    arguments: dict[str, Any],
    bucket: str,
    step_id: str | None,
    config: Any,
) -> Any | None:
    """Create and wait for a human approval if policy matches this tool call.

    This function runs inside the LangGraph ReAct tool execution path, before the
    underlying Terrabox tool handler is invoked.
    """
    ctx = current_context()
    if ctx is None:
        return None

    requirement = ConfigurableToolApprovalPolicy.from_config(config).requirement_for(
        tool_slug,
        arguments,
        bucket=bucket,
    )
    if requirement is None:
        return None

    gpu_snapshot = get_gpu_status_snapshot() if requirement.include_gpu_snapshot else {"available": False, "gpus": []}
    metadata = {
        "policy_name": requirement.policy_name,
        "tool_args_preview": _args_preview(arguments),
        "gpu_snapshot": gpu_snapshot,
        "langgraph_gate": "approval_gate_node",
    }
    approval = create_approval(
        tool_slug,
        reason=requirement.reason,
        step_id=step_id,
        status="pending",
        metadata=metadata,
    )
    if approval is None:
        return None

    timeout_s = int(getattr(config, "human_tool_approval_timeout_seconds", 300) or 300)
    poll_s = float(getattr(config, "human_tool_approval_poll_seconds", 1.0) or 1.0)
    deadline = time.time() + timeout_s if timeout_s > 0 else None
    while True:
        ctx.db.expire_all()
        latest = AgentRunService.get_approval(ctx.db, approval.id)
        status = getattr(latest, "status", None)
        if status == "approved":
            emit_event(
                "approval_resolved",
                approval_id=approval.id,
                tool_slug=tool_slug,
                status="approved",
            )
            return latest
        if status == "denied":
            emit_event(
                "approval_resolved",
                approval_id=approval.id,
                tool_slug=tool_slug,
                status="denied",
            )
            raise PermissionError(f"Human denied tool execution: {tool_slug}")
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"Human approval timed out for tool: {tool_slug}")
        time.sleep(max(0.1, poll_s))
