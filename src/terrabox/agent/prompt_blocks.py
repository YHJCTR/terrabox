"""Named prompt blocks for observable standard prompt assembly."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(frozen=True)
class PromptBlock:
    """A named piece of prompt context with enough metadata for tracing."""

    name: str
    content: str
    role: str = "system"
    source: str = "runtime"
    cache_policy: str = "dynamic"
    version: str = "unversioned"
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptRenderer:
    """Render prompt blocks into LangChain messages and trace descriptors."""

    @staticmethod
    def render_system_messages(blocks: list[PromptBlock]) -> list[SystemMessage]:
        return [SystemMessage(content=block.content) for block in blocks if block.role == "system"]

    @staticmethod
    def render_human_context(blocks: list[PromptBlock]) -> str:
        parts: list[str] = []
        for block in blocks:
            if block.role != "human_context":
                continue
            title = block.metadata.get("title") or block.name.replace("_", " ").title()
            parts.append(f"[{title}]\n{block.content}")
        return "\n\n".join(parts)

    @staticmethod
    def render_human_message(user_message: str, context_blocks: list[PromptBlock]) -> HumanMessage:
        context = PromptRenderer.render_human_context(context_blocks)
        content = user_message
        if context:
            content += f"\n\n{context}"
        return HumanMessage(content=content)

    @staticmethod
    def describe(blocks: list[PromptBlock]) -> list[dict[str, Any]]:
        return [
            {
                "name": block.name,
                "role": block.role,
                "source": block.source,
                "cache_policy": block.cache_policy,
                "version": block.version,
                "content_chars": len(block.content),
                "metadata": dict(block.metadata),
            }
            for block in blocks
        ]
