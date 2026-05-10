"""Lazy registry for agent mode handlers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable


RunModeFn = Callable[[str, str, list[str], object, object, object], str]
StreamModeFn = Callable[[str, str, list[str], object, object, object], AsyncIterator[str]]
ModeLoader = Callable[[], "AgentModeHandler"]


@dataclass(frozen=True)
class AgentModeHandler:
    name: str
    run: RunModeFn
    stream: StreamModeFn


_MODE_LOADERS: dict[str, ModeLoader] = {}
_MODE_CACHE: dict[str, AgentModeHandler] = {}


def register_agent_mode(name: str, loader: ModeLoader) -> None:
    _MODE_LOADERS[name] = loader


def list_agent_modes() -> tuple[str, ...]:
    return tuple(sorted(_MODE_LOADERS))


def get_agent_mode(name: str) -> AgentModeHandler:
    handler = _MODE_CACHE.get(name)
    if handler is not None:
        return handler
    loader = _MODE_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown agent mode: {name!r}. Available modes: {sorted(_MODE_LOADERS)}")
    handler = loader()
    _MODE_CACHE[name] = handler
    return handler


def _load_standard_mode() -> AgentModeHandler:
    from .standard import run_standard_agent, stream_standard_agent

    return AgentModeHandler(name="standard", run=run_standard_agent, stream=stream_standard_agent)


def _load_progressive_mode() -> AgentModeHandler:
    from ..progressive_graph import run_progressive_agent, stream_progressive_agent

    return AgentModeHandler(name="progressive", run=run_progressive_agent, stream=stream_progressive_agent)


def _load_category_mode() -> AgentModeHandler:
    from ..category_graph import run_category_agent, stream_category_agent

    return AgentModeHandler(name="category_scoped", run=run_category_agent, stream=stream_category_agent)


def _load_artifact_progressive_mode() -> AgentModeHandler:
    from .artifact_progressive import run_artifact_progressive_agent, stream_artifact_progressive_agent

    return AgentModeHandler(
        name="artifact_progressive",
        run=run_artifact_progressive_agent,
        stream=stream_artifact_progressive_agent,
    )


register_agent_mode("standard", _load_standard_mode)
register_agent_mode("progressive", _load_progressive_mode)
register_agent_mode("category_scoped", _load_category_mode)
register_agent_mode("artifact_progressive", _load_artifact_progressive_mode)
