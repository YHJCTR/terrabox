from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from terrabox.agent import session
from terrabox.agent.modes import standard
from terrabox.agent.prompt_blocks import PromptBlock, PromptRenderer
from terrabox.agent.session import _REACT_SYSTEM_PROMPT


class _FakeQuery:
    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return None


class _FakeDb:
    def add(self, _record):
        pass

    def flush(self):
        pass

    def query(self, *_args, **_kwargs):
        return _FakeQuery()


def test_prompt_renderer_renders_messages_and_describes_blocks():
    blocks = [
        PromptBlock(
            name="react_system",
            content="static instructions",
            source="standard",
            cache_policy="static",
            version="v1",
        ),
        PromptBlock(
            name="rag_context",
            content="dynamic context",
            source="rag",
            cache_policy="dynamic",
        ),
    ]

    messages = PromptRenderer.render_system_messages(blocks)
    description = PromptRenderer.describe(blocks)

    assert [m.content for m in messages] == ["static instructions", "dynamic context"]
    assert all(isinstance(m, SystemMessage) for m in messages)
    assert description == [
        {
            "name": "react_system",
            "role": "system",
            "source": "standard",
            "cache_policy": "static",
            "version": "v1",
            "content_chars": len("static instructions"),
            "metadata": {},
        },
        {
            "name": "rag_context",
            "role": "system",
            "source": "rag",
            "cache_policy": "dynamic",
            "version": "unversioned",
            "content_chars": len("dynamic context"),
            "metadata": {},
        },
    ]


def test_prepare_history_can_return_prompt_blocks_for_dynamic_context(monkeypatch):
    monkeypatch.setattr(session, "get_user_memory_context", lambda *_args, **_kwargs: "likes tif")
    monkeypatch.setattr(session, "_summarize_history", lambda *_args, **_kwargs: "")

    record, history, blocks = session.prepare_history(
        session_id="session-1",
        user_message="compare",
        image_paths=["/tmp/before.png"],
        user=SimpleNamespace(id="user-1"),
        db=_FakeDb(),
        llm=None,
        include_prompt_blocks=True,
    )

    names = [block.name for block in blocks]
    assert record.summary_json == "[]"
    assert names == ["uploaded_files_metadata", "user_memory"]
    assert any(isinstance(msg, HumanMessage) and "[Uploaded files metadata]" in msg.content for msg in history)
    assert any(isinstance(msg, SystemMessage) and "[Terrabox user memories]" in msg.content for msg in history)


def test_standard_prompt_returns_described_prompt_blocks(monkeypatch):
    monkeypatch.setattr(
        standard,
        "_classify_intent",
        lambda *_args, **_kwargs: SimpleNamespace(value="tool_call"),
    )
    monkeypatch.setattr(standard, "_rewrite_query", lambda message, *_args: message)
    monkeypatch.setattr(standard, "_inject_rag_context", lambda *_args: "retrieved context")

    _intent, messages, blocks = standard._prepare_standard_prompt(
        history=[HumanMessage(content="question")],
        user_message="question",
        user=SimpleNamespace(id="user-1"),
        db=SimpleNamespace(query=lambda *_args, **_kwargs: _FakeQuery()),
        llm=object(),
        tools=[object()],
        include_prompt_blocks=True,
    )

    assert messages[0].content == _REACT_SYSTEM_PROMPT
    assert [block.name for block in blocks] == ["react_system", "rag_context"]
    assert blocks[0].cache_policy == "static"
    assert blocks[1].cache_policy == "dynamic"
