from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from terrabox.agent import session
from conftest import RecordDb


class _FakeLLM:
    def __init__(self, content="btw answer"):
        self.messages = []
        self.content = content

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


def _record(messages, summary_messages=None):
    return SimpleNamespace(
        id="session-1",
        user_id_fk="user-1",
        messages_json=session.serialize_messages(messages),
        summary_json=session.serialize_messages(summary_messages or []),
        updated_at=None,
    )


def test_run_btw_query_uses_session_context_without_persisting(monkeypatch):
    monkeypatch.setattr(session, "get_user_memory_context", lambda *_args, **_kwargs: "")
    original_messages = [
        HumanMessage(content="Compare these images"),
        AIMessage(content="The first answer"),
    ]
    record = _record(
        original_messages,
        [SystemMessage(content="[Terrabox conversation memory]\n## 当前任务状态\n- Comparing images")],
    )
    db = RecordDb(record)
    llm = _FakeLLM()
    before = record.messages_json

    result = session.run_btw_query(
        session_id="session-1",
        question="What does CRS mean here?",
        user=SimpleNamespace(id="user-1"),
        db=db,
        llm=llm,
    )

    assert result["response"] == "btw answer"
    assert result["session_id"] == "session-1"
    assert record.messages_json == before
    assert db.committed is False
    assert any("quick aside" in getattr(msg, "content", "") for msg in llm.messages)
    assert any("Compare these images" in getattr(msg, "content", "") for msg in llm.messages)
    assert getattr(llm.messages[-1], "content", "").startswith("/btw")


def test_run_btw_query_requires_existing_session():
    with pytest.raises(session.BtwSessionNotFound):
        session.run_btw_query(
            session_id="missing",
            question="side question",
            user=SimpleNamespace(id="user-1"),
            db=RecordDb(None),
            llm=_FakeLLM(),
        )


def test_run_btw_query_splits_thinking_from_response(monkeypatch):
    monkeypatch.setattr(session, "get_user_memory_context", lambda *_args, **_kwargs: "")
    record = _record([HumanMessage(content="Context")])
    db = RecordDb(record)
    llm = _FakeLLM("<think>need context</think>final answer")

    result = session.run_btw_query(
        session_id="session-1",
        question="side question",
        user=SimpleNamespace(id="user-1"),
        db=db,
        llm=llm,
    )

    assert result["thinking"] == "need context"
    assert result["response"] == "final answer"
