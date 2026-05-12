from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from terrabox.agent import session


class _FakeQuery:
    def __init__(self, record):
        self.record = record
        self.filters = {}

    def filter_by(self, **kwargs):
        self.filters.update(kwargs)
        return self

    def first(self):
        if self.record and self.filters.get("id") == self.record.id:
            return self.record
        return None


class _FakeDb:
    def __init__(self, record):
        self.record = record
        self.committed = False

    def query(self, *_args, **_kwargs):
        return _FakeQuery(self.record)

    def commit(self):
        self.committed = True


def _record(messages, summary_messages=None):
    return SimpleNamespace(
        id="session-1",
        user_id_fk="user-1",
        messages_json=session.serialize_messages(messages),
        summary_json=session.serialize_messages(summary_messages or []),
        updated_at=None,
    )


def test_get_context_status_reports_message_summary_and_token_budget():
    record = _record(
        [
            HumanMessage(content="hello" * 20),
            AIMessage(content="answer" * 20),
        ],
        [SystemMessage(content="[Terrabox conversation memory]\n## 当前任务状态\n- ok")],
    )

    status = session.get_context_status("session-1", _FakeDb(record), "user-1", max_model_len=100)

    assert status["session_id"] == "session-1"
    assert status["exists"] is True
    assert status["raw_message_count"] == 2
    assert status["summary_message_count"] == 1
    assert status["has_summary"] is True
    assert status["estimated_context_tokens"] > 0
    assert status["remaining_context_tokens"] == 100 - status["estimated_context_tokens"]
    assert status["can_compact"] is False


def test_compact_session_keeps_recent_messages_and_stores_summary(monkeypatch):
    monkeypatch.setattr(session, "_history_limits", lambda: (20, 15, 2))
    messages = [
        HumanMessage(content="user old 1"),
        AIMessage(content="assistant old 1"),
        HumanMessage(content="user old 2"),
        AIMessage(content="assistant old 2"),
        HumanMessage(content="recent user"),
        AIMessage(content="recent assistant"),
    ]
    record = _record(messages)
    db = _FakeDb(record)

    status = session.compact_session("session-1", db, "user-1", llm=None)

    remaining = session.deserialize_messages(record.messages_json)
    summaries = session.deserialize_messages(record.summary_json)

    assert status["compacted"] is True
    assert len(remaining) == 2
    assert remaining[0].content == "recent user"
    assert len(summaries) == 1
    assert "[Terrabox conversation memory]" in summaries[0].content
    assert "user old 1" in summaries[0].content
    assert db.committed is True
