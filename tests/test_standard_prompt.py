from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from terrabox.agent.modes import standard
from terrabox.agent.session import _REACT_SYSTEM_PROMPT


class _FakeQuery:
    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return object()


class _FakeDb:
    def query(self, *_args, **_kwargs):
        return _FakeQuery()


def test_react_system_prompt_stays_first_and_rag_is_separate(monkeypatch):
    monkeypatch.setattr(
        standard,
        "_classify_intent",
        lambda *_args, **_kwargs: SimpleNamespace(value="tool_call"),
    )
    monkeypatch.setattr(standard, "_rewrite_query", lambda message, *_args: message)
    monkeypatch.setattr(standard, "_inject_rag_context", lambda *_args: "retrieved context")

    _intent, messages = standard._prepare_standard_prompt(
        history=[SystemMessage(content="[Terrabox user memories]\nremember me"), HumanMessage(content="question")],
        user_message="question",
        user=SimpleNamespace(id="user-1"),
        db=_FakeDb(),
        llm=object(),
        tools=[object()],
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == _REACT_SYSTEM_PROMPT
    assert "Relevant Knowledge Base Context" not in messages[0].content
    assert isinstance(messages[1], SystemMessage)
    assert messages[1].content == "## Relevant Knowledge Base Context\n\nretrieved context"
