from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from terrabox.agent.modes import standard
from terrabox.agent.session import _REACT_SYSTEM_PROMPT
from conftest import ObjectDb


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
        db=ObjectDb(),
        llm=object(),
        tools=[object()],
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == _REACT_SYSTEM_PROMPT
    assert "Relevant Knowledge Base Context" not in messages[0].content
    assert isinstance(messages[1], SystemMessage)
    assert messages[1].content == "## Relevant Knowledge Base Context\n\nretrieved context"


def test_append_tool_transparency_to_final_answer():
    messages = [
        HumanMessage(content="run tool"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "example.echo",
                    "args": {"text": "hello"},
                }
            ],
        ),
        ToolMessage(content="hello", tool_call_id="call-1", name="example__echo"),
        AIMessage(content="Here is the answer."),
    ]
    state = {"messages": messages}

    standard._append_tool_transparency_to_final(state)

    assert "Here is the answer." in messages[-1].content
    assert "本次工具调用" in messages[-1].content
    assert "example.echo" in messages[-1].content
    assert "成功" in messages[-1].content


def test_append_tool_transparency_marks_failed_tool_result():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "example.echo",
                    "args": {"text": "hello"},
                }
            ],
        ),
        ToolMessage(content="Tool execution error: boom", tool_call_id="call-1", name="example__echo"),
        AIMessage(content="Could not finish."),
    ]
    state = {"messages": messages}

    standard._append_tool_transparency_to_final(state)

    assert "example.echo" in messages[-1].content
    assert "失败" in messages[-1].content
