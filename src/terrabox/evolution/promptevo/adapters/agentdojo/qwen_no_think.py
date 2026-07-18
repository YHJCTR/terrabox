"""AgentDojo module-load patch that makes Qwen3 tool calls explicitly no-think."""
from __future__ import annotations

from collections.abc import Sequence
import os

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionReasoningEffort, ChatCompletionToolParam
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.llms import openai_llm


DEFAULT_MAX_TOKENS = int(os.getenv("TERRABOX_AGENTDOJO_QWEN_MAX_TOKENS", "2048"))


# Pydantic 2.13 no longer resolves this TypedDict forward reference lazily when
# AgentDojo resumes JSON results produced by an earlier process.
try:
    from agentdojo.benchmark import TaskResults
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.default_suites.v1.tools.types import CalendarEvent
except ModuleNotFoundError as exc:
    if exc.name not in {"agentdojo.benchmark", "agentdojo.functions_runtime"}:
        raise
else:
    TaskResults.model_rebuild(_types_namespace={"FunctionCall": FunctionCall})

    # AgentDojo v1.2.2 can put CalendarEvent objects inside Email attachments.
    # DeepDiff/evaluator paths may hash those attachments under Pydantic 2.x,
    # where BaseModel instances are unhashable by default. Keep the upstream
    # checkout untouched and patch the runtime class only for this adapter.
    def _calendar_event_hash(self: CalendarEvent) -> int:
        return hash(self.model_dump_json())

    CalendarEvent.__hash__ = _calendar_event_hash


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def _qwen_no_think_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None = 0.0,
):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or NOT_GIVEN,
            tool_choice="auto" if tools else NOT_GIVEN,
            temperature=temperature if temperature is not None else NOT_GIVEN,
            reasoning_effort=reasoning_effort or NOT_GIVEN,
            max_tokens=DEFAULT_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except openai.BadRequestError as exc:
        # AgentDojo already treats OpenAI-style context_length_exceeded as a
        # per-sample failure and continues. vLLM returns the same condition as
        # a generic 400 with slightly different wording, so normalize it here.
        text = repr(exc).lower()
        if "maximum context length" in text or "input tokens" in text or "reduce the length" in text:
            try:
                exc.code = "context_length_exceeded"
            except Exception:
                object.__setattr__(exc, "code", "context_length_exceeded")
        raise


openai_llm.chat_completion_request = _qwen_no_think_request
