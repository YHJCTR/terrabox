"""AgentDojo module-load patch that makes Qwen3 tool calls explicitly no-think."""
from __future__ import annotations

from collections.abc import Sequence

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionReasoningEffort, ChatCompletionToolParam
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.llms import openai_llm


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
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature if temperature is not None else NOT_GIVEN,
        reasoning_effort=reasoning_effort or NOT_GIVEN,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


openai_llm.chat_completion_request = _qwen_no_think_request
