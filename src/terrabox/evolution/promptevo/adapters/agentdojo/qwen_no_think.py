"""AgentDojo module-load patch that makes Qwen3 tool calls explicitly no-think."""
from __future__ import annotations

from collections.abc import Sequence
import fcntl
import os
from pathlib import Path
import time

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionReasoningEffort, ChatCompletionToolParam
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.llms import openai_llm


DEFAULT_MAX_TOKENS = int(os.getenv("TERRABOX_AGENTDOJO_QWEN_MAX_TOKENS", "2048"))
REQUEST_PROFILE = os.getenv("TERRABOX_AGENTDOJO_REQUEST_PROFILE", "qwen").strip().lower()
OPENAI_MAX_RETRIES = int(os.getenv("TERRABOX_AGENTDOJO_OPENAI_MAX_RETRIES", "6"))
OPENAI_RETRY_MAX_WAIT = int(os.getenv("TERRABOX_AGENTDOJO_OPENAI_RETRY_MAX_WAIT", "60"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("TERRABOX_AGENTDOJO_OPENAI_TIMEOUT_SECONDS", os.getenv("TERRABOX_REMOTE_LLM_TIMEOUT_SECONDS", "300")))


def _float_env(names: Sequence[str], default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                return max(0.0, float(raw))
            except ValueError:
                return default
    return default


def _default_api_min_interval() -> float:
    if REQUEST_PROFILE in {"", "qwen", "local", "vllm", "vllm_parsed"}:
        return 0.0
    if REQUEST_PROFILE == "longcat":
        return 10.0
    return 1.0


API_MIN_INTERVAL_SECONDS = _float_env(
    [
        "TERRABOX_AGENTDOJO_API_MIN_INTERVAL_SECONDS",
        f"TERRABOX_AGENTDOJO_{REQUEST_PROFILE.upper()}_MIN_INTERVAL_SECONDS",
        "TERRABOX_AGENTDOJO_LONGCAT_MIN_INTERVAL_SECONDS",
        f"TERRABOX_{REQUEST_PROFILE.upper()}_MIN_INTERVAL_SECONDS",
        "TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS",
    ],
    _default_api_min_interval(),
)
API_RATE_LOCK = Path(
    os.getenv(
        "TERRABOX_AGENTDOJO_API_RATE_LOCK",
        os.getenv(
            f"TERRABOX_AGENTDOJO_{REQUEST_PROFILE.upper()}_RATE_LOCK",
            os.getenv(
                "TERRABOX_AGENTDOJO_LONGCAT_RATE_LOCK",
                os.getenv(
                    "TERRABOX_REMOTE_LLM_RATE_LOCK",
                    str(
                        Path.cwd()
                        / "tmp"
                        / "service_locks"
                        / (
                            "remote_llm_longcat.lock"
                            if REQUEST_PROFILE == "longcat"
                            else f"agentdojo_{REQUEST_PROFILE}_api_rate.lock"
                        )
                    ),
                ),
            ),
        ),
    )
)


def _pace_external_api_request() -> None:
    """Cross-process pacing for external API rollouts.

    AgentDojo can issue several chat calls per sample even with max_workers=1,
    and upstream OpenAI-compatible SDK calls bypass Terrabox's RemoteChatClient.
    Serialize external-provider requests across API lanes and leave a
    configurable gap between calls. Set the interval to 0 only for a deliberate
    high-concurrency experiment.
    """
    if REQUEST_PROFILE in {"", "qwen", "local", "vllm", "vllm_parsed"} or API_MIN_INTERVAL_SECONDS <= 0:
        return
    API_RATE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with API_RATE_LOCK.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        try:
            last = float(raw) if raw else 0.0
        except ValueError:
            last = 0.0
        now = time.monotonic()
        wait_s = API_MIN_INTERVAL_SECONDS - (now - last)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        handle.seek(0)
        handle.truncate()
        handle.write(f"{now:.6f}")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    wait=wait_random_exponential(multiplier=1, max=OPENAI_RETRY_MAX_WAIT),
    stop=stop_after_attempt(max(1, OPENAI_MAX_RETRIES)),
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
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    if REQUEST_PROFILE == "longcat":
        extra_body = {"thinking": {"type": "disabled"}}
        # LongCat's OpenAI-compatible endpoint does not need OpenAI reasoning
        # effort hints. Passing them through can turn a normal rollout into a
        # provider-specific 400, so keep the upstream value only for local Qwen.
        reasoning_effort = None
    try:
        _pace_external_api_request()
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or NOT_GIVEN,
            tool_choice="auto" if tools else NOT_GIVEN,
            temperature=temperature if temperature is not None else NOT_GIVEN,
            reasoning_effort=reasoning_effort or NOT_GIVEN,
            max_tokens=DEFAULT_MAX_TOKENS,
            extra_body=extra_body,
            timeout=OPENAI_TIMEOUT_SECONDS,
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
