import json
import urllib.error

from terrabox.agent.llm_provider import (
    ProviderSpec,
    RemoteChatClient,
    is_retryable_remote_error,
    is_retryable_remote_error_text,
)


def test_remote_payment_and_quota_errors_are_not_retryable():
    assert not is_retryable_remote_error_text("HTTP Error 402: Payment Required")
    assert not is_retryable_remote_error_text("insufficient_quota: please recharge")
    assert not is_retryable_remote_error_text("调用失败：Token 额度不足")
    assert not is_retryable_remote_error_text("invalid api key")

    exc = urllib.error.HTTPError(
        url="https://api.longcat.chat/openai/chat/completions",
        code=402,
        msg="Payment Required",
        hdrs={},
        fp=None,
    )
    assert not is_retryable_remote_error(exc)


def test_remote_transient_errors_remain_retryable():
    assert is_retryable_remote_error_text("rate limit: too many requests")
    assert is_retryable_remote_error_text("gateway timeout 504")

    exc = urllib.error.HTTPError(
        url="https://api.longcat.chat/openai/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None,
    )
    assert is_retryable_remote_error(exc)


def test_longcat_per_request_thinking_override(monkeypatch):
    """Adapters may select thinking mode without changing process-wide env."""
    payloads = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "ok"}}]}'

    def fake_urlopen(request, timeout):
        del timeout
        payloads.append(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr("terrabox.agent.llm_provider.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("terrabox.agent.llm_provider._pace_remote_llm_request", lambda _provider: None)
    client = RemoteChatClient(
        ProviderSpec("longcat", is_local=False, base_url="https://example.test", api_key="test", model="LongCat-2.0")
    )

    assert client.call("hello", enable_thinking=False) == "ok"
    assert client.call("hello", enable_thinking=True) == "ok"
    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert payloads[1]["thinking"] == {"type": "enabled"}
