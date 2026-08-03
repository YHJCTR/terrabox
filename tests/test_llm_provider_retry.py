import urllib.error

from terrabox.agent.llm_provider import (
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
