import json
import urllib.error
from pathlib import Path

from terrabox.agent.llm_provider import (
    ProviderSpec,
    RemoteChatClient,
    _remote_llm_next_workload,
    pace_remote_llm_request,
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


def test_remote_pacing_without_workload_keeps_legacy_timestamp_lock(tmp_path):
    lock_path = tmp_path / "remote_llm_longcat.lock"

    pace_remote_llm_request("longcat", workload="", interval=0.0, lock_path=lock_path)
    assert not lock_path.exists()

    pace_remote_llm_request("longcat", workload="", interval=0.001, lock_path=lock_path)
    float(lock_path.read_text(encoding="utf-8").strip())
    assert not Path(f"{lock_path}.json").exists()


def test_remote_workload_queue_rotates_between_waiting_workloads():
    state = {
        "version": 1,
        "last_grant": {"workload": "tau2", "mono": 1.0, "pid": 1, "host": "h"},
        "order": ["tau2", "experienceevo"],
        "queues": {
            "tau2": [{"request_id": "t1"}],
            "experienceevo": [{"request_id": "e1"}],
        },
    }
    assert _remote_llm_next_workload(state, "tau2") == "experienceevo"

    state["last_grant"]["workload"] = "experienceevo"
    assert _remote_llm_next_workload(state, "experienceevo") == "tau2"


def test_remote_workload_queue_allows_single_active_workload_to_continue():
    state = {
        "version": 1,
        "last_grant": {"workload": "experienceevo", "mono": 1.0, "pid": 1, "host": "h"},
        "order": ["tau2"],
        "queues": {"tau2": [{"request_id": "t1"}, {"request_id": "t2"}]},
    }
    assert _remote_llm_next_workload(state, "tau2") == "tau2"
