from types import SimpleNamespace

from terrabox.toolkits import geo_perception


class _FakeManager:
    API_URL = "http://127.0.0.1:9999"

    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start_service(self):
        self.started += 1

    def stop_service(self):
        self.stopped += 1


class _FakeResponse:
    status_code = 200
    text = "{}"

    @staticmethod
    def json():
        return {"success": True}


def test_call_service_stops_manager_when_tool_scope_is_call(monkeypatch):
    manager = _FakeManager()
    monkeypatch.setenv("TERRABOX_TOOL_SERVICE_SCOPE", "call")
    monkeypatch.setattr(geo_perception.requests, "post", lambda *args, **kwargs: _FakeResponse())

    result = geo_perception._call_service(manager, "http://127.0.0.1:9999/run", {})

    assert result == {"success": True}
    assert manager.started == 1
    assert manager.stopped == 1


def test_call_service_keeps_manager_running_by_default(monkeypatch):
    manager = _FakeManager()
    monkeypatch.delenv("TERRABOX_TOOL_SERVICE_SCOPE", raising=False)
    monkeypatch.setattr(geo_perception.requests, "post", lambda *args, **kwargs: _FakeResponse())

    result = geo_perception._call_service(manager, "http://127.0.0.1:9999/run", {})

    assert result == {"success": True}
    assert manager.started == 1
    assert manager.stopped == 0

