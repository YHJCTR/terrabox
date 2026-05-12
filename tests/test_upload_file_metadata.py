from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from terrabox.agent import harness
from terrabox.agent import session
from terrabox.core.utils.uploads import inspect_uploaded_file


class _FakeQuery:
    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return None


class _FakeDb:
    def add(self, _record):
        pass

    def flush(self):
        pass

    def query(self, *_args, **_kwargs):
        return _FakeQuery()


def test_inspect_uploaded_file_reports_basic_png_facts(tmp_path):
    path = tmp_path / "before.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    metadata = inspect_uploaded_file(str(path))

    assert metadata["path"] == str(path)
    assert metadata["extension"] == ".png"
    assert metadata["mime_type"] == "image/png"
    assert metadata["size_bytes"] == 8
    assert metadata["semantic_type"] == "image"
    assert "visual_image" in metadata["capabilities"]
    assert metadata["raster"]["readable"] is False


def test_prepare_history_includes_dynamic_uploaded_file_metadata(monkeypatch):
    monkeypatch.setattr(session, "get_user_memory_context", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(session, "_summarize_history", lambda *_args, **_kwargs: "")

    record, history = session.prepare_history(
        session_id="session-1",
        user_message="compare these files",
        image_paths=["/tmp/before.png", "/tmp/after.png"],
        user=SimpleNamespace(id="user-1"),
        db=_FakeDb(),
        llm=None,
    )

    assert record.summary_json == "[]"
    message = next(msg for msg in history if isinstance(msg, HumanMessage))
    assert "[Uploaded files metadata]" in message.content
    assert '"extension": ".png"' in message.content
    assert '"semantic_type": "image"' in message.content
    assert "geo_perception" not in message.content
    assert "geo_raster" not in message.content


def test_start_run_stores_uploaded_file_metadata(monkeypatch, tmp_path):
    path = tmp_path / "before.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured = {}

    class _Runtime:
        def configure(self, _config):
            pass

        def acquire_run(self):
            return True

        def release_run(self):
            pass

    def fake_create_run(db, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="run-1")

    monkeypatch.setattr(harness, "get_runtime", lambda: _Runtime())
    monkeypatch.setattr(harness.AgentRunService, "create_run", fake_create_run)

    try:
        harness.start_run(
            session_id="session-1",
            user_message="compare",
            image_paths=[str(path)],
            user=SimpleNamespace(id="user-1"),
            db=object(),
            config=SimpleNamespace(agent_mode="standard"),
        )
    finally:
        harness.close_context()

    assert "uploaded_files" in captured["metadata"]
    uploaded = captured["metadata"]["uploaded_files"][0]
    assert uploaded["extension"] == ".png"
    assert uploaded["semantic_type"] == "image"
