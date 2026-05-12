from __future__ import annotations

from terrabox.agent.events import drain_events_as_sse


def test_drain_events_as_sse_serializes_all_pending_events():
    events = [
        {"type": "tool_start", "tool_slug": "geo_raster.raster_diff", "args": {"path_a": "a.png"}},
        {"type": "tool_result", "tool_slug": "geo_raster.raster_diff", "ok": True, "duration_ms": 42},
    ]

    frames = drain_events_as_sse(lambda: events)

    assert len(frames) == 2
    assert frames[0].startswith("data: ")
    assert '"type": "tool_start"' in frames[0]
    assert '"tool_slug": "geo_raster.raster_diff"' in frames[0]
    assert '"duration_ms": 42' in frames[1]
