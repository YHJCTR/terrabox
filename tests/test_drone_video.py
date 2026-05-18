from __future__ import annotations

import csv

import pytest

from terrabox.toolkits import drone_video


class MockRegistrar:
    def __init__(self):
        self.tools = {}
        self.handlers = {}

    def toolkit(self, *args, **kwargs):
        pass

    def tool(self, spec, handler):
        self.tools[spec.slug] = spec
        self.handlers[spec.slug] = handler

    def call(self, slug, arguments):
        return self.handlers[slug](arguments, {}, None)


def _registered_drone_tools() -> MockRegistrar:
    registrar = MockRegistrar()
    drone_video.setup(registrar)
    return registrar


def test_drone_video_registers_only_stable_non_perception_tools():
    registrar = _registered_drone_tools()

    assert set(registrar.tools) == {
        "drone_video.extract_frames",
        "drone_video.pixel_to_gps",
        "drone_video.parse_telemetry",
    }


def test_pixel_to_gps_maps_image_center_to_drone_position():
    registrar = _registered_drone_tools()

    result = registrar.call(
        "drone_video.pixel_to_gps",
        {
            "pixel_x": 500,
            "pixel_y": 500,
            "image_width": 1000,
            "image_height": 1000,
            "drone_lat": 30.0,
            "drone_lon": 120.0,
            "drone_alt_m": 100,
            "drone_heading_deg": 0,
            "fov_deg": 84,
        },
    )

    assert result["status"] == "success"
    assert result["lat"] == 30.0
    assert result["lon"] == 120.0
    assert result["gsd_m"] > 0


def test_parse_telemetry_reads_dji_csv_and_writes_json(tmp_path):
    registrar = _registered_drone_tools()
    telemetry_path = tmp_path / "flight.csv"
    output_path = tmp_path / "track.json"

    with telemetry_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latitude",
                "longitude",
                "altitude(m)",
                "compass_heading(degrees)",
                "time(millisecond)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "latitude": 30.0,
                "longitude": 120.0,
                "altitude(m)": 50,
                "compass_heading(degrees)": 12,
                "time(millisecond)": 0,
            }
        )
        writer.writerow(
            {
                "latitude": 30.001,
                "longitude": 120.002,
                "altitude(m)": 55,
                "compass_heading(degrees)": 15,
                "time(millisecond)": 2000,
            }
        )

    result = registrar.call(
        "drone_video.parse_telemetry",
        {
            "telemetry_path": str(telemetry_path),
            "format": "dji_csv",
            "output_path": str(output_path),
        },
    )

    assert result["status"] == "success"
    assert result["track_points"] == 2
    assert result["duration_s"] == 2.0
    assert output_path.exists()


def test_extract_frames_writes_video_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    registrar = _registered_drone_tools()
    video_path = tmp_path / "input.avi"
    output_dir = tmp_path / "frames"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (16, 16),
    )
    assert writer.isOpened()
    for value in (0, 64, 128, 192, 255):
        frame = np.full((16, 16, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    result = registrar.call(
        "drone_video.extract_frames",
        {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "fps": 5,
            "max_frames": 5,
            "format": "png",
        },
    )

    assert result["status"] == "success"
    assert result["frame_count"] == 5
    assert len(result["frame_paths"]) == 5
    assert all((tmp_path / "frames" / f"frame_{idx:06d}.png").exists() for idx in range(5))
