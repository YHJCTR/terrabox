"""
Drone Video Toolkit
-------------------
Drone video frame processing and geospatial coordinate conversion tools.

Key features:
1. Video frame extraction by frame rate or time window
2. Pixel-to-GPS coordinate conversion based on drone pose
3. Drone telemetry parsing for DJI CSV, GPX, and MAVLink JSON
"""

import csv
import json
import math
import os
from typing import Any, Dict

from ..core.registry import ToolSpec


# ------------------------------------------------------------------------------
# Lazy Imports & Helpers
# ------------------------------------------------------------------------------

def _lazy_cv2():
    try:
        import cv2
        import numpy as np
        return cv2, np
    except ImportError:
        raise ImportError(
            "Missing opencv-python. Install: pip install opencv-python numpy"
        )


# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------

def extract_frames_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Extract frames from a video at the requested frame rate and save them as images.
    Supports formats such as mp4, mov, and avi.
    """
    cv2, np = _lazy_cv2()

    video_path = arguments.get("video_path")
    output_dir = arguments.get("output_dir")
    fps_target = float(arguments.get("fps", 1.0))
    start_time = arguments.get("start_time")
    end_time = arguments.get("end_time")
    max_frames = int(arguments.get("max_frames", 100))
    fmt = arguments.get("format", "jpg").lower().lstrip(".")

    if not video_path or not output_dir:
        return {"status": "error", "message": "Missing required: video_path, output_dir"}
    if not os.path.exists(video_path):
        return {"status": "error", "message": f"Video not found: {video_path}"}
    if fmt not in ("jpg", "jpeg", "png"):
        return {"status": "error", "message": f"Unsupported format '{fmt}'. Use jpg or png."}

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": f"Cannot open video: {video_path}"}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / native_fps

    start_frame = int((start_time or 0) * native_fps)
    end_frame = int((end_time or duration_s) * native_fps)
    end_frame = min(end_frame, total_frames)

    # Compute frame interval
    frame_interval = max(1, int(native_fps / fps_target))

    frame_paths = []
    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_idx < end_frame and len(frame_paths) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            fname = os.path.join(output_dir, f"frame_{frame_idx:06d}.{fmt}")
            cv2.imwrite(fname, frame)
            frame_paths.append(fname)
        frame_idx += 1

    cap.release()

    actual_duration = (end_frame - start_frame) / native_fps
    fps_extracted = len(frame_paths) / actual_duration if actual_duration > 0 else 0

    return {
        "status": "success",
        "frame_paths": frame_paths,
        "frame_count": len(frame_paths),
        "fps_extracted": round(fps_extracted, 3),
        "duration_s": round(actual_duration, 2),
        "native_fps": round(native_fps, 2),
    }


def pixel_to_gps_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Convert an image pixel coordinate to a WGS84 geographic coordinate using
    drone flight parameters such as GPS position, altitude, and heading.
    Uses a pinhole camera projection model for nadir-looking drone cameras
    with an approximate gimbal pitch of -90 degrees.

    Formula:
      GSD (m/px) = 2 * alt_m * tan(fov_rad / 2) / image_width
      dx_m = (pixel_x - cx) * GSD
      dy_m = (pixel_y - cy) * GSD
      Rotate by heading_deg, then add the offset to the drone coordinate.
    """
    pixel_x = arguments.get("pixel_x")
    pixel_y = arguments.get("pixel_y")
    image_width = arguments.get("image_width")
    image_height = arguments.get("image_height")
    drone_lat = arguments.get("drone_lat")
    drone_lon = arguments.get("drone_lon")
    drone_alt_m = arguments.get("drone_alt_m")
    heading_deg = float(arguments.get("drone_heading_deg", 0.0))
    fov_deg = float(arguments.get("fov_deg", 84.0))

    for name, val in [("pixel_x", pixel_x), ("pixel_y", pixel_y),
                      ("image_width", image_width), ("image_height", image_height),
                      ("drone_lat", drone_lat), ("drone_lon", drone_lon),
                      ("drone_alt_m", drone_alt_m)]:
        if val is None:
            return {"status": "error", "message": f"Missing required parameter: {name}"}

    pixel_x = float(pixel_x)
    pixel_y = float(pixel_y)
    image_width = int(image_width)
    image_height = int(image_height)
    drone_lat = float(drone_lat)
    drone_lon = float(drone_lon)
    drone_alt_m = float(drone_alt_m)

    if drone_alt_m <= 0:
        return {"status": "error", "message": "drone_alt_m must be positive"}

    fov_rad = math.radians(fov_deg)
    # GSD (ground sampling distance, meters per pixel), horizontally.
    gsd_m = 2.0 * drone_alt_m * math.tan(fov_rad / 2.0) / image_width

    # Offset from image center (in pixels)
    cx = image_width / 2.0
    cy = image_height / 2.0
    dx_px = pixel_x - cx
    dy_px = pixel_y - cy  # positive means down in the image, toward the camera front

    # Convert to meters
    dx_m = dx_px * gsd_m   # positive east before heading rotation
    dy_m = -dy_px * gsd_m  # positive north; negate because image y points down

    # Rotate by heading
    heading_rad = math.radians(heading_deg)
    north_m = dx_m * (-math.sin(heading_rad)) + dy_m * math.cos(heading_rad)
    east_m = dx_m * math.cos(heading_rad) + dy_m * math.sin(heading_rad)

    # Convert meters to degrees
    # 1 degree latitude is approximately 111320 meters.
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(drone_lat)))

    target_lat = drone_lat + north_m * lat_per_m
    target_lon = drone_lon + east_m * lon_per_m

    # Accuracy estimate: +/-2 pixels, converted to meters.
    accuracy_m = round(2.0 * gsd_m, 2)

    return {
        "status": "success",
        "lat": round(target_lat, 7),
        "lon": round(target_lon, 7),
        "gsd_m": round(gsd_m, 4),
        "accuracy_m": accuracy_m,
    }


def parse_telemetry_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Parse a drone flight telemetry file and extract a normalized GPS track
    with time, latitude, longitude, altitude, and heading.

    Supported formats:
    - dji_csv: exported DJI flight log CSV from DJI Assistant or DatCon
    - gpx: standard GPX XML
    - mavlink_json: MAVLink JSON exported by pymavlink or a compatible tool
    """
    telemetry_path = arguments.get("telemetry_path")
    fmt = arguments.get("format", "dji_csv").lower()
    output_path = arguments.get("output_path")

    if not telemetry_path:
        return {"status": "error", "message": "Missing required: telemetry_path"}
    if not os.path.exists(telemetry_path):
        return {"status": "error", "message": f"File not found: {telemetry_path}"}
    if fmt not in ("dji_csv", "gpx", "mavlink_json"):
        return {"status": "error", "message": f"Unknown format '{fmt}'. Use dji_csv/gpx/mavlink_json."}

    records = []

    if fmt == "dji_csv":
        # DJI CSV columns (varies by firmware, try common names)
        LAT_KEYS = ("latitude", "OSD.latitude", "lat")
        LON_KEYS = ("longitude", "OSD.longitude", "lon", "lng")
        ALT_KEYS = ("altitude(m)", "OSD.height [m]", "altitude", "alt_m", "height")
        HEAD_KEYS = ("compass_heading(degrees)", "OSD.yaw", "heading", "compass")
        TIME_KEYS = ("time(millisecond)", "OSD.flyTime [s]", "time_ms", "timestamp")

        def _first(row, keys, default=None):
            for k in keys:
                if k in row and row[k] not in ("", None):
                    try:
                        return float(row[k])
                    except (ValueError, TypeError):
                            continue
            return default

        def _first_with_key(row, keys, default=None):
            for k in keys:
                if k in row and row[k] not in ("", None):
                    try:
                        return k, float(row[k])
                    except (ValueError, TypeError):
                        continue
            return None, default

        with open(telemetry_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                lat = _first(row, LAT_KEYS)
                lon = _first(row, LON_KEYS)
                if lat is None or lon is None:
                    continue
                alt = _first(row, ALT_KEYS, 0.0)
                heading = _first(row, HEAD_KEYS, 0.0)
                time_key, t_raw = _first_with_key(row, TIME_KEYS, i * 1000)
                if time_key in {"time(millisecond)", "time_ms"} or t_raw > 10000:
                    t_s = t_raw / 1000.0
                else:
                    t_s = t_raw
                records.append({
                    "time_s": round(t_s, 3),
                    "lat": lat,
                    "lon": lon,
                    "alt_m": alt,
                    "heading_deg": heading,
                })

    elif fmt == "gpx":
        import xml.etree.ElementTree as ET
        tree = ET.parse(telemetry_path)
        root = tree.getroot()
        ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
        t0 = None
        for trkpt in root.findall(".//gpx:trkpt", ns):
            lat = float(trkpt.get("lat", 0))
            lon = float(trkpt.get("lon", 0))
            ele_el = trkpt.find("gpx:ele", ns)
            alt = float(ele_el.text) if ele_el is not None else 0.0
            time_el = trkpt.find("gpx:time", ns)
            if time_el is not None:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
                    if t0 is None:
                        t0 = dt
                    t_s = (dt - t0).total_seconds()
                except Exception:
                    t_s = len(records)
            else:
                t_s = float(len(records))
            records.append({"time_s": round(t_s, 3), "lat": lat, "lon": lon,
                            "alt_m": alt, "heading_deg": 0.0})

    elif fmt == "mavlink_json":
        with open(telemetry_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Expect list of dicts with "time_boot_ms", "lat" (int x 1e7), "lon" (int x 1e7), "alt" (mm).
        msgs = raw if isinstance(raw, list) else raw.get("messages", [])
        t0 = None
        for msg in msgs:
            lat_raw = msg.get("lat")
            lon_raw = msg.get("lon")
            if lat_raw is None or lon_raw is None:
                continue
            lat = lat_raw / 1e7 if abs(lat_raw) > 360 else lat_raw
            lon = lon_raw / 1e7 if abs(lon_raw) > 360 else lon_raw
            alt_raw = msg.get("alt", 0)
            alt = alt_raw / 1000.0 if abs(alt_raw) > 1000 else alt_raw
            t_ms = msg.get("time_boot_ms", len(records) * 1000)
            if t0 is None:
                t0 = t_ms
            records.append({
                "time_s": round((t_ms - t0) / 1000.0, 3),
                "lat": lat,
                "lon": lon,
                "alt_m": round(alt, 2),
                "heading_deg": float(msg.get("yaw", 0)),
            })

    if not records:
        return {"status": "error", "message": "No valid GPS records found in telemetry file"}

    duration_s = records[-1]["time_s"] - records[0]["time_s"]
    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"records": records, "bbox": bbox, "duration_s": duration_s}, f, indent=2)

    return {
        "status": "success",
        "track_points": len(records),
        "duration_s": round(duration_s, 2),
        "bbox": [round(v, 6) for v in bbox],
        "records": records,
    }


# ------------------------------------------------------------------------------
# Toolkit Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    """Register the drone_video toolkit and its tools."""
    registrar.toolkit(
        name="drone_video",
        description=(
            "Drone video frame processing and geospatial coordinate conversion tools: "
            "frame extraction, pixel-to-GPS conversion, and telemetry parsing."
        ),
        version="1.0.0",
    )

    # 1. extract_frames
    registrar.tool(
        ToolSpec(
            slug="drone_video.extract_frames",
            name="Extract Video Frames",
            description=(
                "Extract frames from a drone video at the requested frame rate and save them "
                "as JPG or PNG images. Supports mp4, mov, and avi files, with optional start "
                "time, end time, and maximum frame count limits."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Input video file path, such as mp4, mov, or avi.",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory for extracted frame images; created if missing.",
                    },
                    "fps": {
                        "type": "number",
                        "description": "Extraction frame rate in frames per second. Defaults to 1.0; larger values sample more densely.",
                        "default": 1.0,
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start timestamp in seconds. Defaults to the beginning of the video.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End timestamp in seconds. Defaults to the end of the video.",
                    },
                    "max_frames": {
                        "type": "integer",
                        "description": "Maximum number of frames to extract. Defaults to 100 to avoid excessive output files.",
                        "default": 100,
                    },
                    "format": {
                        "type": "string",
                        "description": "Output image format: 'jpg' or 'png'. Defaults to 'jpg'.",
                        "default": "jpg",
                    },
                },
                "required": ["video_path", "output_dir"],
            },
            requires_connection=False,
        ),
        extract_frames_handler,
    )

    # 2. pixel_to_gps
    registrar.tool(
        ToolSpec(
            slug="drone_video.pixel_to_gps",
            name="Pixel to GPS Coordinate",
            description=(
                "Convert an image pixel coordinate to a WGS84 geographic coordinate using "
                "drone GPS position, altitude, heading, and camera field of view. Intended "
                "for nadir-looking drone cameras with an approximate -90 degree gimbal pitch. "
                "Returns the target coordinate, ground sampling distance, and an accuracy estimate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "number",
                        "description": "Target pixel column coordinate, measured from the left edge of the image.",
                    },
                    "pixel_y": {
                        "type": "number",
                        "description": "Target pixel row coordinate, measured from the top edge of the image.",
                    },
                    "image_width": {
                        "type": "integer",
                        "description": "Image width in pixels.",
                    },
                    "image_height": {
                        "type": "integer",
                        "description": "Image height in pixels.",
                    },
                    "drone_lat": {
                        "type": "number",
                        "description": "Current drone latitude in WGS84 degrees.",
                    },
                    "drone_lon": {
                        "type": "number",
                        "description": "Current drone longitude in WGS84 degrees.",
                    },
                    "drone_alt_m": {
                        "type": "number",
                        "description": "Drone altitude above ground level, in meters.",
                    },
                    "drone_heading_deg": {
                        "type": "number",
                        "description": "Drone heading in degrees, where north is 0 and values increase clockwise. Defaults to 0.",
                        "default": 0.0,
                    },
                    "fov_deg": {
                        "type": "number",
                        "description": "Horizontal camera field of view in degrees. DJI Mini and Mavic 3 are about 84 degrees. Defaults to 84.",
                        "default": 84.0,
                    },
                },
                "required": [
                    "pixel_x", "pixel_y", "image_width", "image_height",
                    "drone_lat", "drone_lon", "drone_alt_m",
                ],
            },
            requires_connection=False,
        ),
        pixel_to_gps_handler,
    )

    # 3. parse_telemetry
    registrar.tool(
        ToolSpec(
            slug="drone_video.parse_telemetry",
            name="Parse Drone Telemetry",
            description=(
                "Parse a drone flight telemetry file and extract a normalized GPS track with "
                "time, coordinates, altitude, and heading. Supports DJI CSV exported by DJI "
                "Assistant 2 or DatCon, GPX XML, and MAVLink JSON. Returns track points and "
                "a flight bounding box, with optional JSON output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "telemetry_path": {
                        "type": "string",
                        "description": "Telemetry file path, such as .csv, .gpx, or .json.",
                    },
                    "format": {
                        "type": "string",
                        "description": "File format: 'dji_csv' for DJI flight log CSV, 'gpx' for standard GPX XML, or 'mavlink_json' for exported MAVLink JSON. Defaults to 'dji_csv'.",
                        "default": "dji_csv",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output path for the normalized track JSON. No file is written when omitted.",
                    },
                },
                "required": ["telemetry_path"],
            },
            requires_connection=False,
        ),
        parse_telemetry_handler,
    )
