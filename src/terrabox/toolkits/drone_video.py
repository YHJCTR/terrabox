"""
Drone Video Toolkit
-------------------
无人机视频帧处理与地理坐标转换工具集。

Key features:
1. 视频帧提取（按帧率或时间段）
2. 帧间时序差分（变化/运动检测）
3. 关键帧自动筛选（场景变化或均匀采样）
4. 像素坐标转 GPS 地理坐标（基于无人机姿态）
5. 无人机遥测数据解析（DJI CSV / GPX / MAVLink JSON）
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


def _lazy_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("Missing numpy. Install: pip install numpy")


# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------

def extract_frames_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    从视频文件中按指定帧率提取帧序列，保存为图像文件。
    支持 mp4 / mov / avi 等格式。
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


def temporal_diff_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    计算两帧图像的像素差分图，用于检测场景变化或运动区域。
    输出二值差分掩膜（超过阈值的像素为 255，其余为 0）。
    """
    cv2, np = _lazy_cv2()

    frame_a_path = arguments.get("frame_a_path")
    frame_b_path = arguments.get("frame_b_path")
    output_path = arguments.get("output_path")
    threshold = int(arguments.get("threshold", 30))
    blur_radius = int(arguments.get("blur_radius", 5))

    if not (frame_a_path and frame_b_path and output_path):
        return {"status": "error", "message": "Missing required: frame_a_path, frame_b_path, output_path"}
    for p in (frame_a_path, frame_b_path):
        if not os.path.exists(p):
            return {"status": "error", "message": f"File not found: {p}"}

    img_a = cv2.imread(frame_a_path)
    img_b = cv2.imread(frame_b_path)
    if img_a is None:
        return {"status": "error", "message": f"Cannot read image: {frame_a_path}"}
    if img_b is None:
        return {"status": "error", "message": f"Cannot read image: {frame_b_path}"}

    # Resize to same shape if needed
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    # Grayscale diff
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_a, gray_b)

    # Optional blur to reduce noise
    if blur_radius > 1:
        ksize = blur_radius | 1  # ensure odd
        diff = cv2.GaussianBlur(diff, (ksize, ksize), 0)

    # Threshold to binary mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, mask)

    total_pixels = mask.size
    changed_pixels = int(np.sum(mask > 0))
    changed_ratio = round(changed_pixels / total_pixels, 4)
    max_diff = int(diff.max())

    return {
        "status": "success",
        "output_path": output_path,
        "changed_pixels": changed_pixels,
        "changed_ratio": changed_ratio,
        "max_diff": max_diff,
    }


def key_frame_select_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    从帧序列中筛选关键帧。
    scene_change 模式：基于相邻帧直方图差异选取变化最大的帧。
    uniform 模式：均匀间隔采样。
    """
    cv2, np = _lazy_cv2()

    frame_paths = arguments.get("frame_paths", [])
    method = arguments.get("method", "scene_change")
    max_frames = int(arguments.get("max_frames", 10))
    threshold = float(arguments.get("threshold", 0.3))

    if not frame_paths:
        return {"status": "error", "message": "frame_paths is empty"}
    if method not in ("scene_change", "uniform"):
        return {"status": "error", "message": f"Unknown method '{method}'. Use 'scene_change' or 'uniform'."}

    # Filter existing files
    valid_paths = [p for p in frame_paths if os.path.exists(p)]
    if not valid_paths:
        return {"status": "error", "message": "None of the frame_paths exist on disk"}

    total = len(valid_paths)

    if method == "uniform":
        if max_frames >= total:
            selected_indices = list(range(total))
        else:
            step = total / max_frames
            selected_indices = [int(i * step) for i in range(max_frames)]
    else:
        # Scene-change: compute histogram distance between consecutive frames
        def hist_diff(img1, img2):
            h1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
            h2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
            h1 = cv2.normalize(h1, h1).flatten()
            h2 = cv2.normalize(h2, h2).flatten()
            return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

        scores = [0.0]  # First frame always kept
        prev_img = cv2.cvtColor(cv2.imread(valid_paths[0]), cv2.COLOR_BGR2GRAY)
        for i in range(1, total):
            curr_img = cv2.cvtColor(cv2.imread(valid_paths[i]), cv2.COLOR_BGR2GRAY)
            if prev_img is not None and curr_img is not None:
                scores.append(hist_diff(prev_img, curr_img))
            else:
                scores.append(0.0)
            prev_img = curr_img

        # Always include first frame; then pick top (max_frames-1) by score
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = set([0])
        for idx, score in indexed:
            if score >= threshold:
                top_indices.add(idx)
            if len(top_indices) >= max_frames:
                break
        # Fallback: if too few, add by score
        for idx, score in indexed:
            if len(top_indices) >= max_frames:
                break
            top_indices.add(idx)
        selected_indices = sorted(top_indices)

    selected_paths = [valid_paths[i] for i in selected_indices]

    return {
        "status": "success",
        "selected_paths": selected_paths,
        "selected_indices": selected_indices,
        "total_input": total,
    }


def pixel_to_gps_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    基于无人机飞行参数（GPS坐标、飞行高度、朝向）将图像中的像素坐标转换为地理坐标（WGS84）。
    使用针孔相机投影模型，适用于垂直向下拍摄的无人机相机（云台俯角 -90°）。

    公式：
      GSD (m/px) = 2 * alt_m * tan(fov_rad / 2) / image_width
      dx_m = (pixel_x - cx) * GSD
      dy_m = (pixel_y - cy) * GSD
      旋转 heading_deg 后加到无人机坐标
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
    # GSD (ground sampling distance, meters per pixel) — horizontal
    gsd_m = 2.0 * drone_alt_m * math.tan(fov_rad / 2.0) / image_width

    # Offset from image center (in pixels)
    cx = image_width / 2.0
    cy = image_height / 2.0
    dx_px = pixel_x - cx
    dy_px = pixel_y - cy  # positive = down in image = toward camera front

    # Convert to meters
    dx_m = dx_px * gsd_m   # positive East (before heading rotation)
    dy_m = -dy_px * gsd_m  # positive North (image y down → negate)

    # Rotate by heading
    heading_rad = math.radians(heading_deg)
    north_m = dx_m * (-math.sin(heading_rad)) + dy_m * math.cos(heading_rad)
    east_m = dx_m * math.cos(heading_rad) + dy_m * math.sin(heading_rad)

    # Convert meters to degrees
    # 1 degree latitude ≈ 111320 m
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(drone_lat)))

    target_lat = drone_lat + north_m * lat_per_m
    target_lon = drone_lon + east_m * lon_per_m

    # Accuracy estimate: ±2 pixels → ±2 * gsd
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
    解析无人机飞行遥测文件，提取标准化 GPS 轨迹（时间/经纬度/高度/朝向）。

    支持格式：
    - dji_csv: DJI 飞行记录导出 CSV（DJI Assistant / DatCon）
    - gpx: 标准 GPX 格式（XML）
    - mavlink_json: MAVLink 导出 JSON（pymavlink）
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

        with open(telemetry_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                lat = _first(row, LAT_KEYS)
                lon = _first(row, LON_KEYS)
                if lat is None or lon is None:
                    continue
                alt = _first(row, ALT_KEYS, 0.0)
                heading = _first(row, HEAD_KEYS, 0.0)
                t_raw = _first(row, TIME_KEYS, i * 1000)
                # Convert ms → s if large
                t_s = t_raw / 1000.0 if t_raw > 10000 else t_raw
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
        # Expect list of dicts with "time_boot_ms", "lat" (int×1e7), "lon" (int×1e7), "alt" (mm)
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
        description="无人机视频帧处理与地理坐标转换工具集：视频帧提取、时序差分、关键帧筛选、像素→GPS 坐标转换、遥测数据解析。",
        version="1.0.0",
    )

    # 1. extract_frames
    registrar.tool(
        ToolSpec(
            slug="drone_video.extract_frames",
            name="Extract Video Frames",
            description=(
                "从无人机视频文件中按指定帧率提取帧序列，保存为图像文件（JPG/PNG）。"
                "支持 mp4/mov/avi 格式，可指定起止时间和最大帧数。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "输入视频文件路径（mp4/mov/avi）。",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "帧图像输出目录，不存在时自动创建。",
                    },
                    "fps": {
                        "type": "number",
                        "description": "提取帧率（帧/秒），默认 1.0。值越大提取越密集。",
                        "default": 1.0,
                    },
                    "start_time": {
                        "type": "number",
                        "description": "起始时间戳（秒），默认从头开始。",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "结束时间戳（秒），默认到视频末尾。",
                    },
                    "max_frames": {
                        "type": "integer",
                        "description": "最大提取帧数，默认 100，防止输出过多文件。",
                        "default": 100,
                    },
                    "format": {
                        "type": "string",
                        "description": "输出图像格式，'jpg' 或 'png'，默认 'jpg'。",
                        "default": "jpg",
                    },
                },
                "required": ["video_path", "output_dir"],
            },
            requires_connection=False,
        ),
        extract_frames_handler,
    )

    # 2. temporal_diff
    registrar.tool(
        ToolSpec(
            slug="drone_video.temporal_diff",
            name="Temporal Frame Difference",
            description=(
                "计算两帧图像的像素差分掩膜，用于检测场景变化或运动区域。"
                "输出二值图像（变化区域=255，无变化=0）及变化统计。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "frame_a_path": {
                        "type": "string",
                        "description": "前一帧图像路径。",
                    },
                    "frame_b_path": {
                        "type": "string",
                        "description": "后一帧图像路径。",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "差分掩膜输出路径（PNG/JPG）。",
                    },
                    "threshold": {
                        "type": "integer",
                        "description": "差分阈值（0-255），超过此值视为变化，默认 30。",
                        "default": 30,
                    },
                    "blur_radius": {
                        "type": "integer",
                        "description": "降噪高斯模糊核大小（像素），默认 5，设为 0 或 1 关闭。",
                        "default": 5,
                    },
                },
                "required": ["frame_a_path", "frame_b_path", "output_path"],
            },
            requires_connection=False,
        ),
        temporal_diff_handler,
    )

    # 3. key_frame_select
    registrar.tool(
        ToolSpec(
            slug="drone_video.key_frame_select",
            name="Key Frame Selection",
            description=(
                "从帧序列中自动筛选关键帧，去除冗余帧。"
                "scene_change 模式基于直方图差异选取场景变化最大的帧；"
                "uniform 模式均匀间隔采样。"
                "返回筛选后的帧路径列表。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "frame_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入帧路径列表（通常来自 extract_frames）。",
                    },
                    "method": {
                        "type": "string",
                        "description": "筛选方法：'scene_change'（基于场景变化）或 'uniform'（均匀采样），默认 'scene_change'。",
                        "default": "scene_change",
                    },
                    "max_frames": {
                        "type": "integer",
                        "description": "最大保留帧数，默认 10。",
                        "default": 10,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "场景变化阈值（0-1，Bhattacharyya 距离），仅 scene_change 模式有效，默认 0.3。",
                        "default": 0.3,
                    },
                },
                "required": ["frame_paths"],
            },
            requires_connection=False,
        ),
        key_frame_select_handler,
    )

    # 4. pixel_to_gps
    registrar.tool(
        ToolSpec(
            slug="drone_video.pixel_to_gps",
            name="Pixel to GPS Coordinate",
            description=(
                "基于无人机飞行参数（GPS坐标、飞行高度、朝向角、相机视角）将图像像素坐标转换为 WGS84 地理坐标。"
                "适用于垂直向下拍摄的无人机相机（云台俯角约 -90°），使用针孔相机投影模型。"
                "返回目标点经纬度、地面采样距离（GSD）和精度估计。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "number",
                        "description": "目标点像素列坐标（从图像左边缘起）。",
                    },
                    "pixel_y": {
                        "type": "number",
                        "description": "目标点像素行坐标（从图像上边缘起）。",
                    },
                    "image_width": {
                        "type": "integer",
                        "description": "图像宽度（像素）。",
                    },
                    "image_height": {
                        "type": "integer",
                        "description": "图像高度（像素）。",
                    },
                    "drone_lat": {
                        "type": "number",
                        "description": "无人机当前纬度（度，WGS84）。",
                    },
                    "drone_lon": {
                        "type": "number",
                        "description": "无人机当前经度（度，WGS84）。",
                    },
                    "drone_alt_m": {
                        "type": "number",
                        "description": "无人机飞行高度（米，相对地面）。",
                    },
                    "drone_heading_deg": {
                        "type": "number",
                        "description": "无人机朝向角（度，正北为 0，顺时针），默认 0。",
                        "default": 0.0,
                    },
                    "fov_deg": {
                        "type": "number",
                        "description": "相机水平视角（度），DJI Mini 系列约 84°，Mavic 3 约 84°，默认 84。",
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

    # 5. parse_telemetry
    registrar.tool(
        ToolSpec(
            slug="drone_video.parse_telemetry",
            name="Parse Drone Telemetry",
            description=(
                "解析无人机飞行遥测文件，提取标准化 GPS 轨迹（时间/经纬度/高度/朝向）。"
                "支持 DJI CSV（DJI Assistant 2 或 DatCon 导出）、GPX 格式、MAVLink JSON 格式。"
                "返回轨迹点列表及飞行包围框，可选将标准化结果保存为 JSON 文件。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "telemetry_path": {
                        "type": "string",
                        "description": "遥测文件路径（.csv / .gpx / .json）。",
                    },
                    "format": {
                        "type": "string",
                        "description": "文件格式：'dji_csv'（DJI 飞行日志 CSV）/ 'gpx'（标准 GPX XML）/ 'mavlink_json'（MAVLink 导出 JSON），默认 'dji_csv'。",
                        "default": "dji_csv",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "（可选）标准化轨迹输出路径（JSON），不提供则不写文件。",
                    },
                },
                "required": ["telemetry_path"],
            },
            requires_connection=False,
        ),
        parse_telemetry_handler,
    )
