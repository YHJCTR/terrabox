import importlib.util
import json
import os
import sys
import types

import pytest

from terrabox.toolkits import geo_perception


def test_resolve_image_path_returns_container_visible_absolute_path(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        resolved = geo_perception._resolve_image_path({"image": "sample.jpg"})
    finally:
        os.chdir(cwd)

    assert resolved == str(image_path)


def test_remotesam_handler_accepts_scalar_json_result(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")
    captured = {}

    def fake_call_service(manager, url, payload, timeout=120):
        captured["payload"] = payload
        captured["timeout"] = timeout
        return "residential"

    monkeypatch.setattr(geo_perception, "_call_service", fake_call_service)
    result = geo_perception.remotesam_handler(
        {"image": str(image_path), "task_type": "multi_class_cls", "classnames": ["residential"]},
        {"artifact_dir": str(tmp_path / "artifacts")},
        None,
    )

    assert result["status"] == "success"
    assert result["result"] == "residential"
    assert result["result_summary"] == {"value": "residential"}
    assert captured["payload"]["image_path"] == str(image_path)
    assert captured["timeout"] >= 300


def test_remotesam_handler_extracts_detection_bboxes(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")

    def fake_call_service(manager, url, payload, timeout=120):
        return {"building": [[1, 2, 3, 4, 0.9]], "water": []}

    monkeypatch.setattr(geo_perception, "_call_service", fake_call_service)
    result = geo_perception.remotesam_handler(
        {"image": str(image_path), "task_type": "detection", "classnames": ["building", "water"]},
        {"artifact_dir": str(tmp_path / "artifacts")},
        None,
    )

    assert result["status"] == "success"
    assert result["bboxes"] == [{"label": "building", "bbox": [1, 2, 3, 4], "score": 0.9}]


def test_remotesam_handler_normalizes_detect_alias_and_sentence_classname(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")
    captured = {}

    def fake_call_service(manager, url, payload, timeout=120):
        captured["url"] = url
        captured["payload"] = payload
        return {"bridge": [[1, 2, 3, 4, 0.9]]}

    monkeypatch.setattr(geo_perception, "_call_service", fake_call_service)
    result = geo_perception.remotesam_handler(
        {"image": str(image_path), "task_type": "detect", "sentence": "bridge"},
        {"artifact_dir": str(tmp_path / "artifacts")},
        None,
    )

    assert result["status"] == "success"
    assert captured["url"].endswith("/detection")
    assert captured["payload"]["classnames"] == ["bridge"]
    assert result["bboxes"] == [{"label": "bridge", "bbox": [1, 2, 3, 4], "score": 0.9}]


def test_remotesam_handler_normalizes_text_segmentation_alias(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")
    captured = {}

    def fake_call_service(manager, url, payload, timeout=120):
        captured["url"] = url
        captured["payload"] = payload
        return {"mask": "ok"}

    monkeypatch.setattr(geo_perception, "_call_service", fake_call_service)
    result = geo_perception.remotesam_handler(
        {"image": str(image_path), "task_type": "segmentation", "sentence": "baseball field"},
        {"artifact_dir": str(tmp_path / "artifacts")},
        None,
    )

    assert result["status"] == "success"
    assert captured["url"].endswith("/referring_seg")
    assert captured["payload"]["sentence"] == "baseball field"


def test_ocr_gpu_setting_maps_tool_gpu_to_visible_cuda_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("TERRABOX_TOOL_GPU_DEVICES", "1")

    assert geo_perception._easyocr_gpu_setting() == "cuda:1"


def test_ocr_gpu_setting_can_be_forced_to_cpu(monkeypatch):
    monkeypatch.setenv("TERRABOX_OCR_USE_GPU", "false")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("TERRABOX_TOOL_GPU_DEVICES", "1")

    assert geo_perception._easyocr_gpu_setting() is False


def test_ocr_extract_runs_in_subprocess_with_tool_gpu(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not a real image")
    captured = {}

    class FakeCompleted:
        returncode = 0
        stdout = json.dumps({
            "texts": [{
                "text": "HELLO",
                "confidence": 0.9,
                "bbox": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            }]
        })
        stderr = ""

    def fake_run(cmd, input=None, text=None, capture_output=None, timeout=None, env=None):
        captured["cmd"] = cmd
        captured["payload"] = json.loads(input)
        captured["env"] = env
        return FakeCompleted()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("TERRABOX_TOOL_GPU_DEVICES", "1")
    monkeypatch.setattr(geo_perception.subprocess, "run", fake_run)

    result = geo_perception.ocr_extract_handler({"image": str(image_path), "languages": ["en"]}, {}, None)

    assert result["status"] == "success"
    assert result["texts"][0]["text"] == "HELLO"
    assert result["ocr_subprocess"] is True
    assert captured["payload"]["gpu"] == "cuda:1"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_call_service_returns_structured_tool_oom_and_stops(monkeypatch):
    class FakeManager:
        stopped = False

        @classmethod
        def start_service(cls):
            return None

        @classmethod
        def stop_service(cls):
            cls.stopped = True

    class FakeResponse:
        status_code = 500
        text = '{"error":"CUDA out of memory. Tried to allocate 4.16 GiB."}'

    monkeypatch.setenv("TERRABOX_TOOL_SERVICE_SCOPE", "call")
    monkeypatch.setattr(geo_perception.requests, "post", lambda *args, **kwargs: FakeResponse())

    result = geo_perception._call_service(FakeManager, "http://127.0.0.1:9006/segment", {})

    assert result["status"] == "error"
    assert result["error_type"] == "tool_oom"
    assert result["retryable"] is False
    assert "Do not repeat" in result["recovery_suggestions"][0]
    assert FakeManager.stopped is True


def test_instructsam_handler_resizes_large_input_and_rescales_bboxes(monkeypatch, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (2000, 1000), "white").save(image_path)
    captured = {}

    def fake_call_service(manager, url, payload, timeout=120):
        captured["payload"] = payload
        return {
            "count": 1,
            "objects": [{"bbox": [100, 50, 200, 150]}],
            "detections": [{"bbox": [100, 50, 200, 150]}],
        }

    monkeypatch.setenv("TERRABOX_PERCEPTION_MAX_IMAGE_LONG_SIDE", "1000")
    monkeypatch.setattr(geo_perception, "_call_service", fake_call_service)

    result = geo_perception.instructsam_handler(
        {"image": str(image_path), "text_prompt": "ship"},
        {"artifact_dir": str(tmp_path / "artifacts")},
        None,
    )

    assert captured["payload"]["image_path"] != str(image_path)
    assert result["image_preprocessing"]["resized"] is True
    assert result["objects"][0]["bbox"] == [200.0, 100.0, 400.0, 300.0]
    assert result["detections"][0]["bbox"] == [200.0, 100.0, 400.0, 300.0]


def test_remotesam_server_json_safe_converts_numpy_scalars(monkeypatch):
    class FakeFlask:
        def __init__(self, name):
            self.name = name

        def get(self, path):
            return lambda fn: fn

        def route(self, path, methods=None):
            return lambda fn: fn

        def run(self, *args, **kwargs):
            return None

    fake_flask = types.ModuleType("flask")
    fake_flask.Flask = FakeFlask
    fake_flask.request = types.SimpleNamespace(json={})
    fake_flask.jsonify = lambda value=None, **kwargs: value if value is not None else kwargs
    monkeypatch.setitem(sys.modules, "flask", fake_flask)

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.imread = lambda path: None
    fake_cv2.cvtColor = lambda img, code: img
    fake_cv2.COLOR_BGR2RGB = 1
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    fake_model_mod = types.ModuleType("tasks.code.model")
    fake_model_mod.init_demo_model = lambda checkpoint, device: object()
    fake_model_mod.RemoteSAM = lambda base_model, device, use_EPOC=True: object()
    monkeypatch.setitem(sys.modules, "tasks", types.ModuleType("tasks"))
    monkeypatch.setitem(sys.modules, "tasks.code", types.ModuleType("tasks.code"))
    monkeypatch.setitem(sys.modules, "tasks.code.model", fake_model_mod)

    spec = importlib.util.spec_from_file_location("remotesam_server_for_test", "docker/remotesam/server.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    np = pytest.importorskip("numpy")
    payload = {"box": [np.int32(1), np.float32(0.5)], "mask": np.array([1, 2], dtype=np.int32)}

    assert module._json_safe(payload) == {"box": [1, 0.5], "mask": [1, 2]}
