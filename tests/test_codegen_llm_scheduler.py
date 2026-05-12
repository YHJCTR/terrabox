from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest import mock

from terrabox.managers.resource_allocator import ResourceLease
from terrabox.toolkits import codegen


def _lease(port: int = 9123) -> ResourceLease:
    return ResourceLease(
        service="codegen-llm",
        container_name=f"terrabox-codegen-llm-{port}",
        image="terrabox/agent-llm:latest",
        host="127.0.0.1",
        port=port,
        internal_port=8000,
        gpu_devices="2",
        created_at="2026-05-12T00:00:00",
    )


def test_codegen_llm_service_uses_resource_lease_and_dynamic_port(monkeypatch):
    lease = _lease(9127)
    commands: list[list[str]] = []
    checks = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")

    def fake_health(port: int) -> bool:
        checks["n"] += 1
        return port == 9127

    monkeypatch.setattr(codegen, "acquire_docker_lease", lambda **_kwargs: lease)
    monkeypatch.setattr(codegen, "remove_container_if_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(codegen, "labels_for_lease", lambda lease: ["--label", f"terrabox.service={lease.service}"])
    monkeypatch.setattr(codegen, "_codegen_llm_is_healthy", fake_health)
    monkeypatch.setattr(codegen.subprocess, "run", fake_run)
    monkeypatch.setattr(codegen.time, "sleep", lambda _seconds: None)

    api_base, started_lease = codegen._start_codegen_llm_service(
        SimpleNamespace(
            local_llm_model_path="/models/qwen",
            local_llm_docker_image="terrabox/agent-llm:latest",
            local_llm_port=9100,
            local_llm_tensor_parallel=1,
            local_llm_gpu_devices="0",
            local_llm_max_model_len=24576,
        )
    )

    assert api_base == "http://127.0.0.1:9127/v1"
    assert started_lease == lease
    run_cmd = next(cmd for cmd in commands if cmd[:3] == ["docker", "run", "-d"])
    assert "9127:8000" in run_cmd
    assert "CUDA_VISIBLE_DEVICES=2" in run_cmd
    assert "terrabox-codegen-llm-9127" in run_cmd
    assert "--model" in run_cmd
    assert "/model" in run_cmd


def test_codegen_handler_stops_codegen_llm_after_success(monkeypatch):
    lease = _lease(9131)
    stopped: list[str] = []
    calls: list[str] = []

    monkeypatch.setattr(codegen, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(codegen, "_start_codegen_llm_service", lambda _config: ("http://127.0.0.1:9131/v1", lease))
    monkeypatch.setattr(codegen, "_stop_codegen_llm_service", lambda lease: stopped.append(lease.container_name))
    monkeypatch.setattr(codegen, "_generate_code", lambda api_base, description, input_data: calls.append(api_base) or "print(json.dumps({'ok': True}))")
    monkeypatch.setattr(codegen, "_run_in_sandbox", lambda code, input_data: {"success": True, "output": "{\"ok\": true}", "error": ""})

    result = codegen.codegen_generate_and_run_handler(
        {
            "description": "return ok",
            "input_data": json.dumps({"x": 1}),
        },
        {},
    )

    assert result["success"] is True
    assert result["executed_code"] == "print(json.dumps({'ok': True}))"
    assert "executed_code" in result["final_answer_instruction"]
    assert calls == ["http://127.0.0.1:9131/v1"]
    assert stopped == ["terrabox-codegen-llm-9131"]


def test_codegen_handler_stops_codegen_llm_after_retry_exhaustion(monkeypatch):
    lease = _lease(9132)
    stopped: list[str] = []

    monkeypatch.setattr(codegen, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(codegen, "_start_codegen_llm_service", lambda _config: ("http://127.0.0.1:9132/v1", lease))
    monkeypatch.setattr(codegen, "_stop_codegen_llm_service", lambda lease: stopped.append(lease.container_name))
    monkeypatch.setattr(codegen, "_generate_code", lambda api_base, description, input_data: "raise RuntimeError('boom')")
    monkeypatch.setattr(codegen, "_fix_code", lambda api_base, original_code, error, description: "raise RuntimeError('boom')")
    monkeypatch.setattr(codegen, "_run_in_sandbox", lambda code, input_data: {"success": False, "output": "", "error": "boom"})

    result = codegen.codegen_generate_and_run_handler(
        {
            "description": "fail",
            "input_data": json.dumps({"x": 1}),
        },
        {},
    )

    assert result["success"] is False
    assert result["attempts"] == codegen.MAX_RETRIES
    assert stopped == ["terrabox-codegen-llm-9132"]


def test_generate_code_prompt_requires_exact_requested_output_schema(monkeypatch):
    captured = {}

    def fake_call_llm(api_base, messages, no_thinking=False):
        captured["api_base"] = api_base
        captured["messages"] = messages
        captured["no_thinking"] = no_thinking
        return "```python\nresult = {'selected_ids': []}\nprint(json.dumps(result))\n```"

    monkeypatch.setattr(codegen, "_call_llm", fake_call_llm)

    generated = codegen._generate_code(
        "http://127.0.0.1:9133/v1",
        "Output exactly selected_ids, selected_count, mean_selected_ndvi. Do not include std_ndvi.",
        json.dumps({"pixels": []}),
    )

    prompt = captured["messages"][1]["content"]
    assert "Match the requested output schema exactly" in prompt
    assert "Do NOT add extra output fields" in prompt
    assert "selected_ids, selected_count, mean_selected_ndvi" in prompt
    assert captured["api_base"] == "http://127.0.0.1:9133/v1"
    assert captured["no_thinking"] is True
    assert "selected_ids" in generated
