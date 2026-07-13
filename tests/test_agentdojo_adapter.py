import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path


def _write_result(
    root: Path,
    *,
    suite: str = "workspace",
    user_task: str = "user_task_0",
    attack: str | None = None,
    injection: str | None = None,
    utility: bool = True,
    security: bool = True,
) -> Path:
    path = root / "vllm_parsed" / suite / user_task / (attack or "none") / f"{injection or 'none'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "suite_name": suite,
        "pipeline_name": "vllm_parsed",
        "user_task_id": user_task,
        "injection_task_id": injection,
        "attack_type": attack,
        "messages": [
            {"role": "system", "content": [{"type": "text", "content": "Use tools carefully."}]},
            {"role": "user", "content": [{"type": "text", "content": "Find the file."}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "content": "I will search."}],
                "tool_calls": [{"id": "call-1", "function": "search_files", "args": {"query": "report"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [{"type": "text", "content": "report.pdf"}],
                "error": None,
            },
            {"role": "assistant", "content": [{"type": "text", "content": "Found report.pdf."}], "tool_calls": []},
        ],
        "error": None,
        "duration": 1.25,
        "utility": utility,
        "security": security,
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    return path


def test_agentdojo_metrics_distinguish_clean_and_attacked_none_strings(tmp_path):
    from terrabox.evolution.promptevo.adapters.agentdojo import AgentDojoMetricProvider

    _write_result(tmp_path, user_task="user_task_0", attack=None, utility=True, security=True)
    _write_result(tmp_path, user_task="injection_task_1", attack=None, utility=False, security=True)
    _write_result(
        tmp_path,
        user_task="user_task_1",
        attack="important_instructions",
        injection="injection_task_0",
        utility=True,
        security=False,
    )
    provider = AgentDojoMetricProvider(results_path_fn=lambda _exp: str(tmp_path))
    tasks = provider.per_task("sample")
    summary = provider.aggregate("sample")

    assert set(tasks) == {
        "workspace/user_task_0/none/none",
        "workspace/injection_task_1/none/none",
        "workspace/user_task_1/important_instructions/injection_task_0",
    }
    assert summary["n_clean"] == 1
    assert summary["n_injection_utility"] == 1
    assert summary["n_attacked"] == 1
    assert summary["clean_utility_rate"] == 1.0
    assert summary["injection_task_utility_rate"] == 0.0
    assert summary["attacked_utility_rate"] == 1.0
    assert summary["attacked_security_rate"] == 0.0
    assert summary["attack_success_rate"] == 1.0
    assert summary["balanced_score"] == 2 / 3


def test_agentdojo_trace_recovers_tool_name_from_tool_call_id(tmp_path):
    from terrabox.evolution.promptevo.adapters.agentdojo import AgentDojoTrajectorySource

    _write_result(tmp_path)
    trace = next(iter(AgentDojoTrajectorySource(results_path_fn=lambda _exp: str(tmp_path)).traces("sample")))
    assistant_call = next(step for step in trace.steps if step.role == "assistant" and step.tool)
    tool_result = next(step for step in trace.steps if step.role == "tool")

    assert trace.query == "Find the file."
    assert trace.final_answer == "Found report.pdf."
    assert assistant_call.tool == "search_files"
    assert tool_result.tool == "search_files"


def test_agentdojo_run_config_is_resumable_and_matches_upstream_cli():
    from terrabox.evolution.promptevo.adapters.agentdojo import AgentDojoRunConfig
    from terrabox.evolution.promptevo.adapters.agentdojo.pipeline import NO_THINK_PATCH

    config = AgentDojoRunConfig(
        suite="banking",
        model="vllm_parsed",
        attack="important_instructions",
        modules_to_load=[NO_THINK_PATCH],
    )
    args = config.cli_args("/tmp/results", "system prompt")

    assert args[:2] == ["-m", "agentdojo.scripts.benchmark"]
    assert "--force-rerun" not in args
    assert "--tool-output-format" not in args
    assert args[args.index("--suite") + 1] == "banking"
    assert args[args.index("--attack") + 1] == "important_instructions"
    assert args[args.index("--module-to-load") + 1] == NO_THINK_PATCH


def test_agentdojo_pipeline_vllm_command_pins_one_gpu_and_native_tool_parser():
    from terrabox.evolution.promptevo.adapters.agentdojo.pipeline import build_vllm_command

    command = build_vllm_command("base", "workspace", 2, 9200, "/models/qwen3")

    assert command[command.index("--gpus") + 1] == "device=2"
    assert command[command.index("-p") + 1] == "9200:8000"
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-auto-tool-choice" in command


def test_agentdojo_pipeline_shards_attacks_without_duplicate_injection_pairs():
    from terrabox.evolution.promptevo.adapters.agentdojo.pipeline import build_jobs

    jobs = build_jobs(
        {
            "workspace": {
                "user_tasks": ["user_task_0", "user_task_1"],
                "injection_tasks": ["injection_task_0", "injection_task_1"],
            }
        },
        "important_instructions",
    )

    assert {job["name"] for job in jobs} == {
        "workspace_clean",
        "workspace_attack_injection_task_0",
        "workspace_attack_injection_task_1",
    }
    assert sum(job["expected_results"] for job in jobs) == 8
    assert [job["injection_tasks"] for job in jobs if job["attack"]] == [
        ["injection_task_0"],
        ["injection_task_1"],
    ]


def test_agentdojo_qwen_patch_sends_no_think_on_every_request(monkeypatch):
    from terrabox.evolution.promptevo.adapters.agentdojo import pipeline

    agentdojo = types.ModuleType("agentdojo")
    agent_pipeline = types.ModuleType("agentdojo.agent_pipeline")
    llms = types.ModuleType("agentdojo.agent_pipeline.llms")
    openai_llm = types.ModuleType("agentdojo.agent_pipeline.llms.openai_llm")
    openai_llm.chat_completion_request = object()
    llms.openai_llm = openai_llm
    for name, module in {
        "agentdojo": agentdojo,
        "agentdojo.agent_pipeline": agent_pipeline,
        "agentdojo.agent_pipeline.llms": llms,
        "agentdojo.agent_pipeline.llms.openai_llm": openai_llm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    patch_path = Path(pipeline.__file__).with_name("qwen_no_think.py")
    spec = importlib.util.spec_from_file_location("agentdojo_qwen_patch_test", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    module._qwen_no_think_request(client, "model", [], [], None)

    assert openai_llm.chat_completion_request is module._qwen_no_think_request
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_agentdojo_runner_uses_absolute_terrabox_pythonpath(tmp_path, monkeypatch):
    from terrabox.evolution.promptevo.adapters.agentdojo import core

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(core.subprocess, "run", fake_run)
    agentdojo_root = tmp_path / "agentdojo"
    (agentdojo_root / "src" / "agentdojo" / "data").mkdir(parents=True)
    runner = core.AgentDojoRolloutRunner(
        agentdojo_root=str(agentdojo_root),
        output_dir=str(tmp_path / "experiments"),
        check_dependencies=False,
    )
    runner.run(
        "System prompt",
        experiment="sample",
        run_config=core.AgentDojoRunConfig(suite="workspace", model="vllm_parsed"),
    )

    python_paths = captured["env"]["PYTHONPATH"].split(os.pathsep)
    terrabox_src = str(Path(core.__file__).resolve().parents[5])
    assert python_paths[0] == str(agentdojo_root / "src")
    assert terrabox_src in python_paths


def test_metric_viewer_discovers_adapter_owned_agentdojo_group(tmp_path, monkeypatch):
    from metricViewer.adapters import AgentDojoSceneAdapter

    group = tmp_path / "promptevo_agentdojo_base_qwen3_20260713"
    _write_result(group / "workspace_clean" / "runs")
    (group / "active_system_message.txt").write_text("Use tools carefully.\n", encoding="utf-8")

    adapter = AgentDojoSceneAdapter()
    monkeypatch.setattr(adapter, "root", tmp_path)
    monkeypatch.setattr(adapter, "upstream_root", tmp_path / "missing_upstream")
    refs = adapter.discover()

    assert [ref.name for ref in refs] == [group.name]
    assert refs[0].kind == "base"
    assert refs[0].prompt_path == group / "active_system_message.txt"


def test_agentdojo_job_does_not_skip_incomplete_complete_status(tmp_path):
    from terrabox.evolution.promptevo.adapters.agentdojo import pipeline

    output_dir = tmp_path / "experiments"
    adapter_dir = output_dir / "group" / "workspace_clean"
    (adapter_dir / "runs").mkdir(parents=True)
    (adapter_dir / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    _write_result(adapter_dir / "runs", user_task="user_task_0")

    class Runner:
        def __init__(self):
            self.output_dir = str(output_dir)
            self.called = 0

        def run(self, *args, **kwargs):
            self.called += 1
            _write_result(adapter_dir / "runs", user_task="user_task_1")
            (adapter_dir / "run_status.json").write_text(
                json.dumps({"status": "complete", "returncode": 0}), encoding="utf-8"
            )
            return str(adapter_dir)

    runner = Runner()
    result = pipeline._run_job(
        runner,
        "group",
        {
            "name": "workspace_clean",
            "suite": "workspace",
            "attack": None,
            "injection_tasks": [],
            "expected_results": 2,
        },
        "prompt",
        9200,
    )

    assert result == str(adapter_dir)
    assert runner.called == 1
    status = json.loads((adapter_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["actual_results"] == status["expected_results"] == 2


def test_agentdojo_runner_records_timeout_status(tmp_path, monkeypatch):
    from terrabox.evolution.promptevo.adapters.agentdojo import core

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(core.subprocess, "run", fake_run)
    agentdojo_root = tmp_path / "agentdojo"
    (agentdojo_root / "src" / "agentdojo" / "data").mkdir(parents=True)
    runner = core.AgentDojoRolloutRunner(
        agentdojo_root=str(agentdojo_root),
        output_dir=str(tmp_path / "experiments"),
        check_dependencies=False,
    )

    try:
        runner.run("prompt", experiment="timeout", timeout=5)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("timeout should fail the AgentDojo job")
    status = json.loads(
        (tmp_path / "experiments" / "timeout" / "run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["reason"] == "timeout"


def test_agentdojo_pipeline_reuses_saved_stage_prompts(tmp_path, monkeypatch):
    from terrabox.evolution.promptevo.adapters.agentdojo import pipeline

    output_dir = tmp_path / "experiments"
    stage1_record = output_dir / "stage1"
    stage2_record = output_dir / "stage2"
    stage1_record.mkdir(parents=True)
    stage2_record.mkdir(parents=True)
    stage1_prompt = tmp_path / "stage1.txt"
    stage2_prompt = tmp_path / "stage2.txt"
    stage1_prompt.write_text("stage1", encoding="utf-8")
    stage2_prompt.write_text("stage2", encoding="utf-8")
    (stage1_record / "stage1_proposal.json").write_text(
        json.dumps({"version": "s1", "prompt_path": str(stage1_prompt)}), encoding="utf-8"
    )
    (stage2_record / "stage2_contrastive.json").write_text(
        json.dumps({"version": "s2", "prompt_path": str(stage2_prompt)}), encoding="utf-8"
    )

    assert pipeline.optimize_stage1("base", "s1", "stage1", output_dir=str(output_dir)) == str(stage1_prompt)
    assert pipeline.optimize_stage2(
        "base", "stage1", "s1", "s2", "stage2", output_dir=str(output_dir)
    ) == str(stage2_prompt)


def test_agentdojo_rollout_lock_rejects_duplicate_pipeline(tmp_path, monkeypatch):
    from terrabox.evolution.promptevo.adapters.agentdojo import pipeline

    monkeypatch.setattr(pipeline, "ROLLOUT_LOCK_PATH", tmp_path / "agentdojo.lock")
    first = pipeline._acquire_rollout_lock()
    try:
        try:
            pipeline._acquire_rollout_lock()
        except RuntimeError as exc:
            assert "already running" in str(exc)
        else:
            raise AssertionError("duplicate AgentDojo rollout lock should fail")
    finally:
        pipeline._release_rollout_lock(first)
