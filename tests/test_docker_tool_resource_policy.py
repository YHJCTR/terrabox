import importlib


def test_tool_gpu_pool_limits_non_agent_services(monkeypatch):
    allocator = importlib.import_module("terrabox.managers.resource_allocator")

    monkeypatch.setenv("TERRABOX_TOOL_GPU_DEVICES", "2")
    monkeypatch.setenv("TERRABOX_TOOL_MAX_GPUS", "1")

    assert allocator.tool_gpu_override_for_service("vlm", 1) == "2"


def test_tool_gpu_pool_does_not_affect_agent_llm(monkeypatch):
    allocator = importlib.import_module("terrabox.managers.resource_allocator")

    monkeypatch.setenv("TERRABOX_TOOL_GPU_DEVICES", "2")
    monkeypatch.setenv("TERRABOX_TOOL_MAX_GPUS", "1")

    assert allocator.tool_gpu_override_for_service("agent-llm", 1) is None


def test_tool_gpu_pool_rejects_services_that_need_too_many_gpus(monkeypatch):
    allocator = importlib.import_module("terrabox.managers.resource_allocator")

    monkeypatch.setenv("TERRABOX_TOOL_GPU_DEVICES", "2")
    monkeypatch.setenv("TERRABOX_TOOL_MAX_GPUS", "1")

    try:
        allocator.tool_gpu_override_for_service("vlm", 2)
    except RuntimeError as exc:
        assert "requires 2 GPU" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_vlm_env_overrides_allow_single_gpu_experiment_mode(monkeypatch):
    manager_mod = importlib.import_module("terrabox.managers.docker.vllm_manager")
    manager = manager_mod.VLLMDockerManager
    original = (
        manager.TENSOR_PARALLEL_SIZE,
        manager.MAX_MODEL_LEN,
        manager.GPU_MEMORY_UTILIZATION,
        manager.MIN_IMAGE_MODEL_LEN,
        getattr(manager, "LIMIT_MM_PER_PROMPT", None),
        getattr(manager, "SKIP_MM_PROFILING", None),
        getattr(manager, "MAX_NUM_SEQS", None),
    )
    try:
        monkeypatch.setenv("VLM_TENSOR_PARALLEL_SIZE", "1")
        monkeypatch.setenv("VLM_MAX_MODEL_LEN", "8192")
        monkeypatch.setenv("VLM_GPU_MEMORY_UTILIZATION", "0.55")
        monkeypatch.setenv("VLM_MIN_IMAGE_MODEL_LEN", "8192")
        monkeypatch.setenv("VLM_LIMIT_MM_PER_PROMPT", '{"image":1,"video":0}')
        monkeypatch.setenv("VLM_SKIP_MM_PROFILING", "true")
        monkeypatch.setenv("VLM_MAX_NUM_SEQS", "1")

        manager._apply_env_overrides()

        assert manager.TENSOR_PARALLEL_SIZE == "1"
        assert manager.MAX_MODEL_LEN == 8192
        assert manager.MIN_IMAGE_MODEL_LEN == 8192
        assert manager.GPU_MEMORY_UTILIZATION == 0.55
        assert manager.LIMIT_MM_PER_PROMPT == '{"image":1,"video":0}'
        assert manager.SKIP_MM_PROFILING is True
        assert manager.MAX_NUM_SEQS == 1
        assert manager._extra_engine_args() == [
            "--limit-mm-per-prompt", '{"image":1,"video":0}',
            "--skip-mm-profiling",
            "--max-num-seqs", "1",
        ]
    finally:
        (
            manager.TENSOR_PARALLEL_SIZE,
            manager.MAX_MODEL_LEN,
            manager.GPU_MEMORY_UTILIZATION,
            manager.MIN_IMAGE_MODEL_LEN,
            manager.LIMIT_MM_PER_PROMPT,
            manager.SKIP_MM_PROFILING,
            manager.MAX_NUM_SEQS,
        ) = original
