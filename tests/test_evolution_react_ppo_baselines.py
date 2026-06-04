import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_row(task_id: str = "sample_1") -> dict:
    return {
        "id": task_id,
        "source": "openearth",
        "task_type": "type_distance",
        "question": "Measure the distance between two baseball fields.",
        "images": ["/data/image.jpg"],
        "data_files": [],
        "data_dir": None,
        "ground_truth": "42 meters",
        "messages": [
            {"role": "system", "content": "Tool catalog: geo_perception.instructsam"},
            {"role": "user", "content": "Measure the distance between two baseball fields."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": "Detect baseball fields.",
                        "actions": [
                            {
                                "tool": "geo_perception.instructsam",
                                "function_name": "geo_perception__instructsam",
                                "arguments": {
                                    "image": "/data/image.jpg",
                                    "text_prompt": "baseballfield",
                                },
                            }
                        ],
                    }
                ),
            },
        ],
        "expected_tools": ["geo_perception.instructsam", "ipython.execute"],
        "gold_tool_calls": [
            {
                "tool": "geo_perception.instructsam",
                "function_name": "geo_perception__instructsam",
                "arguments": {"image": "/data/image.jpg", "text_prompt": "baseballfield"},
                "is_executable_under_current_schema": True,
            },
            {
                "tool": "ipython.execute",
                "function_name": "ipython__execute",
                "arguments": {"action": "print(42)"},
                "is_executable_under_current_schema": True,
            },
        ],
        "all_gold_calls_executable": True,
    }


def test_react_task_adapter_does_not_leak_gold_messages(tmp_path):
    from terrabox.evolution.ReAct.data_adapter import sample_to_task, write_task_file
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row()])
    sample = load_sft_samples(data_path)[0]

    task = sample_to_task(sample)

    assert task["task_id"] == "sample_1"
    assert task["question"] == "Measure the distance between two baseball fields."
    assert task["images"] == ["/data/image.jpg"]
    assert task["expected_tools"] == ["geo_perception.instructsam", "ipython.execute"]
    assert "messages" not in task
    assert "gold_tool_calls" not in task
    assert "ground_truth" not in task

    out = tmp_path / "tasks.json"
    written = write_task_file([sample], out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert written == 1
    assert payload["tasks"][0] == task


def test_react_rollout_command_uses_existing_trajectory_script_and_gpu_split(tmp_path):
    from terrabox.evolution.ReAct.runner import build_rollout_command, build_rollout_env

    cmd = build_rollout_command(
        task_file=tmp_path / "tasks.json",
        experiment="react_smoke",
        output_dir=tmp_path / "exp",
        mode="standard",
        port=9100,
        limit=5,
        start_index=2,
        end_index=None,
        max_iterations=12,
        resume=True,
        no_restrict_tools=False,
        include_osm=False,
        include_bing=False,
        include_mock=False,
        include_vlm=False,
        include_changeos=False,
        python_executable="python",
    )
    assert cmd[:3] == ["python", "scripts/run_trajectory_experiment.py", "rollout"]
    assert "--use-docker" in cmd
    assert "--resume" in cmd
    assert "--no-restrict-tools" not in cmd
    assert "--no-skip-osm" not in cmd

    env = build_rollout_env(agent_gpu=0, tool_gpu=1)
    assert env["AGENT_LLM_GPU_DEVICES"] == "0"
    assert env["TERRABOX_TOOL_GPU_DEVICES"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["TERRABOX_TOOL_SERVICE_SCOPE"] == "call"


def test_react_metrics_marks_task_success_as_unverified():
    from terrabox.evolution.ReAct.metrics import aggregate_results

    metrics = aggregate_results(
        [
            {
                "status": "completed",
                "success": True,
                "real_success": True,
                "tools_called": ["ipython.execute"],
                "metrics": {"f1": 1.0, "exact_match": True},
            }
        ]
    )

    assert metrics["success_count"] == 1
    assert metrics["verified_task_success_counts"] == {"None": 1}
    assert metrics["task_success_basis"] == "not_auto_verifiable"
    assert "not semantic task correctness" in metrics["success_metric_basis"]


def test_ppo_adapter_builds_prompt_without_gold_assistant_and_keeps_ground_truth(tmp_path):
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples
    from terrabox.evolution.ppo.data_adapter import sample_to_verl_row

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row()])
    sample = load_sft_samples(data_path)[0]

    row = sample_to_verl_row(sample, tool_catalog=[])

    roles = [message["role"] for message in row["prompt"]]
    assert roles == ["system", "user"]
    prompt_text = json.dumps(row["prompt"], ensure_ascii=False)
    assert "Detect baseball fields" not in prompt_text
    assert "print(42)" not in prompt_text

    ground_truth = json.loads(row["reward_model"]["ground_truth"])
    assert ground_truth["task_id"] == "sample_1"
    assert ground_truth["expected_tools"] == ["geo_perception.instructsam", "ipython.execute"]
    assert ground_truth["gold_tool_calls"][0]["arguments"]["text_prompt"] == "baseballfield"


def test_ppo_reward_scores_correct_tool_sequence_above_empty_or_wrong():
    from terrabox.evolution.ppo.reward_fn import compute_score

    ground_truth = json.dumps(
        {
            "expected_tools": ["geo_perception.instructsam", "ipython.execute"],
            "gold_tool_calls": [],
        }
    )
    correct = json.dumps(
        {
            "thought": "Need perception then calculation.",
            "actions": [
                {"tool": "geo_perception.instructsam", "arguments": {"text_prompt": "baseballfield"}},
                {"tool": "ipython.execute", "arguments": {"action": "print(42)"}},
            ],
        }
    )
    wrong = json.dumps(
        {
            "thought": "Use unrelated tool.",
            "actions": [
                {"tool": "bash.execute", "arguments": {"commands": "echo hi"}},
            ],
        }
    )

    assert compute_score("terrabox", correct, ground_truth) > compute_score("terrabox", wrong, ground_truth)
    assert compute_score("terrabox", wrong, ground_truth) > compute_score("terrabox", "", ground_truth)


def test_ppo_reward_classifies_tool_execution_without_calling_it_task_success():
    from terrabox.evolution.ppo.reward_fn import classify_tool_result, summarize_executed_trace

    ok = classify_tool_result('{"status": "success", "value": 42}')
    error = classify_tool_result('Tool execution error: {"status": "error", "error_type": "tool_timeout"}')

    assert ok["tool_success"] is True
    assert error["tool_success"] is False
    assert error["error_type"] == "tool_timeout"

    summary = summarize_executed_trace(
        [
            {"tool": "ipython.execute", "tool_success": True},
            {"tool": "geo_perception.instructsam", "tool_success": False, "error_type": "tool_oom"},
        ]
    )
    assert summary["all_tools_succeeded"] is False
    assert summary["tool_success_rate"] == 0.5
    assert summary["verified_task_success"] is None
    assert summary["task_success_basis"] == "not_auto_verifiable"


def test_ppo_dynamic_reward_uses_real_executor_signal(monkeypatch):
    from terrabox.evolution.ppo import reward_fn

    calls = []

    def fake_execute_tool(call, extra_info):
        calls.append(call)
        if call["tool"] == "ipython.execute":
            return {
                "tool": call["tool"],
                "arguments": call["arguments"],
                "result": '{"status": "success", "value": 42}',
                "tool_success": True,
                "error_type": None,
            }
        return {
            "tool": call["tool"],
            "arguments": call["arguments"],
            "result": 'Tool execution error: {"status": "error", "error_type": "tool_timeout"}',
            "tool_success": False,
            "error_type": "tool_timeout",
        }

    monkeypatch.setattr(reward_fn, "execute_tool_call", fake_execute_tool)
    ground_truth = json.dumps({"expected_tools": ["ipython.execute"]})
    ok_solution = json.dumps({"actions": [{"tool": "ipython.execute", "arguments": {"action": "print(42)"}}]})
    bad_solution = json.dumps({"actions": [{"tool": "geo_perception.instructsam", "arguments": {}}]})

    ok_score = reward_fn.compute_score("terrabox", ok_solution, ground_truth, execute_tools=True)
    bad_score = reward_fn.compute_score("terrabox", bad_solution, ground_truth, execute_tools=True)

    assert ok_score > bad_score
    assert calls == [
        {"tool": "ipython.execute", "arguments": {"action": "print(42)"}},
        {"tool": "geo_perception.instructsam", "arguments": {}},
    ]


def test_ppo_grpo_command_uses_custom_reward_and_low_memory_lora_defaults(tmp_path):
    from terrabox.evolution.ppo.runner import build_grpo_command

    cmd = build_grpo_command(
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        model_dir=tmp_path / "model",
        reward_path=tmp_path / "reward_fn.py",
        model_path="/models/qwen3-8b/",
        n_gpus=2,
        max_steps=20,
    )
    assert "algorithm.adv_estimator=grpo" in cmd
    assert f"reward.custom_reward_function.path={tmp_path / 'reward_fn.py'}" in cmd
    assert "+reward.custom_reward_function.reward_kwargs.execute_tools=True" in cmd
    assert "actor_rollout_ref.model.lora_rank=16" in cmd
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in cmd
    assert "actor_rollout_ref.rollout.n=2" in cmd
    assert "actor_rollout_ref.rollout.layered_summon=True" in cmd
    assert "reward.num_workers=1" in cmd
    assert "data.train_batch_size=4" in cmd
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in cmd


def test_reflection_shuffle_slice_and_memory_topk(tmp_path):
    from terrabox.evolution.reflection.data_split import load_shuffled_samples, write_task_slice
    from terrabox.evolution.reflection.memory import ReflectionEntry, ReflectionMemoryBank

    rows = [_sample_row(f"sample_{i}") for i in range(6)]
    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, rows)

    first = [sample.task_id for sample in load_shuffled_samples(data_path, seed=7)]
    second = [sample.task_id for sample in load_shuffled_samples(data_path, seed=7)]
    assert first == second
    assert first != [f"sample_{i}" for i in range(6)]

    task_path = tmp_path / "tasks.json"
    written = write_task_slice(
        data_path,
        task_path,
        seed=7,
        start=1,
        limit=2,
        split_name="eval",
    )
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    assert written == 2
    assert len(payload["tasks"]) == 2
    assert payload["metadata"]["shuffle_seed"] == 7
    assert payload["metadata"]["slice_start"] == 1
    assert "messages" not in payload["tasks"][0]
    assert "gold_tool_calls" not in payload["tasks"][0]

    bank_path = tmp_path / "reflection_memory.jsonl"
    bank = ReflectionMemoryBank(bank_path)
    bank.add(
        ReflectionEntry(
            task_id="a",
            question="Measure the distance between baseball fields.",
            task_type="type_distance",
            kind="failure",
            reflection="Use perception before computation.",
            tools_called=["ipython.execute"],
            expected_tools=["geo_perception.instructsam", "ipython.execute"],
            f1=0.2,
            error_types=[],
        )
    )
    bank.add(
        ReflectionEntry(
            task_id="b",
            question="Classify a remote sensing scene.",
            task_type="type_cls",
            kind="success",
            reflection="Use the classifier for scene labels.",
            tools_called=["geo_perception.remoteclip"],
            expected_tools=["geo_perception.remoteclip"],
            f1=1.0,
            error_types=[],
        )
    )
    bank.save()
    loaded = ReflectionMemoryBank(bank_path)
    retrieved = loaded.retrieve("How far apart are the baseball fields?", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].task_id == "a"


def test_reflection_prompt_augmenter_loads_topk_memory(tmp_path):
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.reflection.memory import ReflectionEntry, ReflectionMemoryBank

    store = tmp_path / "reflection_store"
    bank = ReflectionMemoryBank(store / "memory" / "reflection_memory.jsonl")
    for i in range(3):
        bank.add(
            ReflectionEntry(
                task_id=f"m{i}",
                question="Detect storage tanks in an image.",
                task_type="type_detection",
                kind="failure",
                reflection=f"Reflection {i}",
                tools_called=[],
                expected_tools=["geo_perception.instructsam"],
                f1=0.0,
                error_types=["no_tool_call"],
            )
        )
    bank.save()

    augmenter = get_prompt_augmenter("reflection", store_dir=str(store), top_k=2)
    prompt = augmenter.augment("Count storage tanks in this image.")

    assert "Retrieved Reflection Memories" in prompt
    assert "Reflection 0" in prompt
    assert "Reflection 1" in prompt
    assert "Reflection 2" not in prompt


def test_reflection_rollout_command_supports_full_task_file_index_slices(tmp_path):
    from terrabox.evolution.reflection.runner import build_rollout_command

    cmd = build_rollout_command(
        task_file=tmp_path / "all_tasks.json",
        experiment="reflection_train",
        output_dir=tmp_path / "train",
        port=9102,
        start_index=200,
        limit=400,
    )

    assert "--task-file" in cmd
    assert str(tmp_path / "all_tasks.json") in cmd
    assert "--start-index" in cmd
    assert "200" in cmd
    assert "--limit" in cmd
    assert "400" in cmd


def test_sft_adapter_writes_chat_rows_and_prompt_only_eval_tasks(tmp_path):
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples
    from terrabox.evolution.sft.data_adapter import write_chat_jsonl, write_eval_task_file

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row("sample_a"), _sample_row("sample_b")])
    samples = load_sft_samples(data_path)

    train_path = tmp_path / "train.jsonl"
    written = write_chat_jsonl(samples[:1], train_path, source_path=data_path, split_name="train")
    row = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])

    assert written == 1
    assert row["messages"] == samples[0].messages
    assert row["task_id"] == "sample_a"
    assert row["expected_tools"] == ["geo_perception.instructsam", "ipython.execute"]
    assert "gold_tool_calls" in row

    eval_path = tmp_path / "eval_tasks.json"
    eval_written = write_eval_task_file(samples[:1], eval_path, source_path=data_path, split_name="eval")
    payload = json.loads(eval_path.read_text(encoding="utf-8"))

    assert eval_written == 1
    assert payload["tasks"][0]["task_id"] == "sample_a"
    assert "messages" not in payload["tasks"][0]
    assert "gold_tool_calls" not in payload["tasks"][0]
    assert "ground_truth" not in payload["tasks"][0]
    assert payload["metadata"]["gold_leakage_policy"].startswith("messages/gold_tool_calls")


def test_sft_compaction_keeps_all_tool_slugs_and_compacts_long_context():
    from terrabox.evolution.sft.data_adapter import compact_messages_for_sft

    catalog = [
        {
            "slug": "tool.alpha",
            "function_name": "tool__alpha",
            "description": "Alpha tool",
            "parameters": {"required": ["path"], "properties": {"path": {"type": "string"}}},
        },
        {
            "slug": "tool.beta",
            "function_name": "tool__beta",
            "description": "Beta tool",
            "parameters": {"required": [], "properties": {"items": {"type": "array"}}},
        },
    ]
    messages = [
        {"role": "system", "content": "Intro\nTool catalog:\n" + json.dumps(catalog)},
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "thought": "Use many files",
                    "actions": [
                        {
                            "tool": "tool.beta",
                            "arguments": {"items": [f"file_{i}.tif" for i in range(20)]},
                        }
                    ],
                }
            ),
        },
        {"role": "user", "content": "OBSERVATION:\n" + ("x" * 5000)},
    ]

    compacted, stats = compact_messages_for_sft(
        messages,
        max_observation_chars=200,
        max_string_chars=100,
        max_list_items=4,
    )
    system_content = compacted[0]["content"]
    assistant_payload = json.loads(compacted[2]["content"])

    assert "tool.alpha" in system_content
    assert "tool.beta" in system_content
    assert stats["system_catalog_compacted"] is True
    assert stats["observation_messages_compacted"] == 1
    assert stats["assistant_messages_compacted"] == 1
    assert assistant_payload["actions"][0]["arguments"]["items"]["__sft_compacted_list__"] is True
    assert "SFT_COMPACTED" in compacted[3]["content"]


def test_sft_prepare_data_uses_fixed_shuffle_slices(tmp_path):
    from terrabox.evolution.sft.runner import prepare_sft_files

    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [_sample_row(f"sample_{i}") for i in range(8)])
    out_dir = tmp_path / "exp"

    stats = prepare_sft_files(
        strict_data=data_path,
        output_dir=out_dir,
        seed=11,
        train_start=2,
        train_limit=3,
        val_start=0,
        val_limit=2,
    )

    assert stats["train_rows"] == 3
    assert stats["val_rows"] == 2
    assert stats["shuffle_seed"] == 11
    assert (out_dir / "sft_data" / "train.jsonl").exists()
    assert (out_dir / "sft_data" / "val.jsonl").exists()
    assert (out_dir / "eval_tasks.json").exists()


def test_sft_train_command_and_rollout_command_use_model_dir_and_gpu_split(tmp_path):
    from terrabox.evolution.sft.runner import build_sft_train_command, build_sft_rollout_command

    train_cmd = build_sft_train_command(
        train_file=tmp_path / "train.jsonl",
        val_file=tmp_path / "val.jsonl",
        model_path="/models/qwen3-8b",
        output_dir=tmp_path / "model",
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        python_executable="python",
    )

    assert train_cmd[:3] == ["python", "-m", "terrabox.evolution.sft.train_lora"]
    assert "--train-file" in train_cmd
    assert "--model-path" in train_cmd
    assert "/models/qwen3-8b" in train_cmd
    assert "--max-steps" in train_cmd
    assert "20" in train_cmd
    assert "--save-merged-model" in train_cmd

    rollout_cmd = build_sft_rollout_command(
        task_file=tmp_path / "all_tasks.json",
        experiment="sft_eval",
        output_dir=tmp_path / "eval",
        model_path=tmp_path / "model",
        port=9100,
        start_index=0,
        limit=200,
        python_executable="python",
    )

    assert rollout_cmd[:3] == ["python", "scripts/run_trajectory_experiment.py", "rollout"]
    assert "--start-index" in rollout_cmd
    assert "--limit" in rollout_cmd
    assert "200" in rollout_cmd
    assert "AGENT_LLM_MODEL_PATH" not in " ".join(rollout_cmd)


def test_sft_verl_rows_keep_messages_and_trace_metadata():
    from terrabox.evolution.sft.verl_backend import chat_row_to_verl_row

    row = {
        "task_id": "sample_a",
        "source": "openearth",
        "task_type": "type_distance",
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "gold_tool_calls": [{"tool": "ipython.execute"}],
        "expected_tools": ["ipython.execute"],
    }

    verl_row = chat_row_to_verl_row(row)

    assert verl_row["messages"] == row["messages"]
    assert verl_row["task_id"] == "sample_a"
    assert verl_row["source"] == "openearth"
    assert verl_row["task_type"] == "type_distance"
    assert verl_row["extra_info"]["expected_tools"] == ["ipython.execute"]
    assert "gold_tool_calls" not in verl_row


def test_sft_verl_command_uses_torchrun_fsdp_lora_and_sharded_checkpoint(tmp_path):
    from terrabox.evolution.sft.verl_backend import build_verl_sft_command

    cmd = build_verl_sft_command(
        verl_dir="/data1/yuhongjie2/verl",
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        model_path="/models/qwen3-8b",
        output_dir=tmp_path / "model",
        nproc_per_node=4,
        max_length=13312,
        max_token_len_per_gpu=13312,
        micro_batch_size_per_gpu=1,
        train_batch_size=4,
        total_epochs=1,
        learning_rate=2e-5,
        lora_rank=16,
        lora_alpha=32,
    )
    joined = " ".join(str(part) for part in cmd)

    assert cmd[:4] == ["torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4"]
    assert "-m" in cmd
    assert "verl.trainer.sft_trainer" in cmd
    assert "engine=fsdp" in cmd
    assert "model.lora_rank=16" in cmd
    assert "model.lora_alpha=32" in cmd
    assert "checkpoint.save_contents=['model','extra']" in cmd
    assert "data.truncation=error" in cmd
    assert "trainer.logger=['console']" in cmd
    assert "model.path=/models/qwen3-8b" in joined
    assert "model.path=/models/qwen3-8b/" not in joined


def test_sft_verl_command_can_explicitly_save_hf_checkpoint(tmp_path):
    from terrabox.evolution.sft.verl_backend import build_verl_sft_command

    cmd = build_verl_sft_command(
        verl_dir="/data1/yuhongjie2/verl",
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        model_path="/models/qwen3-8b",
        output_dir=tmp_path / "model",
        save_hf_model=True,
    )

    assert "checkpoint.save_contents=['hf_model','model','extra']" in cmd


def test_sft_verl_3090_safe_command_enables_memory_savers(tmp_path):
    from terrabox.evolution.sft.verl_backend import build_verl_sft_command

    cmd = build_verl_sft_command(
        verl_dir="/data1/yuhongjie2/verl",
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        model_path="/models/qwen3-8b/",
        output_dir=tmp_path / "model",
        nproc_per_node=4,
        train_batch_size=1,
        lora_rank=8,
        lora_alpha=16,
        sequence_parallel_size=4,
        param_offload=True,
        activation_offload=True,
        use_torch_compile=False,
    )

    assert "data.train_batch_size=1" in cmd
    assert "engine.param_offload=True" in cmd
    assert "engine.optimizer_offload=True" in cmd
    assert "engine.use_torch_compile=False" in cmd
    assert "engine.ulysses_sequence_parallel_size=4" in cmd
    assert "model.enable_activation_offload=True" in cmd
    assert "model.lora_rank=8" in cmd
    assert "model.lora_alpha=16" in cmd


def test_sft_verl_runner_3090_safe_preset_saves_every_1000_and_skips_validation_by_default():
    from argparse import Namespace

    from terrabox.evolution.sft.runner import apply_verl_preset

    args = Namespace(
        preset="3090-safe",
        nproc_per_node=4,
        train_batch_size=4,
        micro_batch_size_per_gpu=1,
        lora_rank=16,
        lora_alpha=32,
        sequence_parallel_size=1,
        param_offload=False,
        optimizer_offload=True,
        activation_offload=False,
        use_torch_compile=True,
        save_freq=None,
        test_freq=None,
    )

    apply_verl_preset(args)

    assert args.train_batch_size == 1
    assert args.sequence_parallel_size == 4
    assert args.save_freq == "1000"
    assert args.test_freq == "-1"


def test_sft_verl_runner_3090_safe_preset_keeps_explicit_save_and_test_freq():
    from argparse import Namespace

    from terrabox.evolution.sft.runner import apply_verl_preset

    args = Namespace(
        preset="3090-safe",
        nproc_per_node=4,
        train_batch_size=4,
        micro_batch_size_per_gpu=1,
        lora_rank=16,
        lora_alpha=32,
        sequence_parallel_size=1,
        param_offload=False,
        optimizer_offload=True,
        activation_offload=False,
        use_torch_compile=True,
        save_freq="after_each_epoch",
        test_freq="after_each_epoch",
    )

    apply_verl_preset(args)

    assert args.save_freq == "after_each_epoch"
    assert args.test_freq == "after_each_epoch"


def test_sft_length_preflight_rejects_overlong_samples():
    from terrabox.evolution.sft.train_lora import assert_no_overlength_rows

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            class Encoded:
                def __init__(self, n):
                    self.input_ids = list(range(n))

            return Encoded(len(text.split()))

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return " ".join(str(message.get("content", "")) for message in messages)

    rows = [
        {"task_id": "short", "messages": [{"role": "system", "content": "a b c"}]},
        {"task_id": "long", "messages": [{"role": "system", "content": " ".join(["x"] * 7)}]},
    ]

    try:
        assert_no_overlength_rows(rows, FakeTokenizer(), max_seq_length=5, split_name="train")
    except ValueError as exc:
        text = str(exc)
    else:
        raise AssertionError("expected overlength preflight to fail")

    assert "max_seq_length=5" in text
    assert "long" in text
    assert "7 tokens" in text


def test_ppo_reward_trace_metrics_keep_task_success_unknown(tmp_path):
    from terrabox.evolution.ppo.metrics import write_reward_metrics

    trace_path = tmp_path / "reward_traces.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "score": 1.0,
                        "called_tools": ["ipython.execute"],
                        "summary": {
                            "tool_success_rate": 1.0,
                            "all_tools_succeeded": True,
                            "verified_task_success": None,
                            "error_types": [],
                        },
                    }
                ),
                json.dumps(
                    {
                        "score": -0.5,
                        "called_tools": ["geo_perception.instructsam"],
                        "summary": {
                            "tool_success_rate": 0.0,
                            "all_tools_succeeded": False,
                            "verified_task_success": None,
                            "error_types": ["tool_oom"],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = write_reward_metrics(trace_path, tmp_path / "metrics.json")

    assert metrics["total_reward_evaluations"] == 2
    assert metrics["avg_tool_success_rate"] == 0.5
    assert metrics["verified_task_success_counts"] == {"None": 2}
    assert metrics["error_counts"] == {"tool_oom": 1}
