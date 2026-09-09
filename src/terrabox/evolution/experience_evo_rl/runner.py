"""CLI helpers for strict ExperienceEvo-guided veRL GRPO runs."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .data_builder import DEFAULT_STORE, DEFAULT_TOOL_CATALOG, DEFAULT_TRAIN_DATA, load_tool_catalog, prepare_dataset
from .reward_fn import compute_score


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_VERL_DIR = Path("/data1/yuhongjie2/verl")
DEFAULT_MODEL_PATH = Path("/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Instruct_2507")
DEFAULT_EXP_ROOT = REPO_ROOT / "tmp/experience_evo_rl"


def build_grpo_command(
    *,
    train_file: str | Path,
    val_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    reward_path: str | Path,
    n_gpus: int = 2,
    max_steps: int | None = 5,
    rollout_backend: str = "vllm",
    rollout_tensor_parallel_size: int = 1,
    free_cache_engine: bool = True,
    train_batch_size: int = 1,
    val_max_samples: int = 8,
    val_batch_size: int | None = None,
    dataloader_num_workers: int = 0,
    val_before_train: bool = False,
    max_prompt_length: int = 8192,
    max_response_length: int = 4096,
    rollout_n: int = 2,
    rollout_gpu_memory_utilization: float = 0.25,
    rollout_max_model_len: int = 10240,
    rollout_max_num_batched_tokens: int = 10240,
    rollout_max_num_seqs: int = 1,
    actor_ppo_max_token_len_per_gpu: int = 10240,
    rollout_log_prob_max_token_len_per_gpu: int = 10240,
    ref_log_prob_max_token_len_per_gpu: int = 10240,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    save_freq: int = 100,
    test_freq: int = 100,
    max_actor_ckpt_to_keep: int = 1,
    resume_mode: str = "disable",
    resume_from_path: str | Path | None = None,
    checkpoint_save_contents: tuple[str, ...] = ("model", "extra"),
    checkpoint_load_contents: tuple[str, ...] = ("model", "extra"),
    reward_trace_path: str | Path | None = None,
    online: bool = False,
    tool_config_path: str | Path | None = None,
    agent_loop_config_path: str | Path | None = None,
    online_max_turns: int = 12,
    online_max_tool_response_length: int = 8192,
    python_executable: str | Path | None = None,
) -> list[str]:
    if rollout_backend == "hf":
        raise ValueError(
            "Current veRL main_ppo only registers async server rollouts; "
            "hf rollout is not available here. Use vllm/sglang/trtllm."
        )
    rollout_top_k = "-1"
    # veRL requires PPO mini-batch size to be divisible by the actor data-parallel
    # size. For our 4x3090 smoke runs, using the full train batch avoids the
    # ``mini_batch_size % dp_size`` assertion while keeping the CLI simple.
    ppo_mini_batch_size = max(1, train_batch_size)
    cmd = [
        str(python_executable or sys.executable),
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_max_samples={val_max_samples}",
        f"data.dataloader_num_workers={dataloader_num_workers}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.model.path={model_path}",
        # Current unsloth env does not have flash_attn installed. veRL's remove-padding
        # path imports flash_attn.bert_padding during old-logprob computation, so keep
        # it disabled for the strict smoke unless the environment is upgraded.
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.calculate_entropy=False",
        # Long multi-turn tool trajectories can make the vocab-sized entropy
        # tensor exceed a 24GB card. Chunking/checkpointing preserves the
        # training objective while reducing peak activation memory.
        "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True",
        "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        f"actor_rollout_ref.rollout.name={rollout_backend}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={rollout_max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={rollout_max_num_seqs}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=True",
        f"actor_rollout_ref.rollout.free_cache_engine={str(free_cache_engine)}",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        f"actor_rollout_ref.rollout.top_k={rollout_top_k}",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={rollout_log_prob_max_token_len_per_gpu}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={ref_log_prob_max_token_len_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        "reward.num_workers=1",
        "trainer.balance_batch=True",
        "trainer.logger=['console']",
        "trainer.project_name=experience_evo_rl",
        f"trainer.experiment_name={Path(output_dir).name}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.max_actor_ckpt_to_keep={max_actor_ckpt_to_keep}",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.val_before_train={str(val_before_train)}",
        "trainer.total_epochs=1",
    ]
    if resume_from_path:
        cmd.append(f"trainer.resume_from_path={resume_from_path}")
    save_contents = "[" + ",".join(checkpoint_save_contents) + "]"
    load_contents = "[" + ",".join(checkpoint_load_contents) + "]"
    cmd.extend(
        [
            f"actor_rollout_ref.actor.checkpoint.save_contents={save_contents}",
            f"actor_rollout_ref.actor.checkpoint.load_contents={load_contents}",
        ]
    )
    if val_batch_size is not None:
        cmd.append(f"data.val_batch_size={val_batch_size}")
    if online:
        if not tool_config_path or not agent_loop_config_path:
            raise ValueError("online=True requires tool_config_path and agent_loop_config_path")
        cmd.extend(
            [
                f"data.tool_config_path={tool_config_path}",
                "data.function_tool_path=null",
                "+data.need_tools_kwargs=True",
                "actor_rollout_ref.rollout.multi_turn.enable=True",
                f"actor_rollout_ref.rollout.multi_turn.tool_config_path={tool_config_path}",
                "actor_rollout_ref.rollout.multi_turn.function_tool_path=null",
                f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={online_max_turns}",
                "actor_rollout_ref.rollout.multi_turn.max_user_turns=0",
                "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1",
                f"actor_rollout_ref.rollout.multi_turn.max_tool_response_length={online_max_tool_response_length}",
                "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle",
                "actor_rollout_ref.rollout.multi_turn.format=hermes",
                # On 24GB cards, collecting a full FSDP state dict just to transfer
                # LoRA adapters can OOM before the first rollout. veRL's layered
                # path materializes one wrapped layer at a time instead.
                "actor_rollout_ref.rollout.layered_summon=True",
                # Terrabox GPU services are individually serialized. Multiple veRL
                # agent-loop workers only turn service-lock wait time into false
                # timeout rewards, so online OEA training uses one worker per lane.
                "actor_rollout_ref.rollout.agent.num_workers=1",
                "actor_rollout_ref.rollout.agent.default_agent_loop=terrabox_tool_agent",
                f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop_config_path}",
            ]
        )
        # Pure online GRPO does not use a KL reward. Disable the actor
        # KL/entropy branch to avoid full-vocabulary entropy allocations on 24GB
        # cards; this does not change the tool reward or rollout behavior.
        cmd.extend(
            [
                "actor_rollout_ref.actor.use_kl_loss=False",
                "actor_rollout_ref.actor.calculate_entropy=False",
                "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False",
                "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False",
            ]
        )
    if max_steps is not None:
        cmd.append(f"trainer.total_training_steps={max_steps}")
    if reward_trace_path is not None:
        cmd.append(f"+reward.custom_reward_function.reward_kwargs.trace_path={reward_trace_path}")
    return cmd


def _exp_dir(name: str) -> Path:
    return DEFAULT_EXP_ROOT / name


def _sanitize_openai_parameters(parameters: Any) -> dict[str, Any]:
    """Return a veRL/Pydantic-compatible OpenAI function parameters schema."""
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    schema = json.loads(json.dumps(parameters, ensure_ascii=False))
    schema.setdefault("type", "object")
    props = schema.get("properties")
    if isinstance(props, dict):
        for spec in props.values():
            if isinstance(spec, dict):
                spec.setdefault("type", "string")
    required = schema.get("required")
    if required is not None and not isinstance(required, list):
        schema["required"] = []
    return schema


def write_online_configs(*, tool_catalog: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Write veRL multi-turn tool and agent-loop configs for Terrabox OEA tools."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tools = []
    for tool in load_tool_catalog(tool_catalog):
        slug = str(tool.get("slug") or "")
        function_name = str(tool.get("function_name") or slug.replace(".", "__"))
        if not slug or not function_name:
            continue
        tools.append(
            {
                "class_name": "terrabox.evolution.experience_evo_rl.online_tools.TerraboxOeaTool",
                "config": {"type": "native", "slug": slug},
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": str(tool.get("description") or slug),
                        "parameters": _sanitize_openai_parameters(tool.get("parameters")),
                    },
                },
            }
        )
    tool_config = out / "terrabox_oea_tool_config.yaml"
    tool_config.write_text(json.dumps({"tools": tools}, ensure_ascii=False, indent=2), encoding="utf-8")
    agent_loop_config = out / "terrabox_agent_loop_config.json"
    agent_loop_config.write_text(
        json.dumps(
            [
                {
                    "name": "terrabox_tool_agent",
                    "_target_": "terrabox.evolution.experience_evo_rl.online_agent_loop.TerraboxToolAgentLoop",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = out / "online_config_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tool_config": str(tool_config),
                "agent_loop_config": str(agent_loop_config),
                "num_tools": len(tools),
                "tool_catalog": str(tool_catalog),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"tool_config": str(tool_config), "agent_loop_config": str(agent_loop_config), "manifest": str(manifest)}


def cmd_write_online_configs(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir or _exp_dir(args.experiment) / "configs")
    result = write_online_configs(tool_catalog=args.tool_catalog, output_dir=out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_prepare_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir or _exp_dir(args.experiment) / "verl_data")
    artifact_root = args.artifact_root or str(_exp_dir(args.experiment) / "artifacts" / "online_train")
    stats = prepare_dataset(
        task_file=args.task_file,
        tool_catalog_file=args.tool_catalog,
        store_dir=args.store_dir,
        output_dir=out_dir,
        limit=args.limit,
        train_ratio=args.train_ratio,
        top_k=args.top_k,
        prompt_top_k=args.prompt_top_k,
        reward_top_k=args.reward_top_k,
        parquet=True,
        online=args.online,
        artifact_root=artifact_root if args.online else None,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_reward_dry_run(args: argparse.Namespace) -> None:
    row_path = Path(args.train_file or _exp_dir(args.experiment) / "verl_data" / "train.jsonl")
    trace_path = Path(args.trace_output or _exp_dir(args.experiment) / "metrics" / "reward_dry_run_traces.jsonl")
    if trace_path.exists():
        trace_path.unlink()
    rows = []
    with row_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    outputs: list[dict[str, Any]] = []
    for row in rows:
        policy = json.loads(row["reward_model"]["ground_truth"])
        recs = policy.get("recommended_tools") or []
        tool = recs[0]["tool"] if recs else (policy.get("allowed_tools") or [""])[0]
        schema = (policy.get("tool_schemas") or {}).get(tool) or {}
        args_payload = {key: "<from_current_task>" for key in (schema.get("required") or [])}
        solution = json.dumps({"thought": "Use the strongest ExperienceEvo policy.", "actions": [{"tool": tool, "arguments": args_payload}]}, ensure_ascii=False)
        score = compute_score(row["data_source"], solution, row["reward_model"]["ground_truth"], trace_path=str(trace_path))
        outputs.append({"tool": tool, "score": score})
    out = Path(args.output or _exp_dir(args.experiment) / "metrics" / "reward_dry_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"num_rows": len(rows), "scores": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "trace_output": str(trace_path), "num_rows": len(rows), "scores": outputs}, ensure_ascii=False, indent=2))


def cmd_train_grpo(args: argparse.Namespace) -> None:
    exp_dir = _exp_dir(args.experiment)
    data_dir = exp_dir / "verl_data"
    model_dir = Path(args.output_dir or exp_dir / "model")
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train_file = Path(args.train_file or data_dir / "train.parquet").resolve()
    val_file = Path(args.val_file or data_dir / "val.parquet").resolve()
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}. Run prepare-data first.")
    config_dir = exp_dir / "configs"
    online_configs = None
    if args.online:
        online_configs = write_online_configs(tool_catalog=args.tool_catalog, output_dir=config_dir)
    cmd = build_grpo_command(
        train_file=train_file,
        val_file=val_file,
        model_path=args.model_path,
        output_dir=model_dir,
        reward_path=MODULE_ROOT / "reward_fn.py",
        n_gpus=args.n_gpus,
        max_steps=args.max_steps,
        rollout_backend=args.rollout_backend,
        rollout_tensor_parallel_size=args.rollout_tensor_parallel_size,
        free_cache_engine=not args.keep_rollout_loaded,
        train_batch_size=args.train_batch_size,
        val_max_samples=args.val_max_samples,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        val_before_train=not args.no_val_before_train,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        rollout_n=args.rollout_n,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        rollout_max_model_len=args.rollout_max_model_len,
        rollout_max_num_batched_tokens=args.rollout_max_num_batched_tokens,
        rollout_max_num_seqs=args.rollout_max_num_seqs,
        actor_ppo_max_token_len_per_gpu=args.actor_ppo_max_token_len_per_gpu,
        rollout_log_prob_max_token_len_per_gpu=args.rollout_log_prob_max_token_len_per_gpu,
        ref_log_prob_max_token_len_per_gpu=args.ref_log_prob_max_token_len_per_gpu,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        max_actor_ckpt_to_keep=args.max_actor_ckpt_to_keep,
        resume_mode=args.resume_mode,
        resume_from_path=args.resume_from_path,
        checkpoint_save_contents=tuple(args.checkpoint_save_contents.split(",")),
        checkpoint_load_contents=tuple(args.checkpoint_load_contents.split(",")),
        reward_trace_path=metrics_dir / "reward_traces.jsonl",
        online=args.online,
        tool_config_path=online_configs["tool_config"] if online_configs else None,
        agent_loop_config_path=online_configs["agent_loop_config"] if online_configs else None,
        online_max_turns=args.online_max_turns,
        online_max_tool_response_length=args.online_max_tool_response_length,
        python_executable=sys.executable,
    )
    script = exp_dir / "run_grpo_command.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    online_env = ""
    if args.online:
        online_env = (
            # Keep tool-service fallback on the existing service lane. The
            # training process itself is pinned separately through CUDA_VISIBLE_DEVICES.
            "export TERRABOX_GPU_ALLOWED_DEVICES=${TERRABOX_GPU_ALLOWED_DEVICES:-0,1}\n"
            "export TERRABOX_GPU_FALLBACK_DEVICE=${TERRABOX_GPU_FALLBACK_DEVICE:-1}\n"
            "export VLM_GPU_DEVICES=${VLM_GPU_DEVICES:-0}\n"
            "export VLM_TENSOR_PARALLEL_SIZE=${VLM_TENSOR_PARALLEL_SIZE:-1}\n"
            "export VLM_MAX_MODEL_LEN=${VLM_MAX_MODEL_LEN:-8192}\n"
            "export VLM_GPU_MEMORY_UTILIZATION=${VLM_GPU_MEMORY_UTILIZATION:-0.95}\n"
            "export VLM_MAX_NUM_SEQS=${VLM_MAX_NUM_SEQS:-1}\n"
            "export TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS=${TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS:-4096}\n"
            "export SAM2_GPU_DEVICES=${SAM2_GPU_DEVICES:-1}\n"
            "export REMOTESAM_GPU_DEVICES=${REMOTESAM_GPU_DEVICES:-1}\n"
            "export REMOTECLIP_GPU_DEVICES=${REMOTECLIP_GPU_DEVICES:-1}\n"
            "export STRIP_RCNN_GPU_DEVICES=${STRIP_RCNN_GPU_DEVICES:-1}\n"
            "export INSTRUCTSAM_GPU_DEVICES=${INSTRUCTSAM_GPU_DEVICES:-1}\n"
            "export CHANGEOS_GPU_DEVICES=${CHANGEOS_GPU_DEVICES:-1}\n"
            "export REMOTESAM_CHECKPOINT_HOST=${REMOTESAM_CHECKPOINT_HOST:-/data1/yuhongjie2/RemoteSAM/pretrained_weights}\n"
            "export REMOTESAM_USE_EPOC=${REMOTESAM_USE_EPOC:-false}\n"
            "export TERRABOX_OCR_USE_GPU=${TERRABOX_OCR_USE_GPU:-0}\n"
            "export TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_SAM2_SEGMENT=${TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_SAM2_SEGMENT:-420}\n"
            "export TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_VLM_ANALYZE=${TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_VLM_ANALYZE:-420}\n"
            "export TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_INSTRUCTSAM=${TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_INSTRUCTSAM:-420}\n"
            "export TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_STRIP_RCNN_DETECT=${TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_STRIP_RCNN_DETECT:-420}\n"
            "export TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_CHANGE_OS_DETECT=${TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_CHANGE_OS_DETECT:-420}\n"
        )
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(args.verl_dir))}\n"
        f"export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}\n"
        f"export PYTHONPATH={shlex.quote(str(REPO_ROOT / 'src'))}:{shlex.quote(str(args.verl_dir))}:${{PYTHONPATH:-}}\n"
        # vLLM 0.12 V1's CuMem sleep path is incompatible with the local
        # CUDA/PyTorch build; V0 retains the compatible weight-offload path.
        "export VLLM_USE_V1=0\n"
        "unset PYTORCH_CUDA_ALLOC_CONF\n"
        "export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}\n"
        "export WANDB_DISABLED=${WANDB_DISABLED:-true}\n"
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}\n"
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}\n"
        "export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.90}\n"
        "export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-1000}\n"
        "export TERRABOX_USE_DOCKER=true\n"
        "export no_proxy=localhost,127.0.0.1\n"
        "export TERRABOX_SERVICE_CALL_LOCKS=1\n"
        f"export TERRABOX_SERVICE_LOCK_DIR={shlex.quote(str(REPO_ROOT / 'tmp/service_locks'))}\n"
        f"export TERRABOX_EXPEVO_RL_REWARD_TRACE_PATH={shlex.quote(str(metrics_dir / 'reward_traces.jsonl'))}\n"
        f"export TERRABOX_ONLINE_RL_TRACE_PATH={shlex.quote(str(metrics_dir / 'online_episode_traces.jsonl'))}\n"
        f"export TERRABOX_ONLINE_RAW_OBSERVATION_TRACE_PATH={shlex.quote(str(metrics_dir / 'raw_tool_observations.jsonl'))}\n"
        f"export TERRABOX_RESOURCE_MONITOR_PATH={shlex.quote(str(metrics_dir / 'resource_monitor.jsonl'))}\n"
        "export TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS=${TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS:-8192}\n"
        f"{online_env}"
        + "\n"
        + "monitor_pid=''\n"
        + "monitor_resources() {\n"
        + "  local target_pid=\"$1\"\n"
        + "  while kill -0 \"$target_pid\" 2>/dev/null; do\n"
        + "    ts=\"$(date -Is)\"\n"
        + "    cpu=\"$(awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} END{if(t>0) printf \"%.4f\", 1-a/t; else print \"-1\"}' /proc/meminfo)\"\n"
        + "    gpu=\"$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\\n' ';' || true)\"\n"
        + "    printf '%s\\n' \"{\\\"ts\\\":\\\"$ts\\\",\\\"cpu_memory_fraction\\\":$cpu,\\\"gpu\\\":\\\"$gpu\\\"}\" >> \"$TERRABOX_RESOURCE_MONITOR_PATH\"\n"
        + "    sleep 10\n"
        + "  done\n"
        + "}\n"
        + "set +e\n"
        + "("
        + " ".join(shlex.quote(str(part)) for part in cmd)
        + ") > >(tee -a \"${TRAIN_LOG_PATH:-"
        + shlex.quote(str(exp_dir / 'train_online.log'))
        + "}\") 2>&1 &\n"
        + "main_pid=$!\n"
        + "monitor_resources \"$main_pid\" &\n"
        + "monitor_pid=$!\n"
        + "wait \"$main_pid\"\n"
        + "status=$?\n"
        + "kill \"$monitor_pid\" 2>/dev/null || true\n"
        + "wait \"$monitor_pid\" 2>/dev/null || true\n"
        + "exit \"$status\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    print(
        json.dumps(
            {
                "script": str(script),
                "launch": bool(args.launch),
                "max_steps": args.max_steps,
                "online": bool(args.online),
                "online_configs": online_configs,
                "metrics_dir": str(metrics_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.launch:
        subprocess.run(["bash", str(script)], cwd=args.verl_dir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict ExperienceEvo RL runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("prepare-data", help="Build veRL JSONL/Parquet")
    p_data.add_argument("--experiment", required=True)
    p_data.add_argument("--task-file", default=str(DEFAULT_TRAIN_DATA))
    p_data.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    p_data.add_argument("--store-dir", default=str(DEFAULT_STORE))
    p_data.add_argument("--output-dir")
    p_data.add_argument("--limit", type=int, default=100)
    p_data.add_argument("--train-ratio", type=float, default=0.95)
    p_data.add_argument("--top-k", type=int, default=5)
    p_data.add_argument("--prompt-top-k", type=int, help="ExperienceEvo families included in the model prompt; defaults to --top-k")
    p_data.add_argument("--reward-top-k", type=int, help="ExperienceEvo families used by the reward policy; defaults to --top-k")
    p_data.add_argument("--online", action="store_true", help="Build veRL multi-turn online tool-agent rows")
    p_data.add_argument("--artifact-root", help="Root directory for per-episode online tool artifacts")
    p_data.set_defaults(func=cmd_prepare_data)

    p_cfg = sub.add_parser("write-online-configs", help="Write veRL Terrabox tool and agent-loop configs")
    p_cfg.add_argument("--experiment", required=True)
    p_cfg.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    p_cfg.add_argument("--output-dir")
    p_cfg.set_defaults(func=cmd_write_online_configs)

    p_dry = sub.add_parser("reward-dry-run", help="Score synthetic policy-following completions")
    p_dry.add_argument("--experiment", required=True)
    p_dry.add_argument("--train-file")
    p_dry.add_argument("--limit", type=int, default=5)
    p_dry.add_argument("--output")
    p_dry.add_argument("--trace-output")
    p_dry.set_defaults(func=cmd_reward_dry_run)

    p_train = sub.add_parser("train-grpo", help="Write and optionally launch veRL GRPO smoke")
    p_train.add_argument("--experiment", required=True)
    p_train.add_argument("--verl-dir", default=str(DEFAULT_VERL_DIR))
    p_train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_train.add_argument("--train-file")
    p_train.add_argument("--val-file")
    p_train.add_argument("--output-dir")
    p_train.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    p_train.add_argument("--n-gpus", type=int, default=2)
    p_train.add_argument("--max-steps", type=int, default=5)
    p_train.add_argument("--cuda-visible-devices", default="2,3")
    p_train.add_argument("--rollout-backend", choices=["vllm", "sglang", "trtllm"], default="vllm")
    p_train.add_argument("--rollout-tensor-parallel-size", type=int, default=1)
    p_train.add_argument(
        "--keep-rollout-loaded",
        action="store_true",
        help="Disable vLLM sleep/offload when its CuMem sleep path is unavailable.",
    )
    p_train.add_argument("--train-batch-size", type=int, default=1)
    p_train.add_argument(
        "--val-max-samples",
        type=int,
        default=8,
        help="Maximum validation samples loaded by veRL; use a small value for online tool RL health checks.",
    )
    p_train.add_argument(
        "--val-batch-size",
        type=int,
        help="Validation batch size. For online multi-turn OEA, set to 1 to avoid gathering many episodes in memory.",
    )
    p_train.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="veRL dataloader workers. Use 0/1 for memory-constrained online tool RL.",
    )
    p_train.add_argument(
        "--no-val-before-train",
        action="store_true",
        help="Skip veRL's initial validation before the first training update.",
    )
    p_train.add_argument("--max-prompt-length", type=int, default=8192)
    p_train.add_argument("--max-response-length", type=int, default=4096)
    p_train.add_argument("--rollout-n", type=int, default=2)
    p_train.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.25)
    p_train.add_argument("--rollout-max-model-len", type=int, default=10240)
    p_train.add_argument("--rollout-max-num-batched-tokens", type=int, default=10240)
    p_train.add_argument("--rollout-max-num-seqs", type=int, default=1)
    p_train.add_argument(
        "--actor-ppo-max-token-len-per-gpu",
        type=int,
        default=10240,
        help="Maximum actor PPO tokens per GPU during update; lower this to reduce training-side peak VRAM.",
    )
    p_train.add_argument(
        "--rollout-log-prob-max-token-len-per-gpu",
        type=int,
        default=10240,
        help="Maximum tokens per GPU for old rollout log-prob computation; lower this first after old-logprob OOM.",
    )
    p_train.add_argument(
        "--ref-log-prob-max-token-len-per-gpu",
        type=int,
        default=10240,
        help="Maximum tokens per GPU for reference log-prob computation; lower this if ref log-prob OOMs.",
    )
    p_train.add_argument("--lora-rank", type=int, default=8)
    p_train.add_argument("--lora-alpha", type=int, default=16)
    p_train.add_argument("--save-freq", type=int, default=100)
    p_train.add_argument("--test-freq", type=int, default=100)
    p_train.add_argument("--max-actor-ckpt-to-keep", type=int, default=1)
    p_train.add_argument("--resume-mode", choices=["disable", "auto", "resume_path"], default="disable")
    p_train.add_argument("--resume-from-path")
    p_train.add_argument(
        "--checkpoint-save-contents",
        default="model,extra",
        help="逗号分隔的 veRL 分片 checkpoint 内容；默认不保存 optimizer，降低 CPU 峰值。",
    )
    p_train.add_argument(
        "--checkpoint-load-contents",
        default="model,extra",
        help="恢复时加载的 checkpoint 内容；model,extra 可用于模型状态恢复。",
    )
    p_train.add_argument("--online", action="store_true", help="Use veRL multi-turn Terrabox tool-agent loop")
    p_train.add_argument("--online-max-turns", type=int, default=12)
    p_train.add_argument("--online-max-tool-response-length", type=int, default=8192)
    p_train.add_argument("--launch", action="store_true")
    p_train.set_defaults(func=cmd_train_grpo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
