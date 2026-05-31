#!/usr/bin/env python3
"""
统一轨迹实验脚本：批量跑真实 agent 轨迹，保存完整 trajectory。

子命令：
  rollout  — 跑真实 agent 轨迹，保存完整 trajectory
  stats    — 统计已保存轨迹的信息（工具覆盖、成功率等）

用法：
  cd /data1/yuhongjie2/terrabox

  # 在合并数据集上跑标准模式轨迹（前 10 条）
  python scripts/run_trajectory_experiment.py rollout \\
      --task-file data/merged/merged_train_tasks.json \\
      --experiment merged_standard \\
      --mode standard --limit 10

  # 断点续跑（自动跳过已有结果）
  python scripts/run_trajectory_experiment.py rollout \\
      --task-file data/merged/merged_train_tasks.json \\
      --experiment merged_standard \\
      --mode standard --resume

  # 带进化增强 prompt
  python scripts/run_trajectory_experiment.py rollout \\
      --task-file data/merged/merged_train_tasks.json \\
      --experiment merged_seqgraphevo \\
      --mode standard \\
      --evolution-method seqgraphevo \\
      --evolution-store evo_res/merged/seqgraphevo/store

  # 查看轨迹统计
  python scripts/run_trajectory_experiment.py stats \\
      --trajectory-dir tmp/trajectories/merged_standard
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tool filtering constants (shared with prepare_merged_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

MOCK_TOOLS = {
    "geo_perception.mscn_classify",
    "geo_perception.sm3det_detect",
    "geo_perception.change_os_detect",
}

API_KEY_TOOLS = {"bing_search.search"}

OSM_TOOLS_PREFIX = "osm_gis."

VLM_TOOLS = {"geo_perception.vlm_analyze"}

CHANGEOS_KEYWORDS = {"change_os", "changeos", "changedetection", "change detection"}

TOOL_ALIASES = {"ipython_code.execute": "ipython.execute"}


def canonical_slug(slug: str) -> str:
    return TOOL_ALIASES.get(slug, slug)


def cleanup_gpu_memory():
    """Per-call Docker tool services are released by their managers.

    Do not stop all Docker containers here: parallel rollout workers keep their
    own agent LLM containers alive, and a global docker stop kills the sibling
    worker mid-run.
    """
    log.debug("GPU cleanup delegated to per-call Docker managers")


def should_skip_task(
    task: dict,
    *,
    skip_mock: bool = True,
    skip_bing: bool = True,
    skip_osm: bool = True,
    skip_vlm: bool = True,
    skip_changeos: bool = True,
) -> Optional[str]:
    """Return skip reason or None."""
    expected = [canonical_slug(t) for t in task.get("expected_tools", [])]
    if skip_mock and any(t in MOCK_TOOLS for t in expected):
        return "mock"
    if skip_bing and any(t in API_KEY_TOOLS for t in expected):
        return "bing_api"
    if skip_osm and any(t.startswith(OSM_TOOLS_PREFIX) for t in expected):
        return "osm_online"
    if skip_vlm and any(t in VLM_TOOLS for t in expected):
        return "vlm"
    if skip_changeos:
        if any("change_os" in t for t in expected):
            return "changeos"
        question = task.get("question", "").lower()
        if any(kw in question for kw in CHANGEOS_KEYWORDS):
            return "changeos"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Task loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tasks_from_file(path: str) -> list[dict]:
    """Load tasks from a JSON file (merged or OpenEarth task-file format)."""
    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Expected list or {{'tasks': [...]}}: {path}")
    # Normalize
    for i, task in enumerate(tasks):
        task.setdefault("task_id", task.get("id", f"task_{i}"))
        task.setdefault("id", task["task_id"])
        task.setdefault("question", task.get("prompt", ""))
        task.setdefault("images", [])
        task.setdefault("expected_tools", [])
        task["canonical_expected_tools"] = [canonical_slug(t) for t in task["expected_tools"]]
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Agent setup (reuses core infra from test_single_task_3modes.py)
# ──────────────────────────────────────────────────────────────────────────────

def setup_agent(port: int = 9100, use_docker: bool = False, gpu_devices: str = "0", max_iterations: int = 15):
    """Load toolkits, return config and registry."""
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import registry
    from terrabox.agent.config import AgentConfig

    if use_docker:
        os.environ["TERRABOX_USE_DOCKER"] = "true"
        # Set environment variables for Docker LLM manager
        os.environ["AGENT_LLM_PORT"] = str(port)
        agent_gpu_devices = os.environ.get("AGENT_LLM_GPU_DEVICES") or gpu_devices
        os.environ["AGENT_LLM_GPU_DEVICES"] = agent_gpu_devices
        gpu_devices = agent_gpu_devices
        # Ensure model path is set
        if "AGENT_LLM_MODEL_PATH" not in os.environ:
            os.environ["AGENT_LLM_MODEL_PATH"] = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B/"

    if not registry.list_toolkits():
        load_builtin_toolkits()

    log.info(f"Registry: {len(registry.list_toolkits())} toolkits, {len(registry.list_tools())} tools")

    config = AgentConfig(
        use_local_llm=True,
        use_docker=use_docker,
        local_llm_port=port,
        local_llm_gpu_devices=gpu_devices,
        max_iterations=max_iterations,
        max_retries_on_error=3,
        max_progressive_steps=10,
        max_category_expansions=2,
    )
    return config, registry


def build_llm(config):
    """Build LLM with token-tracking tracer, using AgentLLMDockerManager if Docker is enabled."""
    from langchain_core.callbacks import BaseCallbackHandler

    class TokenTracer(BaseCallbackHandler):
        def __init__(self):
            self.call_count = 0
            self.usages: list[dict[str, int]] = []

        def reset(self):
            self.call_count = 0
            self.usages = []

        def token_totals(self) -> dict[str, int]:
            return {
                "input_tokens": sum(u.get("input_tokens", 0) for u in self.usages),
                "output_tokens": sum(u.get("output_tokens", 0) for u in self.usages),
                "total_tokens": sum(u.get("total_tokens", 0) for u in self.usages),
            }

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.call_count += 1

        def on_llm_end(self, response, **kwargs):
            try:
                gen = response.generations[0][0]
                msg = gen.message if hasattr(gen, "message") else None
                usage = None
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    usage = msg.usage_metadata
                elif response.llm_output and "token_usage" in response.llm_output:
                    usage = response.llm_output["token_usage"]
                if usage:
                    data = dict(usage)
                    inp = data.get("input_tokens", data.get("prompt_tokens", 0)) or 0
                    out = data.get("output_tokens", data.get("completion_tokens", 0)) or 0
                    self.usages.append({
                        "input_tokens": int(inp),
                        "output_tokens": int(out),
                        "total_tokens": int(inp + out),
                    })
            except Exception:
                pass

    tracer = TokenTracer()

    # Use agent.llm.get_llm() which handles Docker container startup
    from terrabox.agent.llm import get_llm
    llm_base = get_llm(config)

    # Wrap with tracer. ChatOpenAI leaves callbacks as None by default, so
    # normalize it before appending; otherwise a batch rollout can instantly
    # mark every task as an exception before the LLM is even called.
    callbacks = list(getattr(llm_base, "callbacks", None) or [])
    callbacks.append(tracer)
    llm_base.callbacks = callbacks
    if hasattr(llm_base, "streaming"):
        llm_base.streaming = False

    return llm_base, tracer


# ──────────────────────────────────────────────────────────────────────────────
# Tool call extraction from agent messages
# ──────────────────────────────────────────────────────────────────────────────

def extract_tool_calls(messages: list) -> list[str]:
    """Extract tool slugs from agent messages."""
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                calls.append(name.replace("__", "."))
    return calls


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def classify_rollout_status(final: str, messages: list) -> dict[str, Any]:
    """Classify rollout status, keeping system-limit acknowledgements separate from real success."""
    final_clean = _strip_think(final)
    tool_error_markers = (
        'Tool execution error:',
        '"status": "error"',
        '"status":"error"',
        '"error_type": "tool_oom"',
        '"error_type":"tool_oom"',
    )
    oom_markers = ('"error_type": "tool_oom"', "CUDA out of memory", "OutOfMemoryError", "out of memory")

    message_texts = [
        msg.content
        for msg in messages
        if hasattr(msg, "content") and isinstance(msg.content, str)
    ]
    has_error = any(any(marker in text for marker in tool_error_markers) for text in message_texts)
    has_tool_oom = any(any(marker in text for marker in oom_markers) for text in message_texts)
    limitation_ack = bool(
        has_tool_oom
        and final_clean
        and re.search(
            r"out of memory|oom|cuda|gpu memory|显存|内存|system|technical|系统|技术限制|cannot be completed|无法完成",
            final_clean,
            flags=re.IGNORECASE,
        )
    )

    if limitation_ack:
        status = "system_limited"
    elif final.startswith("ERROR:"):
        status = "failed"
    elif has_error and not final_clean:
        status = "failed"
    elif "Sorry, need more steps" in final:
        status = "incomplete"
    elif not final_clean:
        status = "empty_final"
    else:
        status = "completed"

    return {
        "status": status,
        "has_tool_error": has_error,
        "has_tool_oom": has_tool_oom,
        "system_limitation_acknowledged": limitation_ack,
        "final_clean": final_clean,
    }


def compute_tool_metrics(called: list[str], expected: list[str]) -> dict:
    """Compute precision, recall, F1 for tool match."""
    called_set = set(called)
    expected_set = set(expected)
    if not expected_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": not called_set}
    tp = len(called_set & expected_set)
    precision = tp / len(called_set) if called_set else 0.0
    recall = tp / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = called_set == expected_set
    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}


# ──────────────────────────────────────────────────────────────────────────────
# Single task rollout
# ──────────────────────────────────────────────────────────────────────────────

def _serialize_message(msg) -> dict:
    """Convert a message object to a serializable dict."""
    result = {"type": type(msg).__name__}

    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            result["content"] = content[:2000]  # Truncate very long content
        elif isinstance(content, list):
            result["content"] = str(content)[:2000]
        else:
            result["content"] = str(content)[:2000]

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = []
        for tc in msg.tool_calls:
            result["tool_calls"].append({
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "id": tc.get("id", ""),
            })

    if hasattr(msg, "response_metadata"):
        result["response_metadata"] = str(msg.response_metadata)[:500]

    return result


@contextmanager
def task_data_runtime_env(task: dict):
    """Expose the current task's data files to tool path resolution."""
    keys = ("TERRABOX_TASK_DATA_DIR", "TERRABOX_TASK_DATA_FILES")
    previous = {key: os.environ.get(key) for key in keys}

    data_dir = str(task.get("data_dir") or "").strip()
    data_files = [str(path) for path in task.get("data_files", []) if path]

    try:
        if data_dir:
            os.environ["TERRABOX_TASK_DATA_DIR"] = data_dir
        else:
            os.environ.pop("TERRABOX_TASK_DATA_DIR", None)
        if data_files:
            os.environ["TERRABOX_TASK_DATA_FILES"] = json.dumps(data_files, ensure_ascii=False)
        else:
            os.environ.pop("TERRABOX_TASK_DATA_FILES", None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_single_task(
    task: dict,
    *,
    mode: str,
    config,
    llm,
    tracer,
    allowed_slugs: list[str] | None = None,
    evolution_prompt: str = "",
) -> dict:
    """Run a single task through the agent, return result dict with complete conversation history."""
    from terrabox.agent.eval_modes import EvalModeContext, get_eval_mode_runner

    tracer.reset()

    question = task["question"]
    images = task.get("images", [])
    if images:
        question += f"\n\n[Image files: {', '.join(images)}]"

    # Inject data_files info if present (EarthBench tasks)
    data_files = task.get("data_files", [])
    if data_files:
        question += f"\n\n[Data files available in: {task.get('data_dir', '')}]\n"
        question += f"[Files: {', '.join(Path(f).name for f in data_files[:20])}]"
        if len(data_files) > 20:
            question += f" ... and {len(data_files) - 20} more"

    # Inject evolution prompt if provided
    if evolution_prompt:
        question = f"{evolution_prompt}\n\n{question}"

    runner = get_eval_mode_runner(mode)
    with task_data_runtime_env(task):
        result = runner.run(
            EvalModeContext(
                question=question,
                config=config,
                llm=llm,
                allowed_slugs=allowed_slugs,
                image_paths=images,
                sequential_tool_turns=True,
                user=None,
                verbose=False,
            )
        )

    tool_calls = extract_tool_calls(result.messages)
    expected = [canonical_slug(t) for t in task.get("expected_tools", [])]
    metrics = compute_tool_metrics(tool_calls, expected)
    status_info = classify_rollout_status(result.final, result.messages)
    final_clean = status_info["final_clean"]
    status = status_info["status"]
    has_error = status_info["has_tool_error"]
    has_tool_oom = status_info["has_tool_oom"]
    system_limitation_acknowledged = status_info["system_limitation_acknowledged"]

    # Serialize complete conversation history
    conversation_history = [_serialize_message(msg) for msg in result.messages]

    return {
        "task_id": task.get("task_id") or task.get("id"),
        "source": task.get("source", "unknown"),
        "question": task["question"],
        "expected_tools": expected,
        "allowed_slugs": allowed_slugs,
        "task_type": task.get("type") or task.get("task_type", "general"),
        "tool_calls": tool_calls,
        "tool_calls_deduped": list(dict.fromkeys(tool_calls)),
        "metrics": metrics,
        "status": status,
        "success": status in {"completed", "completed_with_recovery"} and metrics["f1"] > 0 and not has_tool_oom,
        "real_success": status in {"completed", "completed_with_recovery"} and metrics["f1"] > 0 and not has_tool_oom,
        "system_limitation_acknowledged": system_limitation_acknowledged,
        "llm_calls": tracer.call_count,
        "tokens": tracer.token_totals(),
        "time": result.elapsed,
        "final_answer_preview": final_clean[:500] if final_clean else "",
        "final_answer_full": final_clean,
        "has_tool_error": has_error,
        "has_tool_oom": has_tool_oom,
        # Complete conversation history with parameters
        "conversation_history": conversation_history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rollout subcommand
# ──────────────────────────────────────────────────────────────────────────────

def cmd_rollout(args):
    """Run batch rollout over a task file."""
    # Load tasks
    all_tasks = load_tasks_from_file(args.task_file)
    log.info(f"Loaded {len(all_tasks)} tasks from {args.task_file}")

    # Filter
    filter_kwargs = dict(
        skip_mock=args.skip_mock,
        skip_bing=args.skip_bing,
        skip_osm=args.skip_osm,
        skip_vlm=args.skip_vlm,
        skip_changeos=args.skip_changeos,
    )
    skip_stats: Counter = Counter()
    tasks = []
    for task in all_tasks:
        reason = should_skip_task(task, **filter_kwargs)
        if reason:
            skip_stats[reason] += 1
        else:
            tasks.append(task)
    log.info(f"After filtering: {len(tasks)} tasks (skipped: {dict(skip_stats)})")

    # Apply index range
    start = args.start_index
    end = args.end_index if args.end_index is not None else len(tasks)
    end = min(end, len(tasks))
    if args.limit:
        end = min(start + args.limit, end)
    tasks = tasks[start:end]
    log.info(f"Running tasks [{start}:{end}] ({len(tasks)} tasks)")

    # Setup output
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup agent
    # Determine agent GPU explicitly. CUDA_VISIBLE_DEVICES may contain the
    # worker's full pair (agent GPU + tool GPU), so do not feed it directly
    # into the agent LLM Docker manager when AGENT_LLM_GPU_DEVICES is set.
    gpu_devices = os.environ.get("AGENT_LLM_GPU_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if args.use_docker:
        os.environ.setdefault("TERRABOX_TOOL_SERVICE_SCOPE", "call")
        os.environ.setdefault("TERRABOX_TOOL_MAX_GPUS", "1")
    config, registry = setup_agent(
        port=args.port,
        use_docker=args.use_docker,
        gpu_devices=gpu_devices,
        max_iterations=args.max_iterations,
    )

    # Determine allowed tools
    # Collect all tools from the task file's expected_tools, then exclude mock/bing/osm.
    # This gives agent access to ~40+ tools (full dataset diversity) minus unavailable ones.
    all_expected = set()
    for task in all_tasks:  # Use all_tasks (before filtering) to get full tool set
        all_expected.update(canonical_slug(t) for t in task.get("expected_tools", []))
    registered = {spec.slug for spec in registry.list_tools()}

    if args.no_restrict_tools:
        allowed_slugs = None
        log.info("Tool restriction disabled: agent sees all %d registered tools", len(registered))
    else:
        # Start with all tools that appear in the dataset
        allowed_slugs_set = all_expected & registered

        # Remove blocked tool slugs based on skip flags
        excluded = set()
        if args.skip_mock:
            excluded |= MOCK_TOOLS
        if args.skip_bing:
            excluded |= API_KEY_TOOLS
        if args.skip_osm:
            excluded |= {s for s in allowed_slugs_set if s.startswith(OSM_TOOLS_PREFIX)}
        if args.skip_vlm:
            excluded |= VLM_TOOLS
        if args.skip_changeos:
            excluded |= {s for s in allowed_slugs_set if "change_os" in s}

        allowed_slugs_set -= excluded
        allowed_slugs = sorted(allowed_slugs_set)
        log.info(f"Dataset tools: {len(all_expected)} total, {len(excluded)} excluded, "
                 f"{len(allowed_slugs)} allowed: {allowed_slugs}")

    # Load evolution augmenter if specified
    evolution_prompt = ""
    if args.evolution_method:
        from terrabox.evolution import get_prompt_augmenter
        augmenter = get_prompt_augmenter(
            args.evolution_method,
            store_dir=args.evolution_store,
        )
        log.info(f"Evolution augmenter loaded: {args.evolution_method}")

    # Run rollouts
    results = []
    success_count = 0
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for i, task in enumerate(tasks):
        task_id = task.get("task_id") or task.get("id", f"task_{i}")
        result_path = results_dir / f"{task_id}.json"

        # Resume: skip existing
        if args.resume and result_path.exists() and result_path.stat().st_size > 0:
            log.info(f"[{i+1}/{len(tasks)}] Skipping {task_id} (result exists)")
            try:
                with open(result_path) as f:
                    existing = json.load(f)
                results.append(existing)
                if existing.get("success"):
                    success_count += 1
            except Exception:
                pass
            continue

        log.info(f"[{i+1}/{len(tasks)}] Running {task_id} ...")

        # Get evolution prompt for this query
        evo_prompt = ""
        if args.evolution_method:
            try:
                evo_prompt = augmenter.augment(task.get("question", ""))
            except Exception as e:
                log.warning(f"Evolution augment failed: {e}")

        try:
            llm, tracer = build_llm(config)
            result = run_single_task(
                task,
                mode=args.mode,
                config=config,
                llm=llm,
                tracer=tracer,
                allowed_slugs=allowed_slugs,
                evolution_prompt=evo_prompt,
            )
            # 清理GPU显存（释放perception模型占用的容器）
            cleanup_gpu_memory()
        except Exception as e:
            log.error(f"Task {task_id} failed with exception: {e}")
            traceback.print_exc()
            # 即使出错也要清理GPU
            cleanup_gpu_memory()
            result = {
                "task_id": task_id,
                "source": task.get("source", "unknown"),
                "question": task["question"],
                "expected_tools": task.get("expected_tools", []),
                "tool_calls": [],
                "tool_calls_deduped": [],
                "metrics": {"precision": 0, "recall": 0, "f1": 0, "exact_match": False},
                "status": "exception",
                "success": False,
                "error": str(e),
                "llm_calls": 0,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "time": 0,
            }

        # Save per-task result
        with open(result_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        results.append(result)
        if result.get("success"):
            success_count += 1
        for k in total_tokens:
            total_tokens[k] += result.get("tokens", {}).get(k, 0)

        f1 = result.get("metrics", {}).get("f1", 0)
        log.info(
            f"  → {result['status']} | F1={f1:.2f} | "
            f"tools={result.get('tool_calls_deduped', [])} | "
            f"time={result.get('time', 0):.1f}s"
        )

    # Write consolidated trajectory JSONL (compact version for evolution)
    traj_path = out_dir / "trajectories.jsonl"
    with open(traj_path, "w") as f:
        for r in results:
            traj = {
                "task_id": r["task_id"],
                "source": r.get("source", "unknown"),
                "query": r.get("question", ""),
                "tool_sequence": r.get("tool_calls_deduped", []),
                "tools_called": r.get("tool_calls", []),
                "expected_tools": r.get("expected_tools", []),
                "f1": r.get("metrics", {}).get("f1", 0),
                "reward": 1.0 if r.get("success") else 0.0,
                "task_type": r.get("task_type", "general"),
                "status": r.get("status", "unknown"),
                "real_success": r.get("real_success", r.get("success", False)),
                "system_limitation_acknowledged": r.get("system_limitation_acknowledged", False),
                "has_tool_oom": r.get("has_tool_oom", False),
            }
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    # Write full trajectory JSONL (with complete conversation history)
    full_traj_path = out_dir / "trajectories_full.jsonl"
    with open(full_traj_path, "w") as f:
        for r in results:
            traj = {
                "task_id": r["task_id"],
                "source": r.get("source", "unknown"),
                "question": r.get("question", ""),
                "expected_tools": r.get("expected_tools", []),
                "tool_sequence": r.get("tool_calls_deduped", []),
                "tools_called": r.get("tool_calls", []),
                "metrics": r.get("metrics", {}),
                "status": r.get("status", "unknown"),
                "success": r.get("success", False),
                "real_success": r.get("real_success", r.get("success", False)),
                "system_limitation_acknowledged": r.get("system_limitation_acknowledged", False),
                "has_tool_oom": r.get("has_tool_oom", False),
                "tokens": r.get("tokens", {}),
                "llm_calls": r.get("llm_calls", 0),
                "time": r.get("time", 0),
                "final_answer": r.get("final_answer_full", ""),
                # Complete conversation with parameters
                "conversation_history": r.get("conversation_history", []),
            }
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    # Write report
    report = {
        "experiment": args.experiment,
        "mode": args.mode,
        "task_file": args.task_file,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_tasks": len(results),
        "success_count": success_count,
        "success_rate": success_count / len(results) if results else 0,
        "total_tokens": total_tokens,
        "avg_f1": sum(r.get("metrics", {}).get("f1", 0) for r in results) / len(results) if results else 0,
        "status_distribution": dict(Counter(r.get("status", "unknown") for r in results)),
        "unique_tools_called": sorted(set(
            t for r in results for t in r.get("tool_calls_deduped", [])
        )),
        "filter_settings": filter_kwargs,
        "skip_stats": dict(skip_stats),
        "evolution_method": args.evolution_method or None,
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  ROLLOUT COMPLETE: {args.experiment}")
    print(f"{'='*70}")
    print(f"  Tasks:       {len(results)}")
    print(f"  Success:     {success_count} ({report['success_rate']:.1%})")
    print(f"  Avg F1:      {report['avg_f1']:.3f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Status: {report['status_distribution']}")
    print(f"  Unique tools called: {len(report['unique_tools_called'])}")
    print(f"\n  Output:")
    print(f"    Results:           {results_dir}/")
    print(f"    Trajectories:      {traj_path} (compact, for evolution)")
    print(f"    Trajectories Full: {full_traj_path} (complete with conversation history)")
    print(f"    Report:            {report_path}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Stats subcommand
# ──────────────────────────────────────────────────────────────────────────────

def cmd_stats(args):
    """Compute statistics from saved trajectory results."""
    traj_dir = Path(args.trajectory_dir)
    if not traj_dir.is_absolute():
        traj_dir = REPO_ROOT / traj_dir

    # Try reading trajectories.jsonl
    traj_path = traj_dir / "trajectories.jsonl"
    results = []
    if traj_path.exists():
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    else:
        # Fall back to reading individual result files
        results_dir = traj_dir / "results"
        if results_dir.exists():
            for p in sorted(results_dir.glob("*.json")):
                with open(p) as f:
                    results.append(json.load(f))

    if not results:
        print(f"No trajectories found in {traj_dir}")
        return

    # Compute stats
    all_called = set()
    all_expected = set()
    f1_scores = []
    status_counts: Counter = Counter()
    source_counts: Counter = Counter()
    tool_freq: Counter = Counter()

    for r in results:
        tools = r.get("tool_sequence") or r.get("tool_calls_deduped") or r.get("tool_calls", [])
        expected = r.get("expected_tools", [])
        all_called.update(tools)
        all_expected.update(expected)
        for t in tools:
            tool_freq[t] += 1
        f1 = r.get("f1") or r.get("metrics", {}).get("f1", 0)
        f1_scores.append(f1)
        status_counts[r.get("status", "unknown")] += 1
        source_counts[r.get("source", "unknown")] += 1

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    success = sum(1 for f in f1_scores if f > 0)

    print(f"\n{'='*70}")
    print(f"  TRAJECTORY STATISTICS: {traj_dir.name}")
    print(f"{'='*70}")
    print(f"\n  Total trajectories: {len(results)}")
    print(f"  Success (F1>0):     {success} ({success/len(results):.1%})")
    print(f"  Average F1:         {avg_f1:.3f}")
    print(f"\n  Source distribution: {dict(source_counts)}")
    print(f"  Status distribution: {dict(status_counts)}")
    print(f"\n  Unique tools called:    {len(all_called)}")
    print(f"  Unique tools expected:  {len(all_expected)}")
    print(f"\n  Tool frequency (called):")
    for tool, count in tool_freq.most_common():
        print(f"    {tool}: {count}")

    # Group by toolkit
    toolkit_groups: dict[str, list[str]] = {}
    for tool in sorted(all_called):
        prefix = tool.split(".")[0] if "." in tool else "other"
        toolkit_groups.setdefault(prefix, []).append(tool)

    print(f"\n  Tools by toolkit:")
    for toolkit, tools in sorted(toolkit_groups.items()):
        print(f"    {toolkit}: {len(tools)} tools — {tools}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="统一轨迹实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── rollout ──
    p_rollout = subparsers.add_parser("rollout", help="批量跑真实 agent 轨迹")
    p_rollout.add_argument("--task-file", required=True, help="任务文件路径（JSON）")
    p_rollout.add_argument("--experiment", required=True, help="实验名")
    p_rollout.add_argument(
        "--mode", default="standard",
        choices=["standard", "progressive", "category_scoped"],
        help="Agent 模式",
    )
    p_rollout.add_argument(
        "--output-dir", default="",
        help="轨迹输出目录（默认 tmp/trajectories/{experiment}/{mode}）",
    )
    p_rollout.add_argument("--start-index", type=int, default=0, help="起始任务索引")
    p_rollout.add_argument("--end-index", type=int, default=None, help="结束任务索引")
    p_rollout.add_argument("--limit", type=int, default=None, help="最多跑 N 条")
    p_rollout.add_argument("--port", type=int, default=9100, help="LLM 端口")
    p_rollout.add_argument("--max-iterations", type=int, default=15, help="Agent 最大迭代次数")
    p_rollout.add_argument("--resume", action="store_true", help="跳过已有结果的任务")
    p_rollout.add_argument("--use-docker", action="store_true", help="使用 Docker 感知服务")

    # Filtering
    p_rollout.add_argument("--skip-mock", action="store_true", default=True)
    p_rollout.add_argument("--no-skip-mock", dest="skip_mock", action="store_false")
    p_rollout.add_argument("--skip-bing", action="store_true", default=True)
    p_rollout.add_argument("--no-skip-bing", dest="skip_bing", action="store_false")
    p_rollout.add_argument("--skip-osm", action="store_true", default=True)
    p_rollout.add_argument("--no-skip-osm", dest="skip_osm", action="store_false")
    p_rollout.add_argument("--skip-vlm", action="store_true", default=True)
    p_rollout.add_argument("--no-skip-vlm", dest="skip_vlm", action="store_false")
    p_rollout.add_argument("--skip-changeos", action="store_true", default=True)
    p_rollout.add_argument("--no-skip-changeos", dest="skip_changeos", action="store_false")

    # Tool restriction (default: restrict to dataset tools)
    p_rollout.add_argument(
        "--no-restrict-tools", action="store_true",
        help="关闭工具限制，暴露全部注册工具（默认只暴露任务文件中出现的工具）",
    )

    # Evolution
    p_rollout.add_argument("--evolution-method", default="", help="进化方法名（如 seqgraphevo）")
    p_rollout.add_argument("--evolution-store", default="", help="进化知识库目录")

    # ── stats ──
    p_stats = subparsers.add_parser("stats", help="统计已保存轨迹")
    p_stats.add_argument("--trajectory-dir", required=True, help="轨迹目录")

    args = parser.parse_args()

    if args.command == "rollout":
        if not args.output_dir:
            args.output_dir = str(REPO_ROOT / "tmp" / "trajectories" / args.experiment / args.mode)
        cmd_rollout(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
