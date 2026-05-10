#!/usr/bin/env python3
"""
Run a single eval task through all 3 agent modes, printing the full
execution trace: prompts sent to LLM, tool calls, results, and routing decisions.

Usage:
  cd /data1/yuhongjie2/terrabox
  python scripts/eval/test_single_task_3modes.py --dataset disaster_v2 --task-index 0
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Setup paths and env
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from terrabox.agent.artifacts import infer_artifact_kind as shared_infer_artifact_kind
from terrabox.agent.eval_modes import EvalModeContext, get_eval_mode_runner
from terrabox.agent.eval_validation import validate_result_with_llm

# ── Verbose LLM callback ──────────────────────────────────────────────────

class VerboseTracer(BaseCallbackHandler):
    """Print every LLM call's input and output for debugging."""

    def __init__(self):
        self.call_count = 0
        self.usages: list[dict[str, int]] = []

    def reset(self):
        self.call_count = 0
        self.usages = []

    @staticmethod
    def _normalize_usage(usage: Any) -> dict[str, int]:
        if not usage:
            return {}

        data = dict(usage)
        input_tokens = data.get("input_tokens", data.get("prompt_tokens", 0)) or 0
        output_tokens = data.get("output_tokens", data.get("completion_tokens", 0)) or 0
        total_tokens = data.get("total_tokens", input_tokens + output_tokens) or 0
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    def token_totals(self) -> dict[str, int]:
        return {
            "input_tokens": sum(u.get("input_tokens", 0) for u in self.usages),
            "output_tokens": sum(u.get("output_tokens", 0) for u in self.usages),
            "total_tokens": sum(u.get("total_tokens", 0) for u in self.usages),
        }

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.call_count += 1
        print(f"\n{'='*70}")
        print(f"  LLM CALL #{self.call_count}")
        print(f"{'='*70}")
        # prompts is for non-chat models; for chat models we use on_chat_model_start
        if prompts:
            for i, p in enumerate(prompts):
                print(f"  [prompt {i}] {p[:300]}...")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.call_count += 1
        print(f"\n{'='*70}")
        print(f"  LLM CALL #{self.call_count}")
        print(f"{'='*70}")
        for msg_batch in messages:
            for msg in msg_batch:
                role = msg.__class__.__name__
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Truncate tool schemas but show structure
                if len(content) > 600:
                    print(f"  [{role}] ({len(content)} chars) {content[:300]}...")
                    print(f"           ...{content[-200:]}")
                else:
                    print(f"  [{role}] {content}")
                # Show tool_calls if present
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    -> tool_call: {tc.get('name')} args={json.dumps(tc.get('args', {}), ensure_ascii=False)[:200]}")

    def on_llm_end(self, response, **kwargs):
        try:
            gen = response.generations[0][0]
            msg = gen.message if hasattr(gen, "message") else None
            text = msg.content if msg else gen.text
            if len(text) > 400:
                text = text[:400] + "..."
            print(f"  [LLM OUTPUT] {text}")
            if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    -> tool_call: {tc.get('name')} args={json.dumps(tc.get('args', {}), ensure_ascii=False)[:200]}")
            # Token usage
            usage = None
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
            elif response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
            if usage:
                normalized = self._normalize_usage(usage)
                if normalized:
                    self.usages.append(normalized)
                print(f"  [TOKENS] {usage}")
        except Exception as e:
            print(f"  [LLM OUTPUT] (parse error: {e})")


# ── Load task ──────────────────────────────────────────────────────────────

API_REQUIRED_TOOL_KEYWORDS = (
    "bing_search",
    "google",
    "search",
)

CHANGEOS_TOOLS = {
    "geo_perception.change_os_detect",
    "geo_perception__change_os_detect",
}

TOOL_ALIASES = {
    "ipython_code.execute": "ipython.execute",
}


def canonical_slug(slug: str) -> str:
    return TOOL_ALIASES.get(slug, slug)


def _load_disaster_image_mapping() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "data" / "sft_augmented_image_mapping.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {m["sft_id"]: m for m in data.get("mappings", []) if "sft_id" in m}


def _expected_tools(task: dict) -> list[str]:
    tools = task.get("expected_tools") or task.get("tools") or []
    return [str(t) for t in tools]


def _canonical_expected_tools(task: dict) -> list[str]:
    return [canonical_slug(t) for t in _expected_tools(task) if canonical_slug(t)]


def _normalize_task(task: dict, index: int = 0, source: str = "task_file") -> dict:
    normalized = dict(task)
    normalized.setdefault("id", normalized.get("task_id") or f"{source}_{index}")
    normalized.setdefault("question", normalized.get("prompt", ""))
    normalized.setdefault("images", [])
    normalized["expected_tools"] = _canonical_expected_tools(normalized)
    normalized["canonical_expected_tools"] = list(normalized["expected_tools"])
    normalized.setdefault("raw_index", normalized.get("raw_index", index))
    normalized.setdefault("filtered_index", normalized.get("filtered_index", index))
    return normalized


def load_task_from_file(path: str, index: int = 0) -> dict:
    task_path = Path(path)
    if not task_path.is_absolute():
        task_path = REPO_ROOT / task_path
    with open(task_path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Task file must contain a list or {{'tasks': [...]}}: {task_path}")
    if not 0 <= index < len(tasks):
        raise IndexError(f"Task index {index} out of range for {task_path} ({len(tasks)} tasks)")
    task = _normalize_task(tasks[index], index=index, source=task_path.stem)
    task["task_file"] = str(task_path)
    task["task_file_index"] = index
    return task


def _openearth_skip_reason(task: dict, skip_changeos: bool, skip_api_tools: bool) -> str | None:
    expected = _expected_tools(task)
    task_type = str(task.get("type") or task.get("task_type") or "")
    haystack = " ".join([task_type, task.get("question", ""), *expected]).lower()

    if skip_changeos and (
        any(t in CHANGEOS_TOOLS for t in expected)
        or "change_os" in haystack
        or "changeos" in haystack
        or "changedetection" in haystack
    ):
        return "changeos"

    if skip_api_tools and any(keyword in haystack for keyword in API_REQUIRED_TOOL_KEYWORDS):
        return "api_required"

    return None


def load_task(
    dataset: str,
    index: int = 0,
    skip_changeos: bool = False,
    skip_api_tools: bool = False,
) -> dict:
    if dataset == "openearth":
        path = REPO_ROOT / "data" / "openearth" / "eval.jsonl"
        kept_index = 0
        skipped: dict[str, int] = {}
        with open(path) as f:
            for raw_index, line in enumerate(f):
                task = json.loads(line)
                reason = _openearth_skip_reason(task, skip_changeos, skip_api_tools)
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                if kept_index == index:
                    task.setdefault("question", task.get("prompt", ""))
                    task.setdefault("expected_tools", _expected_tools(task))
                    task["canonical_expected_tools"] = _canonical_expected_tools(task)
                    task["raw_index"] = raw_index
                    task["filtered_index"] = kept_index
                    task["filter_skipped_before"] = skipped
                    return task
                kept_index += 1
        raise IndexError(f"Task index {index} out of range for {path}")

    if dataset == "disaster_v2":
        path = REPO_ROOT / "data" / "disaster_sft_dataset_v2.json"
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        if not 0 <= index < len(samples):
            raise IndexError(f"Task index {index} out of range for {path} ({len(samples)} samples)")

        sample = samples[index]
        mapping = _load_disaster_image_mapping().get(sample["id"], {})
        images = [
            mapping[key]
            for key in ("image_pre", "image_post")
            if mapping.get(key)
        ]
        return {
            "id": sample["id"],
            "question": sample["prompt"],
            "expected_tools": [tc["tool"] for tc in sample.get("tool_calls", []) if tc.get("tool")],
            "canonical_expected_tools": [
                canonical_slug(tc["tool"])
                for tc in sample.get("tool_calls", [])
                if tc.get("tool")
            ],
            "images": images,
            "source_sample": sample,
            "image_mapping": mapping,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def format_question(task: dict) -> str:
    question = task["question"]
    images = task.get("images", [])
    if images:
        question += f"\n\n[Image files: {', '.join(images)}]"
    return question


# ── Setup ──────────────────────────────────────────────────────────────────

def setup():
    """Load toolkits, return config and LLM."""
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import registry
    from terrabox.agent.config import AgentConfig

    if not registry.list_toolkits():
        load_builtin_toolkits()

    print(f"Registry: {len(registry.list_toolkits())} toolkits, {len(registry.list_tools())} tools")

    config = AgentConfig(
        use_local_llm=True,
        use_docker=False,
        local_llm_port=int(os.environ.get("AGENT_LLM_PORT", "9100")),
        max_iterations=15,
        max_retries_on_error=3,
        max_progressive_steps=10,
        max_category_expansions=2,
    )
    return config, registry


def dataset_allowed_slugs(dataset: str, args, registry) -> list[str]:
    """Return canonical tools used by the selected dataset after eval filters."""
    allowed: set[str] = set()

    if dataset == "openearth":
        path = REPO_ROOT / "data" / "openearth" / "eval.jsonl"
        with open(path) as f:
            for line in f:
                task = json.loads(line)
                if _openearth_skip_reason(task, args.skip_changeos, args.skip_api_tools):
                    continue
                allowed.update(_canonical_expected_tools(task))
    elif dataset == "disaster_v2":
        path = REPO_ROOT / "data" / "disaster_sft_dataset_v2.json"
        with open(path) as f:
            data = json.load(f)
        for sample in data.get("samples", []):
            allowed.update(
                canonical_slug(tc["tool"])
                for tc in sample.get("tool_calls", [])
                if tc.get("tool")
            )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    registered = {spec.slug for spec in registry.list_tools()}
    return sorted(slug for slug in allowed if slug in registered)


def resolve_allowed_slugs(args, registry) -> list[str] | None:
    exclude = {canonical_slug(slug) for slug in (args.exclude_tools or [])}
    if getattr(args, "task_file", "") and args.restrict_to_dataset_tools:
        task_path = Path(args.task_file)
        if not task_path.is_absolute():
            task_path = REPO_ROOT / task_path
        with open(task_path) as f:
            data = json.load(f)
        tasks = data.get("tasks", data) if isinstance(data, dict) else data
        allowed = {
            canonical_slug(tool)
            for task in tasks
            for tool in _expected_tools(task)
            if canonical_slug(tool)
        }
    elif args.restrict_to_dataset_tools:
        allowed = set(dataset_allowed_slugs(args.dataset, args, registry))
    elif exclude:
        allowed = {spec.slug for spec in registry.list_tools()}
    else:
        return None

    allowed.difference_update(exclude)
    return sorted(allowed)


def apply_runtime_overrides(config, args):
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.max_progressive_steps is not None:
        config.max_progressive_steps = args.max_progressive_steps
    if args.max_category_expansions is not None:
        config.max_category_expansions = args.max_category_expansions


def _safe_path_part(value: Any, fallback: str = "item") -> str:
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def _token_artifact_root(
    *,
    experiment_dir: Path,
    output_path: Path,
    args,
    task: dict,
) -> Path:
    """Return the persistent artifact directory paired with this token run.

    JSON results stay under tmp/tool_discovery_tokens/results. Large generated
    files and full observations go under tmp_tokens/<experiment>/<mode>/<task>.
    This keeps the repo root clean while preserving enough paths for replay.
    """
    mode_name = experiment_dir.name if experiment_dir.name != "json" else args.mode
    experiment_name = experiment_dir.parent.name if experiment_dir.name in {
        "standard",
        "progressive",
        "category",
        "artifact_progressive",
    } else experiment_dir.name
    if output_path.parent.name == "json" and output_path.parent.parent.name in {
        "standard",
        "progressive",
        "category",
        "artifact_progressive",
    }:
        mode_name = output_path.parent.parent.name
        experiment_name = output_path.parent.parent.parent.name

    explicit = os.environ.get("TERRABOX_TOKEN_ARTIFACT_ROOT", "").strip()
    base = Path(explicit) if explicit else REPO_ROOT / "tmp_tokens"
    if not base.is_absolute():
        base = REPO_ROOT / base
    task_name = _safe_path_part(task.get("id") or f"task_{args.task_index}", "task")
    return base / _safe_path_part(experiment_name, "experiment") / _safe_path_part(mode_name, args.mode) / task_name


def get_llm_with_tracer(config):
    """Build LLM with verbose tracer callback."""
    import httpx
    from langchain_openai import ChatOpenAI

    tracer = VerboseTracer()
    api_base = f"http://{config.local_llm_host}:{config.local_llm_port}/v1"
    llm = ChatOpenAI(
        base_url=api_base,
        api_key="EMPTY",
        model="/model",
        temperature=0.7,
        streaming=False,  # ensure usage metadata is returned
        callbacks=[tracer],
        http_client=httpx.Client(trust_env=False, transport=httpx.HTTPTransport(retries=3)),
        http_async_client=httpx.AsyncClient(trust_env=False, transport=httpx.AsyncHTTPTransport(retries=3)),
    )
    return llm, tracer


# ── Extract tool calls from messages ───────────────────────────────────────

def extract_tool_calls(messages: list) -> list[str]:
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                # Convert back from LangChain format (__ → .)
                calls.append(name.replace("__", "."))
    return calls


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _observation_inline_limit() -> int:
    return int(os.environ.get("TERRABOX_OBSERVATION_INLINE_LIMIT", "12000"))


def _observation_output_dir() -> Path | None:
    value = os.environ.get("TERRABOX_OBSERVATION_OUTPUT_DIR", "")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _looks_like_json_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
        return True
    except Exception:
        return False


def _store_observation_content(
    content: str,
    *,
    task_id: str,
    mode: str,
    step: int,
    tool: str,
) -> dict[str, Any]:
    """Persist full tool observations without forcing huge strings into JSON."""
    payload: dict[str, Any] = {
        "observation_preview": content[:1000],
        "observation_chars": len(content),
        "observation_sha256": hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
    }
    if len(content) <= _observation_inline_limit():
        payload["observation"] = content
        payload["observation_storage"] = "inline"
        return payload

    out_dir = _observation_output_dir()
    if out_dir is None:
        payload["observation"] = content
        payload["observation_storage"] = "inline_no_output_dir"
        return payload

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tool = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool or "tool")
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id or "task")
    suffix = ".json" if _looks_like_json_text(content) else ".txt"
    path = out_dir / f"{safe_task}_{mode}_step{step:02d}_{safe_tool}{suffix}"
    path.write_text(content, encoding="utf-8")
    payload["observation_path"] = str(path)
    payload["observation_storage"] = "file"
    payload["observation_format"] = "json" if suffix == ".json" else "text"
    return payload


def _paths_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [p for p in _extract_paths_from_text(value) if _looks_like_path(p)]
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            paths.extend(_paths_from_value(item))
        return paths
    if isinstance(value, dict):
        paths = []
        for item in value.values():
            paths.extend(_paths_from_value(item))
        return paths
    return []


def _artifact_entry(path: str, *, source_tool: str, source_field: str = "") -> dict[str, Any]:
    entry = {
        "path": path,
        "kind": _infer_artifact_kind(path, source_field),
        "source_tool": source_tool,
        "source_field": source_field,
    }
    try:
        stat = Path(path).stat()
        entry["exists"] = True
        entry["size_bytes"] = stat.st_size
    except OSError:
        entry["exists"] = False
    return entry


def _artifact_paths_from_observation(content: str) -> list[str]:
    parsed = _safe_json_loads(content)
    if isinstance(parsed, dict):
        paths = _paths_from_value(parsed)
        if paths:
            return paths
    return [p for p in _extract_paths_from_text(content) if _looks_like_path(p)]


def tool_trace_from_messages(messages: list) -> list[dict[str, Any]]:
    calls_by_id: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    task_id = os.environ.get("TERRABOX_CURRENT_TASK_ID", "task")
    mode = os.environ.get("TERRABOX_CURRENT_MODE", "mode")
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    calls_by_id[tc_id] = tc
        elif isinstance(msg, ToolMessage):
            tc = calls_by_id.get(getattr(msg, "tool_call_id", ""))
            raw_name = tc.get("name", "") if tc else ""
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            is_error = (
                "Tool execution error:" in content
                or '"status": "error"' in content
                or content.startswith("Error:")
                or "not a valid tool" in content
            )
            slug = raw_name.replace("__", ".")
            step = len(trace) + 1
            args = tc.get("args", {}) if tc else {}
            artifact_entries = [
                _artifact_entry(path, source_tool=slug, source_field="args")
                for path in _paths_from_value(args)
            ]
            artifact_entries.extend(
                _artifact_entry(path, source_tool=slug, source_field="observation")
                for path in _artifact_paths_from_observation(content)
            )
            deduped_artifacts = {
                (item["path"], item.get("source_field", "")): item
                for item in artifact_entries
            }
            entry = {
                "step": step,
                "tool_call_id": getattr(msg, "tool_call_id", ""),
                "tool": slug,
                "args": args,
                "status": "error" if is_error else "ok",
                "artifacts": list(deduped_artifacts.values()),
            }
            entry.update(
                _store_observation_content(
                    content,
                    task_id=task_id,
                    mode=mode,
                    step=step,
                    tool=slug,
                )
            )
            trace.append(entry)
    return trace


def standardized_turns_from_messages(messages: list, tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_by_id = {step.get("tool_call_id"): step for step in tool_trace if step.get("tool_call_id")}
    turns: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, SystemMessage):
            turns.append({"role": "system", "content": content})
        elif isinstance(msg, HumanMessage):
            turns.append({"role": "human", "content": content})
        elif isinstance(msg, AIMessage):
            turn: dict[str, Any] = {"role": "assistant", "content": content}
            tool_calls = []
            for tc in getattr(msg, "tool_calls", []) or []:
                tool_calls.append({
                    "id": tc.get("id"),
                    "name": str(tc.get("name", "")).replace("__", "."),
                    "args": tc.get("args", {}) or {},
                })
            if tool_calls:
                turn["tool_calls"] = tool_calls
            turns.append(turn)
        elif isinstance(msg, ToolMessage):
            trace = trace_by_id.get(getattr(msg, "tool_call_id", ""))
            if trace:
                turn = {
                    "role": "tool",
                    "content": trace.get("observation_preview", content[:1000]),
                    "tool_call_id": trace.get("tool_call_id"),
                    "tool_name": trace.get("tool"),
                    "tool_args": trace.get("args", {}),
                    "is_error": trace.get("status") == "error",
                    "observation_preview": trace.get("observation_preview", ""),
                    "observation_chars": trace.get("observation_chars", 0),
                    "observation_sha256": trace.get("observation_sha256"),
                }
                if "observation" in trace:
                    turn["tool_result"] = trace["observation"]
                if "observation_path" in trace:
                    turn["tool_result_path"] = trace["observation_path"]
                turns.append(turn)
            else:
                turns.append({
                    "role": "tool",
                    "content": content[:1000],
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "is_error": "Tool execution error:" in content,
                })
        else:
            turns.append({"role": msg.__class__.__name__.lower(), "content": content})
    return turns


def artifact_manifest_from_trace(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for step in tool_trace:
        for artifact in step.get("artifacts", []) or []:
            path = artifact.get("path")
            if not path:
                continue
            current = manifest.setdefault(path, dict(artifact))
            current.setdefault("seen_in_steps", []).append(step.get("step"))
            current.setdefault("source_tools", []).append(step.get("tool"))
    for item in manifest.values():
        item["seen_in_steps"] = list(dict.fromkeys(item.get("seen_in_steps", [])))
        item["source_tools"] = list(dict.fromkeys(item.get("source_tools", [])))
    return list(manifest.values())


def assess_status(final: str, tool_trace: list[dict[str, Any]], expected_tools: list[str]) -> dict[str, Any]:
    called = [step["tool"] for step in tool_trace if step.get("tool")]
    expected = [canonical_slug(t) for t in expected_tools]
    missing = [tool for tool in dict.fromkeys(expected) if tool not in set(called)]
    has_error = any(step.get("status") == "error" for step in tool_trace)
    final_clean = _strip_think(final)
    if final.startswith("ERROR:"):
        status = "failed"
    elif has_error:
        status = "failed" if missing or not final_clean else "completed_with_recovery"
    elif "Sorry, need more steps" in final or missing:
        status = "incomplete"
    elif not final_clean:
        status = "empty_final"
    else:
        status = "completed"
    return {
        "status": status,
        "missing_expected_tools": missing,
        "called_expected_tools": [tool for tool in called if tool in set(expected)],
        "has_tool_error": has_error,
    }


def _base_result(messages: list, final: str, elapsed: float, tracer: VerboseTracer, task: dict) -> dict[str, Any]:
    tool_calls = extract_tool_calls(messages)
    trace = tool_trace_from_messages(messages)
    turns = standardized_turns_from_messages(messages, trace)
    artifact_manifest = artifact_manifest_from_trace(trace)
    status = assess_status(final, trace, task.get("expected_tools", []))
    return {
        "tool_calls": tool_calls,
        "tool_trace": trace,
        "standardized_turns": turns,
        "artifact_manifest": artifact_manifest,
        "llm_calls": tracer.call_count,
        "time": elapsed,
        "tokens": tracer.token_totals(),
        "final": final,
        "evolution_status": {
            "status": status["status"],
            "success": status["status"] in {"completed", "completed_with_recovery"},
            "source": task.get("id"),
        },
        **status,
    }


def _json_from_text(text: str) -> dict[str, Any] | None:
    cleaned = _strip_think(text)
    if cleaned.startswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines() if not line.startswith("```")).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


FILE_PARAM_MARKERS = ("path", "gpkg", "image", "raster", "file")
OUTPUT_PARAM_NAMES = {"output_path", "result_path", "preview_path", "artifact_path"}


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"status": "error", "message": text}


def _extract_paths_from_text(text: str) -> list[str]:
    candidates = re.findall(r"(?:(?:/|\.{1,2}/)[^\s,;:'\")\]]+)", text or "")
    return [p for p in dict.fromkeys(candidates) if p]


def _looks_like_path(value: str) -> bool:
    if not value or any(ch.isspace() for ch in value):
        return False
    return (
        value.startswith(("/", "./", "../", "tmp/"))
        or bool(re.search(r"\.(gpkg|tif|tiff|png|jpg|jpeg|json|geojson|csv|txt)$", value, re.IGNORECASE))
    )


def _infer_artifact_kind(path: str, key: str = "") -> str:
    return shared_infer_artifact_kind(path, key)


EVAL_MODE_ORDER = ("standard", "progressive", "category", "artifact_progressive")


def run_eval_mode(
    *,
    mode: str,
    task: dict,
    config,
    llm,
    tracer: VerboseTracer,
    allowed_slugs: list[str] | None,
    sequential_tool_turns: bool,
) -> dict[str, Any]:
    """Run one token-eval mode through the shared eval runner registry."""
    tracer.reset()
    runner = get_eval_mode_runner(mode)
    result = runner.run(
        EvalModeContext(
            question=format_question(task),
            config=config,
            llm=llm,
            allowed_slugs=allowed_slugs,
            image_paths=task.get("images", []),
            sequential_tool_turns=sequential_tool_turns,
            user=None,
            verbose=True,
        )
    )
    summary = _base_result(result.messages, result.final, result.elapsed, tracer, task)
    summary.update(result.extra)
    print(f"\n{'─'*70}")
    print(f"  {mode.upper()} RESULT")
    print(f"  Time: {result.elapsed:.1f}s | LLM calls: {tracer.call_count} | Tool calls: {summary['tool_calls']}")
    print(f"  Token totals: {tracer.token_totals()}")
    print(f"  Status: {summary['status']} | Missing expected: {summary['missing_expected_tools']}")
    print(f"  Final answer: {result.final[:300]}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="disaster_v2", choices=["openearth", "disaster_v2"])
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--task-file",
        type=str,
        default="",
        help="Optional JSON task list. If set, --task-index selects from this pre-filtered file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "standard", "progressive", "category", "artifact_progressive"],
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        help="Exit successfully without rerunning if --output already exists and is non-empty.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="tmp/tool_discovery_tokens/results",
        help="Root directory for token experiment results when --output is not provided.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Experiment name used under --results-root and recorded in experiment_meta.json.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-progressive-steps", type=int, default=None)
    parser.add_argument("--max-category-expansions", type=int, default=None)
    parser.add_argument(
        "--use-docker-perception",
        action="store_true",
        help="Use Docker managers for perception services such as RemoteSAM/RemoteCLIP/SAM2.",
    )
    parser.add_argument("--skip-changeos", action="store_true", help="For OpenEarth, skip ChangeOS/change detection samples.")
    parser.add_argument("--skip-api-tools", action="store_true", help="For OpenEarth, skip samples that require search/API tools.")
    parser.add_argument(
        "--restrict-to-dataset-tools",
        action="store_true",
        help="Expose only canonical tools used by the selected dataset after filters.",
    )
    parser.add_argument(
        "--exclude-tools",
        nargs="*",
        default=[],
        help="Tool slugs to hide from this experiment, after alias normalization.",
    )
    parser.add_argument(
        "--sequential-tool-turns",
        action="store_true",
        help=(
            "Experiment mode: standard/category execute at most one tool per LLM turn, "
            "so file-path outputs can be observed before downstream calls."
        ),
    )
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help=(
            "Run an independent LLM judge after each rollout. The judge decides whether "
            "the final answer is supported by tool observations; expected_tools are only a reference."
        ),
    )
    parser.add_argument("--judge-max-evidence-steps", type=int, default=12)
    parser.add_argument("--judge-max-observation-chars", type=int, default=1800)
    args = parser.parse_args()
    if args.use_docker_perception:
        os.environ["TERRABOX_USE_DOCKER"] = "true"

    if args.task_file:
        task = load_task_from_file(args.task_file, args.task_index)
    else:
        task = load_task(
            args.dataset,
            args.task_index,
            skip_changeos=args.skip_changeos,
            skip_api_tools=args.skip_api_tools,
        )
    print(f"Dataset: {args.dataset}")
    print(f"Task: {task['id']}")
    if "raw_index" in task:
        print(f"OpenEarth raw index: {task['raw_index']} | filtered index: {task['filtered_index']}")
        print(f"Skipped before selected sample: {task.get('filter_skipped_before', {})}")
    print(f"Question: {task['question']}")
    print(f"Expected tools: {task['expected_tools']}")
    print(f"Images: {task.get('images', [])}")

    output = args.output
    experiment_name = args.experiment_name.strip()
    if not experiment_name:
        mode_part = args.mode
        task_part = task.get("id") or f"task_{args.task_index}"
        experiment_name = f"{args.dataset}_{mode_part}_{task_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not output:
        results_root = Path(args.results_root)
        if not results_root.is_absolute():
            results_root = REPO_ROOT / results_root
        output = str(results_root / experiment_name / "json" / f"{task['id']}_{args.mode}.json")

    output_for_sidecars = Path(output)
    if not output_for_sidecars.is_absolute():
        output_for_sidecars = REPO_ROOT / output_for_sidecars
    if args.skip_existing_output and output_for_sidecars.is_file() and output_for_sidecars.stat().st_size > 0:
        print(f"Skipping existing output: {output_for_sidecars}")
        return

    experiment_dir_for_sidecars = (
        output_for_sidecars.parent.parent
        if output_for_sidecars.parent.name == "json"
        else output_for_sidecars.parent
    )
    token_artifact_dir = _token_artifact_root(
        experiment_dir=experiment_dir_for_sidecars,
        output_path=output_for_sidecars,
        args=args,
        task=task,
    )
    token_artifact_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TERRABOX_CURRENT_TASK_ID"] = str(task.get("id") or f"task_{args.task_index}")
    os.environ["TERRABOX_CURRENT_MODE"] = args.mode
    os.environ["TERRABOX_TOKEN_ARTIFACT_DIR"] = str(token_artifact_dir)
    os.environ["TERRABOX_ARTIFACT_OUTPUT_DIR"] = str(token_artifact_dir / "artifacts")
    os.environ["TERRABOX_ARTIFACT_INDEX_PATH"] = str(token_artifact_dir / "artifact_index.json")
    os.environ["TERRABOX_GPKG_OUTPUT_DIR"] = str(token_artifact_dir / "gpkg")
    os.environ["TERRABOX_OBSERVATION_OUTPUT_DIR"] = str(token_artifact_dir / "observations")
    if "TERRABOX_SERVICE_EVENTS_PATH" not in os.environ:
        os.environ["TERRABOX_SERVICE_EVENTS_PATH"] = str(output_for_sidecars.with_suffix(".services.jsonl"))
    if "TERRABOX_RUN_MANIFEST_PATH" not in os.environ:
        os.environ["TERRABOX_RUN_MANIFEST_PATH"] = str(output_for_sidecars.with_suffix(".run_manifest.json"))

    config, registry = setup()
    apply_runtime_overrides(config, args)
    allowed_slugs = resolve_allowed_slugs(args, registry)
    if allowed_slugs is None:
        print("Allowed tools: all registered tools")
    else:
        print(f"Allowed tools ({len(allowed_slugs)}): {allowed_slugs}")

    results = {}
    selected_modes = EVAL_MODE_ORDER if args.mode == "all" else (args.mode,)
    for mode in selected_modes:
        llm, tracer = get_llm_with_tracer(config)
        results[mode] = run_eval_mode(
            mode=mode,
            task=task,
            config=config,
            llm=llm,
            tracer=tracer,
            allowed_slugs=allowed_slugs,
            sequential_tool_turns=args.sequential_tool_turns,
        )

    if args.enable_judge:
        judge_payload = {
            "dataset": args.dataset,
            "task_file": args.task_file,
            "task_index": args.task_index,
            "task_id": task["id"],
            "task_type": task.get("source_sample", {}).get("task_type") or task.get("task_type"),
            "raw_index": task.get("raw_index"),
            "filtered_index": task.get("filtered_index"),
            "question": task["question"],
            "expected_tools": task["expected_tools"],
            "canonical_expected_tools": task.get("canonical_expected_tools"),
            "images": task.get("images", []),
        }
        for mode, result in results.items():
            print(f"\n{'='*70}")
            print(f"  JUDGE VALIDATION: {mode}")
            print(f"{'='*70}")
            judge_llm, judge_tracer = get_llm_with_tracer(config)
            validation = validate_result_with_llm(
                llm=judge_llm,
                payload=judge_payload,
                mode=mode,
                result=result,
                max_evidence_steps=args.judge_max_evidence_steps,
                max_observation_chars=args.judge_max_observation_chars,
            )
            validation["judge_llm_calls"] = judge_tracer.call_count
            validation["judge_tokens"] = judge_tracer.token_totals()
            result["validation"] = validation
            result["evolution_status"] = {
                "status": validation["trajectory_label"],
                "success": bool(validation["success"]),
                "source": task.get("id"),
                "legacy_tool_status": result.get("status"),
                "answer_status": validation["answer_status"],
                "failure_type": validation["failure_type"],
            }
            print(
                "  Judge: "
                f"label={validation['trajectory_label']} "
                f"answer={validation['answer_status']} "
                f"confidence={validation['judge_confidence']:.2f} "
                f"failure_type={validation['failure_type']}"
            )

    # Summary
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    print(f"  Expected tools: {task['expected_tools']}")
    for mode, r in results.items():
        tokens = r.get("tokens", {})
        print(
            f"  {mode:20s} | LLM calls: {r['llm_calls']:2d} | "
            f"input/output/total tokens: "
            f"{tokens.get('input_tokens', 0)}/{tokens.get('output_tokens', 0)}/{tokens.get('total_tokens', 0)} | "
            f"Status: {r.get('status')} | Tool calls: {r['tool_calls']} | Time: {r['time']:.1f}s"
        )

    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_dir = output_path.parent.parent if output_path.parent.name == "json" else output_path.parent
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_name": experiment_name,
            "experiment_dir": str(experiment_dir),
            "token_artifact_dir": str(token_artifact_dir),
            "artifact_index_path": str(token_artifact_dir / "artifact_index.json"),
            "dataset": args.dataset,
            "task_file": args.task_file,
            "task_index": args.task_index,
            "mode": args.mode,
            "task_id": task["id"],
            "task_type": task.get("source_sample", {}).get("task_type") or task.get("task_type"),
            "source_sample": task.get("source_sample"),
            "image_mapping": task.get("image_mapping"),
            "raw_index": task.get("raw_index"),
            "filtered_index": task.get("filtered_index"),
            "filter_skipped_before": task.get("filter_skipped_before"),
            "question": task["question"],
            "expected_tools": task["expected_tools"],
            "canonical_expected_tools": task.get("canonical_expected_tools"),
            "allowed_slugs": allowed_slugs,
            "sequential_tool_turns": args.sequential_tool_turns,
            "images": task.get("images", []),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        meta_path = experiment_dir / "experiment_meta.json"
        if not meta_path.exists():
            meta_payload = {
                "created_at": payload["created_at"],
                "experiment_name": experiment_name,
                "experiment_dir": str(experiment_dir),
                "token_artifact_root": str(token_artifact_dir.parent.parent),
                "results_root": str(experiment_dir.parent),
                "dataset": args.dataset,
                "mode": args.mode,
                "task_file": args.task_file,
                "restrict_to_dataset_tools": args.restrict_to_dataset_tools,
                "exclude_tools": [canonical_slug(t) for t in args.exclude_tools],
                "skip_changeos": args.skip_changeos,
                "skip_api_tools": args.skip_api_tools,
                "sequential_tool_turns": args.sequential_tool_turns,
                "use_docker_perception": args.use_docker_perception,
                "enable_judge": args.enable_judge,
                "judge_max_evidence_steps": args.judge_max_evidence_steps,
                "judge_max_observation_chars": args.judge_max_observation_chars,
                "max_iterations": config.max_iterations,
                "max_progressive_steps": config.max_progressive_steps,
                "max_category_expansions": config.max_category_expansions,
                "sidecar_patterns": {
                    "trajectory_json": "json/*.json",
                    "service_events": "json/*.services.jsonl",
                    "run_manifest": "json/*.run_manifest.json",
                    "task_artifacts": f"{token_artifact_dir.parent.name}/*",
                },
            }
            with open(meta_path, "w") as f:
                json.dump(meta_payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary JSON: {output_path}")
        print(f"Saved experiment metadata: {meta_path}")


if __name__ == "__main__":
    main()
