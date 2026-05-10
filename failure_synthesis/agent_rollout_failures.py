#!/usr/bin/env python3
"""Run real agent-LLM rollouts and save source-format failure trajectories."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        OPENEARTH_TO_SLUG,
        add_common_args,
        append_openearth_observation,
        attach_disaster_images,
        build_image_index,
        canonical_tool,
        collect_path_errors,
        collect_produced_paths,
        execute_registry_tool,
        import_terrabox_runtime,
        is_error_result,
        limited,
        load_json,
        make_disaster_wrapper,
        normalize_skip_tools,
        openearth_action_names,
        parse_openearth_actions,
        registry_has_tool,
        repo_path,
        result_to_jsonable,
        save_json,
        should_skip_tools,
        write_summary,
    )
except ImportError:
    from common import (
        OPENEARTH_TO_SLUG,
        add_common_args,
        append_openearth_observation,
        attach_disaster_images,
        build_image_index,
        canonical_tool,
        collect_path_errors,
        collect_produced_paths,
        execute_registry_tool,
        import_terrabox_runtime,
        is_error_result,
        limited,
        load_json,
        make_disaster_wrapper,
        normalize_skip_tools,
        openearth_action_names,
        parse_openearth_actions,
        registry_has_tool,
        repo_path,
        result_to_jsonable,
        save_json,
        should_skip_tools,
        write_summary,
    )


JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


class SampleTimeoutError(TimeoutError):
    """Raised when one source sample exceeds the configured rollout timeout."""


def _handle_sample_timeout(signum: int, frame: Any) -> None:
    raise SampleTimeoutError("sample rollout timed out")


def _no_proxy_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class AgentLLMClient:
    """Small OpenAI-compatible chat client for the deployed agent LLM."""

    def __init__(self, base_url: str, temperature: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.model = self._detect_model()

    def _detect_model(self) -> str:
        opener = _no_proxy_opener()
        with opener.open(f"{self.base_url}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["id"]

    def call(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
        opener = _no_proxy_opener()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _compact_schema(schema: dict[str, Any], max_chars: int = 900) -> dict[str, Any] | str:
    try:
        text = json.dumps(schema, ensure_ascii=False)
    except Exception:
        return {}
    if len(text) <= max_chars:
        return schema
    return text[:max_chars] + "...<truncated>"


def _load_tool_specs(slugs: set[str]) -> dict[str, dict[str, Any]]:
    import_terrabox_runtime()
    from terrabox.core.registry import get_tool

    specs: dict[str, dict[str, Any]] = {}
    for slug in sorted(slugs):
        if not slug or not registry_has_tool(slug):
            continue
        spec = get_tool(slug)
        if spec is None:
            continue
        specs[slug] = {
            "slug": spec.slug,
            "name": spec.name,
            "description": spec.description,
            "parameters": _compact_schema(spec.parameters),
        }
    return specs


def _disaster_gold_tools(sample: dict[str, Any]) -> list[str]:
    return [step.get("tool", "") for step in sample.get("tool_calls", []) if step.get("tool")]


def _openearth_gold_actions(record: dict[str, Any]) -> list[str]:
    return [a["name"] for a in parse_openearth_actions(record)]


def _openearth_action_catalog(records: list[dict[str, Any]], skip_tools: set[str]) -> dict[str, str]:
    actions: dict[str, str] = {}
    for record in records:
        raw_tools = _openearth_gold_actions(record)
        if should_skip_tools(raw_tools, skip_tools):
            continue
        for raw in raw_tools:
            mapped = canonical_tool(raw)
            if mapped:
                actions[raw] = mapped
    return actions


def _tool_f1(predicted: list[str], expected: list[str]) -> float:
    pred = set(predicted)
    exp = set(expected)
    if not exp:
        return 0.0 if pred else 1.0
    if not pred:
        return 0.0
    tp = len(pred & exp)
    precision = tp / len(pred)
    recall = tp / len(exp)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _result_text(result: Any) -> str:
    parsed = result_to_jsonable(result)
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, ensure_ascii=False)


def _resolve_disaster_arg_value(value: Any, images: list[str]) -> Any:
    if isinstance(value, str):
        low = value.lower()
        pre = images[0] if images else ""
        post = images[-1] if images else ""
        if low in {"rgb.tif", "image.tif", "image.png", "post.tif", "post.png", "post_image.png"} and post:
            return post
        if low in {"pre.tif", "pre.png", "pre_image.png"} and pre:
            return pre
    if isinstance(value, list):
        return [_resolve_disaster_arg_value(v, images) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_disaster_arg_value(v, images) for k, v in value.items()}
    return value


def _prepare_args(args: dict[str, Any], images: list[str], produced_paths: set[str]) -> dict[str, Any]:
    prepared = _resolve_disaster_arg_value(copy.deepcopy(args), images)
    for key in ("image", "image_path"):
        if key in prepared and isinstance(prepared[key], str):
            prepared[key] = _resolve_disaster_arg_value(prepared[key], images)
    for key in ("images", "image_paths"):
        if key in prepared:
            prepared[key] = _resolve_disaster_arg_value(prepared[key], images)
    return prepared


def _safe_artifact_path(value: str, artifact_dir: Path) -> str:
    name = Path(value).name if value else "artifact"
    if not name:
        name = "artifact"
    return str(artifact_dir / name)


def _redirect_output_args(args: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    out = copy.deepcopy(args)
    for key in ("output_path", "result_path", "preview_path", "gpkg"):
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = _safe_artifact_path(value, artifact_dir)
    return out


def _execute_step(tool: str, args: dict[str, Any], images: list[str], produced_paths: set[str], artifact_dir: Path) -> tuple[bool, Any, str, dict[str, Any]]:
    slug = canonical_tool(tool) or tool
    if not registry_has_tool(slug):
        return False, {"status": "error", "message": f"tool not registered: {slug}", "tool": slug}, "unregistered_tool", args
    prepared = _redirect_output_args(_prepare_args(args, images, produced_paths), artifact_dir)
    path_errors = collect_path_errors(prepared, produced_paths)
    if path_errors:
        return False, {"status": "error", "message": "; ".join(path_errors), "tool": slug}, "missing_path", prepared
    result = execute_registry_tool(slug, prepared)
    if is_error_result(result):
        return False, result_to_jsonable(result), "tool_error", prepared
    produced_paths.update(collect_produced_paths(prepared, result))
    return True, result_to_jsonable(result), "", prepared


def _system_prompt(dataset: str) -> str:
    action_field = "tool" if dataset == "disaster" else "action_name"
    return f"""You are a Terrabox geospatial tool-using agent.
Choose one tool call at a time using only the allowed tools.
Return ONLY JSON, no markdown and no extra text.

Schema:
{{
  "thought": "brief reason",
  "action": "tool" | "final",
  "{action_field}": "exact allowed tool/action name when action=tool",
  "arguments": {{}},
  "final_answer": "answer text when action=final"
}}

Rules:
- Use exact names from the allowed list.
- Use provided image paths exactly when the task needs images.
- Stop with action=final when enough observations are available or no useful tool remains.
- Do not call ChangeOS/change detection tools."""


def _tool_prompt_disaster(tool_specs: dict[str, dict[str, Any]]) -> str:
    return json.dumps(list(tool_specs.values()), ensure_ascii=False, indent=2)


def _tool_prompt_openearth(action_catalog: dict[str, str], tool_specs: dict[str, dict[str, Any]]) -> str:
    rows = []
    for action_name, slug in sorted(action_catalog.items()):
        spec = tool_specs.get(slug, {"slug": slug})
        rows.append({
            "action_name": action_name,
            "executes_slug": slug,
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _build_user_prompt(
    dataset: str,
    question: str,
    images: list[str],
    tools_text: str,
    history: list[dict[str, Any]],
    max_history_chars: int,
) -> str:
    hist_text = json.dumps(history[-8:], ensure_ascii=False, indent=2)
    if len(hist_text) > max_history_chars:
        hist_text = hist_text[-max_history_chars:]
    return f"""Task:
{question}

Available images:
{json.dumps(images, ensure_ascii=False)}

Allowed tools/actions:
{tools_text}

Previous steps and observations:
{hist_text if history else "[]"}

Select the next step as JSON."""


def _call_next_action(
    client: AgentLLMClient,
    dataset: str,
    question: str,
    images: list[str],
    tools_text: str,
    history: list[dict[str, Any]],
    max_history_chars: int,
) -> tuple[dict[str, Any] | None, str]:
    messages = [
        {"role": "system", "content": _system_prompt(dataset)},
        {
            "role": "user",
            "content": _build_user_prompt(dataset, question, images, tools_text, history, max_history_chars),
        },
    ]
    raw = client.call(messages)
    return _parse_llm_json(raw), raw


def _default_llm_url(args: argparse.Namespace) -> str:
    if args.llm_url:
        return args.llm_url
    if os.environ.get("AGENT_LLM_URL"):
        return os.environ["AGENT_LLM_URL"]
    port = os.environ.get("AGENT_LLM_PORT", "9100")
    return f"http://localhost:{port}"


def _rollout(
    *,
    client: AgentLLMClient,
    dataset: str,
    question: str,
    images: list[str],
    tools_text: str,
    action_to_slug: dict[str, str],
    max_steps: int,
    max_history_chars: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    produced_paths: set[str] = set()
    tool_calls: list[dict[str, Any]] = []
    final_answer = ""
    failure_type = ""
    error = None

    for step in range(1, max_steps + 1):
        parsed, raw = _call_next_action(client, dataset, question, images, tools_text, history, max_history_chars)
        if parsed is None:
            failure_type = "bad_llm_json"
            error = raw
            break
        thought = str(parsed.get("thought", ""))
        if parsed.get("action") == "final":
            final_answer = str(parsed.get("final_answer", ""))
            history.append({"step": step, "thought": thought, "action": "final", "final_answer": final_answer})
            break
        name_key = "tool" if dataset == "disaster" else "action_name"
        chosen = str(parsed.get(name_key, "")).strip()
        args = parsed.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        slug = action_to_slug.get(chosen, canonical_tool(chosen) or chosen)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ok, result, step_failure, executed_args = _execute_step(slug, args, images, produced_paths, artifact_dir)
        result_text = _result_text(result)
        tool_calls.append({
            "step": step,
            "name": chosen,
            "tool": slug,
            "args": executed_args,
            "llm_args": args,
            "thought": thought,
            "sample_output": result,
            "observation": result_text,
            "is_error": not ok,
        })
        history.append({
            "step": step,
            "thought": thought,
            "tool": slug,
            "action_name": chosen,
            "arguments": executed_args,
            "observation": result_text[:1200],
            "is_error": not ok,
        })
        if not ok:
            failure_type = step_failure
            error = result
            break
    else:
        failure_type = "max_steps"
        error = f"reached max_steps={max_steps}"

    return {
        "history": history,
        "tool_calls": tool_calls,
        "tools_called": [c["tool"] for c in tool_calls],
        "final_answer": final_answer,
        "failure_type": failure_type,
        "error": error,
    }


def _timeout_rollout(seconds: int, sample_id: Any) -> dict[str, Any]:
    message = f"sample {sample_id} timed out after {seconds}s"
    return {
        "history": [{"step": 0, "action": "timeout", "observation": message, "is_error": True}],
        "tool_calls": [],
        "tools_called": [],
        "final_answer": "",
        "failure_type": "sample_timeout",
        "error": message,
    }


def _rollout_with_timeout(timeout_seconds: int, sample_id: Any, **kwargs: Any) -> dict[str, Any]:
    if timeout_seconds <= 0:
        return _rollout(**kwargs)
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_sample_timeout)
    signal.alarm(timeout_seconds)
    try:
        return _rollout(**kwargs)
    except SampleTimeoutError:
        return _timeout_rollout(timeout_seconds, sample_id)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _disaster_record_from_rollout(sample: dict[str, Any], rollout: dict[str, Any], images: list[str], f1: float) -> dict[str, Any]:
    out = copy.deepcopy(sample)
    out["id"] = f"{sample.get('id')}__agent_rollout"
    out["source_id"] = sample.get("id")
    out["images"] = images
    out["tool_calls"] = [
        {
            "step": call["step"],
            "tool": call["tool"],
            "args": call["args"],
            "sample_output": call["sample_output"],
            "thought": call.get("thought", ""),
            "is_error": call.get("is_error", False),
        }
        for call in rollout["tool_calls"]
    ]
    out["agent_final_answer"] = rollout.get("final_answer", "")
    out["failure_type"] = rollout.get("failure_type") or ("low_tool_f1" if f1 < 1.0 else "")
    out["failure_meta"] = {
        "method": "agent_rollout",
        "failure_type": out["failure_type"],
        "tool_f1": round(f1, 4),
        "expected_tools": _disaster_gold_tools(sample),
        "tools_called": rollout["tools_called"],
        "error": rollout.get("error"),
        "synthetic": False,
    }
    return out


def _openearth_record_from_rollout(record: dict[str, Any], rollout: dict[str, Any], f1: float) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out["failure_id"] = f"oea_{record.get('idx')}__agent_rollout"
    out["source_idx"] = record.get("idx")
    conversation: list[dict[str, Any]] = []
    first_human = next((t for t in record.get("conversation", []) if t.get("from") == "human"), None)
    if first_human:
        conversation.append(copy.deepcopy(first_human))
    else:
        conversation.append({"from": "human", "value": record.get("question", "")})
    for call in rollout["tool_calls"]:
        payload = {
            "thought": call.get("thought", ""),
            "actions": [{"name": call["name"], "arguments": call["args"]}],
        }
        conversation.append({"from": "gpt", "value": json.dumps(payload, ensure_ascii=False)})
        prefix = "ERROR" if call.get("is_error") else "OBSERVATION"
        conversation.append({"from": "human", "value": f"{prefix}:\n{call.get('observation', '')}"})
    if rollout.get("final_answer"):
        conversation.append({
            "from": "gpt",
            "value": json.dumps({"thought": "Final answer.", "actions": [], "answer": rollout["final_answer"]}, ensure_ascii=False),
        })
    out["conversation"] = conversation
    out["failure_meta"] = {
        "method": "agent_rollout",
        "failure_type": rollout.get("failure_type") or ("low_tool_f1" if f1 < 1.0 else ""),
        "tool_f1": round(f1, 4),
        "expected_actions": _openearth_gold_actions(record),
        "expected_tools": [canonical_tool(t) for t in _openearth_gold_actions(record) if canonical_tool(t)],
        "tools_called": rollout["tools_called"],
        "error": rollout.get("error"),
        "synthetic": False,
    }
    return out


def _checkpoint_disaster(args: argparse.Namespace, source: dict[str, Any], failures: list[dict[str, Any]], successes: list[dict[str, Any]]) -> None:
    if args.checkpoint_every <= 0:
        return
    save_json(args.output, make_disaster_wrapper(source, failures, "Disaster v2 agent rollout failure trajectories."))
    if args.success_output:
        save_json(args.success_output, make_disaster_wrapper(source, successes, "Disaster v2 agent rollout successful trajectories."))


def _checkpoint_openearth(args: argparse.Namespace, failures: list[dict[str, Any]], successes: list[dict[str, Any]]) -> None:
    if args.checkpoint_every <= 0:
        return
    save_json(args.output, failures)
    if args.success_output:
        save_json(args.success_output, successes)


def _load_existing_disaster(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not args.resume:
        return failures, successes, seen
    for path, target in ((args.output, failures), (args.success_output, successes)):
        if not path:
            continue
        p = repo_path(path)
        if not p.exists() or not p.stat().st_size:
            continue
        data = load_json(path)
        loaded = data.get("samples", []) if isinstance(data, dict) else []
        target.extend(loaded)
        for sample in loaded:
            sid = sample.get("source_id") or str(sample.get("id", "")).replace("__agent_rollout", "")
            if sid:
                seen.add(str(sid))
    return failures, successes, seen


def _load_existing_openearth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not args.resume:
        return failures, successes, seen
    for path, target in ((args.output, failures), (args.success_output, successes)):
        if not path:
            continue
        p = repo_path(path)
        if not p.exists() or not p.stat().st_size:
            continue
        loaded = load_json(path)
        if not isinstance(loaded, list):
            continue
        target.extend(loaded)
        for record in loaded:
            sid = record.get("source_idx")
            if sid is not None:
                seen.add(str(sid))
    return failures, successes, seen


def run_disaster(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    input_path = args.input or "data/disaster_sft_dataset_v2.json"
    mapping_path = args.mapping or "data/sft_augmented_image_mapping.json"
    source = load_json(input_path)
    image_index = build_image_index(load_json(mapping_path))
    samples = limited(source.get("samples", []), args.limit)
    skip_tools = normalize_skip_tools(args.skip_tools, "disaster")
    skip_tools.add("geo_perception.change_os_detect")

    used_tools = {
        canonical_tool(tool) or tool
        for sample in source.get("samples", [])
        for tool in _disaster_gold_tools(sample)
        if not should_skip_tools([tool], skip_tools)
    }
    tool_specs = _load_tool_specs(used_tools)
    tools_text = _tool_prompt_disaster(tool_specs)
    action_to_slug = {slug: slug for slug in tool_specs}
    client = AgentLLMClient(_default_llm_url(args), temperature=args.temperature)

    failures, successes, seen_source_ids = _load_existing_disaster(args)
    counters = {
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
        "skipped_total": 0,
        "skipped_changeos": 0,
        "success_replay": 0,
    }

    for index, sample in enumerate(samples, start=1):
        if str(sample.get("id", "")) in seen_source_ids:
            continue
        gold = _disaster_gold_tools(sample)
        if should_skip_tools(gold, skip_tools):
            counters["skipped_total"] += 1
            if any((canonical_tool(t) or t) == "geo_perception.change_os_detect" for t in gold):
                counters["skipped_changeos"] += 1
            if args.checkpoint_every and index % args.checkpoint_every == 0:
                _checkpoint_disaster(args, source, failures, successes)
            continue
        images = attach_disaster_images(sample, image_index)
        rollout = _rollout_with_timeout(
            args.sample_timeout,
            sample.get("id", "unknown"),
            client=client,
            dataset="disaster",
            question=sample.get("prompt", ""),
            images=images,
            tools_text=tools_text,
            action_to_slug=action_to_slug,
            max_steps=args.max_steps,
            max_history_chars=args.max_history_chars,
            artifact_dir=repo_path(f"failure_synthesis/outputs/artifacts/disaster/{sample.get('id', 'unknown')}"),
        )
        f1 = _tool_f1(rollout["tools_called"], gold)
        record = _disaster_record_from_rollout(sample, rollout, images, f1)
        for tool in rollout["tools_called"]:
            counters["by_tool"][tool] += 1
        counters["by_task_type"][sample.get("task_type", "unknown")] += 1
        failed = bool(rollout.get("failure_type")) or f1 < args.max_success_f1
        if failed:
            if not record["failure_type"]:
                record["failure_type"] = "low_tool_f1"
                record["failure_meta"]["failure_type"] = "low_tool_f1"
            failures.append(record)
            counters["by_failure_type"][record["failure_type"]] += 1
        else:
            successes.append(record)
            counters["success_replay"] += 1
        if args.checkpoint_every and index % args.checkpoint_every == 0:
            _checkpoint_disaster(args, source, failures, successes)
            print(f"Checkpoint disaster {index}/{len(samples)}: failures={len(failures)}, successes={len(successes)}", flush=True)

    failure_data = make_disaster_wrapper(source, failures, "Disaster v2 agent rollout failure trajectories.")
    success_data = make_disaster_wrapper(source, successes, "Disaster v2 agent rollout successful trajectories.") if args.success_output else None
    summary = {
        "dataset": "disaster",
        "method": "agent_rollout",
        "input": str(input_path),
        "output": str(args.output),
        "failures": len(failures),
        **{k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }
    return failure_data, summary, success_data


def run_openearth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]] | None]:
    input_path = args.input or "data/openearth/train.json"
    all_records = load_json(input_path)
    records = limited(all_records, args.limit)
    skip_tools = normalize_skip_tools(args.skip_tools, "openearth")
    skip_tools.update({"ChangeDetection", "geo_perception.change_os_detect"})

    action_catalog = _openearth_action_catalog(all_records, skip_tools)
    used_slugs = set(action_catalog.values())
    tool_specs = _load_tool_specs(used_slugs)
    tools_text = _tool_prompt_openearth(action_catalog, tool_specs)
    action_to_slug = dict(action_catalog)
    client = AgentLLMClient(_default_llm_url(args), temperature=args.temperature)

    failures, successes, seen_source_ids = _load_existing_openearth(args)
    counters = {
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
        "skipped_total": 0,
        "skipped_changeos": 0,
        "success_replay": 0,
    }

    for index, record in enumerate(records, start=1):
        if str(record.get("idx", "")) in seen_source_ids:
            continue
        gold_actions = _openearth_gold_actions(record)
        if not gold_actions:
            counters["skipped_total"] += 1
            if args.checkpoint_every and index % args.checkpoint_every == 0:
                _checkpoint_openearth(args, failures, successes)
            continue
        if should_skip_tools(gold_actions, skip_tools):
            counters["skipped_total"] += 1
            if any((canonical_tool(t) or t) == "geo_perception.change_os_detect" for t in gold_actions):
                counters["skipped_changeos"] += 1
            if args.checkpoint_every and index % args.checkpoint_every == 0:
                _checkpoint_openearth(args, failures, successes)
            continue
        gold_slugs = [canonical_tool(t) for t in gold_actions if canonical_tool(t)]
        rollout = _rollout_with_timeout(
            args.sample_timeout,
            record.get("idx", "unknown"),
            client=client,
            dataset="openearth",
            question=record.get("question", ""),
            images=record.get("images", []),
            tools_text=tools_text,
            action_to_slug=action_to_slug,
            max_steps=args.max_steps,
            max_history_chars=args.max_history_chars,
            artifact_dir=repo_path(f"failure_synthesis/outputs/artifacts/openearth/{record.get('idx', 'unknown')}"),
        )
        f1 = _tool_f1(rollout["tools_called"], gold_slugs)
        out = _openearth_record_from_rollout(record, rollout, f1)
        for tool in rollout["tools_called"]:
            counters["by_tool"][tool] += 1
        counters["by_task_type"][record.get("type", "unknown")] += 1
        failed = bool(rollout.get("failure_type")) or f1 < args.max_success_f1
        if failed:
            if not out["failure_meta"]["failure_type"]:
                out["failure_meta"]["failure_type"] = "low_tool_f1"
            failures.append(out)
            counters["by_failure_type"][out["failure_meta"]["failure_type"]] += 1
        else:
            successes.append(out)
            counters["success_replay"] += 1
        if args.checkpoint_every and index % args.checkpoint_every == 0:
            _checkpoint_openearth(args, failures, successes)
            print(f"Checkpoint openearth {index}/{len(records)}: failures={len(failures)}, successes={len(successes)}", flush=True)

    summary = {
        "dataset": "openearth",
        "method": "agent_rollout",
        "input": str(input_path),
        "output": str(args.output),
        "failures": len(failures),
        **{k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }
    return failures, summary, successes if args.success_output else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--llm-url", default=None, help="OpenAI-compatible agent LLM base URL, default http://localhost:$AGENT_LLM_PORT or :9100")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-success-f1", type=float, default=0.8, help="Rollouts below this tool F1 are saved as failures.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-history-chars", type=int, default=6000)
    parser.add_argument("--success-output", default=None, help="Optional path for successful rollout trajectories in source format.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write partial output every N source samples; set 0 to disable.")
    parser.add_argument("--sample-timeout", type=int, default=600, help="Max seconds per source sample; set 0 to disable.")
    parser.add_argument("--resume", action="store_true", help="Load existing output/success-output and skip completed source samples.")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"failure_synthesis/outputs/{args.dataset}_agent_rollout_failures.json"
    if args.dataset == "disaster":
        failures, summary, successes = run_disaster(args)
    else:
        failures, summary, successes = run_openearth(args)
    save_json(args.output, failures)
    if args.success_output and successes is not None:
        save_json(args.success_output, successes)
    write_summary(summary)
    print(f"Saved {summary['failures']} agent rollout failures to {args.output}")
    if args.success_output and successes is not None:
        n_success = len(successes.get("samples", [])) if isinstance(successes, dict) else len(successes)
        print(f"Saved {n_success} successful rollouts to {args.success_output}")


if __name__ == "__main__":
    main()
