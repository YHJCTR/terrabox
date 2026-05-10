#!/usr/bin/env python3
"""Generate source-format failure trajectories by replaying SFT tool calls."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from typing import Any

try:
    from .common import (
        add_common_args,
        append_openearth_observation,
        attach_disaster_images,
        build_image_index,
        canonical_tool,
        collect_path_errors,
        collect_produced_paths,
        execute_registry_tool,
        is_error_result,
        limited,
        load_json,
        make_disaster_wrapper,
        normalize_skip_tools,
        parse_openearth_actions,
        prepare_disaster_replay_args,
        registry_has_tool,
        repo_path,
        result_to_jsonable,
        save_json,
        should_skip_tools,
        write_summary,
    )
except ImportError:
    from common import (
        add_common_args,
        append_openearth_observation,
        attach_disaster_images,
        build_image_index,
        canonical_tool,
        collect_path_errors,
        collect_produced_paths,
        execute_registry_tool,
        is_error_result,
        limited,
        load_json,
        make_disaster_wrapper,
        normalize_skip_tools,
        parse_openearth_actions,
        prepare_disaster_replay_args,
        registry_has_tool,
        repo_path,
        result_to_jsonable,
        save_json,
        should_skip_tools,
        write_summary,
    )


def _error_payload(message: str, tool: str) -> dict[str, Any]:
    return {"status": "error", "tool": tool, "message": message}


def _result_text(result: Any) -> str:
    parsed = result_to_jsonable(result)
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, ensure_ascii=False)


def _run_step(tool: str, args: dict[str, Any], produced_paths: set[str]) -> tuple[bool, Any, str]:
    canonical = canonical_tool(tool)
    if not canonical:
        return False, _error_payload(f"tool maps to terminal/no-op action: {tool}", tool), "terminal_tool"
    try:
        has_tool = registry_has_tool(canonical)
    except RuntimeError as exc:
        return False, _error_payload(str(exc), canonical), "runtime_import_error"
    if not has_tool:
        return False, _error_payload(f"tool not registered: {canonical}", canonical), "unregistered_tool"

    path_errors = collect_path_errors(args, produced_paths)
    if path_errors:
        return False, _error_payload("; ".join(path_errors), canonical), "missing_path"

    try:
        result = execute_registry_tool(canonical, args)
    except RuntimeError as exc:
        return False, _error_payload(str(exc), canonical), "runtime_import_error"
    if is_error_result(result):
        return False, result_to_jsonable(result), "tool_error"
    produced_paths.update(collect_produced_paths(args, result))
    return True, result_to_jsonable(result), ""


def _disaster_tools(sample: dict[str, Any]) -> list[str]:
    return [step.get("tool", "") for step in sample.get("tool_calls", []) if step.get("tool")]


def replay_disaster(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    input_path = args.input or "data/disaster_sft_augmented.json"
    mapping_path = args.mapping or "data/sft_augmented_image_mapping.json"
    source = load_json(input_path)
    image_index = build_image_index(load_json(mapping_path)) if mapping_path else {}
    samples = limited(source.get("samples", []), args.limit)
    skip_tools = normalize_skip_tools(args.skip_tools, "disaster")

    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    counters = {
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
        "skipped_total": 0,
        "skipped_changeos": 0,
        "success_replay": 0,
    }

    for sample in samples:
        tools = _disaster_tools(sample)
        if should_skip_tools(tools, skip_tools):
            counters["skipped_total"] += 1
            if any((canonical_tool(t) or t) == "geo_perception.change_os_detect" for t in tools):
                counters["skipped_changeos"] += 1
            continue

        produced_paths: set[str] = set()
        step_outputs: dict[int, Any] = {}
        replayed_calls: list[dict[str, Any]] = []
        failure_record: dict[str, Any] | None = None
        images = attach_disaster_images(sample, image_index)
        artifact_dir = repo_path(f"failure_synthesis/outputs/artifacts/disaster_replay/{sample.get('id', 'unknown')}")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for idx, step in enumerate(sample.get("tool_calls", []), start=1):
            tool = step.get("tool", "")
            step_args = prepare_disaster_replay_args(
                step.get("args", {}),
                images=images,
                step_outputs=step_outputs,
                artifact_dir=artifact_dir,
            )
            ok, result, failure_type = _run_step(tool, step_args, produced_paths)
            replay_step = copy.deepcopy(step)
            replay_step["step"] = idx
            replay_step["args"] = step_args
            replay_step["sample_output"] = result
            replay_step["is_error"] = not ok
            replayed_calls.append(replay_step)
            counters["by_tool"][canonical_tool(tool) or tool] += 1
            if not ok:
                failure_record = copy.deepcopy(sample)
                failure_record["id"] = f"{sample.get('id')}__replay_fail_step{idx}"
                failure_record["source_id"] = sample.get("id")
                failure_record["images"] = images
                failure_record["tool_calls"] = replayed_calls
                failure_record["failure_type"] = "source_gold_invalid"
                failure_record["failure_meta"] = {
                    "method": "replay",
                    "failure_type": "source_gold_invalid",
                    "replay_failure_type": failure_type,
                    "failed_step": idx,
                    "failed_tool": canonical_tool(tool) or tool,
                    "error": result,
                    "synthetic": False,
                }
                break
            step_outputs[idx] = result

        if failure_record is None:
            success_record = copy.deepcopy(sample)
            success_record["id"] = f"{sample.get('id')}__validated_gold"
            success_record["source_id"] = sample.get("id")
            success_record["images"] = images
            success_record["tool_calls"] = replayed_calls
            success_record["success"] = True
            success_record["validation_meta"] = {
                "method": "replay",
                "status": "validated_gold",
                "synthetic": False,
            }
            successes.append(success_record)
            counters["success_replay"] += 1
            continue
        failures.append(failure_record)
        counters["by_failure_type"][failure_record["failure_type"]] += 1
        counters["by_task_type"][failure_record.get("task_type", "unknown")] += 1

    wrapped = make_disaster_wrapper(source, failures, "Disaster SFT replay failure trajectories generated from real Terrabox tool execution.")
    success_wrapped = (
        make_disaster_wrapper(source, successes, "Disaster SFT validated gold trajectories generated from real Terrabox replay.")
        if args.success_output else None
    )
    return wrapped, {
        "dataset": "disaster",
        "method": "replay",
        "input": str(input_path),
        "output": str(args.output),
        "failures": len(failures),
        "validated_gold": len(successes),
        **{k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }, success_wrapped


def replay_openearth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]] | None]:
    input_path = args.input or "data/openearth/train.json"
    records = limited(load_json(input_path), args.limit)
    skip_tools = normalize_skip_tools(args.skip_tools, "openearth")

    failures: list[dict[str, Any]] = []
    counters = {
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
        "skipped_total": 0,
        "skipped_changeos": 0,
        "success_replay": 0,
    }

    for record in records:
        actions = parse_openearth_actions(record)
        raw_tools = [a["name"] for a in actions]
        if not actions:
            counters["skipped_total"] += 1
            continue
        if should_skip_tools(raw_tools, skip_tools):
            counters["skipped_total"] += 1
            if any((canonical_tool(t) or t) == "geo_perception.change_os_detect" for t in raw_tools):
                counters["skipped_changeos"] += 1
            continue

        produced_paths: set[str] = set()
        failure_item: dict[str, Any] | None = None

        for step_idx, action in enumerate(actions, start=1):
            tool = action["canonical"] or action["name"]
            ok, result, failure_type = _run_step(tool, copy.deepcopy(action.get("arguments", {})), produced_paths)
            counters["by_tool"][tool] += 1
            if ok:
                continue
            error_text = _result_text(result)
            failure_item = append_openearth_observation(record, error_text)
            failure_item["failure_id"] = f"oea_{record.get('idx')}__replay_fail_step{step_idx}"
            failure_item["source_idx"] = record.get("idx")
            failure_item["failure_meta"] = {
                "method": "replay",
                "failure_type": failure_type,
                "failed_step": step_idx,
                "failed_tool": tool,
                "error": result,
                "synthetic": False,
            }
            break

        if failure_item is None:
            counters["success_replay"] += 1
            continue
        failures.append(failure_item)
        counters["by_failure_type"][failure_item["failure_meta"]["failure_type"]] += 1
        counters["by_task_type"][record.get("type", "unknown")] += 1

    return failures, {
        "dataset": "openearth",
        "method": "replay",
        "input": str(input_path),
        "output": str(args.output),
        "failures": len(failures),
        **{k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--success-output", default=None, help="Optional path for validated replay successes.")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"failure_synthesis/outputs/{args.dataset}_replay_failures.json"

    data, summary, successes = replay_disaster(args) if args.dataset == "disaster" else replay_openearth(args)
    save_json(args.output, data)
    if args.success_output and successes is not None:
        save_json(args.success_output, successes)
    write_summary(summary)
    print(f"Saved {summary['failures']} replay failures to {args.output}")
    if args.success_output and successes is not None:
        n_success = len(successes.get("samples", [])) if isinstance(successes, dict) else len(successes)
        print(f"Saved {n_success} validated replay successes to {args.success_output}")


if __name__ == "__main__":
    main()
