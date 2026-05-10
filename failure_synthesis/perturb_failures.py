#!/usr/bin/env python3
"""Generate source-format synthetic failure trajectories by perturbing SFT demos."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        add_common_args,
        append_openearth_observation,
        canonical_tool,
        delete_openearth_action,
        init_random,
        limited,
        load_json,
        make_disaster_wrapper,
        mutate_numeric_args,
        normalize_skip_tools,
        openearth_action_names,
        parse_openearth_actions,
        replacement_openearth_action,
        replacement_tool,
        replace_openearth_action,
        save_json,
        should_skip_tools,
        write_summary,
    )
except ImportError:
    from common import (
        add_common_args,
        append_openearth_observation,
        canonical_tool,
        delete_openearth_action,
        init_random,
        limited,
        load_json,
        make_disaster_wrapper,
        mutate_numeric_args,
        normalize_skip_tools,
        openearth_action_names,
        parse_openearth_actions,
        replacement_openearth_action,
        replacement_tool,
        replace_openearth_action,
        save_json,
        should_skip_tools,
        write_summary,
    )


def _disaster_tool_sequence(sample: dict[str, Any]) -> list[str]:
    return [step.get("tool", "") for step in sample.get("tool_calls", []) if step.get("tool")]


def _renumber_steps(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(tool_calls)
    for idx, step in enumerate(out, start=1):
        step["step"] = idx
    return out


def _mark_disaster_failure(sample: dict[str, Any], source_id: str, failure_type: str, detail: str) -> dict[str, Any]:
    sample["source_id"] = source_id
    sample["failure_type"] = failure_type
    sample["failure_meta"] = {
        "method": "perturb",
        "failure_type": failure_type,
        "detail": detail,
        "expected_tools": _disaster_tool_sequence(sample),
        "synthetic": True,
    }
    return sample


def _disaster_drop_step(sample: dict[str, Any]) -> dict[str, Any] | None:
    calls = sample.get("tool_calls", [])
    if len(calls) < 2:
        return None
    drop_idx = 1 if len(calls) > 2 else 0
    out = copy.deepcopy(sample)
    removed = out["tool_calls"].pop(drop_idx)
    out["tool_calls"] = _renumber_steps(out["tool_calls"])
    out["id"] = f"{sample.get('id')}__fail_drop_step{drop_idx + 1}"
    return _mark_disaster_failure(out, sample.get("id", ""), "drop_step", f"removed {removed.get('tool')}")


def _disaster_swap_adjacent(sample: dict[str, Any]) -> dict[str, Any] | None:
    calls = sample.get("tool_calls", [])
    if len(calls) < 2:
        return None
    out = copy.deepcopy(sample)
    idx = 0
    out["tool_calls"][idx], out["tool_calls"][idx + 1] = out["tool_calls"][idx + 1], out["tool_calls"][idx]
    out["tool_calls"] = _renumber_steps(out["tool_calls"])
    out["id"] = f"{sample.get('id')}__fail_swap_step{idx + 1}_{idx + 2}"
    return _mark_disaster_failure(out, sample.get("id", ""), "swap_adjacent", "swapped adjacent tool order")


def _disaster_replace_tool(sample: dict[str, Any]) -> dict[str, Any] | None:
    calls = sample.get("tool_calls", [])
    if not calls:
        return None
    out = copy.deepcopy(sample)
    idx = 0
    original = out["tool_calls"][idx].get("tool", "")
    replacement = replacement_tool(original)
    out["tool_calls"][idx]["tool"] = replacement
    out["tool_calls"][idx]["note"] = f"[FAILURE SYNTHESIS] replaced {original} with {replacement}"
    out["id"] = f"{sample.get('id')}__fail_replace_step{idx + 1}"
    return _mark_disaster_failure(out, sample.get("id", ""), "replace_tool", f"{original} -> {replacement}")


def _disaster_mutate_args(sample: dict[str, Any]) -> dict[str, Any] | None:
    for idx, step in enumerate(sample.get("tool_calls", [])):
        changed = mutate_numeric_args(step.get("args", {}))
        if changed is None:
            continue
        new_args, detail = changed
        out = copy.deepcopy(sample)
        out["tool_calls"][idx]["args"] = new_args
        out["tool_calls"][idx]["note"] = f"[FAILURE SYNTHESIS] {detail}"
        out["id"] = f"{sample.get('id')}__fail_args_step{idx + 1}"
        return _mark_disaster_failure(out, sample.get("id", ""), "mutate_args", detail)
    return None


def generate_disaster(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    input_path = args.input or "data/disaster_sft_augmented.json"
    source = load_json(input_path)
    samples = limited(source.get("samples", []), args.limit)
    skip_tools = normalize_skip_tools(args.skip_tools, "disaster")
    variants = [_disaster_drop_step, _disaster_swap_adjacent, _disaster_replace_tool, _disaster_mutate_args]
    records: list[dict[str, Any]] = []
    counters = {
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
        "skipped_total": 0,
        "skipped_changeos": 0,
    }

    for sample in samples:
        tools = _disaster_tool_sequence(sample)
        if should_skip_tools(tools, skip_tools):
            counters["skipped_total"] += 1
            if any((canonical_tool(t) or t) == "geo_perception.change_os_detect" for t in tools):
                counters["skipped_changeos"] += 1
            continue
        made = 0
        for maker in variants:
            if made >= args.variants_per_sample:
                break
            record = maker(sample)
            if record is None:
                continue
            records.append(record)
            made += 1
            counters["by_failure_type"][record["failure_type"]] += 1
            counters["by_task_type"][record.get("task_type", "unknown")] += 1
            for tool in _disaster_tool_sequence(record):
                counters["by_tool"][tool] += 1

    wrapped = make_disaster_wrapper(source, records, "Disaster SFT synthetic failure trajectories generated by deterministic perturbation.")
    return wrapped, {
        "dataset": "disaster",
        "method": "perturb",
        "input": str(input_path),
        "output": str(args.output),
        "failures": len(records),
        **{k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }


def _openearth_drop_action(record: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    out = delete_openearth_action(record, action)
    return append_openearth_observation(out, f"synthetic failure: removed key action {action['name']}")


def _openearth_replace_action(record: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    replacement = replacement_openearth_action(action["name"])
    new_action = {
        "name": replacement,
        "arguments": copy.deepcopy(action.get("arguments", {})),
    }
    out = replace_openearth_action(record, action, new_action)
    return append_openearth_observation(out, f"synthetic failure: replaced {action['name']} with {replacement}")


def _openearth_mutate_args(record: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    changed = mutate_numeric_args(action.get("arguments", {}))
    if changed is None:
        return None
    new_args, detail = changed
    new_action = {"name": action["name"], "arguments": new_args}
    out = replace_openearth_action(record, action, new_action)
    return append_openearth_observation(out, f"synthetic failure: {detail} in {action['name']}")


def _openearth_swap_actions(record: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(actions) < 2:
        return None
    first, second = actions[0], actions[1]
    if first["turn_index"] == second["turn_index"]:
        out = copy.deepcopy(record)
        turn = out["conversation"][first["turn_index"]]
        payload = json.loads(turn.get("value", "{}"))
        items = payload.get("actions", [])
        i, j = first["action_index"], second["action_index"]
        items[i], items[j] = items[j], items[i]
        turn["value"] = json.dumps(payload, ensure_ascii=False)
    else:
        out = copy.deepcopy(record)
        first_turn = out["conversation"][first["turn_index"]]
        second_turn = out["conversation"][second["turn_index"]]
        first_payload = json.loads(first_turn.get("value", "{}"))
        second_payload = json.loads(second_turn.get("value", "{}"))
        first_action = first_payload["actions"][first["action_index"]]
        second_action = second_payload["actions"][second["action_index"]]
        first_payload["actions"][first["action_index"]] = second_action
        second_payload["actions"][second["action_index"]] = first_action
        first_turn["value"] = json.dumps(first_payload, ensure_ascii=False)
        second_turn["value"] = json.dumps(second_payload, ensure_ascii=False)
    return append_openearth_observation(out, "synthetic failure: swapped adjacent tool order")


def _mark_openearth_failure(record: dict[str, Any], source_idx: Any, failure_id: str, failure_type: str, detail: str) -> dict[str, Any]:
    record["failure_id"] = failure_id
    record["source_idx"] = source_idx
    record["failure_meta"] = {
        "method": "perturb",
        "failure_type": failure_type,
        "detail": detail,
        "synthetic": True,
    }
    return record


def generate_openearth(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = args.input or "data/openearth/train.json"
    records = limited(load_json(input_path), args.limit)
    skip_tools = normalize_skip_tools(args.skip_tools, "openearth")
    output: list[dict[str, Any]] = []
    counters = {
        "by_failure_type": Counter(),
        "by_tool": Counter(),
        "by_task_type": Counter(),
        "skipped_total": 0,
        "skipped_changeos": 0,
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

        source_idx = record.get("idx")
        variants: list[tuple[str, str, dict[str, Any] | None]] = [
            ("drop_action", f"removed {actions[0]['name']}", _openearth_drop_action(record, actions[0])),
            ("swap_adjacent", "swapped adjacent tool order", _openearth_swap_actions(record, actions)),
            ("replace_tool", f"replaced {actions[0]['name']}", _openearth_replace_action(record, actions[0])),
        ]
        mutated = _openearth_mutate_args(record, actions[0])
        if mutated is not None:
            variants.append(("mutate_args", f"mutated args for {actions[0]['name']}", mutated))

        made = 0
        for failure_type, detail, item in variants:
            if made >= args.variants_per_sample:
                break
            if item is None:
                continue
            failure_id = f"oea_{source_idx}__fail_{failure_type}_{made + 1}"
            marked = _mark_openearth_failure(item, source_idx, failure_id, failure_type, detail)
            output.append(marked)
            made += 1
            counters["by_failure_type"][failure_type] += 1
            counters["by_task_type"][record.get("type", "unknown")] += 1
            for tool in openearth_action_names(marked):
                mapped = canonical_tool(tool)
                if mapped:
                    counters["by_tool"][mapped] += 1

    return output, {
        "dataset": "openearth",
        "method": "perturb",
        "input": str(input_path),
        "output": str(args.output),
        "failures": len(output),
        **{k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--variants-per-sample", type=int, default=2)
    args = parser.parse_args()
    init_random(args.seed)

    if args.output is None:
        args.output = f"failure_synthesis/outputs/{args.dataset}_perturb_failures.json"

    data, summary = generate_disaster(args) if args.dataset == "disaster" else generate_openearth(args)
    save_json(args.output, data)
    write_summary(summary)
    print(f"Saved {summary['failures']} perturb failures to {args.output}")


if __name__ == "__main__":
    main()
