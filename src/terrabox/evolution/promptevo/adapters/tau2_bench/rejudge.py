"""Post-hoc LLM judging for tau2 natural-language assertions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .files import _iter_result_paths, _load_results_like, _write_json
from .traces import _message_text, _sim_task_id, _task_lookup


_SYSTEM = """TASK
- You will be given expected outcomes and a conversation collected during a customer-service test case.
- Evaluate whether the agent satisfies each expected outcome.
- Grade each expected outcome individually using only the conversation.

FORMAT
Return strict JSON only. Refer to outcomes by their numeric index; do not repeat the outcome text:
{"results":[{"index":0,"met":true,"reason":"short explanation"}]}
"""


def _conversation_text(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '')}: {_message_text(m)}" for m in messages)


def _cache_key(domain: str, sim: dict, assertions: list[str], model: str) -> str:
    payload = {
        "domain": domain,
        "task_id": sim.get("task_id"),
        "trial": sim.get("trial"),
        "conversation": _conversation_text(sim.get("messages") or []),
        "assertions": assertions,
        "model": model,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _extract_json_object(raw: str) -> dict[str, Any]:
    for value in _extract_json_values(raw):
        if isinstance(value, dict):
            return value
    raise ValueError("judge response contains no valid JSON object")


def _extract_json_values(raw: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    text = raw or ""
    while index < len(text):
        starts = [pos for pos in (text.find("{", index), text.find("[", index)) if pos >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + max(end, 1)
    if not values:
        raise ValueError("judge response contains no valid JSON value")
    return values


def _judge_payloads(raw: str) -> list[dict[str, Any]]:
    values = _extract_json_values(raw)
    payloads: list[dict[str, Any]] = []
    row_objects: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, list):
            payloads.append({"results": value})
        elif isinstance(value, dict):
            payloads.append(value)
            if "index" in value:
                row_objects.append(value)
    if len(row_objects) > 1:
        payloads.insert(0, {"results": row_objects})
    return payloads


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"judge returned a non-boolean verdict: {value!r}")


def _normalize_checks(data: dict[str, Any], assertions: list[str]) -> list[dict[str, Any]]:
    rows = None
    if isinstance(data, dict):
        for key in ("results", "evaluations", "checks", "outcomes", "judgments"):
            if key in data:
                rows = data[key]
                break
        if rows is None and "index" in data:
            rows = [data]
        if isinstance(rows, dict):
            rows = [
                ({"index": index, **value} if isinstance(value, dict) else {"index": index, "met": value})
                for index, value in rows.items()
            ]
    if not isinstance(rows, list):
        raise ValueError("judge response is missing results[]")
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "index" not in row:
            continue
        try:
            by_index[int(row["index"])] = row
        except (TypeError, ValueError):
            continue
    by_expected = {
        str(row.get("expectedOutcome", "")).strip(): row
        for row in rows
        if isinstance(row, dict)
    }
    checks = []
    for index, assertion in enumerate(assertions):
        row = by_index.get(index) or by_expected.get(assertion.strip())
        if row is None:
            raise ValueError(f"judge omitted expected outcome index {index}: {assertion}")
        checks.append(
            {
                "nl_assertion": assertion,
                "met": _as_bool(row.get("met", row.get("metExpectation"))),
                "justification": str(row.get("reason") or row.get("reasoning") or ""),
            }
        )
    return checks


def _recompute_reward(reward_info: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    nl_reward = 1.0 if all(row["met"] for row in checks) else 0.0
    reward_info["nl_assertions"] = checks
    breakdown = dict(reward_info.get("reward_breakdown") or {})
    breakdown["NL_ASSERTION"] = nl_reward
    reward_info["reward_breakdown"] = breakdown
    basis = [str(item) for item in (reward_info.get("reward_basis") or [])]
    if "NL_ASSERTION" in basis:
        reward_info["reward"] = math.prod(float(breakdown.get(item, 0.0)) for item in basis)


def rejudge_results(
    source: str,
    output_dir: str,
    *,
    client,
    provider: str,
    model: str,
    max_tokens: int = 3000,
    retries: int = 5,
) -> dict[str, Any]:
    """Rejudge all NL assertions and materialize a separate results tree."""

    source_root = Path(source).resolve()
    output_root = Path(output_dir).resolve()
    cache_dir = output_root / "judge_cache" / provider
    cache_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "simulations": 0,
        "judged": 0,
        "cache_hits": 0,
        "changed": 0,
        "files": 0,
        "skipped_non_scoring": 0,
        "judge_failures": 0,
    }

    for result_path in _iter_result_paths(str(source_root)):
        path = Path(result_path).resolve()
        if path == output_root or output_root in path.parents:
            continue
        meta, simulations = _load_results_like(str(path))
        if not simulations:
            continue
        tasks = _task_lookup(meta.get("tasks") or [])
        domain = (((meta.get("info") or {}).get("environment_info") or {}).get("domain_name") or "")
        copied = json.loads(json.dumps(meta, ensure_ascii=False))
        copied_sims = copied.get("simulations") or []

        for index, sim in enumerate(simulations):
            totals["simulations"] += 1
            task = tasks.get(str(sim.get("task_id"))) or {}
            criteria = task.get("evaluation_criteria") or {}
            assertions = [str(x) for x in (criteria.get("nl_assertions") or []) if str(x).strip()]
            if not assertions:
                continue
            reward_basis = [
                str(item)
                for item in (
                    criteria.get("reward_basis")
                    or (sim.get("reward_info") or {}).get("reward_basis")
                    or []
                )
            ]
            if "NL_ASSERTION" not in reward_basis:
                totals["skipped_non_scoring"] += 1
                continue
            key = _cache_key(domain, sim, assertions, model)
            cache_path = cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                checks = cached["checks"]
                totals["cache_hits"] += 1
            else:
                prompt = (
                    f"conversation:\n{_conversation_text(sim.get('messages') or [])}\n\n"
                    f"expectedOutcomesByIndex:\n"
                    f"{json.dumps(list(enumerate(assertions)), ensure_ascii=False)}"
                )
                last_error: Exception | None = None
                last_raw = ""
                for attempt in range(retries):
                    try:
                        retry_note = ""
                        if last_error is not None:
                            retry_note = (
                                f"\n\nThe previous response was invalid: {last_error}. "
                                "Return one compact valid JSON object only."
                            )
                        last_raw = client.call(prompt + retry_note, system=_SYSTEM, max_tokens=max_tokens)
                        payload_errors = []
                        for data in _judge_payloads(last_raw):
                            try:
                                checks = _normalize_checks(data, assertions)
                                break
                            except Exception as payload_exc:
                                payload_errors.append(str(payload_exc))
                        else:
                            raise ValueError("; ".join(payload_errors) or "judge response has no usable payload")
                        _write_json(
                            cache_path,
                            {
                                "provider": provider,
                                "model": model,
                                "task_key": _sim_task_id(sim, str(path), domain),
                                "checks": checks,
                            },
                        )
                        break
                    except Exception as exc:  # external API/transient parse failure
                        last_error = exc
                        if attempt + 1 < retries:
                            time.sleep(2 ** attempt)
                else:
                    totals["judge_failures"] += 1
                    failure_key = _sim_task_id(sim, str(path), domain).replace("/", "_").replace(":", "_")
                    _write_json(
                        output_root / "judge_failures" / provider / f"{failure_key}.json",
                        {
                            "provider": provider,
                            "model": model,
                            "domain": domain,
                            "task_id": sim.get("task_id"),
                            "trial": sim.get("trial"),
                            "assertions": assertions,
                            "error": str(last_error),
                            "raw_response": last_raw,
                            "policy": "preserve_original_reward_info",
                        },
                    )
                    continue
            totals["judged"] += 1
            target = copied_sims[index]
            before = float((target.get("reward_info") or {}).get("reward") or 0.0)
            target.setdefault("reward_info", {})
            _recompute_reward(target["reward_info"], checks)
            after = float(target["reward_info"].get("reward") or 0.0)
            totals["changed"] += int(before != after)

        try:
            relative = path.relative_to(source_root)
        except ValueError:
            relative = Path(path.parent.name) / path.name
        out_path = output_root / "results" / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8")
        totals["files"] += 1

    summary = {
        **totals,
        "source": str(source_root),
        "results_dir": str(output_root / "results"),
        "provider": provider,
        "model": model,
    }
    _write_json(output_root / "rejudge_summary.json", summary)
    return summary
