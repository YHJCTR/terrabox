"""Build an ACE-style playbook from Terrabox rollout results.

This is a dependency-light adaptation of ACE for OEA: instead of running ACE's
full Generator/Reflector/Curator stack, it converts train rollout feedback into
playbook bullets with helpful/harmful counters. It intentionally reads only
historical rollout behavior and metrics, never eval gold answers.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .playbook import ACEPlaybook, PlaybookBullet, tokenize


logger = logging.getLogger(__name__)


_INFRA_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "quota",
    "billing",
    "payment",
    "oom",
    "out of memory",
    "cuda",
    "connection",
    "network",
    "provider",
    "context length",
    "docker",
    "service health",
)


def _load_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    paths = sorted(root.glob("*.json")) if root.is_dir() else [root]
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id") or "unknown")


def _metrics_f1(row: dict[str, Any]) -> float:
    try:
        return float((row.get("metrics") or {}).get("f1", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _task_type(row: dict[str, Any]) -> str:
    return str(row.get("task_type") or "unknown")


def _tool_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("name") or item.get("tool") or item.get("slug") or "")
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item or "")


def _compact_sequence(row: dict[str, Any], max_len: int = 10) -> list[str]:
    raw = row.get("tool_calls_deduped") or row.get("tool_sequence") or row.get("tool_calls") or []
    out: list[str] = []
    seen_consecutive = None
    for item in raw:
        name = _tool_name(item).strip()
        if not name or name == "final_answer":
            continue
        if name == seen_consecutive:
            continue
        out.append(name)
        seen_consecutive = name
        if len(out) >= max_len:
            break
    return out


def _is_infra_failure(row: dict[str, Any]) -> bool:
    if row.get("has_tool_oom"):
        return True
    text = json.dumps(
        {
            "status": row.get("status"),
            "error": row.get("error"),
            "final": row.get("final_answer_full") or row.get("final_answer_preview"),
            "history": row.get("conversation_history") or [],
        },
        ensure_ascii=False,
    ).lower()
    return any(marker in text for marker in _INFRA_MARKERS)


def _success_like(row: dict[str, Any], min_f1: float) -> bool:
    return bool(row.get("real_success") or row.get("success")) and _metrics_f1(row) >= min_f1


def _keywords(rows: list[dict[str, Any]], tools: list[str], limit: int = 18) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(tokenize(row.get("question") or ""))
        counter.update(tokenize(_task_type(row)))
    counter.update(tokenize(" ".join(tools)))
    return [token for token, _ in counter.most_common(limit)]


def _sanitize(text: object, limit: int = 480) -> str:
    value = str(text or "")
    value = value.replace("```", "`")
    value = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", value)
    value = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_id>", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def _tool_family(tool: str) -> str:
    if tool.startswith("geo_perception."):
        return "perception"
    if tool.startswith("osm_gis."):
        return "gis"
    if tool.startswith("compute.") or tool.startswith("ipython."):
        return "logic"
    if tool.startswith("bing_search."):
        return "search"
    return tool.split(".", 1)[0]


def _flow_text(task_type: str, tools: list[str]) -> str:
    flow = " -> ".join(tools)
    families = {_tool_family(tool) for tool in tools}
    clauses: list[str] = []
    if "gis" in families:
        clauses.append("keep the generated GeoPackage/layer artifact from each GIS step and pass it to the next GIS or visualization step")
    if "perception" in families:
        clauses.append("obtain object masks/counts/attribute evidence with perception tools before any numeric or final answer")
    if "logic" in families:
        clauses.append("route arithmetic, thresholds, distances, and percentages through compute tools instead of estimating them in text")
    if any(tool.endswith(("display_on_map", "show_index_layer", "draw_bboxes", "add_text", "plot")) for tool in tools):
        clauses.append("finish generation/display tasks with the required artifact-producing tool rather than stopping at intermediate analysis")
    detail = "; ".join(clauses) if clauses else "preserve intermediate artifacts and answer only after tool evidence exists"
    return f"For {task_type}-like OEA tasks, prefer the proven tool flow `{flow}`; {detail}."


def _caution_text(task_type: str, tools: list[str], reason: str) -> str:
    tail = " -> ".join(tools[-4:]) if tools else "the previous failing tool call"
    if reason == "tool_error":
        return (
            f"For {task_type}-like tasks, if `{tail}` returns a schema/artifact error, change the missing or invalid input "
            "before retrying; do not repeat the same call or finalize from a failed observation."
        )
    if reason == "low_f1":
        return (
            f"For {task_type}-like tasks, a partial flow ending in `{tail}` often misses required tools; check whether a downstream "
            "computation or generation artifact is still needed before finalizing."
        )
    return f"For {task_type}-like tasks, avoid overusing `{tail}` without new evidence or a changed artifact state."


def _episode_summary(row: dict[str, Any]) -> dict[str, Any]:
    tools = _compact_sequence(row)
    status = str(row.get("status") or "unknown")
    f1 = _metrics_f1(row)
    outcome = "success" if _success_like(row, 0.6) else "partial_or_failed"
    if _is_infra_failure(row):
        outcome = "infra_issue"
    return {
        "task_id": _row_id(row),
        "task_type": _task_type(row),
        "request": _sanitize(row.get("question"), 520),
        "actual_tool_flow": tools,
        "status": status,
        "tool_f1_feedback": round(f1, 3),
        "has_tool_error": bool(row.get("has_tool_error")),
        "infra_issue": bool(_is_infra_failure(row)),
        "outcome": outcome,
        "final_answer_excerpt": _sanitize(row.get("final_answer_full") or row.get("final_answer_preview"), 300),
    }


def _select_diverse_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one representative per task_type/tool-flow first, then the rest."""
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tools = tuple(_compact_sequence(row))
        if not tools:
            continue
        groups[(_task_type(row), tools)].append(row)
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _, items in sorted(groups.items(), key=lambda item: str(item[0])):
        items.sort(key=lambda row: (_success_like(row, 0.6), _metrics_f1(row)), reverse=True)
        chosen = items[0]
        selected.append(chosen)
        seen_ids.add(_row_id(chosen))
    rest = [row for row in rows if _row_id(row) not in seen_ids]
    rest.sort(key=lambda row: (_task_type(row), -_metrics_f1(row), _row_id(row)))
    return selected + rest


def _playbook_stats(playbook: ACEPlaybook) -> dict[str, Any]:
    return {
        "n_bullets": len(playbook.bullets),
        "sections": dict(Counter(b.section for b in playbook.bullets)),
        "task_types": len({t for b in playbook.bullets for t in b.task_types}),
    }


def _playbook_excerpt(playbook: ACEPlaybook, max_chars: int = 7000) -> str:
    lines = [bullet.as_prompt_line() for bullet in playbook.bullets[-80:]]
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _call_json_resilient(llm: Any, prompt: str, *, system: str, max_tokens: int, retries: int = 2) -> dict[str, Any] | None:
    for attempt in range(retries + 1):
        data = llm.call_json(prompt, system=system, max_tokens=max_tokens)
        if isinstance(data, dict):
            return data
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


def _candidate_bullet_ids(playbook: ACEPlaybook, episodes: list[dict[str, Any]], *, per_episode: int = 4, limit: int = 24) -> list[str]:
    """Select playbook bullets that play the role of ACE `bullets_used`.

    In official ACE, the Generator returns bullet IDs it used while answering a
    sample. OEA train episodes here already exist, so we retrieve the bullets
    that would have been visible for the same question/task and let Reflector
    tag those bullets against environment feedback.
    """
    selected: list[str] = []
    seen: set[str] = set()
    for episode in episodes:
        matches = playbook.retrieve(
            str(episode.get("request") or ""),
            task_type=str(episode.get("task_type") or "unknown"),
            available_tools=episode.get("actual_tool_flow") or [],
            top_k=per_episode,
        )
        for _, bullet in matches:
            if bullet.id in seen:
                continue
            seen.add(bullet.id)
            selected.append(bullet.id)
            if len(selected) >= limit:
                return selected
    return selected


def _write_batch_trace(output_dir: str | Path, batch_idx: int, payload: dict[str, Any]) -> None:
    trace_dir = Path(output_dir) / "ace_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / "batches.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"batch": batch_idx, **payload}, ensure_ascii=False) + "\n")


def _write_intermediate_playbook(playbook: ACEPlaybook, batch_idx: int) -> None:
    playbook_dir = playbook.store_dir / "intermediate_playbooks"
    playbook_dir.mkdir(parents=True, exist_ok=True)
    (playbook_dir / f"batch_{batch_idx:04d}_playbook.txt").write_text(playbook.as_text(), encoding="utf-8")


def _reflect_batch(llm: Any, episodes: list[dict[str, Any]], bullets_used: str) -> dict[str, Any] | None:
    prompt = (
        "You are the REFLECTOR in the ACE framework, adapted to a geospatial tool agent.\n"
        "Analyze historical train rollout episodes and diagnose reusable tool-use lessons from environment feedback.\n"
        "Tag the supplied playbook bullets as helpful, harmful, or neutral for these episodes, matching ACE's counter update layer.\n"
        "Do not use or mention gold tool traces, ground-truth answers, task ids, exact paths, or place-specific facts. Treat tool_f1_feedback/status/tool errors only as environment feedback.\n"
        "Return JSON only with fields: reasoning, error_identification, root_cause_analysis, correct_approach, key_insight, bullet_tags, insights.\n"
        "Each bullet_tags item has id and tag in ['helpful','harmful','neutral']. Each insights item has: section, content, task_types, tools, polarity ('helpful' or 'caution'), evidence_strength (0-1).\n"
        "Insights must be actionable playbook candidates, not summaries of individual episodes.\n\n"
        f"Part of Playbook used by the generator:\n{bullets_used}\n\n"
        f"Episodes:\n{json.dumps(episodes, ensure_ascii=False, indent=2)}"
    )
    return _call_json_resilient(
        llm,
        prompt,
        system="You are ACE Reflector for a Terrabox geospatial tool-use agent.",
        max_tokens=1800,
    )


def _curate_batch(llm: Any, playbook: ACEPlaybook, reflection: dict[str, Any], episodes: list[dict[str, Any]], token_budget: int, step: int, total: int) -> dict[str, Any] | None:
    prompt = (
        "You are the CURATOR in the ACE framework. Convert the reflection into incremental playbook operations.\n"
        "Only ADD genuinely new bullets missing from the current playbook. Do not rewrite the full playbook. Avoid duplicate or over-specific bullets.\n"
        "Each bullet must be transferable across OEA-style geospatial tasks, concise, and directly useful at inference time.\n"
        "Never include task ids, exact locations, file paths, exact numeric answers, gold traces, or benchmark leakage.\n"
        "Return JSON only: {\"reasoning\": \"brief rationale\", \"operations\": [{\"type\": \"ADD\", \"section\": \"tool_flow|artifact_handoff|parameter_discipline|termination|caution\", \"content\": \"...\", \"task_types\": [\"...\"], \"tools\": [\"tool.slug\"], \"polarity\": \"helpful|caution\"}]}.\n\n"
        f"Training progress: sample batch {step} of {total}; token budget {token_budget}\n"
        f"Current playbook stats:\n{json.dumps(_playbook_stats(playbook), ensure_ascii=False)}\n"
        f"Current playbook excerpt:\n{_playbook_excerpt(playbook)}\n\n"
        f"Recent reflection:\n{json.dumps(reflection, ensure_ascii=False, indent=2)}\n\n"
        f"Question context summaries:\n{json.dumps([{'task_type': e['task_type'], 'tool_flow': e['actual_tool_flow'], 'outcome': e['outcome']} for e in episodes], ensure_ascii=False, indent=2)}"
    )
    return _call_json_resilient(
        llm,
        prompt,
        system="You are ACE Curator for a Terrabox geospatial playbook.",
        max_tokens=1600,
    )


def _operation_to_bullet(op: dict[str, Any], batch_rows: list[dict[str, Any]], next_id: int) -> PlaybookBullet | None:
    content = str(op.get("content") or "").strip()
    if not content:
        return None
    section = str(op.get("section") or "tool_flow").strip().lower().replace(" ", "_")
    if section not in {"tool_flow", "artifact_handoff", "parameter_discipline", "termination", "caution"}:
        section = "tool_flow"
    polarity = str(op.get("polarity") or "helpful").lower()
    task_types = [str(x) for x in op.get("task_types") or [] if str(x).strip()]
    if not task_types:
        task_types = sorted({_task_type(row) for row in batch_rows})[:6]
    tools = [str(x) for x in op.get("tools") or [] if str(x).strip()]
    if not tools:
        counter: Counter[str] = Counter()
        for row in batch_rows:
            counter.update(_compact_sequence(row))
        tools = [tool for tool, _ in counter.most_common(8)]
    helpful = sum(1 for row in batch_rows if _success_like(row, 0.6)) if polarity != "caution" else 0
    harmful = sum(1 for row in batch_rows if not _success_like(row, 0.6) and not _is_infra_failure(row)) if polarity == "caution" else 0
    prefix = {"caution": "cau", "termination": "ter", "artifact_handoff": "art", "parameter_discipline": "par"}.get(section, "str")
    return PlaybookBullet(
        id=f"{prefix}-{next_id:05d}",
        text=content,
        helpful=max(0, helpful),
        harmful=max(0, harmful),
        support=len(batch_rows),
        task_types=task_types,
        tools=tools,
        keywords=sorted(tokenize(" ".join([content, " ".join(task_types), " ".join(tools)])))[:24],
        source_tasks=[_row_id(row) for row in batch_rows[:12]],
        section=section,
    )


def _fallback_ops_from_reflection(reflection: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for insight in reflection.get("insights") or []:
        if not isinstance(insight, dict):
            continue
        content = str(insight.get("content") or "").strip()
        if not content:
            continue
        ops.append(
            {
                "type": "ADD",
                "section": insight.get("section") or ("caution" if insight.get("polarity") == "caution" else "tool_flow"),
                "content": content,
                "task_types": insight.get("task_types") or [],
                "tools": insight.get("tools") or [],
                "polarity": insight.get("polarity") or "helpful",
            }
        )
    return ops


def build_playbook_ace_official_style(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    provider: str = "longcat",
    max_bullets: int = 160,
    batch_size: int = 10,
    max_batches: int | None = None,
    token_budget: int = 80000,
    force: bool = False,
) -> ACEPlaybook:
    """Build an ACE-style playbook with LongCat Reflector/Curator calls.

    This follows ACE's split between reflection and curation while adapting the
    generator step to pre-existing train rollout episodes.
    """
    from terrabox.agent.llm_provider import make_llm_client

    rows = _select_diverse_rows(_load_rows(results_dir))
    if max_batches is not None:
        rows = rows[: max(0, max_batches) * batch_size]

    playbook = ACEPlaybook(output_dir)
    if force:
        root = Path(output_dir)
        for stale_file in (root / "playbook.json", root / "playbook.txt"):
            if stale_file.exists():
                stale_file.unlink()
        for stale_dir in (root / "ace_traces", root / "intermediate_playbooks"):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        playbook.bullets = []
        playbook.manifest = {}
    processed = set(playbook.manifest.get("processed_task_ids") or [])
    completed = playbook.manifest.get("build_status") == "complete"
    if completed and not force:
        logger.info("ACE official-style playbook already complete: %s", playbook.path)
        return playbook

    llm = make_llm_client(provider)
    total_batches = (len(rows) + batch_size - 1) // batch_size
    existing_text = {bullet.text.strip().lower() for bullet in playbook.bullets}
    next_id = len(playbook.bullets) + 1
    playbook.manifest = {
        **playbook.manifest,
        "method": "ace_playbook",
        "adapter": "ACE official-style Terrabox playbook",
        "official_source": "/data1/yuhongjie2/external_repos/ace",
        "source_results": str(Path(results_dir).resolve()),
        "uses_gold": False,
        "provider": provider,
        "build_mode": "ace_official_style",
        "build_status": "in_progress",
        "n_rows_available": len(rows),
        "batch_size": batch_size,
        "max_batches": max_batches,
        "created_at": playbook.manifest.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        "processed_task_ids": sorted(processed),
        "llm_failures": list(playbook.manifest.get("llm_failures") or []),
        "counter_updates": playbook.manifest.get(
            "counter_updates", {"helpful": 0, "harmful": 0, "neutral": 0, "unknown": 0}
        ),
        "ace_mechanisms_reproduced": [
            "playbook bullets in official '[id] helpful=X harmful=Y :: content' format",
            "Reflector diagnostics from generator episodes and environment feedback",
            "Reflector bullet_tags for helpful/harmful/neutral counter updates",
            "Curator incremental ADD operations rather than full playbook rewriting",
            "intermediate playbook snapshots and batch trace logging",
            "offline/eval-only split with no eval gold leakage",
        ],
        "oea_adaptations": [
            "Generator episodes are pre-existing LongCat OEA train rollouts instead of rerunning ACE's native generator class",
            "bullets_used are approximated by retrieving bullets that would be visible for each historical train episode",
            "OEA environment feedback is rollout status, tool F1, tool errors, and final-answer excerpt; gold answers are not used",
            "Terrabox eval injects retrieved playbook bullets into the ReAct system prompt rather than requiring generator JSON with bullet_ids",
        ],
        "notes": "Uses LongCat Reflector then Curator with ACE-style bullet tagging/counter updates from train rollout feedback; no eval gold/answer is used.",
    }
    playbook.save()

    for batch_idx in range(total_batches):
        batch_rows = rows[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_rows = [row for row in batch_rows if _row_id(row) not in processed]
        if not batch_rows:
            continue
        episodes = [_episode_summary(row) for row in batch_rows]
        logger.info("ACE official-style batch %d/%d (%d episodes)", batch_idx + 1, total_batches, len(episodes))
        batch_number = batch_idx + 1
        used_ids = _candidate_bullet_ids(playbook, episodes)
        bullets_used = playbook.format_bullets(used_ids)
        try:
            reflection = _reflect_batch(llm, episodes, bullets_used)
            if not reflection:
                raise RuntimeError("empty reflection JSON")
            tag_counts = playbook.update_bullet_counts(reflection.get("bullet_tags") or [])
            total_counter_updates = dict(playbook.manifest.get("counter_updates") or {})
            for key in ("helpful", "harmful", "neutral", "unknown"):
                total_counter_updates[key] = int(total_counter_updates.get(key, 0)) + int(tag_counts.get(key, 0))
            playbook.manifest["counter_updates"] = total_counter_updates
            curated = _curate_batch(llm, playbook, reflection, episodes, token_budget, batch_idx + 1, total_batches)
            operations = (curated or {}).get("operations") or []
            if not operations:
                operations = _fallback_ops_from_reflection(reflection)
            added = 0
            for op in operations:
                if not isinstance(op, dict) or str(op.get("type") or "ADD").upper() != "ADD":
                    continue
                bullet = _operation_to_bullet(op, batch_rows, next_id)
                if bullet is None:
                    continue
                key = bullet.text.strip().lower()
                if key in existing_text:
                    continue
                existing_text.add(key)
                playbook.bullets.append(bullet)
                next_id += 1
                added += 1
                if len(playbook.bullets) >= max_bullets:
                    break
            _write_batch_trace(
                output_dir,
                batch_number,
                {
                    "task_ids": [_row_id(row) for row in batch_rows],
                    "used_bullet_ids": used_ids,
                    "tag_counts": tag_counts,
                    "reflection": reflection,
                    "curator": curated,
                    "operations": operations,
                    "added": added,
                    "n_bullets_after": len(playbook.bullets),
                },
            )
            _write_intermediate_playbook(playbook, batch_number)
            logger.info(
                "ACE official-style batch %d tagged=%s added=%d bullets",
                batch_idx + 1,
                tag_counts,
                added,
            )
        except Exception as exc:  # keep long builds resumable; do not hide in manifest.
            failure = {"batch": batch_idx + 1, "error": repr(exc), "task_ids": [_row_id(row) for row in batch_rows]}
            logger.warning("ACE official-style batch failed: %s", failure)
            playbook.manifest.setdefault("llm_failures", []).append(failure)
            _write_batch_trace(
                output_dir,
                batch_number,
                {
                    "task_ids": [_row_id(row) for row in batch_rows],
                    "used_bullet_ids": used_ids,
                    "error": repr(exc),
                    "n_bullets_after": len(playbook.bullets),
                },
            )
        processed.update(_row_id(row) for row in batch_rows)
        playbook.manifest["processed_task_ids"] = sorted(processed)
        playbook.manifest["n_bullets"] = len(playbook.bullets)
        playbook.save()
        if len(playbook.bullets) >= max_bullets:
            break

    playbook.manifest["build_status"] = "complete"
    playbook.manifest["n_bullets"] = len(playbook.bullets)
    playbook.manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    playbook.save()
    return playbook


def build_playbook(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    max_bullets: int = 120,
    min_success_f1: float = 0.6,
    max_source_tasks: int = 12,
) -> ACEPlaybook:
    rows = _load_rows(results_dir)
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tools = _compact_sequence(row)
        if not tools:
            continue
        groups[(_task_type(row), tuple(tools))].append(row)

    bullets: list[PlaybookBullet] = []
    for (task_type, tools_tuple), items in groups.items():
        tools = list(tools_tuple)
        helpful_rows = [row for row in items if _success_like(row, min_success_f1)]
        harmful_rows = [row for row in items if not _success_like(row, min_success_f1) and not _is_infra_failure(row)]
        if helpful_rows:
            bullets.append(
                PlaybookBullet(
                    id="",
                    text=_flow_text(task_type, tools),
                    helpful=len(helpful_rows),
                    harmful=len(harmful_rows),
                    support=len(items),
                    task_types=[task_type],
                    tools=tools,
                    keywords=_keywords(helpful_rows or items, tools),
                    source_tasks=[_row_id(row) for row in helpful_rows[:max_source_tasks]],
                    section="tool_flow",
                )
            )
        if harmful_rows and len(harmful_rows) >= 2:
            reason = "tool_error" if any(row.get("has_tool_error") for row in harmful_rows) else "low_f1"
            bullets.append(
                PlaybookBullet(
                    id="",
                    text=_caution_text(task_type, tools, reason),
                    helpful=0,
                    harmful=len(harmful_rows),
                    support=len(harmful_rows),
                    task_types=[task_type],
                    tools=tools,
                    keywords=_keywords(harmful_rows, tools),
                    source_tasks=[_row_id(row) for row in harmful_rows[:max_source_tasks]],
                    section="caution",
                )
            )

    bullets.sort(key=lambda b: (b.prior_score, b.support, len(b.tools)), reverse=True)
    deduped: list[PlaybookBullet] = []
    seen: set[str] = set()

    # Coverage first: high-support GIS flows otherwise dominate and hide long-tail
    # task/tool families, which is exactly the failure mode this baseline should
    # avoid when compared with product-transition methods.
    best_by_task: dict[str, PlaybookBullet] = {}
    for bullet in bullets:
        if bullet.section != "tool_flow":
            continue
        for task_type in bullet.task_types or ["unknown"]:
            current = best_by_task.get(task_type)
            if current is None or bullet.prior_score > current.prior_score:
                best_by_task[task_type] = bullet

    for bullet in sorted(best_by_task.values(), key=lambda b: (b.prior_score, b.support), reverse=True):
        if bullet.text in seen:
            continue
        seen.add(bullet.text)
        deduped.append(bullet)
        if len(deduped) >= max_bullets:
            break

    for bullet in bullets:
        if len(deduped) >= max_bullets:
            break
        if bullet.text in seen:
            continue
        seen.add(bullet.text)
        deduped.append(bullet)
    for idx, bullet in enumerate(deduped, 1):
        prefix = "cau" if bullet.section == "caution" else "str"
        bullet.id = f"{prefix}-{idx:05d}"

    playbook = ACEPlaybook(output_dir)
    playbook.bullets = deduped
    playbook.manifest = {
        "method": "ace_playbook",
        "adapter": "ACE-style Terrabox lightweight playbook",
        "official_source": "/data1/yuhongjie2/external_repos/ace",
        "source_results": str(Path(results_dir).resolve()),
        "uses_gold": False,
        "n_rows": len(rows),
        "n_groups": len(groups),
        "n_bullets": len(deduped),
        "min_success_f1": min_success_f1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Built from train rollout tool flows and environment feedback; no eval gold/answer is used.",
    }
    playbook.save()
    return playbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ACE-style playbook from rollout results")
    parser.add_argument("--results-dir", required=True, help="Directory containing rollout results/*.json")
    parser.add_argument("--output-dir", required=True, help="Output evolution store directory")
    parser.add_argument("--mode", choices=["deterministic", "ace-official"], default="deterministic")
    parser.add_argument("--provider", default="longcat", help="LLM provider for --mode ace-official")
    parser.add_argument("--batch-size", type=int, default=10, help="Episodes per Reflector/Curator batch")
    parser.add_argument("--max-batches", type=int, default=None, help="Cap ACE official-style LLM batches")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing ACE official-style store")
    parser.add_argument("--max-bullets", type=int, default=120)
    parser.add_argument("--min-success-f1", type=float, default=0.6)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.mode == "ace-official":
        playbook = build_playbook_ace_official_style(
            args.results_dir,
            args.output_dir,
            provider=args.provider,
            max_bullets=args.max_bullets,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            force=args.force,
        )
    else:
        playbook = build_playbook(
            args.results_dir,
            args.output_dir,
            max_bullets=args.max_bullets,
            min_success_f1=args.min_success_f1,
        )
    print(json.dumps(playbook.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
