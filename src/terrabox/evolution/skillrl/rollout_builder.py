"""Build a rollout-only SkillRL-style skill bank for live OEA evaluation.

This adapter implements SkillRL's offline, non-RL skill-distillation phase.
It deliberately consumes only what an agent could observe in its own train
rollouts: task text, actual tool calls, completion state, and observable tool
errors.  It never reads or forwards ``expected_tools``, gold calls, answers,
F1, or task identifiers to the distillation model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .skill_bank import HierarchicalSkillBank


logger = logging.getLogger(__name__)

_INFRA_MARKERS = (
    "timeout", "timed out", "rate limit", "quota", "billing", "payment",
    "oom", "out of memory", "cuda", "connection", "network", "provider",
    "context length", "docker", "service health",
)


def _load_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    paths = sorted(root.glob("*.json")) if root.is_dir() else [root]
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _sanitize(text: object, limit: int = 360) -> str:
    value = str(text or "")
    value = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", value)
    value = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_id>", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit].rstrip()


def _tool_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("name") or item.get("tool") or item.get("slug") or "")
    return str(item or "")


def _compact_tools(row: dict[str, Any], limit: int = 10) -> list[str]:
    raw = row.get("tool_calls_deduped") or row.get("tool_calls") or row.get("tool_sequence") or []
    tools: list[str] = []
    last = ""
    for item in raw:
        tool = _tool_name(item).strip()
        if not tool or tool == "final_answer" or tool == last:
            continue
        tools.append(tool)
        last = tool
        if len(tools) >= limit:
            break
    return tools


def _is_infra(row: dict[str, Any]) -> bool:
    if row.get("has_tool_oom"):
        return True
    visible = {
        "status": row.get("status"),
        "error": row.get("error"),
        "answer": row.get("final_answer_preview"),
        "history": row.get("conversation_history") or [],
    }
    return any(marker in json.dumps(visible, ensure_ascii=False).lower() for marker in _INFRA_MARKERS)


def _outcome(row: dict[str, Any]) -> str:
    if _is_infra(row):
        return "infra_filtered"
    if str(row.get("status") or "") in {"completed", "completed_with_recovery"} and not row.get("has_tool_error"):
        return "completed_without_observable_tool_error"
    if row.get("has_tool_error"):
        return "observable_tool_error"
    return "incomplete_or_unverified"


def _episode(row: dict[str, Any]) -> dict[str, Any] | None:
    tools = _compact_tools(row)
    if not tools:
        return None
    return {
        "task_type": str(row.get("task_type") or "unknown"),
        "request": _sanitize(row.get("question"), 360),
        "actual_tool_flow": tools,
        "outcome": _outcome(row),
        "observable_tool_error": bool(row.get("has_tool_error")) and not _is_infra(row),
    }


def _select_diverse_episodes(rows: list[dict[str, Any]], max_episodes: int) -> list[dict[str, Any]]:
    """Cover task-type/tool-flow groups without using label-derived quality."""
    groups: dict[tuple[str, tuple[str, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ep = _episode(row)
        if ep is None or ep["outcome"] == "infra_filtered":
            continue
        groups[(ep["task_type"], tuple(ep["actual_tool_flow"]), ep["outcome"])].append(ep)

    selected: list[dict[str, Any]] = []
    # One representative per group handles the long tail before high-frequency
    # OSM and common perception flows consume the distillation budget.
    for _, candidates in sorted(groups.items(), key=lambda pair: pair[0]):
        selected.append(candidates[0])
        if len(selected) >= max_episodes:
            return selected

    if len(selected) >= max_episodes:
        return selected[:max_episodes]
    for _, candidates in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        selected.extend(candidates[1:])
        if len(selected) >= max_episodes:
            break
    return selected[:max_episodes]


def _visible_catalog(rows: list[dict[str, Any]]) -> list[str]:
    tools: set[str] = set()
    for row in rows:
        tools.update(_compact_tools(row, limit=40))
    return sorted(tools)


def _distill_prompt(episodes: list[dict[str, Any]], catalog: list[str]) -> str:
    return f"""Distill reusable tool-use skills from these agent rollouts. The rollouts are imperfect observations, not gold demonstrations.

Available tool slugs: {json.dumps(catalog, ensure_ascii=False)}

Episodes (only actual agent behavior and observable outcome):
{json.dumps(episodes, ensure_ascii=False, indent=2)}

Return one JSON object with exactly these keys:
{{
  "general_skills": [{{"content":"one concise transferable strategy", "tools":["tool.slug"], "tags":["keywords"]}}],
  "task_skills": [{{"task_type":"observed task type", "content":"when to apply and how to execute", "tools":["tool.slug"], "tags":["keywords"]}}],
  "mistakes": [{{"content":"a concise avoid/repair rule grounded only in observable tool-use failures", "failed_tool":"tool.slug", "tags":["keywords"]}}]
}}

Rules: keep skills compact and actionable; distinguish reliable process advice from uncertain outcomes; never invent a tool; never mention task IDs, exact places, paths, numeric answers, gold labels, or hidden expected steps."""


def _valid_tools(value: object, catalog: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(tool) for tool in value if str(tool) in catalog]


def _clean_content(value: object, limit: int = 700) -> str:
    text = _sanitize(value, limit)
    text = re.sub(r"\b(?:expected_tools|gold_tool_calls|ground_truth)\b.*", "", text, flags=re.I)
    return text.strip()


def _add_distilled(bank: HierarchicalSkillBank, data: dict[str, Any], catalog: set[str]) -> int:
    created = 0
    for item in data.get("general_skills") or []:
        if not isinstance(item, dict):
            continue
        content = _clean_content(item.get("content"))
        if content and bank.add_general_skill(content, _valid_tools(item.get("tools"), catalog) + list(item.get("tags") or [])):
            created += 1
    for item in data.get("task_skills") or []:
        if not isinstance(item, dict):
            continue
        content = _clean_content(item.get("content"))
        task_type = _clean_content(item.get("task_type"), 80) or "unknown"
        if content and bank.add_task_skill(task_type, content):
            created += 1
    for item in data.get("mistakes") or []:
        if not isinstance(item, dict):
            continue
        content = _clean_content(item.get("content"))
        failed_tool = str(item.get("failed_tool") or "unknown")
        if failed_tool not in catalog:
            failed_tool = "unknown"
        if content and bank.add_mistake("Observed rollout pattern", failed_tool, content):
            created += 1
    return created


def build_rollout_skill_bank(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    provider: str = "longcat",
    max_episodes: int = 240,
    batch_size: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    """Build a compact, rollout-only SkillRL bank with LongCat distillation."""
    output = Path(output_dir)
    manifest_path = output / "rollout_skillrl_manifest.json"
    if force and output.exists():
        shutil.rmtree(output)
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("build_status") == "complete":
            return manifest

    rows = _load_rows(results_dir)
    catalog = _visible_catalog(rows)
    episodes = _select_diverse_episodes(rows, max_episodes)
    if not episodes:
        raise RuntimeError(f"No usable rollout episodes found in {results_dir}")

    from terrabox.agent.llm_provider import make_llm_client

    bank = HierarchicalSkillBank(str(output))
    bank.clear()
    llm = make_llm_client(provider)
    traces_dir = output / "distillation_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    created = 0
    for start in range(0, len(episodes), batch_size):
        batch_index = start // batch_size + 1
        batch = episodes[start : start + batch_size]
        prompt = _distill_prompt(batch, catalog)
        try:
            data = llm.call_json(
                prompt,
                system="You are a skill-distillation teacher. Return valid JSON only, with no hidden reasoning.",
                max_tokens=1800,
            )
            if not isinstance(data, dict):
                raise RuntimeError("LongCat returned no valid JSON object")
            added = _add_distilled(bank, data, set(catalog))
            created += added
            trace = {"batch": batch_index, "episodes": batch, "distilled": data, "created": added}
        except Exception as exc:
            logger.warning("SkillRL rollout distillation batch %s failed: %s", batch_index, exc)
            failures.append({"batch": batch_index, "error": repr(exc)})
            trace = {"batch": batch_index, "episodes": batch, "error": repr(exc), "created": 0}
        (traces_dir / f"batch_{batch_index:04d}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest = {
        "method": "skillrl_rollout",
        "adapter": "SkillRL offline skill-distillation adaptation for Terrabox OEA",
        "paper": "SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning (arXiv:2602.08234)",
        "reproduction_scope": "adapted_non_rl",
        "official_source": None,
        "source_results": str(Path(results_dir).resolve()),
        "uses_gold": False,
        "forbidden_source_fields": ["expected_tools", "gold_tool_calls", "ground_truth", "metrics", "task_id"],
        "provider": provider,
        "n_rollout_rows": len(rows),
        "n_selected_episodes": len(episodes),
        "n_visible_tools": len(catalog),
        "batch_size": batch_size,
        "n_distilled_skills": created,
        "bank_counts": bank.counts(),
        "distillation_failures": failures,
        "build_status": "complete",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Reproduces SkillRL's frozen-policy offline skill-library arm only. It excludes SkillRL's teacher-generated SFT demonstrations and GRPO recursive evolution, which require model training and are out of scope for this no-training comparison.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build strict rollout-only SkillRL bank")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", default="longcat")
    parser.add_argument("--max-episodes", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    result = build_rollout_skill_bank(
        args.results_dir,
        args.output_dir,
        provider=args.provider,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
