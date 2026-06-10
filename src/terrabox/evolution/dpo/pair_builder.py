"""Build DPO preference pairs (mode A: gold vs the model's own rollout).

For each task we form ``(prompt, chosen, rejected)`` where:
  - prompt  = ChatML(system + initial user question) from the gold sample.
  - chosen  = the gold trajectory (the assistant/observation turns after the
              question), rendered in ChatML.
  - rejected = the model's actual rollout trajectory on the SAME task.

The pair is keyed by ``task_id`` — DPO needs two trajectories on the *same*
prompt with a quality gap, NOT two similar trajectories. See README for the
``rejected-policy`` trade-offs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..full_shared.sft_schema import FullSFTSample


# ChatML rendering ----------------------------------------------------------

def _norm_role(raw: str) -> str:
    r = str(raw).lower()
    if "system" in r:
        return "system"
    if "ai" in r or "assistant" in r:
        return "assistant"
    # HumanMessage / ToolMessage / user / observation → user-side context
    return "user"


def _chatml_turn(role: str, content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def _shorten(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    return text[:head] + "\n...[truncated]...\n" + text[-(max_chars - head):]


def _split_gold(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (prompt_messages, continuation_messages).

    Prompt = leading system message(s) + the FIRST user turn (the question).
    Continuation = everything after it (gold assistant turns + observations).
    """
    first_user = None
    for i, m in enumerate(messages):
        if _norm_role(m.get("role", "")) == "user":
            first_user = i
            break
    if first_user is None:
        return messages, []
    return messages[: first_user + 1], messages[first_user + 1 :]


def _rollout_continuation(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the model's post-question turns from a saved rollout row."""
    history = row.get("conversation_history") or []
    turns: list[dict[str, Any]] = []
    seen_user = False
    for m in history:
        role = _norm_role(m.get("type") or m.get("role") or "")
        content = m.get("content", "")
        if not seen_user:
            if role == "user":
                seen_user = True  # the question; continuation starts after it
            continue
        turns.append({"role": role, "content": content})
    if not turns:
        # No usable history (e.g. no_tool_call): synthesize a single assistant
        # turn from the final answer so the negative is still a valid trajectory.
        final = str(row.get("final_answer") or row.get("final") or "").strip()
        turns = [{"role": "assistant", "content": final or "(no tool call; no answer produced)"}]
    return turns


def _render(messages: list[dict[str, Any]], *, max_obs_chars: int) -> str:
    out = []
    for m in messages:
        role = _norm_role(m.get("role", ""))
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if role == "user":
            content = _shorten(content, max_obs_chars)
        out.append(_chatml_turn(role, content))
    return "".join(out)


# Data structures -----------------------------------------------------------

@dataclass
class DPOPair:
    task_id: str
    source: str
    task_type: str
    prompt: str
    chosen: str
    rejected: str
    rollout_f1: float
    rollout_real_success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rollout_index(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Index rollout rows by task_id from results dirs and/or jsonl files.

    Accepts directories (reads ``trajectories_full.jsonl`` or ``results/*.json``)
    or direct ``.jsonl`` files. Later paths override earlier ones on task_id
    collision, so pass higher-quality caches last if you care.
    """
    index: dict[str, dict[str, Any]] = {}

    def _add(row: dict[str, Any]) -> None:
        if isinstance(row, dict) and isinstance(row.get("result"), dict):
            row = row["result"]
        tid = str(row.get("task_id") or row.get("id") or "")
        if tid:
            index[tid] = row

    for p in paths:
        path = Path(p)
        if path.is_dir():
            full = path / "trajectories_full.jsonl"
            results = path / "results"
            if full.exists():
                with full.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            _add(json.loads(line))
            elif results.exists():
                for rp in sorted(results.glob("*.json")):
                    try:
                        _add(json.loads(rp.read_text(encoding="utf-8")))
                    except Exception:
                        continue
        elif path.suffix == ".jsonl" and path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        _add(json.loads(line))
    return index


def _rollout_f1(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    try:
        return float(metrics.get("f1", row.get("f1", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rollout_success(row: dict[str, Any]) -> bool:
    return bool(row.get("real_success", row.get("success", False)))


def build_pairs(
    gold_samples: list[FullSFTSample],
    rollout_index: dict[str, dict[str, Any]],
    *,
    rejected_policy: str = "below-gold",
    max_rollout_f1: float = 1.0,
    max_obs_chars: int = 1500,
) -> tuple[list[DPOPair], dict[str, Any]]:
    """Build mode-A pairs. Returns (pairs, stats).

    rejected_policy:
      - "below-gold" (default): keep rollouts whose f1 < max_rollout_f1 (i.e. the
        model did not reproduce the gold tool set). Matches "rollouts基本都不匹配
        gold→全是坏轨迹". This is imitation-anchored toward gold tool usage.
      - "failed-only": keep only rollouts with real_success == False (cleanest
        correctness signal; never penalises a valid alternative solution).
      - "all": every matched rollout (penalises even successful rollouts — use
        with care).
    """
    pairs: list[DPOPair] = []
    stats = {
        "gold_samples": len(gold_samples),
        "rollout_index_size": len(rollout_index),
        "matched": 0,
        "kept": 0,
        "skipped_no_rollout": 0,
        "skipped_by_policy": 0,
        "rejected_policy": rejected_policy,
        "max_rollout_f1": max_rollout_f1,
    }
    for sample in gold_samples:
        row = rollout_index.get(sample.task_id)
        if row is None:
            stats["skipped_no_rollout"] += 1
            continue
        stats["matched"] += 1
        f1 = _rollout_f1(row)
        success = _rollout_success(row)
        if rejected_policy == "below-gold" and not (f1 < max_rollout_f1):
            stats["skipped_by_policy"] += 1
            continue
        if rejected_policy == "failed-only" and success:
            stats["skipped_by_policy"] += 1
            continue
        prompt_msgs, gold_cont = _split_gold(sample.messages)
        if not gold_cont:
            stats["skipped_by_policy"] += 1
            continue
        prompt = _render(prompt_msgs, max_obs_chars=max_obs_chars)
        chosen = _render(gold_cont, max_obs_chars=max_obs_chars)
        rejected = _render(_rollout_continuation(row), max_obs_chars=max_obs_chars)
        if chosen.strip() == rejected.strip():
            stats["skipped_by_policy"] += 1  # zero-gap pair: no DPO signal
            continue
        pairs.append(
            DPOPair(
                task_id=sample.task_id,
                source=sample.source,
                task_type=sample.task_type,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                rollout_f1=f1,
                rollout_real_success=success,
            )
        )
    stats["kept"] = len(pairs)
    return pairs, stats


def write_pairs_jsonl(pairs: list[DPOPair], path: str | Path) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return len(pairs)
