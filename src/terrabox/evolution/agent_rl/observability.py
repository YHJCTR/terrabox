"""Trace and resource helpers shared by Terrabox agentic RL adapters."""
from __future__ import annotations

import ctypes
import fcntl
import gc
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


IMPORTANT_LINE_MARKERS = (
    "error",
    "exception",
    "traceback",
    "timeout",
    "output_path",
    "artifact",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".gpkg",
    "positive_pixels",
    "pixel_counts",
    "statistics",
    "summary",
    "count",
    "crs",
    "layer",
)

PATH_RE = re.compile(r"(?:/[^\s,;:'\"\)\]]+|[A-Za-z0-9_./-]+\.(?:png|jpg|jpeg|tif|tiff|gpkg|geojson|json|csv|txt))")
NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
MAX_LIST_ITEMS = 5
MAX_DICT_KEYS = 36
MAX_STRING_CHARS = 480
PLACEHOLDER_SAMPLE_CHARS = 160


def _stable_digest(value: Any) -> str:
    """Return a short deterministic digest for omitted observation payloads."""
    try:
        if isinstance(value, (dict, list, tuple)):
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            payload = str(value)
    except Exception:
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _placeholder(kind: str, *, value: Any, length: int | None = None, sample: Any | None = None) -> dict[str, Any]:
    """Describe omitted raw content without putting it back into model context.

    The full value is already persisted in ``raw_tool_observations.jsonl`` by the
    caller.  The placeholder is intentionally machine-readable so downstream
    prompts can still understand what kind of evidence was omitted.
    """
    payload: dict[str, Any] = {
        "placeholder": f"<omitted_{kind}:{_stable_digest(value)}>",
        "type": kind,
        "raw_saved": "metrics/raw_tool_observations.jsonl",
    }
    if length is not None:
        payload["length"] = length
    if sample not in (None, ""):
        payload["sample"] = sample
    return payload


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    """Append one JSON row with a process-safe file lock."""
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False) + "\n"
    with out.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(payload)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def preview(text: str, max_chars: int = 900) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 16].rstrip() + " ...[truncated]"


def _looks_like_artifact_ref(text: str) -> bool:
    lowered = text.lower()
    return any(suffix in lowered for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gpkg", ".geojson", ".json", ".csv", "output_path", "artifact"))


def _short_string(value: str) -> str | dict[str, Any]:
    text = str(value)
    if len(text) <= MAX_STRING_CHARS:
        return text
    paths = PATH_RE.findall(text)
    if paths:
        return {
            "summary": preview(text, 220),
            "artifact_refs": sorted(set(paths))[:12],
            "omitted": _placeholder(
                "long_text_with_artifact_refs",
                value=text,
                length=len(text),
                sample=preview(text, PLACEHOLDER_SAMPLE_CHARS),
            ),
        }
    return _placeholder("long_text", value=text, length=len(text), sample=preview(text, PLACEHOLDER_SAMPLE_CHARS))


def _compact_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return _short_string(value)
    return value


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Summarize structured tool output without dropping actionable evidence.

    The raw observation is still written to ``raw_tool_observations.jsonl``.
    This function only controls what goes back into the model context during
    online GRPO, where long multi-turn observations otherwise dominate memory.
    """
    if depth >= 4:
        return _compact_scalar(value) if not isinstance(value, (dict, list, tuple)) else _placeholder(
            type(value).__name__, value=value, length=len(value) if hasattr(value, "__len__") else None
        )
    if isinstance(value, dict):
        keep: dict[str, Any] = {}
        deferred: list[tuple[str, Any]] = []
        priority_markers = (
            "status",
            "error",
            "message",
            "summary",
            "count",
            "pixel",
            "stat",
            "bbox",
            "box",
            "detect",
            "object",
            "label",
            "score",
            "path",
            "output",
            "artifact",
            "gpkg",
            "layer",
            "crs",
            "area",
            "distance",
            "text",
            "full_text",
        )
        for key, item in value.items():
            key_s = str(key)
            lowered = key_s.lower()
            pair = (key_s, item)
            if any(marker in lowered for marker in priority_markers):
                deferred.insert(0, pair)
            else:
                deferred.append(pair)
        large_payload_markers = (
            "mask",
            "array",
            "embedding",
            "features",
            "pixels",
            "coordinates",
            "geometry",
            "image_base64",
            "base64",
        )
        for key_s, item in deferred:
            if len(keep) >= MAX_DICT_KEYS:
                break
            lowered = key_s.lower()
            if isinstance(item, (list, tuple, dict, str)) and any(marker in lowered for marker in large_payload_markers):
                try:
                    item_len = len(item)
                except Exception:
                    item_len = None
                if isinstance(item, str) and len(item) <= MAX_STRING_CHARS:
                    keep[key_s] = item
                elif isinstance(item, (list, tuple)) and len(item) <= MAX_LIST_ITEMS:
                    keep[key_s] = _compact_value(item, depth=depth + 1)
                else:
                    keep[key_s] = _placeholder(
                        lowered.replace(" ", "_")[:48] or "large_field",
                        value=item,
                        length=item_len,
                        sample=_compact_value(list(item)[:2], depth=depth + 1) if isinstance(item, (list, tuple)) else None,
                    )
                continue
            keep[key_s] = _compact_value(item, depth=depth + 1)
        omitted = max(0, len(value) - len(keep))
        if omitted:
            keep["_omitted_keys"] = omitted
            keep["_omitted_placeholder"] = _placeholder("dict_keys", value=value, length=len(value))
        return keep
    if isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            return []
        # Long numeric arrays/masks are the main OOM source.  Preserve shape and
        # a tiny sample, not the full mask/vector.
        if all(isinstance(x, (int, float, bool)) or (isinstance(x, str) and NUMERIC_RE.match(x)) for x in values[: min(50, len(values))]):
            if len(values) <= 16:
                return values
            return _placeholder("numeric_list", value=values, length=len(values), sample=values[: min(8, len(values))])
        sample = [_compact_value(item, depth=depth + 1) for item in values[:MAX_LIST_ITEMS]]
        payload: dict[str, Any] = {"length": len(values), "sample": sample}
        if len(values) > MAX_LIST_ITEMS:
            payload["omitted_items"] = len(values) - MAX_LIST_ITEMS
            payload["omitted"] = _placeholder("list_items", value=values, length=len(values))
        return payload
    return _compact_scalar(value)


def _parse_structured_observation(raw: str) -> Any | None:
    text = raw.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def _extract_artifact_refs(raw: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in PATH_RE.findall(raw):
        item = match.strip().strip(",.;:'\"()[]{}")
        if item and item not in seen and _looks_like_artifact_ref(item):
            refs.append(item)
            seen.add(item)
        if len(refs) >= 20:
            break
    return refs


def _structured_observation_summary(raw: str) -> str | None:
    payload = _parse_structured_observation(raw)
    if payload is None:
        return None
    compact = _compact_value(payload)
    refs = _extract_artifact_refs(raw)
    wrapped = {
        "tool_observation_summary": compact,
        "artifact_refs": refs,
        "note": "原始工具输出已完整保存到 raw_tool_observations.jsonl；模型上下文仅使用压缩摘要。",
    }
    return json.dumps(wrapped, ensure_ascii=False, separators=(",", ":"))


def compress_observation(text: str, max_chars: int | None = None) -> tuple[str, dict[str, int]]:
    """Deterministically bound tool observations while keeping actionable state."""
    raw = str(text or "")
    budget = max_chars or int(os.environ.get("TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS", "8192"))
    budget = max(512, budget)
    compact_threshold = int(os.environ.get("TERRABOX_ONLINE_TOOL_COMPACT_THRESHOLD_CHARS", "900"))
    if len(raw) <= min(budget, compact_threshold):
        return raw, {
            "raw_chars": len(raw),
            "compressed_chars": len(raw),
            "kept_lines": raw.count("\n") + bool(raw),
            "strategy": 0,
        }

    structured = _structured_observation_summary(raw)
    if structured:
        if len(structured) <= len(raw) or len(raw) > budget:
            pass
        elif len(raw) <= budget:
            return raw, {
                "raw_chars": len(raw),
                "compressed_chars": len(raw),
                "kept_lines": raw.count("\n") + bool(raw),
                "strategy": 0,
            }
        if len(structured) > budget:
            refs = _extract_artifact_refs(raw)
            fallback = json.dumps(
                {
                    "tool_observation_summary": preview(structured, max(256, budget - 320)),
                    "artifact_refs": refs,
                    "note": "结构化摘要仍超预算，已二次压缩；原始输出保存在 raw_tool_observations.jsonl。",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            structured = fallback[: budget - 24].rstrip() + "...[context_budget]" if len(fallback) > budget else fallback
        return structured, {
            "raw_chars": len(raw),
            "compressed_chars": len(structured),
            "kept_lines": 0,
            "strategy": 2,
        }

    lines = raw.splitlines()
    refs = _extract_artifact_refs(raw)
    ref_block = "\n".join(f"- {ref}" for ref in refs)
    head_budget = max(120, budget // 5)
    tail_budget = max(120, budget // 5)
    head = raw[:head_budget].rstrip()
    tail = raw[-tail_budget:].lstrip()
    middle_budget = max(0, budget - len(head) - len(tail) - len(ref_block) - 260)

    important: list[str] = []
    seen: set[str] = set()
    for line in lines:
        clean = " ".join(line.split())
        lowered = clean.lower()
        if clean and any(marker in lowered for marker in IMPORTANT_LINE_MARKERS) and clean not in seen:
            important.append(clean)
            seen.add(clean)

    middle = "\n".join(important)
    if len(middle) > middle_budget:
        middle = middle[:middle_budget].rstrip()
    compressed = "\n".join(
        part
        for part in (
            "[tool_observation_summary]\n原始工具输出已完整保存；下面仅保留头尾、关键行和 artifact 引用。",
            "[artifact_refs]\n" + ref_block if ref_block else "",
            "[tool_observation_head]\n" + head,
            "[important_lines]\n" + middle if middle else "",
            "[tool_observation_tail]\n" + tail,
        )
        if part
    )
    if len(compressed) > budget:
        compressed = compressed[: budget - 24].rstrip() + "\n...[context_budget]"
    return compressed, {
        "raw_chars": len(raw),
        "compressed_chars": len(compressed),
        "kept_lines": len(important),
        "strategy": 1,
    }


def trim_process_memory() -> None:
    """Release Python and libc arenas after large tool observations."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
