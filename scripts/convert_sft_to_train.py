#!/usr/bin/env python3
"""
Convert disaster_sft_dataset.json → train_sft.py compatible JSONL
=================================================================

PURPOSE
-------
train_sft.py expects data/openearth/train_sft.jsonl where each line is:

  {"messages": [
    {"role": "user",      "content": [{"type":"image","image":"/abs/path.jpg"},
                                      {"type":"text", "text":"<task prompt>"}]},
    {"role": "assistant", "content": [{"type":"text", "text":"<expected output>"}]}
  ]}

disaster_sft_dataset.json stores tool_calls in a structured JSON format — it is
NOT directly usable by train_sft.py. This script performs the conversion.

ABOUT THE PNG→TIF HACK IN run_sft_sample.py
--------------------------------------------
scripts/run_sft_sample.py converts PNG images to TIF at test time (inside
auto_map_inputs() → _extract_band_to_tif()). That is a test-time hack to give
rasterio tools a CRS-bearing file; it is NOT a training data pipeline.

For SFT training, images referenced in sft_image_mapping.json (now pointing to
real GeoTIFF files for most tasks) are passed directly to the VLM's image
processor. The VLM treats them as visual inputs — it does not call rasterio
itself. So training only requires valid image files (PNG, TIF, JPEG all work).

USAGE
-----
  python scripts/convert_sft_to_train.py \\
    --sft      data/disaster_sft_dataset.json \\
    --mapping  data/sft_image_mapping.json \\
    --output   data/disaster_sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_image_index(mapping_data: Any) -> Dict[str, Dict]:
    """Build sft_id → {image_pre, image_post} lookup from mapping JSON."""
    index: Dict[str, Dict] = {}
    if isinstance(mapping_data, dict):
        entries = mapping_data.get("mappings", [])
    else:
        entries = mapping_data
    for entry in entries:
        sid = entry.get("sft_id", "")
        if sid:
            index[sid] = {
                "image_pre":  entry.get("image_pre"),
                "image_post": entry.get("image_post"),
            }
    return index


def _format_tool_calls(tool_calls: List[Dict]) -> str:
    """Format tool_calls list as a readable text block for the user message."""
    lines = []
    for step in tool_calls:
        s = step.get("step", "?")
        tool = step.get("tool", "unknown")
        args = step.get("args", {})
        out = step.get("sample_output", {})
        args_str = json.dumps(args, ensure_ascii=False)
        out_str = json.dumps(out, ensure_ascii=False)
        lines.append(f"Step {s}: {tool}({args_str})")
        lines.append(f"  → {out_str}")
    return "\n".join(lines)


def _format_assistant_response(sample: Dict) -> str:
    """Build the expected assistant response from a SFT sample."""
    task_type = sample.get("task_type", "unknown")
    prompt = sample.get("prompt", "")
    tool_calls = sample.get("tool_calls", [])

    # Extract the final step's output as the "answer"
    final_output = ""
    if tool_calls:
        last = tool_calls[-1]
        out = last.get("sample_output", {})
        final_output = json.dumps(out, ensure_ascii=False, indent=2)

    lines = [
        f"任务类型: {task_type}",
        f"任务描述: {prompt}",
        "",
        "执行工具调用序列:",
        _format_tool_calls(tool_calls),
        "",
        "最终输出:",
        final_output,
    ]
    return "\n".join(lines)


def convert_sample(
    sample: Dict,
    image_index: Dict[str, Dict],
) -> Optional[Dict]:
    """Convert a single SFT sample to a messages dict."""
    sid = sample.get("id", "")
    prompt = sample.get("prompt", "")
    tool_calls = sample.get("tool_calls", [])

    if not prompt or not tool_calls:
        log.debug(f"Skipping {sid}: empty prompt or tool_calls")
        return None

    # Build user message content
    user_content: List[Dict] = []

    # Add images if available
    img_info = image_index.get(sid, {})
    for key in ("image_pre", "image_post"):
        path = img_info.get(key)
        if path:
            user_content.append({"type": "image", "image": path})

    # Add text prompt + tool chain description
    tool_text = _format_tool_calls(tool_calls)
    user_text = (
        f"{prompt}\n\n"
        f"请按以下工具调用序列完成任务:\n"
        f"{tool_text}"
    )
    user_content.append({"type": "text", "text": user_text})

    # Build assistant response
    assistant_text = _format_assistant_response(sample)

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def convert(sft_path: str, mapping_path: str, output_path: str) -> None:
    log.info(f"Loading SFT dataset from {sft_path}")
    sft_data = _load_json(sft_path)
    samples = sft_data if isinstance(sft_data, list) else sft_data.get("samples", [])

    log.info(f"Loading image mapping from {mapping_path}")
    mapping_data = _load_json(mapping_path)
    image_index = _build_image_index(mapping_data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for sample in samples:
            record = convert_sample(sample, image_index)
            if record is None:
                skipped += 1
                continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"Written {written} samples to {output_path} (skipped {skipped})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert disaster_sft_dataset.json → train_sft.py JSONL format"
    )
    parser.add_argument(
        "--sft",
        default=str(REPO_ROOT / "data" / "disaster_sft_dataset.json"),
        help="Path to disaster_sft_dataset.json",
    )
    parser.add_argument(
        "--mapping",
        default=str(REPO_ROOT / "data" / "sft_image_mapping.json"),
        help="Path to sft_image_mapping.json",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "disaster_sft_train.jsonl"),
        help="Output JSONL path (default: data/disaster_sft_train.jsonl)",
    )
    args = parser.parse_args()
    convert(args.sft, args.mapping, args.output)


if __name__ == "__main__":
    main()
