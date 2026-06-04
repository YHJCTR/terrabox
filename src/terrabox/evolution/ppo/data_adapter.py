"""Prepare veRL rows from strict SFT data without leaking gold messages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..full_shared.sft_schema import FullSFTSample


DEFAULT_TOOL_CATALOG = "data/newdata/tools_catalog.json"


def load_tool_catalog(path: str | Path = DEFAULT_TOOL_CATALOG) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list tool catalog: {source}")
    return data


def compact_tool_catalog(tool_catalog: list[dict[str, Any]], *, max_description_chars: int = 240) -> str:
    lines = []
    for tool in tool_catalog:
        slug = str(tool.get("slug") or "")
        if not slug:
            continue
        description = str(tool.get("description") or "").strip().replace("\n", " ")
        if len(description) > max_description_chars:
            description = description[: max_description_chars - 3].rstrip() + "..."
        params = tool.get("parameters") or {}
        required = params.get("required", []) if isinstance(params, dict) else []
        lines.append(f"- {slug}: {description} Required args: {required}")
    return "\n".join(lines)


def build_prompt(sample: FullSFTSample, tool_catalog: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You are a Terrabox geospatial tool-use agent. Solve the task by reasoning "
        "and selecting tools from the available catalog. Do not invent tool names. "
        "When using tools, output JSON with fields thought and actions, where each "
        "action has tool and arguments.\n\n"
        "Available tools:\n"
        f"{compact_tool_catalog(tool_catalog) if tool_catalog else '(tool catalog omitted in this test row)'}"
    )
    user = f"Question: {sample.question}"
    if sample.images:
        user += "\n\nImage files:\n" + "\n".join(f"- {path}" for path in sample.images)
    if sample.data_files:
        user += "\n\nData files:\n" + "\n".join(f"- {path}" for path in sample.data_files[:50])
        if len(sample.data_files) > 50:
            user += f"\n... and {len(sample.data_files) - 50} more files"
    if sample.data_dir:
        user += f"\n\nData directory: {sample.data_dir}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def sample_to_verl_row(
    sample: FullSFTSample,
    *,
    tool_catalog: list[dict[str, Any]] | None = None,
    data_source: str = "terrabox_grpo",
) -> dict[str, Any]:
    catalog = tool_catalog if tool_catalog is not None else load_tool_catalog()
    return {
        "data_source": data_source,
        "prompt": build_prompt(sample, catalog),
        "reward_model": {
            "ground_truth": json.dumps(
                {
                    "task_id": sample.task_id,
                    "expected_tools": sample.tool_sequence,
                    "gold_tool_calls": sample.gold_tool_calls,
                    "final_answer": sample.ground_truth,
                },
                ensure_ascii=False,
            )
        },
        "extra_info": {
            "task_id": sample.task_id,
            "source": sample.source,
            "task_type": sample.task_type,
            "images": sample.images,
            "data_files": sample.data_files,
            "data_dir": sample.data_dir,
        },
    }


def samples_to_verl_rows(
    samples: Iterable[FullSFTSample],
    *,
    tool_catalog: list[dict[str, Any]] | None = None,
    data_source: str = "terrabox_grpo",
) -> list[dict[str, Any]]:
    catalog = tool_catalog if tool_catalog is not None else load_tool_catalog()
    return [sample_to_verl_row(sample, tool_catalog=catalog, data_source=data_source) for sample in samples]


def write_verl_jsonl(rows: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_verl_parquet(rows: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required to write veRL parquet data") from exc

    data = list(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(out)
    return len(data)

