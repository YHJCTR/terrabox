"""Prepare veRL-style supervised/RL data from full SFT samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..full_shared.sft_schema import FullSFTSample


def sample_to_verl_row(sample: FullSFTSample) -> dict[str, Any]:
    return {
        "data_source": "terrabox_skillrl_full",
        "prompt": sample.messages,
        "reward_model": {
            "ground_truth": json.dumps({
                "task_id": sample.task_id,
                "expected_tools": sample.tool_sequence,
                "gold_tool_calls": sample.gold_tool_calls,
                "final_answer": sample.ground_truth,
            }, ensure_ascii=False),
        },
        "extra_info": {
            "task_id": sample.task_id,
            "source": sample.source,
            "task_type": sample.task_type,
            "images": sample.images,
            "data_files": sample.data_files,
        },
    }


def write_verl_jsonl(samples: list[FullSFTSample], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample_to_verl_row(sample), ensure_ascii=False) + "\n")


def write_verl_parquet(samples: list[FullSFTSample], output_path: str | Path) -> None:
    """Write Parquet when pandas/pyarrow are available."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("pandas is required to write parquet") from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([sample_to_verl_row(sample) for sample in samples]).to_parquet(out)
