#!/usr/bin/env python3
"""Deprecated compatibility entry for old dataset conversion commands."""

from __future__ import annotations

import sys


MESSAGE = """
scripts/convert_datasets.py is deprecated and no longer runs conversion logic.

Use the maintained entry points instead:
  - python scripts/prepare_data.py all
      Copy/split/unify OpenEarth + EarthBench eval data.
  - PYTHONPATH=src python scripts/build_oea_full_sft.py
      Build the current oea_full SFT dataset.
  - python scripts/convert_openearth_to_evolution.py
      Convert OpenEarth train trajectories for evolution modules.
  - python scripts/convert_sft_to_evolution.py
      Convert Disaster SFT trajectories for evolution modules.
"""


def main() -> int:
    print(MESSAGE.strip(), file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in sys.argv[1:]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
