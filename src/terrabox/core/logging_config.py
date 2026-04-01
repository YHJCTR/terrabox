"""Centralized logging configuration for the Terrabox platform.

Call ``setup_logging()`` once at application startup (main.py).
All modules should use ``logging.getLogger(__name__)`` and rely on propagation.

Log files:
  logs/app.log   — root logger: all INFO+ messages from every module
  logs/agent.log — agent.io logger: detailed per-session I/O traces
"""
from __future__ import annotations

import logging
import os
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
LOG_DIR = _PROJECT_ROOT / "logs"

_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the root logger to write INFO+ to logs/app.log.

    Safe to call multiple times — skips if a file handler is already attached.
    Does not touch uvicorn's own handlers so existing stdout logging is preserved.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_str = os.environ.get("TL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Skip if a FileHandler for app.log is already registered
    log_file = LOG_DIR / "app.log"
    if any(
        isinstance(h, logging.FileHandler) and pathlib.Path(h.baseFilename) == log_file
        for h in root.handlers
    ):
        return

    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.addHandler(fh)
