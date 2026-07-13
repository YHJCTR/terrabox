"""Backward-compatible public API for the tau2-bench adapter.

The implementation is split by responsibility:
- files.py: project paths, JSON helpers, and experiment directories;
- prompts.py: required PromptStore implementation;
- traces.py: required TrajectorySource implementation;
- metrics.py: optional MetricProvider for experiment comparison;
- runner.py: optional local rollout runner and result summarizers.
"""
from __future__ import annotations

from .files import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .prompts import *  # noqa: F401,F403
from .rejudge import *  # noqa: F401,F403
from .runner import *  # noqa: F401,F403
from .traces import *  # noqa: F401,F403
