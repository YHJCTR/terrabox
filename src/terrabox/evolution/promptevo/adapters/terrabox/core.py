"""Terrabox/OEA promptevo adapter public API."""
from __future__ import annotations

from ...adapters_terrabox import (  # noqa: F401
    TerraboxMetricProvider,
    TerraboxPromptStore,
    TerraboxTrajectorySource,
    is_transient_trace,
)
from .files import *  # noqa: F401,F403


def make_terrabox_components(results_experiment: str = "oe_full_react_offline"):
    """Build the default OEA prompt, trace, and metric adapter components."""

    prompts = TerraboxPromptStore(versions_dir="evolution_store/promptevo/terrabox/versions")
    traces = TerraboxTrajectorySource()
    metrics = TerraboxMetricProvider()
    return prompts, traces, metrics
