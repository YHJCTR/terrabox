"""Artifact-aware tool disclosure primitives."""

from .extractors import update_artifact_state
from .readiness import blocked_tool_text, ready_slugs_by_category, tool_readiness
from .signatures import (
    canonical_tool_slug,
    product_state_delta,
    product_state_signature,
    product_state_tokens,
    state_overlap,
    state_satisfies,
)
from .state import (
    artifact_paths,
    artifact_state_text,
    current_gpkg,
    infer_artifact_kind,
    initial_artifact_state,
    progress_signature,
)

__all__ = [
    "artifact_paths",
    "artifact_state_text",
    "blocked_tool_text",
    "canonical_tool_slug",
    "current_gpkg",
    "infer_artifact_kind",
    "initial_artifact_state",
    "progress_signature",
    "product_state_delta",
    "product_state_signature",
    "product_state_tokens",
    "ready_slugs_by_category",
    "state_overlap",
    "state_satisfies",
    "tool_readiness",
    "update_artifact_state",
]
