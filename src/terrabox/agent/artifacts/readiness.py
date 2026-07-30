"""Readiness checks for artifact-aware tool disclosure."""

from __future__ import annotations

from typing import Any

from ...core.registry import registry
from .contracts import ArtifactNeed, get_contract
from .state import (
    FILE_PARAM_MARKERS,
    OUTPUT_PARAM_NAMES,
    artifact_paths,
    artifact_paths_for_param,
    current_gpkg,
)


def param_required_names(spec) -> list[str]:
    params = spec.parameters or {}
    required = params.get("required") or []
    return [str(name) for name in required]


def is_file_input_param(name: str) -> bool:
    lowered = name.lower()
    if lowered in OUTPUT_PARAM_NAMES:
        return False
    return any(marker in lowered for marker in FILE_PARAM_MARKERS)


def _need_satisfied(need: ArtifactNeed, state: dict[str, Any]) -> bool:
    if need.kind.endswith("_layer"):
        matching = [
            layer
            for layer in state.get("layers", [])
            if str(layer.get("kind") or "vector_layer") == need.kind
        ]
        return len(matching) >= need.count
    paths = artifact_paths(state, need.kind)
    return len(paths) >= need.count


def tool_readiness(spec, state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a tool can be exposed with the current artifact state.

    Contract checks are preferred. Schema-name heuristics are a fallback for
    tools that do not have an explicit contract yet.
    """
    blockers: list[str] = []
    contract = get_contract(spec.slug)
    if contract is not None:
        for need in contract.inputs:
            if not _need_satisfied(need, state):
                if need.kind.endswith("_layer"):
                    blockers.append(f"needs at least {need.count} {need.kind}(s)")
                else:
                    blockers.append(f"needs {need.kind} artifact")
        return not blockers, blockers

    required = param_required_names(spec)
    layers = state.get("layers", [])
    for name in required:
        lowered = name.lower()
        if lowered in OUTPUT_PARAM_NAMES:
            continue
        if lowered == "gpkg":
            if not current_gpkg(state):
                blockers.append("needs GeoPackage artifact")
            continue
        if lowered in {"src_layer", "tar_layer"}:
            if len(layers) < 2:
                blockers.append("needs at least two vector layers")
            continue
        if lowered.endswith("_layer") and lowered != "layer_name":
            if not layers:
                blockers.append("needs a vector layer")
            continue
        if is_file_input_param(lowered) and not artifact_paths_for_param(state, lowered):
            blockers.append(f"needs file artifact for {name}")
    return not blockers, blockers


def is_source_tool(spec) -> bool:
    contract = get_contract(spec.slug)
    if contract is not None:
        return not contract.inputs
    for name in param_required_names(spec):
        lowered = name.lower()
        if lowered in OUTPUT_PARAM_NAMES:
            continue
        if lowered == "gpkg" or lowered.endswith("_layer") or is_file_input_param(lowered):
            return False
    return True


def ready_slugs_by_category(
    state: dict[str, Any],
    allowed_slugs: list[str] | set[str] | None,
    *,
    recovery_mode: bool = False,
) -> dict[str, list[str]]:
    allowed = set(allowed_slugs or [])
    ready: dict[str, list[str]] = {}
    specs_by_slug = {}
    for toolkit in registry.list_toolkits():
        for spec in registry.list_tools(toolkit=toolkit.name):
            specs_by_slug[spec.slug] = spec
            if allowed_slugs is not None and spec.slug not in allowed:
                continue
            is_ready, _ = tool_readiness(spec, state)
            if is_ready:
                ready.setdefault(toolkit.name, []).append(spec.slug)
    successful = set(state.get("successful_calls", []))
    ordered: dict[str, list[str]] = {}
    for category, slugs in ready.items():
        def sort_key(slug: str) -> tuple[int, str]:
            source = is_source_tool(specs_by_slug[slug])
            used_source = source and slug in successful
            if recovery_mode:
                return (0 if source else 1, slug)
            return (1 if used_source else 0, slug)

        ordered[category] = sorted(slugs, key=sort_key)
    return {category: slugs for category, slugs in ordered.items() if slugs}


def blocked_tool_text(
    state: dict[str, Any],
    allowed_slugs: list[str] | set[str] | None,
    category: str | None,
    limit: int = 8,
) -> str:
    allowed = set(allowed_slugs or [])
    rows: list[str] = []
    specs = registry.list_tools(toolkit=category) if category else registry.list_tools()
    for spec in specs:
        if allowed_slugs is not None and spec.slug not in allowed:
            continue
        ready, blockers = tool_readiness(spec, state)
        if ready or not blockers:
            continue
        rows.append(f"- {spec.slug}: blocked because {', '.join(dict.fromkeys(blockers))}.")
        if len(rows) >= limit:
            break
    return "\n".join(rows) if rows else "none"
