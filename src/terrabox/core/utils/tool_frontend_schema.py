"""GUI-only tool schema helpers.

The registry schema stays unchanged for SDK callers and agents.  These helpers
only make newly added tools easier to use from the Terralink tool playground.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

OUTPUT_PARAMETER_NAMES = {"output_path", "output_dir"}
TOOLS_PR_TOOLKIT_PREFIXES = {
    "disaster_response",
    "earth_sci",
    "geo_perception",
    "geo_raster",
    "geo_statistics",
    "geoanalysis",
    "osm_gis",
    "raster_viewer",
}


def _is_output_parameter(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in OUTPUT_PARAMETER_NAMES
        or lowered.endswith("_output_path")
        or lowered.endswith("_output_dir")
    )


def is_tools_pr_tool(slug: str) -> bool:
    """Return True for tools introduced by the tools-only PR branch."""
    toolkit, _, _ = slug.partition(".")
    return toolkit in TOOLS_PR_TOOLKIT_PREFIXES


def prepare_gui_tool_parameters(
    slug: str,
    parameters: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Hide auto-allocated output parameters for newly added tools in GUI schemas."""
    if not parameters or not is_tools_pr_tool(slug):
        return parameters, {}

    prepared = deepcopy(parameters)
    properties = prepared.get("properties")
    if not isinstance(properties, dict):
        return prepared, {}

    auto_output_parameters = [name for name in properties if _is_output_parameter(name)]
    if not auto_output_parameters:
        return prepared, {}

    for name in auto_output_parameters:
        properties.pop(name, None)

    required = prepared.get("required")
    if isinstance(required, list):
        prepared["required"] = [name for name in required if name not in auto_output_parameters]

    return prepared, {"auto_output_parameters": auto_output_parameters}
