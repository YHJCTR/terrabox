"""Pre-execution tool contract checks for the standard agent loop."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..core.registry import registry
from ..core.utils.uploads import inspect_uploaded_file


@dataclass(frozen=True)
class ToolContractIssue:
    code: str
    message: str
    hint: str = ""


@dataclass(frozen=True)
class ToolContractResult:
    ok: bool
    issues: tuple[ToolContractIssue, ...] = ()

    @property
    def issue_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    def to_tool_message(self) -> str:
        if self.ok:
            return "Tool contract validation passed."
        lines = ["Tool contract validation failed before execution:"]
        for issue in self.issues:
            lines.append(f"- {issue.code}: {issue.message}")
            if issue.hint:
                lines.append(f"  Hint: {issue.hint}")
        return "\n".join(lines)


class ToolContractValidator:
    """Validate obvious file/argument contract mismatches before tool execution."""

    _GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}

    def validate(self, slug: str, arguments: dict[str, Any]) -> ToolContractResult:
        issues: list[ToolContractIssue] = []
        issues.extend(self._validate_required_args(slug, arguments))
        if slug == "geo_raster.raster_diff":
            issues.extend(self._validate_raster_diff(arguments))
        return ToolContractResult(ok=not issues, issues=tuple(issues))

    def _validate_required_args(self, slug: str, arguments: dict[str, Any]) -> list[ToolContractIssue]:
        spec = registry.get_tool(slug)
        if spec is None:
            return []
        missing = []
        for name in spec.parameters.get("required", []):
            if arguments.get(name) in (None, ""):
                missing.append(name)
        if not missing:
            return []
        return [
            ToolContractIssue(
                code="missing_required_arguments",
                message=f"Missing required argument(s): {', '.join(missing)}.",
                hint="Provide all required fields from the tool schema before calling the tool.",
            )
        ]

    def _validate_raster_diff(self, arguments: dict[str, Any]) -> list[ToolContractIssue]:
        issues: list[ToolContractIssue] = []
        for key in ("path_a", "path_b"):
            path = arguments.get(key)
            if not isinstance(path, str) or not path:
                continue
            metadata = inspect_uploaded_file(path)
            extension = str(metadata.get("extension") or "").lower()
            semantic_type = metadata.get("semantic_type")
            if not os.path.exists(path):
                issues.append(
                    ToolContractIssue(
                        code="input_path_not_found",
                        message=f"{key} does not exist: {path}",
                        hint="Use one of the uploaded file paths or a previously produced artifact path.",
                    )
                )
                continue
            if extension not in self._GEOTIFF_EXTENSIONS or semantic_type != "geospatial_raster":
                issues.append(
                    ToolContractIssue(
                        code="raster_diff_requires_geospatial_raster",
                        message=(
                            f"{key} must be a geospatial raster such as GeoTIFF; "
                            f"got extension={extension or 'unknown'} semantic_type={semantic_type!r}."
                        ),
                        hint=(
                            "Use a visual-image analysis tool for PNG/JPEG inputs, or convert/register "
                            "the image to a geospatial raster before raster differencing."
                        ),
                    )
                )
        output_path = arguments.get("output_path")
        if isinstance(output_path, str) and output_path:
            out_ext = os.path.splitext(output_path)[1].lower()
            if out_ext not in self._GEOTIFF_EXTENSIONS:
                issues.append(
                    ToolContractIssue(
                        code="raster_diff_output_must_be_geotiff",
                        message=f"output_path should end with .tif or .tiff, got {out_ext or 'no extension'}.",
                        hint="raster_diff writes Float32 raster output, so use a GeoTIFF output path.",
                    )
                )
        return issues
