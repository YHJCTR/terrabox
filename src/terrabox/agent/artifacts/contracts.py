"""Tool-level artifact contracts for stateful tool disclosure.

Contracts describe what a tool needs and what it produces. They are tool-level,
not dataset-level: they should never encode OpenEarth/Disaster gold trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactNeed:
    kind: str
    param: str | None = None
    count: int = 1


@dataclass(frozen=True)
class ArtifactOutput:
    kind: str
    field: str | None = None
    param: str | None = None
    name_param: str | None = None
    name_field: str | None = None
    container_param: str | None = None


@dataclass(frozen=True)
class ToolArtifactContract:
    inputs: tuple[ArtifactNeed, ...] = field(default_factory=tuple)
    outputs: tuple[ArtifactOutput, ...] = field(default_factory=tuple)


TOOL_ARTIFACT_CONTRACTS: dict[str, ToolArtifactContract] = {
    "osm_gis.get_area_boundary": ToolArtifactContract(
        outputs=(ArtifactOutput(kind="gpkg", field="gpkg"),),
    ),
    "osm_gis.add_pois_layer": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="gpkg", param="gpkg"),),
        outputs=(
            ArtifactOutput(
                kind="vector_layer",
                name_param="layer_name",
                container_param="gpkg",
            ),
            ArtifactOutput(kind="gpkg", field="gpkg"),
        ),
    ),
    "osm_gis.compute_route_dist": ToolArtifactContract(
        inputs=(
            ArtifactNeed(kind="gpkg", param="gpkg"),
            ArtifactNeed(kind="vector_layer", count=2),
        ),
    ),
    "osm_gis.show_index_layer": ToolArtifactContract(
        inputs=(
            ArtifactNeed(kind="gpkg", param="gpkg"),
            ArtifactNeed(kind="raster_layer"),
        ),
        outputs=(ArtifactOutput(kind="image", field="out_file"),),
    ),
    "osm_gis.compute_index_change": ToolArtifactContract(
        inputs=(
            ArtifactNeed(kind="gpkg", param="gpkg"),
            ArtifactNeed(kind="raster_layer", count=2),
        ),
        outputs=(
            ArtifactOutput(
                kind="raster_layer",
                name_param="diff_layer_name",
                name_field="diff_layer_name",
                container_param="gpkg",
            ),
        ),
    ),
    "osm_gis.display_on_map": ToolArtifactContract(
        inputs=(
            ArtifactNeed(kind="gpkg", param="gpkg"),
            ArtifactNeed(kind="vector_layer"),
        ),
        outputs=(ArtifactOutput(kind="image", field="out_file"),),
    ),
    "osm_gis.display_on_geotiff": ToolArtifactContract(
        inputs=(
            ArtifactNeed(kind="gpkg", param="gpkg"),
            ArtifactNeed(kind="vector_layer"),
            ArtifactNeed(kind="raster"),
        ),
        outputs=(ArtifactOutput(kind="raster", field="out_file"),),
    ),
    "osm_gis.add_index_layer": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="gpkg", param="gpkg"),),
        outputs=(
            ArtifactOutput(
                kind="raster_layer",
                name_param="layer_name",
                name_field="layer_name",
                container_param="gpkg",
            ),
        ),
    ),
    "osm_gis.get_bbox_from_raster": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="raster"),),
    ),
    "geo_raster.calculate_index": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="raster"),),
        outputs=(ArtifactOutput(kind="raster", field="output_path"),),
    ),
    "geo_raster.raster_diff": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="raster", count=2),),
        outputs=(ArtifactOutput(kind="raster", field="output_path"),),
    ),
    "geo_perception.vlm_analyze": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
    ),
    "geo_perception.ocr_extract": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
    ),
    "geo_perception.draw_bboxes": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
        outputs=(ArtifactOutput(kind="image", field="output_path"),),
    ),
    "geo_perception.add_text": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
        outputs=(ArtifactOutput(kind="image", field="output_path"),),
    ),
    "geo_perception.sam2_segment": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
        outputs=(ArtifactOutput(kind="image", field="output_path"),),
    ),
    "geo_perception.instructsam": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
        outputs=(ArtifactOutput(kind="image", field="output_path"),),
    ),
    "geo_perception.remotesam": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
        outputs=(ArtifactOutput(kind="image", field="output_path"),),
    ),
    "geo_perception.strip_rcnn_detect": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
    ),
    "geo_perception.count_given_object": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
    ),
    "geo_perception.region_attribute_description": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image"),),
    ),
    "geo_perception.change_os_detect": ToolArtifactContract(
        inputs=(ArtifactNeed(kind="image", count=2),),
    ),
    "compute.plot": ToolArtifactContract(
        outputs=(ArtifactOutput(kind="image", field="image_path"),),
    ),
}


def get_contract(slug: str) -> ToolArtifactContract | None:
    return TOOL_ARTIFACT_CONTRACTS.get(slug)


def contract_summary(slug: str) -> dict[str, Any]:
    contract = get_contract(slug)
    if contract is None:
        return {}
    return {
        "inputs": [need.__dict__ for need in contract.inputs],
        "outputs": [output.__dict__ for output in contract.outputs],
    }
