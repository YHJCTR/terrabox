"""Convert CoreRegistry tools into LangChain StructuredTools for Agent use."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from ..core.registry import registry
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)

# Mapping from JSON Schema types to Python types
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_SCHEMA_COMPAT_ALIASES: dict[str, tuple[str, ...]] = {
    "bash.execute": ("get_filelist",),
    "earth_sci.calculate_ati": ("ATI",),
    "earth_sci.calculate_lst_mc": ("lst_multi_channel",),
    "earth_sci.calculate_lst_sc": ("lst_single_channel",),
    "earth_sci.calculate_modis_day_night": ("modis_day_night_lst",),
    "earth_sci.calculate_pwv": ("band_ratio",),
    "earth_sci.calculate_split_window": ("split_window",),
    "earth_sci.calculate_tes": ("temperature_emissivity_separation",),
    "earth_sci.calculate_turbidity": ("calculate_water_turbidity_ntu",),
    "earth_sci.stats_lst_ndvi": (
        "calculate_max_lst_by_ndvi",
        "calculate_mean_lst_by_ndvi",
    ),
    "geo_perception.add_text": ("AddText",),
    "geo_perception.bbox_to_centroid": ("bboxes2centroids",),
    "geo_perception.draw_bboxes": ("DrawBox",),
    "geo_perception.instructsam": ("TextToBbox", "InstructSAM"),
    "geo_perception.ocr_extract": ("OCR",),
    "geo_perception.remotesam": ("RemoteSAM",),
    "geo_perception.sam2_segment": ("SegmentObjectPixels",),
    "geo_perception.strip_rcnn_detect": ("ObjectDetection",),
    "geo_raster.apply_cloud_mask": ("apply_cloud_mask",),
    "geo_raster.calculate_index": ("calculate_batch_ndsi",),
    "geo_raster.compute_tvdi": ("compute_tvdi",),
    "geo_raster.raster_average": ("calculate_tif_average",),
    "geo_statistics.batch_raster_stats": (
        "calc_batch_image_mean",
        "calc_batch_image_sum",
        "calc_batch_image_max",
        "max_value_and_index",
        "min_value_and_index",
    ),
    "geo_statistics.count_images_exceeding": ("count_images_exceeding_threshold_ratio",),
    "geo_statistics.mean_of_means": ("mean", "calc_batch_image_mean_mean"),
    "geo_statistics.percentage_change": ("percentage_change",),
    "geo_statistics.scalar_arithmetic": ("difference", "division", "multiply"),
    "geo_statistics.threshold_ratio": ("calculate_threshold_ratio",),
    "geoanalysis.compute_linear_trend": ("compute_linear_trend",),
    "geoanalysis.count_spikes": ("count_spikes_from_values",),
    "ipython.execute": (
        "Calculator",
        "Solver",
        "Plot",
        "calculate_ndti",
        "calculate_ndwi",
        "argmax",
        "index_to_date_range",
    ),
}

_LOSSY_SCHEMA_COMPAT: set[str] = {
    "geo_perception.instructsam",
    "geo_perception.sam2_segment",
    "geo_raster.calculate_index",
    "geo_statistics.batch_raster_stats",
    "geo_statistics.mean_of_means",
    "geo_statistics.threshold_ratio",
    "ipython.execute",
}

_SOURCE_ALIAS_SCHEMAS: dict[str, dict[str, Any]] = {
    "Calculator": {
        "canonical_slug": "ipython.execute",
        "description": "Source-schema alias for evaluating a Python math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
    "Solver": {
        "canonical_slug": "ipython.execute",
        "description": "Source-schema alias for executing Python/SymPy code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Python code, optionally wrapped in a markdown code block.",
                }
            },
            "required": ["command"],
        },
    },
    "Plot": {
        "canonical_slug": "ipython.execute",
        "description": "Source-schema alias for executing plotting code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Python plotting code, optionally wrapped in a markdown code block.",
                }
            },
            "required": ["command"],
        },
    },
    "TextToBbox": {
        "canonical_slug": "geo_perception.instructsam",
        "description": "Source-schema alias for text-prompted object localization.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image path."},
                "text": {"type": "string", "description": "Object or region description."},
                "top1": {
                    "type": "boolean",
                    "default": False,
                    "description": "Source-schema option; current backend may ignore it.",
                },
            },
            "required": ["image", "text"],
        },
    },
    "DrawBox": {
        "canonical_slug": "geo_perception.draw_bboxes",
        "description": "Source-schema alias for drawing one bounding box on an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image path."},
                "bbox": {
                    "type": "string",
                    "description": "Bounding box as '(x1,y1,x2,y2)' or '[x1,y1,x2,y2]'.",
                },
                "annotation": {"type": "string", "description": "Optional label.", "default": ""},
            },
            "required": ["image", "bbox"],
        },
    },
    "AddText": {
        "canonical_slug": "geo_perception.add_text",
        "description": "Source-schema alias for adding one text annotation to an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image path."},
                "text": {"type": "string", "description": "Text to render."},
                "position": {
                    "type": "string",
                    "description": "Position token such as lt, mm, rb, or 'x,y'.",
                    "default": "lt",
                },
            },
            "required": ["image", "text"],
        },
    },
    "OCR": {
        "canonical_slug": "geo_perception.ocr_extract",
        "description": "Source-schema alias for extracting text from an image.",
        "parameters": {
            "type": "object",
            "properties": {"image": {"type": "string", "description": "Image path."}},
            "required": ["image"],
        },
    },
    "ObjectDetection": {
        "canonical_slug": "geo_perception.strip_rcnn_detect",
        "description": "Source-schema alias for object detection.",
        "parameters": {
            "type": "object",
            "properties": {"image": {"type": "string", "description": "Image path."}},
            "required": ["image"],
        },
    },
    "SegmentObjectPixels": {
        "canonical_slug": "geo_perception.sam2_segment",
        "description": "Source-schema alias for image segmentation.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image path."},
                "text": {
                    "type": "string",
                    "description": "Source-schema object prompt; current SAM2 backend may ignore it.",
                    "default": "",
                },
                "flag": {
                    "type": "boolean",
                    "description": "Source-schema per-object flag; current SAM2 backend may ignore it.",
                    "default": False,
                },
            },
            "required": ["image"],
        },
    },
}


def _parse_bbox(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {
            "x1": float(value.get("x1", value.get("xmin", 0))),
            "y1": float(value.get("y1", value.get("ymin", 0))),
            "x2": float(value.get("x2", value.get("xmax", 0))),
            "y2": float(value.get("y2", value.get("ymax", 0))),
        }
    text = str(value).strip().strip("()[]")
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"Expected bbox with 4 coordinates, got: {value!r}")
    return {"x1": parts[0], "y1": parts[1], "x2": parts[2], "y2": parts[3]}


def _annotation_position(value: Any) -> tuple[float, float]:
    text = str(value or "lt").strip().lower()
    presets = {"lt": (10, 10), "mm": (100, 100), "rb": (200, 200)}
    if text in presets:
        return presets[text]
    if "," in text:
        x, y = [float(part.strip()) for part in text.split(",", 1)]
        return x, y
    return presets["lt"]


def _translate_source_alias(alias: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    canonical = _SOURCE_ALIAS_SCHEMAS[alias]["canonical_slug"]
    if alias == "Calculator":
        return canonical, {"action": f"print({kwargs['expression']})"}
    if alias in {"Solver", "Plot"}:
        return canonical, {"action": kwargs["command"]}
    if alias == "TextToBbox":
        return canonical, {"image": kwargs["image"], "text_prompt": kwargs["text"]}
    if alias == "DrawBox":
        bbox = _parse_bbox(kwargs["bbox"])
        if kwargs.get("annotation"):
            bbox["label"] = str(kwargs["annotation"])
        return canonical, {
            "image": kwargs["image"],
            "bboxes": [bbox],
            "output_path": f"/tmp/terrabox_drawbox_{uuid.uuid4().hex}.png",
        }
    if alias == "AddText":
        x, y = _annotation_position(kwargs.get("position"))
        return canonical, {
            "image": kwargs["image"],
            "annotations": [{"text": kwargs["text"], "x": x, "y": y}],
            "output_path": f"/tmp/terrabox_addtext_{uuid.uuid4().hex}.png",
        }
    if alias == "OCR":
        return canonical, {"image": kwargs["image"]}
    if alias == "ObjectDetection":
        return canonical, {"image": kwargs["image"]}
    if alias == "SegmentObjectPixels":
        return canonical, {"image": kwargs["image"]}
    return canonical, dict(kwargs)


def _make_alias_tool_func(alias: str, user, pre_execute_validator=None):
    def _call(**kwargs):
        canonical, translated = _translate_source_alias(alias, kwargs)
        if pre_execute_validator is not None:
            result = pre_execute_validator.validate(canonical, translated)
            if not result.ok:
                return result.to_tool_message()
        result = AgentToolExecutor.execute(canonical, translated, user)
        if pre_execute_validator is not None and hasattr(pre_execute_validator, "observe_result"):
            pre_execute_validator.observe_result(canonical, translated, result)
        return result
    return _call


def _tool_description(spec) -> str:
    """Return a model-facing description with non-binding schema aliases."""
    parts = [spec.description, f"Canonical Terrabox slug: {spec.slug}"]
    aliases = _SCHEMA_COMPAT_ALIASES.get(spec.slug)
    if aliases:
        parts.append("Compatible source-schema names: " + ", ".join(aliases))
        if spec.slug in _LOSSY_SCHEMA_COMPAT:
            parts.append(
                "Compatibility note: some source-schema behavior may not be exactly identical; "
                "use the current argument schema and current tool observations."
            )
    return "\n".join(part for part in parts if part)


def _oea_experiment_descriptions_enabled() -> bool:
    return os.environ.get("TERRABOX_OEA_EXPERIMENT_TOOL_DESCRIPTIONS", "").lower() in {"1", "true", "yes"}


_OEA_EXPERIMENT_SCHEMA_PROPERTY_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "geo_perception.vlm_analyze": {
        "images": (
            "One or more aerial/satellite image paths for whole-image scene description only. "
            "For object localization, counting, segmentation, OCR, region attributes, measurements, "
            "or change detection, use the dedicated tools instead."
        ),
        "image": (
            "Single aerial/satellite image path for whole-image scene description only. "
            "Do not use this field to answer object count, location, region attribute, mask, measurement, "
            "OCR, or change-detection questions."
        ),
        "prompt": (
            "Optional focus for the whole-image description. This is not a general visual QA interface; "
            "if the user asks for objects, counts, boxes, regions, headings, colors, condition, pixels, "
            "text, or changes, call the specialized tool."
        ),
        "max_tokens": (
            "Maximum response budget for whole-image description output. Keep the default unless a short "
            "scene description is enough; this parameter does not make VLM suitable for counting, "
            "localization, region-attribute, segmentation, OCR, measurement, or change-detection tasks."
        ),
    },
    "geo_perception.instructsam": {
        "image": "Image path. Input example: image plus text='swimming pool' or text='red truck'.",
        "text": "Target object/region to localize. Output includes bounding boxes/confidences/count-like detections.",
        "text_prompt": "Target object/region to localize. Output includes bounding boxes/confidences/count-like detections.",
        "prompt": "Target object/region to localize. Prefer this over VLM for open-vocabulary object grounding.",
        "top1": "Set true when the task asks for one specific object/region; false when multiple instances may be needed.",
    },
    "geo_perception.count_given_object": {
        "image": "Image path.",
        "text": "Object class/name to count. Output is a count and detected objects.",
        "object": "Object class/name to count. Output is a count and detected objects.",
        "object_name": "Object class/name to count. Output is a count and detected objects.",
        "bbox": "Optional region/bounding box to restrict counting.",
    },
    "geo_perception.region_attribute_description": {
        "image": "Image path.",
        "bbox": "Bounding box/region for the localized object. Use after localization when the question asks condition, heading, color, symmetry, or other attributes.",
        "region": "Region or bounding box for the localized object.",
        "attribute": "Specific attribute to inspect, such as color, damage, condition, orientation, heading, or visible state.",
        "prompt": "Attribute question for the specified region; do not use whole-image VLM when a localized region is required.",
        "question": "Attribute question for the specified region; do not use whole-image VLM when a localized region is required.",
    },
    "geo_perception.sam2_segment": {
        "image": "Image path.",
        "text": "Target object to segment. Output is compact pixel-count/mask artifact information.",
        "flag": "Whether per-object pixel counts are requested when available.",
    },
}


def _schema_for_agent(slug: str, schema: dict[str, Any]) -> dict[str, Any]:
    if not _oea_experiment_descriptions_enabled():
        return schema
    overrides = _OEA_EXPERIMENT_SCHEMA_PROPERTY_DESCRIPTIONS.get(slug)
    if not overrides:
        return schema
    patched = dict(schema)
    properties = {
        name: dict(prop)
        for name, prop in (schema.get("properties") or {}).items()
        if isinstance(prop, dict)
    }
    for name, description in overrides.items():
        if name in properties:
            properties[name]["description"] = description
    patched["properties"] = properties
    return patched


def _json_schema_to_pydantic(slug: str, schema: dict[str, Any]):
    """Dynamically build a Pydantic model from a JSON Schema properties block."""
    schema = _schema_for_agent(slug, schema)
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        prop_type = prop.get("type")
        py_type = Any if prop_type is None else _TYPE_MAP.get(prop_type, str)
        description = prop.get("description")
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            default = prop.get("default", None)
            fields[name] = (Optional[py_type], Field(default, description=description))

    if not fields:
        return None

    model_name = "Args_" + slug.replace(".", "_").replace("-", "_")
    return create_model(model_name, **fields)


def _make_tool_func(slug: str, user, pre_execute_validator=None):
    """Return a callable that routes tool execution through AgentToolExecutor."""
    def _call(**kwargs):
        if pre_execute_validator is not None:
            result = pre_execute_validator.validate(slug, kwargs)
            if not result.ok:
                return result.to_tool_message()
        result = AgentToolExecutor.execute(slug, kwargs, user)
        if pre_execute_validator is not None and hasattr(pre_execute_validator, "observe_result"):
            pre_execute_validator.observe_result(slug, kwargs, result)
        return result
    return _call


_OEA_EXPERIMENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    # Keep the live product ToolSpec rich, but make OEA-style rollouts see
    # narrower descriptions that match the source benchmark's tool roles.
    "geo_perception.vlm_analyze": (
        "Generate a detailed natural-language description of the given aerial "
        "or satellite image. Use this for whole-image visual description only; "
        "use dedicated tools for object localization, counting, segmentation, "
        "change detection, OCR, or region-specific attribute checks."
    ),
    "geo_perception.instructsam": (
        "Locate object(s) in an image from a natural-language description and "
        "return bounding boxes/confidences. Set top1=true when only the best "
        "matching object is needed."
    ),
    "geo_perception.count_given_object": (
        "Count instances of a named object in an image, optionally within a bbox "
        "region. Provide the object name in text/object."
    ),
    "geo_perception.region_attribute_description": (
        "Describe a specific attribute of an object or region in an image, such "
        "as color, condition, structure, heading, or visible state. Provide a "
        "bbox/region when the question refers to a localized object."
    ),
    "geo_perception.sam2_segment": (
        "Segment the specified object in an image and return pixel-count "
        "information. Use this when area-from-pixels or mask size is needed; "
        "provide the target object in text."
    ),
    "geo_perception.strip_rcnn_detect": (
        "Detect remote-sensing objects from the DOTA-style object set such as "
        "planes, ships, storage tanks, courts, bridges, vehicles, harbors, "
        "roundabouts, fields, and pools. For open-vocabulary localization, use "
        "the text-guided localization tool instead."
    ),
    "osm_gis.display_on_map": (
        "Render selected GeoPackage vector layer(s), such as boundary, POIs, or "
        "distance lines, to a PNG map. Use after the required layers have been "
        "created in the GeoPackage."
    ),
    "osm_gis.display_on_geotiff": (
        "Render selected GeoPackage vector layer(s) directly over a GeoTIFF and "
        "save the overlay image/raster."
    ),
    "osm_gis.add_index_layer": (
        "Compute an NDVI/NDBI/NBR layer for the AOI stored in a GeoPackage for a "
        "given year and optional month, then save the raster layer to the same "
        "GeoPackage."
    ),
    "osm_gis.get_bbox_from_raster": (
        "Get the bounding box of a GeoTIFF raster."
    ),
}


def _description_for_agent(spec) -> str:
    if _oea_experiment_descriptions_enabled():
        override = _OEA_EXPERIMENT_TOOL_DESCRIPTIONS.get(spec.slug)
        if override:
            return override
    return _tool_description(spec)


def build_langchain_tools(user, slugs: Optional[list[str]] = None, pre_execute_validator=None) -> list[StructuredTool]:
    """
    Wrap every tool registered in CoreRegistry as a LangChain StructuredTool.
    The agent LLM can call any of these tools autonomously.

    Args:
        user: The authenticated user object passed to each tool handler.
        slugs: Optional list of tool slugs to include. If None, all tools are included.
        pre_execute_validator: Optional object with validate(slug, arguments) used to
            short-circuit invalid calls before AgentToolExecutor runs.
    """
    tools = []
    registered_slugs = set()
    for spec in registry.list_tools():
        if slugs is not None and spec.slug not in slugs:
            continue
        if registry.get_handler(spec.slug) is None:
            continue
        registered_slugs.add(spec.slug)

        args_schema = _json_schema_to_pydantic(spec.slug, spec.parameters)
        lc_tool = StructuredTool.from_function(
            func=_make_tool_func(spec.slug, user, pre_execute_validator=pre_execute_validator),
            # LangChain tool names must not contain dots
            name=spec.slug.replace(".", "__"),
            description=_description_for_agent(spec),
            args_schema=args_schema,
        )
        tools.append(lc_tool)
        logger.debug(f"Registered LangChain tool: {lc_tool.name}")

    if os.environ.get("TERRABOX_ENABLE_SOURCE_SCHEMA_TOOL_ALIASES", "").lower() in {"1", "true", "yes"}:
        for alias, meta in _SOURCE_ALIAS_SCHEMAS.items():
            canonical = meta["canonical_slug"]
            if canonical not in registered_slugs:
                continue
            args_schema = _json_schema_to_pydantic(alias, meta["parameters"])
            lc_tool = StructuredTool.from_function(
                func=_make_alias_tool_func(alias, user, pre_execute_validator=pre_execute_validator),
                name=alias,
                description=(
                    f"{meta['description']}\n"
                    f"Routes to canonical Terrabox slug: {canonical}\n"
                    "Use the alias argument schema shown here; execution is recorded through the canonical tool."
                ),
                args_schema=args_schema,
            )
            tools.append(lc_tool)
            logger.debug("Registered source-schema alias tool: %s -> %s", alias, canonical)

    logger.info(f"Built {len(tools)} LangChain tools from CoreRegistry")
    return tools
