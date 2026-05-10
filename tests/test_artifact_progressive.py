import unittest

from terrabox.agent.artifacts.readiness import ready_slugs_by_category
from terrabox.agent.artifacts.state import extract_paths_from_text, initial_artifact_state
from terrabox.core.registry import ToolSpec, registry


class ArtifactProgressiveTests(unittest.TestCase):
    def setUp(self):
        registry.clear()
        registry.register_toolkit("osm_gis", "OSM GIS")
        registry.register_tool(
            ToolSpec(
                slug="osm_gis.get_area_boundary",
                name="Get Area Boundary",
                description="Create an area artifact.",
                parameters={
                    "type": "object",
                    "properties": {"area": {"type": "string"}},
                    "required": ["area"],
                },
            ),
            lambda args, context, user: {},
        )
        registry.register_tool(
            ToolSpec(
                slug="osm_gis.add_pois_layer",
                name="Add POIs Layer",
                description="Add a vector layer to a GeoPackage.",
                parameters={
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string"},
                        "layer_name": {"type": "string"},
                    },
                    "required": ["gpkg", "layer_name"],
                },
            ),
            lambda args, context, user: {},
        )

    def test_units_are_not_extracted_as_paths(self):
        text = "Assume 60 km/h and use /data/example/result.gpkg when provided."

        self.assertNotIn("/h", extract_paths_from_text(text))
        self.assertIn("/data/example/result.gpkg", extract_paths_from_text(text))

    def test_successful_source_tool_is_lower_priority_in_normal_mode(self):
        state = initial_artifact_state("task")
        state["artifacts"].append({"kind": "gpkg", "path": "/tmp/aoi.gpkg", "source": "test"})
        state["successful_calls"].append("osm_gis.get_area_boundary")

        ready = ready_slugs_by_category(
            state,
            {"osm_gis.get_area_boundary", "osm_gis.add_pois_layer"},
        )

        self.assertEqual(ready["osm_gis"], ["osm_gis.add_pois_layer", "osm_gis.get_area_boundary"])

    def test_source_tool_returns_to_front_in_recovery_mode(self):
        state = initial_artifact_state("task")
        state["artifacts"].append({"kind": "gpkg", "path": "/tmp/aoi.gpkg", "source": "test"})
        state["successful_calls"].append("osm_gis.get_area_boundary")
        state["failed_calls"].append({"tool": "osm_gis.add_pois_layer", "args": {}, "error": "failed"})

        ready = ready_slugs_by_category(
            state,
            {"osm_gis.get_area_boundary", "osm_gis.add_pois_layer"},
            recovery_mode=True,
        )

        self.assertEqual(ready["osm_gis"], ["osm_gis.get_area_boundary", "osm_gis.add_pois_layer"])


if __name__ == "__main__":
    unittest.main()
