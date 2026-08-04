import unittest

from terrabox.agent.artifacts.readiness import ready_slugs_by_category
from terrabox.agent.artifacts.extractors import update_artifact_state
from terrabox.agent.artifacts.signatures import product_state_tokens
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

    def test_tool_error_does_not_create_success_product_state(self):
        state = initial_artifact_state("task")

        update_artifact_state(
            state,
            "compute.calculator",
            {"expression": "bad"},
            "Error in calculator: invalid syntax (<string>, line 1)",
        )

        self.assertEqual(state["successful_calls"], [])
        self.assertEqual(state["failed_calls"][0]["tool"], "compute.calculator")
        self.assertNotIn("result:from:compute.calculator", product_state_tokens(state))

    def test_successful_call_records_keep_arguments_for_runtime_guards(self):
        state = initial_artifact_state("task")

        update_artifact_state(
            state,
            "geo_perception.instructsam",
            {"image": "/tmp/a.jpg", "text": "garbage pile"},
            '{"status": "success", "count": 1}',
        )

        self.assertEqual(state["successful_calls"], ["geo_perception.instructsam"])
        self.assertEqual(
            state["successful_call_records"],
            [
                {
                    "tool": "geo_perception.instructsam",
                    "args": {"image": "/tmp/a.jpg", "text": "garbage pile"},
                }
            ],
        )

    def test_truncated_success_json_still_creates_success_product_state(self):
        state = initial_artifact_state("task")

        update_artifact_state(
            state,
            "geo_perception.instructsam",
            {"image": "/tmp/a.jpg", "text": "non-flooded house"},
            '{"status": "success", "count": 15, "objects": [' + ("x" * 2500),
        )

        self.assertEqual(state["failed_calls"], [])
        self.assertEqual(state["successful_calls"], ["geo_perception.instructsam"])
        self.assertIn("result:from:geo_perception.instructsam", product_state_tokens(state))


if __name__ == "__main__":
    unittest.main()
