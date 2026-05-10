import os
import tempfile
import time
import unittest
from unittest import mock

from terrabox.agent.tool_executor import AgentToolExecutor
from terrabox.core.registry import ToolSpec, registry
from terrabox.toolkits import osm_gis


class ToolTimeoutTests(unittest.TestCase):
    def test_agent_tool_executor_returns_timeout_observation(self):
        slug = "timeout_test.sleep"
        registry.register_toolkit("timeout_test", "timeout tests")

        def handler(args, context, account):
            time.sleep(2)
            return {"status": "success"}

        registry.register_tool(
            ToolSpec(
                slug=slug,
                name="Sleep",
                description="Sleep long enough to trigger tool timeout.",
                parameters={"type": "object", "properties": {}},
            ),
            handler,
        )

        with mock.patch.dict(os.environ, {"TERRABOX_TOOL_TIMEOUT_TIMEOUT_TEST_SLEEP": "1"}):
            started = time.time()
            result = AgentToolExecutor.execute(slug, {}, None)

        self.assertLess(time.time() - started, 1.5)
        self.assertIn('"error_type": "tool_timeout"', result)
        self.assertIn('"tool": "timeout_test.sleep"', result)

    def test_compute_route_dist_cost_guard_returns_before_expensive_loop(self):
        class FakeGDF:
            empty = False
            columns = []

            def __init__(self, n):
                self._n = n

            def __len__(self):
                return self._n

            def to_crs(self, crs):
                return self

        class FakeGPD:
            def read_file(self, path, layer=None):
                return FakeGDF(246 if layer == "clinics" else 316)

        with tempfile.NamedTemporaryFile(suffix=".gpkg") as tmp:
            with (
                mock.patch.object(osm_gis, "_lazy_osmnx", return_value=object()),
                mock.patch.object(osm_gis, "_lazy_gpd", return_value=FakeGPD()),
            ):
                result = osm_gis.compute_route_dist_handler(
                    {
                        "gpkg": tmp.name,
                        "src_layer": "clinics",
                        "tar_layer": "pharmacies",
                        "top": 1000,
                        "max_pairs": 5000,
                    },
                    {"timeout": 120, "deadline": time.time() + 120},
                    None,
                )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "tool_cost_guard")
        self.assertEqual(result["estimated_pairs"], 246 * 316)


if __name__ == "__main__":
    unittest.main()
