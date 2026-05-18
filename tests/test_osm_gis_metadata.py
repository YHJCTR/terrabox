from terrabox.toolkits import osm_gis


class MockRegistrar:
    def __init__(self):
        self.toolkits = {}
        self.tools = {}

    def toolkit(self, name, description=None, version=None):
        self.toolkits[name] = {
            "description": description,
            "version": version,
        }

    def tool(self, spec, handler):
        self.tools[spec.slug] = spec


def test_osm_gis_network_tools_do_not_require_account_connection():
    registrar = MockRegistrar()

    osm_gis.setup(registrar)

    assert registrar.tools["osm_gis.get_area_boundary"].requires_connection is False
    assert registrar.tools["osm_gis.add_pois_layer"].requires_connection is False
    assert registrar.tools["osm_gis.compute_route_dist"].requires_connection is False
    assert registrar.tools["osm_gis.get_bbox_from_raster"].requires_connection is False
