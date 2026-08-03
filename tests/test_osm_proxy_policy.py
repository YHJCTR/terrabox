from types import SimpleNamespace
from pathlib import Path

from terrabox.toolkits import osm_gis


class _FakeOsmnx:
    def __init__(self):
        self.settings = SimpleNamespace(requests_kwargs={"headers": {"User-Agent": "keep"}})
        self.seen_requests_kwargs = []

    def geocode_to_gdf(self, area):
        self.seen_requests_kwargs.append(dict(self.settings.requests_kwargs))
        raise RuntimeError("stop after proxy capture")


class _PointFallbackOsmnx:
    def __init__(self):
        self.settings = SimpleNamespace(requests_kwargs={})
        self.geocode_to_gdf_called = False
        self.geocode_called = False

    def geocode_to_gdf(self, area):
        self.geocode_to_gdf_called = True
        raise RuntimeError("Nominatim did not geocode query to a geometry of type (Multi)Polygon")

    def geocode(self, area):
        self.geocode_called = True
        return (45.8208, -64.5763)


def test_osm_proxy_context_sets_and_restores_osmnx_requests_kwargs(monkeypatch):
    fake_ox = _FakeOsmnx()
    monkeypatch.setenv("TERRABOX_OSM_HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("TERRABOX_OSM_HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("TERRABOX_OSM_NO_PROXY", "localhost,127.0.0.1")

    with osm_gis._osm_proxy_context(fake_ox):
        assert fake_ox.settings.requests_kwargs["proxies"] == {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890",
            "no_proxy": "localhost,127.0.0.1",
        }
        assert fake_ox.settings.requests_kwargs["headers"] == {"User-Agent": "keep"}

    assert fake_ox.settings.requests_kwargs == {"headers": {"User-Agent": "keep"}}


def test_osm_proxy_context_defaults_to_direct_requests(monkeypatch):
    fake_ox = _FakeOsmnx()
    monkeypatch.delenv("TERRABOX_OSM_HTTP_PROXY", raising=False)
    monkeypatch.delenv("TERRABOX_OSM_HTTPS_PROXY", raising=False)

    with osm_gis._osm_proxy_context(fake_ox):
        assert fake_ox.settings.requests_kwargs["proxies"]["http"] is None
        assert fake_ox.settings.requests_kwargs["proxies"]["https"] is None
        assert fake_ox.settings.requests_kwargs["headers"] == {"User-Agent": "keep"}

    assert fake_ox.settings.requests_kwargs == {"headers": {"User-Agent": "keep"}}


def test_osm_proxy_context_uses_osmnx_timeout_setting_without_duplicate_kwarg(monkeypatch):
    fake_ox = _FakeOsmnx()
    fake_ox.settings.requests_timeout = 17
    fake_ox.settings.requests_kwargs["timeout"] = 99
    monkeypatch.setenv("TERRABOX_OSM_REQUEST_TIMEOUT", "45")

    with osm_gis._osm_proxy_context(fake_ox):
        assert fake_ox.settings.requests_timeout == 45
        assert "timeout" not in fake_ox.settings.requests_kwargs

    assert fake_ox.settings.requests_timeout == 17
    assert fake_ox.settings.requests_kwargs == {"headers": {"User-Agent": "keep"}, "timeout": 99}


def test_normalise_poi_query_maps_generic_categories():
    assert osm_gis._normalise_poi_query("bar")[0] == {"amenity": "bar"}
    assert osm_gis._normalise_poi_query("bus stops")[0] == {"highway": "bus_stop"}
    assert osm_gis._normalise_poi_query("shop=supermarket")[0] == {"shop": "supermarket"}
    assert osm_gis._normalise_poi_query('{"amenity": "school"}')[0] == {"amenity": "school"}


def test_get_area_boundary_enters_osm_proxy_context(monkeypatch):
    fake_ox = _FakeOsmnx()
    monkeypatch.setenv("TERRABOX_OSM_HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("TERRABOX_OSM_HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setattr(osm_gis, "_lazy_osmnx", lambda: fake_ox)
    monkeypatch.setattr(osm_gis, "_lazy_gpd", lambda: object())

    result = osm_gis.get_area_boundary_handler({"area": "Seattle, Washington, USA"}, {}, None)

    assert result["status"] == "error"
    assert fake_ox.seen_requests_kwargs
    assert fake_ox.seen_requests_kwargs[0]["proxies"]["http"] == "http://127.0.0.1:7890"
    assert fake_ox.settings.requests_kwargs == {"headers": {"User-Agent": "keep"}}


def test_get_area_boundary_buffers_point_geocode_when_polygon_missing(monkeypatch, tmp_path):
    import geopandas as gpd

    fake_ox = _PointFallbackOsmnx()
    output_path = tmp_path / "hopewell.gpkg"

    def fake_to_file(self, path, layer=None, driver=None, **kwargs):
        Path(path).write_bytes(b"gpkg")

    monkeypatch.setattr(
        osm_gis,
        "_nominatim_geocode_point",
        lambda area: (_ for _ in ()).throw(ValueError("no direct point")),
    )
    monkeypatch.setattr(osm_gis, "_lazy_osmnx", lambda: fake_ox)
    monkeypatch.setattr(osm_gis, "_lazy_gpd", lambda: gpd)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file)

    result = osm_gis.get_area_boundary_handler(
        {
            "area": "Hopewell Rocks, New Brunswick, Canada",
            "buffer_m": 2000,
            "output_path": str(output_path),
        },
        {},
        None,
    )

    assert result["status"] == "success"
    assert result["gpkg"] == str(output_path)
    assert result["feature_count"] == 1
    assert output_path.exists()
    assert not fake_ox.geocode_to_gdf_called
    assert fake_ox.geocode_called


def test_get_area_boundary_rejects_excessive_buffer_before_geocode(monkeypatch):
    fake_ox = _PointFallbackOsmnx()
    monkeypatch.setenv("TERRABOX_OSM_MAX_BUFFER_M", "1000")
    monkeypatch.setattr(osm_gis, "_lazy_osmnx", lambda: fake_ox)
    monkeypatch.setattr(osm_gis, "_lazy_gpd", lambda: object())

    result = osm_gis.get_area_boundary_handler(
        {"area": "Parthenon, Athens, Greece", "buffer_m": 5000},
        {},
        None,
    )

    assert result["status"] == "error"
    assert result["error_type"] == "osm_buffer_too_large"
    assert not fake_ox.geocode_to_gdf_called
    assert not fake_ox.geocode_called
