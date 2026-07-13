from types import SimpleNamespace

from terrabox.toolkits import osm_gis


class _FakeOsmnx:
    def __init__(self):
        self.settings = SimpleNamespace(requests_kwargs={"headers": {"User-Agent": "keep"}})
        self.seen_requests_kwargs = []

    def geocode_to_gdf(self, area):
        self.seen_requests_kwargs.append(dict(self.settings.requests_kwargs))
        raise RuntimeError("stop after proxy capture")


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
