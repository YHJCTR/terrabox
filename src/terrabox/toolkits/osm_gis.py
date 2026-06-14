"""
OSM GIS Toolkit
---------------
Geographic data acquisition from OpenStreetMap (OSM) using GeoPackage as the
shared data container between tools — aligned with OpenEarthAgent's architecture.

Data flow:
  1. GetAreaBoundary creates a .gpkg file with an "area_boundary" layer.
  2. AddPoisLayer reads the boundary from the gpkg and appends a new POI layer.
  3. ComputeRouteDistance reads two POI layers from the gpkg and computes
     pairwise road-network distances, appending a distances line layer.
  4. GetBboxFromRaster extracts metadata from a GeoTIFF (standalone).

The agent orchestrator auto-injects the current gpkg path into downstream tools
so the LLM only needs to pass a placeholder reference.
"""

import json
import os
import logging
import re
import copy
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from ..core.registry import ToolSpec

logger = logging.getLogger(__name__)

CRS = "EPSG:4326"
DEFAULT_ROUTE_DIST_MAX_PAIRS = int(os.environ.get("TERRABOX_ROUTE_DIST_MAX_PAIRS", "5000"))


# ------------------------------------------------------------------------------
# Lazy Imports
# ------------------------------------------------------------------------------

def _lazy_osmnx():
    try:
        import osmnx as ox
        return ox
    except ImportError:
        raise ImportError(
            "Missing osmnx. Install: pip install osmnx geopandas networkx"
        )


@contextmanager
def _osm_proxy_context(ox):
    """Temporarily apply proxy settings only to OSMnx external requests."""
    settings = getattr(ox, "settings", None)
    if settings is None or not hasattr(settings, "requests_kwargs"):
        yield
        return

    original_kwargs = copy.deepcopy(getattr(settings, "requests_kwargs", {}) or {})
    http_proxy = os.environ.get("TERRABOX_OSM_HTTP_PROXY", "").strip()
    https_proxy = os.environ.get("TERRABOX_OSM_HTTPS_PROXY", "").strip()
    no_proxy = os.environ.get("TERRABOX_OSM_NO_PROXY", "localhost,127.0.0.1,::1").strip()

    if not http_proxy and not https_proxy:
        yield
        return

    updated_kwargs = copy.deepcopy(original_kwargs)
    proxies = dict(updated_kwargs.get("proxies") or {})
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    if no_proxy:
        proxies["no_proxy"] = no_proxy
    updated_kwargs["proxies"] = proxies

    settings.requests_kwargs = updated_kwargs
    try:
        yield
    finally:
        settings.requests_kwargs = original_kwargs


def _with_osm_proxy(handler):
    @wraps(handler)
    def wrapper(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
        ox = _lazy_osmnx()
        with _osm_proxy_context(ox):
            return handler(arguments, context, account)

    return wrapper


def _lazy_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("Missing rasterio. Install: pip install rasterio")


def _lazy_gpd():
    try:
        import geopandas as gpd
        return gpd
    except ImportError:
        raise ImportError("Missing geopandas. Install: pip install geopandas")


def _is_valid_bbox(bbox):
    """Validate bbox = (west, south, east, north)."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False
    try:
        west, south, east, north = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return False
    return (-180 <= west <= 180 and -180 <= east <= 180
            and -90 <= south <= 90 and -90 <= north <= 90
            and west < east and south < north)


def _get_save_dir() -> str:
    """Return (and create) a directory for gpkg outputs."""
    save_dir = os.environ.get("TERRABOX_GPKG_OUTPUT_DIR") or os.path.join(os.getcwd(), "tmp", "gpkg_output")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _deadline_exceeded(context: Any) -> bool:
    if not isinstance(context, dict):
        return False
    deadline = context.get("deadline")
    if not deadline:
        return False
    try:
        return datetime.now().timestamp() > float(deadline)
    except Exception:
        return False


def _tool_timeout_result(tool: str, context: Any, message: str) -> Dict[str, Any]:
    timeout = context.get("timeout") if isinstance(context, dict) else None
    return {
        "status": "error",
        "error_type": "tool_timeout",
        "tool": tool,
        "timeout_seconds": timeout,
        "message": message,
        "recovery_suggestions": [
            "Retry with a smaller area of interest.",
            "Reduce top/limit if the tool supports it.",
            "Filter source or target layers before retrying.",
            "Do not repeat the same arguments that timed out.",
        ],
    }


def _is_placeholder_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith("/path/")
        or lowered.startswith("path/")
        or "path_to_" in lowered
        or "/path/to/" in lowered
        or "<" in path
        or ">" in path
    )


def _get_name_from_row(row):
    """Return best display name for an OSM feature row."""
    import pandas as pd
    fields = row.index
    for key in ("name:en", "name"):
        if key in fields:
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    for key in fields:
        if key.startswith("name"):
            val = row.get(key, "")
            if pd.notna(val) and str(val).strip():
                return str(val)
    return ""


def _clean_for_gpkg(gdf):
    """Sanitize column names for GeoPackage driver compatibility."""
    gdf = gdf.copy()
    gdf.columns = gdf.columns.str.lower().str[:30]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf


# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------

@_with_osm_proxy
def get_area_boundary_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Create a GeoPackage with the boundary of a place saved as 'area_boundary' layer.
    Returns a short summary and the gpkg path — no large GeoJSON in the response.
    """
    ox = _lazy_osmnx()
    gpd = _lazy_gpd()
    from shapely.geometry import box as shapely_box

    # Accept both "area" (OpenEarthAgent compat) and "place_name" (legacy)
    area = arguments.get("area") or arguments.get("place_name")
    bbox = arguments.get("bbox")
    buffer_m = arguments.get("buffer_m")
    output_path = arguments.get("output_path")
    if isinstance(output_path, str) and _is_placeholder_path(output_path):
        output_path = None

    # Parse bbox from string if needed (e.g. "(1.2, 3.4, 5.6, 7.8)")
    if isinstance(area, str) and not bbox:
        bbox_pattern = r"^[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]$"
        match = re.match(bbox_pattern, area.strip())
        if match:
            try:
                bbox = tuple(float(v) for v in match.groups())
                area = None
            except ValueError:
                pass

    if area and isinstance(area, str):
        try:
            gdf = ox.geocode_to_gdf(area).to_crs(CRS)
        except Exception as e:
            return {"status": "error", "message": f"Failed to geocode '{area}': {e}"}
        geom = gdf.geometry.union_all()
        name = area
    elif bbox and _is_valid_bbox(bbox):
        west, south, east, north = [float(v) for v in bbox]
        geom = shapely_box(west, south, east, north)
        name = f"bbox({west},{south},{east},{north})"
    elif bbox:
        return {"status": "error", "message": f"Invalid bbox: {bbox}"}
    else:
        return {"status": "error", "message": "Provide either 'area' (place name) or 'bbox'."}

    # Apply buffer if requested
    if buffer_m and float(buffer_m) > 0:
        buffer_m = float(buffer_m)
        gseries = gpd.GeoSeries([geom], crs=CRS).to_crs("EPSG:3857")
        geom = gseries.buffer(buffer_m).to_crs(CRS).iloc[0]
        name = f"{name}_buffer{int(buffer_m)}m"

    gdf_out = gpd.GeoDataFrame(
        {"name": [name], "created_at": [datetime.now().isoformat()], "year": [datetime.now().year]},
        geometry=[geom], crs=CRS,
    )

    # Determine save path
    if not output_path:
        gpkg_name = f"aoi_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
        output_path = os.path.join(_get_save_dir(), gpkg_name)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        gdf_out.to_file(output_path, layer="area_boundary", driver="GPKG")
    except Exception as e:
        return {"status": "error", "message": f"Error saving GeoPackage: {e}"}

    gpkg_basename = os.path.basename(output_path)
    bounds = gdf_out.total_bounds.tolist()  # [minx, miny, maxx, maxy]

    return {
        "status": "success",
        "text": f"Saved boundary to {gpkg_basename}",
        "gpkg": output_path,
        "feature_count": 1,
        "bbox": bounds,
    }


@_with_osm_proxy
def add_pois_layer_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Add POIs from OSM into an existing GeoPackage, clipped to the area_boundary.
    The gpkg path is auto-injected by the orchestrator.
    """
    ox = _lazy_osmnx()
    gpd = _lazy_gpd()
    import pandas as pd

    gpkg = arguments.get("gpkg")
    query = arguments.get("query") or arguments.get("tags", {"amenity": True})
    layer_name = arguments.get("layer_name", "pois")

    # Legacy compat: if no gpkg but place_name given, fall back
    if not gpkg:
        place_name = arguments.get("place_name")
        if place_name:
            return _add_pois_legacy(arguments, context, account)
        return {"status": "error", "message": "Missing 'gpkg' parameter. Run get_area_boundary first."}

    if not os.path.exists(gpkg):
        return {"status": "error", "message": f"GeoPackage not found: {gpkg}"}

    # Read boundary
    try:
        area_gdf = gpd.read_file(gpkg, layer="area_boundary")
        geom = area_gdf.geometry.iloc[0]
    except Exception as e:
        return {"status": "error", "message": f"Failed to read area_boundary from gpkg: {e}"}

    # Parse query
    if isinstance(query, str):
        try:
            query = json.loads(query)
        except Exception:
            # Treat as POI name
            try:
                pois = ox.geocode_to_gdf(query).to_crs(CRS)
                pois = pois[pois.geometry.within(geom)]
                if pois.empty:
                    return {"status": "error", "message": f"POI '{query}' not found inside study area."}
            except Exception as e:
                return {"status": "error", "message": f"OSM query failed: {e}"}
            pois["display_name"] = pois.apply(_get_name_from_row, axis=1)
            pois = pois[pois["display_name"] != ""]
            pois = pois.drop_duplicates(subset="display_name", keep="first")
            pois = _clean_for_gpkg(pois)
            pois.to_file(gpkg, layer=layer_name, driver="GPKG")
            gpkg_basename = os.path.basename(gpkg)
            return {
                "status": "success",
                "text": f"Saved {len(pois)} POIs to layer '{layer_name}' in {gpkg_basename}",
                "gpkg": gpkg,
                "poi_count": len(pois),
            }

    if not isinstance(query, dict):
        return {"status": "error", "message": f"query must be dict or str, got {type(query).__name__}"}

    try:
        pois = ox.features_from_polygon(geom, query).to_crs(CRS)
    except Exception as e:
        err_msg = str(e)
        if "No matching" in err_msg or "No data" in err_msg:
            return {"status": "error", "message": f"OSM query failed: No matching features. Check query location, tags, and log."}
        return {"status": "error", "message": f"OSM query failed: {e}"}

    if pois.empty:
        return {"status": "error", "message": f"No POIs found for {query} inside boundary."}

    # Extract display names and deduplicate
    pois["display_name"] = pois.apply(_get_name_from_row, axis=1)
    pois = pois[pois["display_name"] != ""]
    pois = pois.drop_duplicates(subset="display_name", keep="first")

    if pois.empty:
        return {"status": "error", "message": f"No named POIs found for {query} inside boundary."}

    pois = _clean_for_gpkg(pois)
    pois.to_file(gpkg, layer=layer_name, driver="GPKG")

    gpkg_basename = os.path.basename(gpkg)
    return {
        "status": "success",
        "text": f"Saved {len(pois)} POIs to layer '{layer_name}' in {gpkg_basename}",
        "gpkg": gpkg,
        "poi_count": len(pois),
    }


def _add_pois_legacy(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Legacy fallback: query POIs by place_name without gpkg."""
    ox = _lazy_osmnx()
    place_name = arguments.get("place_name")
    tags = arguments.get("tags", {"amenity": True})
    max_results = int(arguments.get("max_results", 500))

    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = {"amenity": tags}

    try:
        gdf = ox.features_from_place(place_name, tags=tags)
    except Exception as e:
        return {"status": "error", "message": f"OSM query failed: {e}"}

    if len(gdf) > max_results:
        gdf = gdf.head(max_results)

    points_gdf = gdf[gdf.geometry.geom_type == "Point"].copy() if len(gdf) > 0 else gdf
    poi_count = len(points_gdf)

    # Save to temp gpkg instead of returning inline GeoJSON
    save_dir = _get_save_dir()
    gpkg_path = os.path.join(save_dir, f"pois_{datetime.now():%Y%m%d_%H%M%S}.gpkg")
    if poi_count > 0:
        _clean_for_gpkg(points_gdf.to_crs(CRS)).to_file(gpkg_path, layer="pois", driver="GPKG")

    return {
        "status": "success",
        "text": f"Found {poi_count} POIs for '{place_name}'",
        "poi_count": poi_count,
        "gpkg": gpkg_path if poi_count > 0 else None,
    }


@_with_osm_proxy
def compute_route_dist_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute pairwise road-network distances between features of two layers
    in a GeoPackage. Saves results as a new line layer. Returns text summary.

    If gpkg/src_layer/tar_layer are given → gpkg-based mode (OpenEarthAgent compat).
    If origin/destination are given → legacy point-to-point mode.
    """
    gpkg = arguments.get("gpkg")
    src_layer = arguments.get("src_layer")
    tar_layer = arguments.get("tar_layer")

    if gpkg and src_layer and tar_layer:
        return _compute_dist_gpkg(arguments, context, account)
    else:
        return _compute_dist_legacy(arguments, context, account)


def _compute_dist_gpkg(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute pairwise distances between two layers in a GeoPackage."""
    ox = _lazy_osmnx()
    gpd = _lazy_gpd()
    import networkx as nx
    from shapely.geometry import LineString, Point

    gpkg = arguments["gpkg"]
    src_layer = arguments["src_layer"]
    tar_layer = arguments["tar_layer"]
    top = arguments.get("top")
    if top is not None:
        top = int(top)
        if top <= 0:
            return {"status": "error", "message": "top must be a positive integer."}

    if not os.path.exists(gpkg):
        return {"status": "error", "message": f"GeoPackage not found: {gpkg}"}

    try:
        src_gdf = gpd.read_file(gpkg, layer=src_layer).to_crs(CRS)
        tar_gdf = gpd.read_file(gpkg, layer=tar_layer).to_crs(CRS)
    except Exception as e:
        return {"status": "error", "message": f"Failed to read layers: {e}"}

    if src_gdf.empty or tar_gdf.empty:
        return {"status": "error", "message": f"One or both layers are empty: {src_layer}={len(src_gdf)}, {tar_layer}={len(tar_gdf)}"}

    per_source_targets = min(len(tar_gdf), top) if top is not None else len(tar_gdf)
    estimated_pairs = len(src_gdf) * per_source_targets
    max_pairs = int(
        arguments.get("max_pairs")
        or (context.get("max_pairs") if isinstance(context, dict) else 0)
        or DEFAULT_ROUTE_DIST_MAX_PAIRS
    )
    if estimated_pairs > max_pairs:
        return {
            "status": "error",
            "error_type": "tool_cost_guard",
            "tool": "osm_gis.compute_route_dist",
            "message": (
                f"Route distance call is too large: estimated {estimated_pairs} pairwise computations "
                f"({src_layer}={len(src_gdf)}, {tar_layer}={len(tar_gdf)}, top={top or 'all'}), "
                f"limit={max_pairs}."
            ),
            "estimated_pairs": estimated_pairs,
            "src_count": len(src_gdf),
            "tar_count": len(tar_gdf),
            "top": top,
            "max_pairs": max_pairs,
            "recovery_suggestions": [
                "Retry with a smaller top value.",
                "Use a smaller area boundary around the actual landmark or target place.",
                "Filter one or both POI layers before computing route distances.",
                "If exact road distance is not required, use a cheaper direct/geodesic calculation.",
            ],
        }

    if _deadline_exceeded(context):
        return _tool_timeout_result(
            "osm_gis.compute_route_dist",
            context,
            "Timed out before route distance computation started.",
        )

    # Get display names
    name_col_src = "display_name" if "display_name" in src_gdf.columns else ("name" if "name" in src_gdf.columns else None)
    name_col_tar = "display_name" if "display_name" in tar_gdf.columns else ("name" if "name" in tar_gdf.columns else None)

    # Build road network from boundary
    use_network = True
    try:
        boundary_gdf = gpd.read_file(gpkg, layer="area_boundary")
        boundary = boundary_gdf.geometry.iloc[0]
        network = ox.graph_from_polygon(boundary, network_type="drive", simplify=True)
        network = ox.project_graph(network)
        network = ox.add_edge_speeds(network, fallback=40)
        for u, v, k, data in network.edges(keys=True, data=True):
            if "speed_kph" not in data or data["speed_kph"] is None:
                data["speed_kph"] = 40
        network = ox.add_edge_travel_times(network)
    except Exception as e:
        logger.warning(f"Failed to build road network, falling back to geodesic: {e}")
        use_network = False

    results = []
    for _, f1 in src_gdf.iterrows():
        if _deadline_exceeded(context):
            return _tool_timeout_result(
                "osm_gis.compute_route_dist",
                context,
                f"Timed out while computing route distances; partial source rows processed: {len(results)} result pairs.",
            )
        name1 = f1[name_col_src] if name_col_src and name_col_src in f1.index else f"src_{_}"
        if not str(name1).strip():
            continue
        pt1 = f1.geometry.centroid

        row_results = []
        for _, f2 in tar_gdf.iterrows():
            if _deadline_exceeded(context):
                return _tool_timeout_result(
                    "osm_gis.compute_route_dist",
                    context,
                    f"Timed out while computing route distances; partial source rows processed: {len(results)} result pairs.",
                )
            name2 = f2[name_col_tar] if name_col_tar and name_col_tar in f2.index else f"tar_{_}"
            if not str(name2).strip():
                continue
            pt2 = f2.geometry.centroid

            dist = None
            travel_time = None
            geom_line = LineString([pt1, pt2])

            if use_network:
                try:
                    graph_crs = network.graph["crs"]
                    pt1_proj = gpd.GeoSeries([pt1], crs=CRS).to_crs(graph_crs).iloc[0]
                    pt2_proj = gpd.GeoSeries([pt2], crs=CRS).to_crs(graph_crs).iloc[0]
                    orig_node = ox.distance.nearest_nodes(network, pt1_proj.x, pt1_proj.y)
                    dest_node = ox.distance.nearest_nodes(network, pt2_proj.x, pt2_proj.y)
                    dist = nx.shortest_path_length(network, orig_node, dest_node, weight="length")
                    travel_time = nx.shortest_path_length(network, orig_node, dest_node, weight="travel_time")
                except Exception:
                    dist = None

            # Fallback to geodesic distance
            if dist is None:
                from pyproj import Geod
                geod = Geod(ellps="WGS84")
                _, _, dist = geod.inv(pt1.x, pt1.y, pt2.x, pt2.y)

            row_results.append({
                src_layer: str(name1),
                tar_layer: str(name2),
                "distance_m": round(dist, 2),
                "travel_time_s": round(travel_time, 1) if travel_time else None,
                "geometry": geom_line,
            })

        row_results.sort(key=lambda r: r["distance_m"])
        if top is not None:
            results.extend(row_results[:top])
        else:
            results.extend(row_results)

    if not results:
        return {"status": "error", "message": "No valid feature pairs found between the two layers."}

    # Save distances as line layer
    dist_layer_name = f"{src_layer}_to_{tar_layer}_distances"
    dist_gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=CRS)
    dist_gdf.to_file(gpkg, layer=dist_layer_name, driver="GPKG")

    # Build text summary
    out_lines = [f"Distances (in meters) saved to line layer: '{dist_layer_name}': "]
    distances = []
    for r in results:
        txt = f"{r[src_layer]} , {r[tar_layer]}, distance={r['distance_m']:.2f} m"
        if r.get("travel_time_s"):
            txt += f", travel_time={r['travel_time_s']:.1f} s"
        out_lines.append(txt)
        distances.append(r["distance_m"])
    out_lines.append(f"distances = {distances}")

    return {
        "status": "success",
        "text": "\n".join(out_lines),
        "gpkg": gpkg,
        "pair_count": len(results),
    }


def _compute_dist_legacy(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Legacy: point-to-point route distance."""
    ox = _lazy_osmnx()

    origin = arguments.get("origin")
    destination = arguments.get("destination")
    travel_mode = arguments.get("travel_mode", "drive")

    if not origin or not destination:
        return {"status": "error", "message": "Provide 'origin' and 'destination' as [lon, lat] pairs"}

    orig_lon, orig_lat = float(origin[0]), float(origin[1])
    dest_lon, dest_lat = float(destination[0]), float(destination[1])

    center_lat = (orig_lat + dest_lat) / 2
    center_lon = (orig_lon + dest_lon) / 2
    import math
    dlat = abs(orig_lat - dest_lat)
    dlon = abs(orig_lon - dest_lon)
    dist_deg = math.sqrt(dlat**2 + dlon**2)
    radius_m = max(int(dist_deg * 111320 * 1.5), 2000)

    try:
        G = ox.graph_from_point((center_lat, center_lon), dist=radius_m, network_type=travel_mode)
        orig_node = ox.nearest_nodes(G, X=orig_lon, Y=orig_lat)
        dest_node = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)

        import networkx as nx
        path_length = nx.shortest_path_length(G, orig_node, dest_node, weight="length")

        speeds = {"drive": 50_000 / 60, "walk": 5_000 / 60, "bike": 15_000 / 60}
        speed = speeds.get(travel_mode, 50_000 / 60)
        travel_time_min = path_length / speed

        return {
            "status": "success",
            "text": f"Distance: {path_length:.1f} m ({path_length/1000:.3f} km), travel time: {travel_time_min:.1f} min ({travel_mode})",
            "distance_m": round(path_length, 1),
            "distance_km": round(path_length / 1000, 3),
            "travel_time_minutes": round(travel_time_min, 1),
            "travel_mode": travel_mode,
        }
    except Exception as e:
        return {"status": "error", "message": f"Route computation failed: {e}"}


def get_bbox_from_raster_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Extract a bounding box (+CRS/resolution) from a GeoTIFF, OR the bounding
    box of a GeoPackage vector layer with an optional buffer.

    Two modes:
      - raster: ``input_path`` (or ``geotiff``) → the raster's bbox/metadata.
      - vector: ``gpkg`` + ``layer`` (+ optional ``buffer_m``) → the layer's
        total bounds expanded by buffer_m metres (OpenEarthAgent GetBboxFromGeotiff).
    """
    rasterio = _lazy_rasterio()
    input_path = arguments.get("input_path") or arguments.get("geotiff")
    gpkg = arguments.get("gpkg")
    layer = arguments.get("layer")
    buffer_m = arguments.get("buffer_m")

    # Vector-layer mode: bbox of a gpkg layer (+ buffer). `layer` here is a layer
    # name, not a raster path; only use it when there is no real raster input.
    if gpkg and layer and not (input_path and os.path.exists(str(input_path))):
        gpd = _lazy_gpd()
        if not os.path.exists(gpkg):
            raise FileNotFoundError(f"GeoPackage not found: {gpkg}")
        gdf = gpd.read_file(gpkg, layer=layer)
        if gdf.empty:
            return {"status": "error", "message": f"layer '{layer}' is empty in {gpkg}"}
        if buffer_m:
            # buffer in metres: project to an equal-distance CRS, buffer, reproject
            src_crs = gdf.crs
            metric = gdf.to_crs(3857) if (src_crs and src_crs.to_epsg() != 3857) else gdf
            metric = metric.buffer(float(buffer_m))
            gdf = metric.to_crs(src_crs) if src_crs else metric
        minx, miny, maxx, maxy = gdf.total_bounds
        return {
            "status": "success",
            "gpkg": gpkg, "layer": layer, "buffer_m": buffer_m,
            "crs": gdf.crs.to_string() if getattr(gdf, "crs", None) is not None else None,
            "bbox": {"west": float(minx), "south": float(miny), "east": float(maxx), "north": float(maxy)},
        }

    if not input_path:
        return {"status": "error", "message": "provide a raster 'input_path'/'geotiff', or 'gpkg'+'layer'."}
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raster file not found: {input_path}")

    with rasterio.open(input_path) as src:
        bounds = src.bounds
        crs = src.crs.to_string() if src.crs else None
        transform = list(src.transform)[:6]
        width, height = src.width, src.height
        count = src.count
        dtype = str(src.dtypes[0])
        res = src.res

    return {
        "status": "success",
        "input_path": input_path,
        "bbox": {
            "west": bounds.left,
            "south": bounds.bottom,
            "east": bounds.right,
            "north": bounds.top,
        },
        "crs": crs,
        "resolution": {"x": res[0], "y": res[1]},
        "size": {"width": width, "height": height},
        "bands": count,
        "dtype": dtype,
        "transform": transform,
    }


# ------------------------------------------------------------------------------
# GeoPackage index-layer + visualization tools (ported from OpenEarthAgent)
#
# Faithful re-implementations of OEA's ComputeIndexChange / ShowIndexLayer /
# DisplayOnMap / DisplayOnGeotiff. The originals used QGIS python bindings
# (unavailable / hard to install here); the raster algebra is reproduced exactly
# on GDAL + numpy instead (identical decode formula, thresholds and statistics).
# Index rasters are stored in the gpkg as uint8 1..254 (0 = nodata) encoding an
# index value in [-1, 1] via index = (v - 1) / 254 * 2 - 1.
# ------------------------------------------------------------------------------

_INF = float("inf")
_INDEX_CHANGE_CLASSES = {
    "NDVI": ([-_INF, -0.3, -0.1, 0.1, 0.3, _INF], {
        1: "Severe vegetation loss", 2: "Moderate vegetation loss",
        3: "Stable / no significant change", 4: "Moderate vegetation gain",
        5: "Strong vegetation gain / regrowth"}),
    "NDBI": ([-_INF, -0.3, -0.1, 0.1, 0.3, _INF], {
        1: "Strong urban decrease", 2: "Moderate urban decrease",
        3: "Stable / no significant change", 4: "Moderate urban growth",
        5: "Strong urban growth"}),
    "NBR": ([-_INF, -0.66, -0.27, -0.1, 0.1, _INF], {
        1: "Severe burn severity", 2: "Moderate burn severity",
        3: "Low burn severity", 4: "Unburned", 5: "Enhanced regrowth"}),
}


def _lazy_gdal():
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        return gdal
    except ImportError:
        raise ImportError("Missing GDAL (osgeo). Install: conda install gdal / pip install gdal")


def _lazy_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Missing matplotlib. Install: pip install matplotlib")


def _decode_index_array(arr, dtype_name):
    """Decode a stored index band to physical index values in [-1, 1]."""
    import numpy as np
    arr = arr.astype("float32")
    if dtype_name.startswith("Float"):
        arr[arr <= -100] = np.nan
    else:
        arr[arr <= 0] = np.nan
        arr = ((arr - 1) / 254.0) * 2.0 - 1.0
    return np.clip(arr, -1, 1)


def show_index_layer_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Render a colorized PNG preview of an index raster layer in a GeoPackage."""
    gdal = _lazy_gdal()
    plt = _lazy_plt()
    gpkg = arguments.get("gpkg")
    layer_name = arguments.get("layer_name")
    index_type = str(arguments.get("index_type", "")).upper()
    if not gpkg or not layer_name:
        return {"status": "error", "message": "show_index_layer requires 'gpkg' and 'layer_name'."}
    if not os.path.exists(gpkg):
        return {"status": "error", "message": f"GeoPackage not found: {gpkg}"}
    out_file = arguments.get("out_file") or os.path.join(_get_save_dir(), f"{layer_name}_preview.png")
    ds = gdal.Open(f"GPKG:{gpkg}:{layer_name}")
    if ds is None:
        return {"status": "error", "message": f"Could not open layer '{layer_name}' from {gpkg}"}
    band = ds.GetRasterBand(1)
    arr = _decode_index_array(band.ReadAsArray(), gdal.GetDataTypeName(band.DataType))
    cmap = {"NDVI": "RdYlGn", "NDBI": "RdYlBu_r", "NBR": "RdYlGn"}.get(index_type, "viridis")
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label=f"{index_type} Value")
    plt.title(f"{index_type} Layer: {layer_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    return {
        "status": "success", "out_file": out_file, "layer_name": layer_name,
        "index_type": index_type,
        "message": f"Saved colorized {index_type} preview to: {out_file}",
    }


def compute_index_change_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """ΔIndex = layer2 - layer1; classify change, report class percentages, and
    save the float difference raster back into the GeoPackage."""
    import numpy as np
    gdal = _lazy_gdal()
    gpkg = arguments.get("gpkg")
    index_type = str(arguments.get("index_type", "")).upper()
    l1 = arguments.get("layer1_name")
    l2 = arguments.get("layer2_name")
    diff_layer_name = arguments.get("diff_layer_name") or f"{index_type}_Change"
    if not all([gpkg, index_type, l1, l2]):
        return {"status": "error", "message": "requires gpkg, index_type, layer1_name, layer2_name."}
    if index_type not in _INDEX_CHANGE_CLASSES:
        return {"status": "error", "message": "index_type must be one of NDVI, NDBI, NBR."}
    if not os.path.exists(gpkg):
        return {"status": "error", "message": f"GeoPackage not found: {gpkg}"}
    ds1 = gdal.Open(f"GPKG:{gpkg}:{l1}")
    ds2 = gdal.Open(f"GPKG:{gpkg}:{l2}")
    if ds1 is None or ds2 is None:
        return {"status": "error", "message": "One or both raster layers could not be loaded."}
    b1, b2 = ds1.GetRasterBand(1), ds2.GetRasterBand(1)
    a1 = _decode_index_array(b1.ReadAsArray(), gdal.GetDataTypeName(b1.DataType))
    a2 = _decode_index_array(b2.ReadAsArray(), gdal.GetDataTypeName(b2.DataType))
    diff = a2 - a1
    breaks, class_names = _INDEX_CHANGE_CLASSES[index_type]
    valid = diff[np.isfinite(diff)]
    total = int(valid.size)
    summary = f"{index_type} Change Statistics:\n"
    classes_pct = {}
    for i in range(1, len(breaks)):
        cnt = int(np.sum((valid >= breaks[i - 1]) & (valid < breaks[i])))
        pct = (cnt / total * 100) if total > 0 else 0.0
        classes_pct[class_names[i]] = round(pct, 2)
        summary += f"  {class_names[i]:<45}: {pct:6.2f} %\n"
    # Save the float difference raster back into the GeoPackage (georeferenced from layer1).
    gt, proj = ds1.GetGeoTransform(), ds1.GetProjection()
    h, w = diff.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    out = np.where(np.isfinite(diff), diff, -9999.0).astype("float32")
    mem.GetRasterBand(1).WriteArray(out)
    mem.GetRasterBand(1).SetNoDataValue(-9999.0)
    try:
        gdal.GetDriverByName("GPKG").CreateCopy(
            gpkg, mem, options=["APPEND_SUBDATASET=YES", f"RASTER_TABLE={diff_layer_name}"]
        )
        saved = True
    except Exception as exc:  # raster save is best-effort; stats are the main output
        saved = False
        summary += f"  (note: could not append raster layer to gpkg: {exc})\n"
    finally:
        mem = ds1 = ds2 = None
    gpkg_name = os.path.basename(gpkg)
    return {
        "status": "success", "diff_layer_name": diff_layer_name, "index_type": index_type,
        "class_percentages": classes_pct, "layer_saved": saved, "summary": summary,
        "message": f"delta-{index_type} layer saved to {gpkg_name} as '{diff_layer_name}'\n" + summary,
    }


def display_on_map_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Render GeoPackage vector layer(s) to a PNG map with an optional basemap."""
    import random
    from datetime import datetime as _dt
    gpd = _lazy_gpd()
    plt = _lazy_plt()
    try:
        import contextily as ctx
    except ImportError:
        ctx = None
    from pyogrio import list_layers
    gpkg = arguments.get("gpkg")
    layers = arguments.get("layers")
    if not gpkg or not layers:
        return {"status": "error", "message": "display_on_map requires 'gpkg' and 'layers'."}
    if isinstance(layers, str):
        layers = [layers]
    if not os.path.exists(gpkg):
        return {"status": "error", "message": f"GeoPackage not found: {gpkg}"}
    available = [str(l[0]) for l in list_layers(gpkg)]
    missing = [l for l in layers if l not in available]
    if missing:
        return {"status": "error", "message": f"Missing layers: {missing}. Available: {available}"}
    if "area_boundary" not in available:
        return {"status": "error", "message": '"area_boundary" layer not found in gpkg'}
    random.seed(42)
    area_gdf = gpd.read_file(gpkg, layer="area_boundary")
    if area_gdf.empty:
        return {"status": "error", "message": '"area_boundary" layer is empty'}
    fig, ax = plt.subplots(figsize=(10, 10))
    legend_added: set = set()

    def _lbl(n):
        if n not in legend_added:
            legend_added.add(n)
            return n
        return None

    for layer in layers:
        gdf = gpd.read_file(gpkg, layer=layer)
        if gdf.empty:
            continue
        color = "#" + "".join(random.choices("0123456789ABCDEF", k=6))
        gt = gdf.geometry.geom_type
        lines = gdf[gt.str.contains("Line", na=False)]
        points = gdf[gt.str.contains("Point", na=False)]
        polys = gdf[gt.str.contains("Polygon", na=False)]
        if not lines.empty:
            has = "distance_m" in lines.columns
            lines.plot(ax=ax, column="distance_m" if has else None,
                       cmap="plasma" if has else None, color=None if has else color,
                       linewidth=2, alpha=0.8, label=_lbl(layer))
        if not points.empty:
            points.plot(ax=ax, color=color, markersize=50, alpha=0.9, label=_lbl(layer))
            for _, row in points.iterrows():
                g = row.geometry
                if g is None or g.is_empty:
                    continue
                nm = _get_name_from_row(row)
                if nm:
                    ax.text(g.x, g.y, nm, fontsize=8, color="black", ha="left", va="bottom",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))
        if not polys.empty:
            polys.boundary.plot(ax=ax, linewidth=1.2, edgecolor=color, label=_lbl(layer))
            polys.plot(ax=ax, alpha=0.05, facecolor=color)
            for _, row in polys.iterrows():
                g = row.geometry
                if g is None or g.is_empty:
                    continue
                nm = _get_name_from_row(row)
                if nm:
                    cp = g.representative_point()
                    ax.text(cp.x, cp.y, nm, fontsize=8, color="black", ha="center", va="center",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))
    minx, miny, maxx, maxy = area_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    if ctx is not None:
        try:
            ctx.add_basemap(ax, crs=area_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass  # basemap needs network; the vector map is still valid without it
    ax.legend(loc="upper left")
    ax.set_axis_off()
    out_path = os.path.join(_get_save_dir(), f"map_{_dt.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"status": "success", "out_file": out_path, "layers": layers,
            "message": f"Rendered {len(layers)} layer(s) to map: {out_path}"}


def display_on_geotiff_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Overlay GeoPackage vector layer(s) onto a GeoTIFF and save a georeferenced
    RGB overlay raster (same CRS / resolution)."""
    import numpy as np
    import random
    from datetime import datetime as _dt
    rasterio = _lazy_rasterio()
    gpd = _lazy_gpd()
    from rasterio.features import rasterize
    from scipy.ndimage import binary_dilation
    from PIL import Image, ImageDraw
    from pyogrio import list_layers
    gpkg = arguments.get("gpkg")
    layers = arguments.get("layers")
    geotiff = arguments.get("geotiff") or arguments.get("input_path")
    show_names = bool(arguments.get("show_names", True))
    if not gpkg or not layers or not geotiff:
        return {"status": "error", "message": "display_on_geotiff requires 'gpkg', 'layers', and 'geotiff'."}
    if isinstance(layers, str):
        layers = [layers]
    for p in (gpkg, geotiff):
        if not os.path.exists(p):
            return {"status": "error", "message": f"Not found: {p}"}
    available = [str(l[0]) for l in list_layers(gpkg)]
    missing = [l for l in layers if l not in available]
    if missing:
        return {"status": "error", "message": f"Missing layers: {missing}. Available: {available}"}
    random.seed(42)
    with rasterio.open(geotiff) as src:
        meta = src.meta.copy()
        raster = src.read()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
    if crs is None:
        return {"status": "error", "message": "GeoTIFF has no CRS; cannot align vector layers."}
    base = (raster[:3] if raster.shape[0] >= 3 else np.repeat(raster[0:1], 3, axis=0)).astype("uint8")
    color_mask = np.zeros((3, height, width), dtype="uint8")
    labels = []
    for layer in layers:
        gdf = gpd.read_file(gpkg, layer=layer)
        if gdf.empty:
            continue
        if gdf.crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        hexc = "".join(random.choices("0123456789ABCDEF", k=6))
        r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
        shapes = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                geom = geom.boundary
            shapes.append((geom, 1))
        if shapes:
            burned = rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                               fill=0, all_touched=True, dtype="uint8")
            mask = binary_dilation(burned == 1, structure=np.ones((3, 3), dtype=bool))
            color_mask[0][mask] = r
            color_mask[1][mask] = g
            color_mask[2][mask] = b
        if show_names:
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                nm = _get_name_from_row(row)
                if not nm:
                    continue
                pt = geom if geom.geom_type == "Point" else geom.representative_point()
                col, rowi = ~transform * (pt.x, pt.y)
                labels.append((int(col), int(rowi), nm))
    overlay = np.any(color_mask > 0, axis=0)
    blended = base.copy()
    for c in range(3):
        blended[c][overlay] = color_mask[c][overlay]
    img = Image.fromarray(np.moveaxis(blended, 0, -1)).convert("RGB")
    if show_names and labels:
        draw = ImageDraw.Draw(img)
        for x, y, nm in labels:
            if 0 <= x < width and 0 <= y < height:
                draw.text((x, y), nm, fill=(255, 255, 0))
    blended = np.moveaxis(np.array(img), -1, 0)
    meta.update({"count": 3, "dtype": "uint8", "height": height, "width": width,
                 "transform": transform, "crs": crs, "driver": "GTiff"})
    out_path = os.path.join(_get_save_dir(), f"overlay_{_dt.now().strftime('%Y%m%d_%H%M%S')}.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(blended)
    return {"status": "success", "out_file": out_path, "layers": layers,
            "message": f"Saved overlay GeoTIFF: {out_path}"}


def add_index_layer_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """Compute a spectral index (NDVI/NDBI/NBR) and save it as a uint8 layer in a GeoPackage.

    Local raster substitute for OpenEarthAgent's GEE-backed AddIndexLayer: instead
    of fetching Sentinel-2 by year/month from Earth Engine, it computes the index
    from two band rasters (band_a_path/band_b_path for the chosen index) and stores
    it with the same uint8 1..254 encoding the other gpkg index tools
    (show_index_layer / compute_index_change) expect. The free public-STAC data
    layer remains available separately via the stac_basic toolkit.
    Band order per index: NDVI = NIR/Red, NDBI = SWIR1/NIR, NBR = NIR/SWIR2.
    """
    import numpy as np
    rasterio = _lazy_rasterio()
    gdal = _lazy_gdal()
    gpkg = arguments.get("gpkg")
    index_type = str(arguments.get("index_type", "")).upper()
    layer_name = arguments.get("layer_name")
    band_a_path = arguments.get("band_a_path") or arguments.get("nir_path")
    band_b_path = (arguments.get("band_b_path") or arguments.get("red_path")
                   or arguments.get("swir_path"))
    if not all([gpkg, index_type, layer_name]):
        return {"status": "error", "message": "add_index_layer requires gpkg, index_type, layer_name."}
    if index_type not in _INDEX_CHANGE_CLASSES:
        return {"status": "error", "message": "index_type must be one of NDVI, NDBI, NBR."}
    if not (band_a_path and band_b_path):
        return {"status": "error", "message": (
            "add_index_layer needs band_a_path and band_b_path (local substitute for the "
            "GEE/STAC fetch). Provide the two Sentinel-2 bands for the index: "
            "NDVI=NIR/Red, NDBI=SWIR1/NIR, NBR=NIR/SWIR2.")}
    if not os.path.exists(gpkg):
        return {"status": "error", "message": f"GeoPackage not found: {gpkg}"}
    with rasterio.open(band_a_path) as sa:
        a = sa.read(1).astype("float32")
        gt, crs = sa.transform, sa.crs
    with rasterio.open(band_b_path) as sb:
        b = sb.read(1).astype("float32")
    if a.shape != b.shape:
        return {"status": "error", "message": "band_a and band_b rasters differ in shape."}
    idx = np.clip((a - b) / (a + b + 1e-6), -1, 1)
    # encode to uint8 1..254 (0 = nodata) — the gpkg index convention shared by
    # show_index_layer / compute_index_change.
    enc = np.where(np.isfinite(idx), np.round(((idx + 1) / 2.0) * 254 + 1), 0).astype("uint8")
    h, w = enc.shape
    geotransform = (gt.c, gt.a, gt.b, gt.f, gt.d, gt.e)
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(geotransform)
    if crs is not None:
        mem.SetProjection(crs.to_wkt())
    mem.GetRasterBand(1).WriteArray(enc)
    try:
        gdal.GetDriverByName("GPKG").CreateCopy(
            gpkg, mem, options=["APPEND_SUBDATASET=YES", f"RASTER_TABLE={layer_name}"]
        )
    finally:
        mem = None
    valid = idx[np.isfinite(idx)]
    stats = {
        "mean": round(float(np.mean(valid)), 4) if valid.size else None,
        "min": round(float(np.min(valid)), 4) if valid.size else None,
        "max": round(float(np.max(valid)), 4) if valid.size else None,
    }
    return {
        "status": "success", "layer_name": layer_name, "index_type": index_type, "stats": stats,
        "message": f"{index_type} index layer '{layer_name}' saved to {os.path.basename(gpkg)} (mean={stats['mean']}).",
    }


# ------------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    """Register all OSM GIS tools."""
    registrar.toolkit(
        name="osm_gis",
        description=(
            "Geospatial data acquisition from OpenStreetMap: area boundaries (saved as GeoPackage), "
            "POI layer queries (fire stations, hospitals, schools, etc.), "
            "pairwise road-network distance computation between POI layers, "
            "raster metadata extraction, spectral-index layer change/preview (NDVI/NDBI/NBR), "
            "and map/GeoTIFF visualization of GeoPackage layers."
        ),
        version="2.1.0"
    )

    # 1. Get area boundary → creates GeoPackage
    registrar.tool(
        ToolSpec(
            slug="osm_gis.get_area_boundary",
            name="Get Area Boundary",
            description=(
                "Fetch the geographic boundary of a place from OpenStreetMap and save as a GeoPackage. "
                "Returns a gpkg file path (not raw GeoJSON). Downstream tools (add_pois_layer, "
                "compute_route_dist) use this gpkg automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Place name (e.g., 'Banff National Park, Alberta, Canada') or bbox as string '(west,south,east,north)'."
                    },
                    "buffer_m": {
                        "type": "number",
                        "description": "Optional buffer distance in meters around the boundary."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional: explicit output path for the GeoPackage file."
                    },
                },
                "required": ["area"]
            },
            requires_connection=False
        ),
        get_area_boundary_handler,
    )

    # 2. Add POIs layer → appends layer to existing GeoPackage
    registrar.tool(
        ToolSpec(
            slug="osm_gis.add_pois_layer",
            name="Add POIs Layer",
            description=(
                "Query Points of Interest (POIs) from OpenStreetMap within the area boundary "
                "stored in a GeoPackage. Appends results as a new named layer. "
                "This tool can be called repeatedly with different queries to create multiple "
                "entity layers before downstream comparison. Common queries: "
                "{\"amenity\": \"restaurant\"}, {\"leisure\": \"park\"}, "
                "{\"amenity\": \"fire_station\"}, {\"amenity\": \"police\"}, "
                "{\"amenity\": \"hospital\"}, {\"amenity\": \"school\"}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {
                        "type": "string",
                        "description": "Path to the GeoPackage (auto-injected from get_area_boundary)."
                    },
                    "query": {
                        "description": "OSM tags as dict (e.g. {\"amenity\": \"fire_station\"}) or POI name as string."
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Layer name for saving POIs in the GeoPackage (e.g. 'fire_stations')."
                    },
                    "area": {"type": "string", "description": "Optional area name (the boundary is normally taken from the gpkg)."},
                    "count": {"type": "integer", "description": "Optional cap on the number of POIs to keep."},
                },
                "required": ["gpkg", "query", "layer_name"]
            },
            requires_connection=False
        ),
        add_pois_layer_handler,
    )

    # 3. Compute route distance → reads two layers, computes distances
    registrar.tool(
        ToolSpec(
            slug="osm_gis.compute_route_dist",
            name="Compute Road Network Distance",
            description=(
                "Compute pairwise road-network distances between features of two POI layers "
                "in a GeoPackage. Saves distance line layer back to the gpkg. "
                "Use 'top' parameter to keep only k nearest per source feature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {
                        "type": "string",
                        "description": "Path to the GeoPackage."
                    },
                    "src_layer": {
                        "type": "string",
                        "description": "Source POI layer name in the GeoPackage."
                    },
                    "tar_layer": {
                        "type": "string",
                        "description": "Target POI layer name in the GeoPackage."
                    },
                    "top": {
                        "type": "integer",
                        "description": "Optional: keep k nearest targets per source (default=all)."
                    },
                },
                "required": ["gpkg", "src_layer", "tar_layer"]
            },
            requires_connection=False
        ),
        compute_route_dist_handler,
    )

    # 4. Get bbox from raster (standalone, unchanged)
    registrar.tool(
        ToolSpec(
            slug="osm_gis.get_bbox_from_raster",
            name="Get Bounding Box from Raster",
            description=(
                "Get a bounding box either from a GeoTIFF (its raster bbox + CRS/resolution "
                "via 'input_path'/'geotiff'), or from a GeoPackage vector layer expanded by a "
                "buffer (via 'gpkg' + 'layer' + optional 'buffer_m' metres)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to a GeoTIFF raster file (raster mode)."},
                    "geotiff": {"type": "string", "description": "Path to a GeoTIFF (alias of input_path)."},
                    "gpkg": {"type": "string", "description": "Path to a GeoPackage (vector-layer mode)."},
                    "layer": {"type": "string", "description": "Vector layer name inside the GeoPackage to bound."},
                    "buffer_m": {"type": "number", "description": "Optional buffer in metres applied around the layer."}
                },
                "required": []
            },
            requires_connection=False
        ),
        get_bbox_from_raster_handler,
    )

    # 5. ShowIndexLayer → colorized PNG preview of a gpkg index raster
    registrar.tool(
        ToolSpec(
            slug="osm_gis.show_index_layer",
            name="Show Index Layer",
            description=(
                "Generate a colorized PNG preview of a spectral-index layer (NDVI/NDBI/NBR) "
                "stored in a GeoPackage. Use after add_index_layer has created the index layer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                    "layer_name": {"type": "string", "description": "Raster layer name inside the GeoPackage."},
                    "index_type": {"type": "string", "enum": ["NDVI", "NDBI", "NBR"], "description": "Index type (chooses default colormap)."},
                    "out_file": {"type": "string", "description": "Optional output image path (.png)."},
                },
                "required": ["gpkg", "layer_name", "index_type"]
            },
            requires_connection=False
        ),
        show_index_layer_handler,
    )

    # 6. ComputeIndexChange → ΔIndex between two gpkg index layers
    registrar.tool(
        ToolSpec(
            slug="osm_gis.compute_index_change",
            name="Compute Index Change",
            description=(
                "Compute ΔIndex (layer2 − layer1) for NDVI/NDBI/NBR from two GeoPackage index "
                "raster layers, classify the change into 5 severity classes (e.g. vegetation "
                "loss/gain, urban decrease/growth, burn severity/regrowth), report per-class "
                "area percentages, and save the difference layer back into the GeoPackage. "
                "Call add_index_layer first to create the two index layers being compared."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                    "index_type": {"type": "string", "enum": ["NDVI", "NDBI", "NBR"], "description": "Index type."},
                    "layer1_name": {"type": "string", "description": "Baseline raster layer name."},
                    "layer2_name": {"type": "string", "description": "Comparison raster layer name."},
                    "diff_layer_name": {"type": "string", "description": "Optional output layer name (defaults to '<index_type>_Change')."},
                },
                "required": ["gpkg", "index_type", "layer1_name", "layer2_name"]
            },
            requires_connection=False
        ),
        compute_index_change_handler,
    )

    # 7. DisplayOnMap → render gpkg vector layers to a PNG map
    registrar.tool(
        ToolSpec(
            slug="osm_gis.display_on_map",
            name="Display On Map",
            description=(
                "Render selected GeoPackage vector layers (boundary, POIs, distance lines) "
                "on a PNG map over a web basemap, with feature-name labels. The GeoPackage "
                "must contain an 'area_boundary' layer (from get_area_boundary)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {"type": "string", "description": "Path to the GeoPackage (must contain 'area_boundary')."},
                    "layers": {"description": "Layer name (string) or list of layer names to render."},
                },
                "required": ["gpkg", "layers"]
            },
            requires_connection=False
        ),
        display_on_map_handler,
    )

    # 8. DisplayOnGeotiff → overlay gpkg vector layers onto a GeoTIFF
    registrar.tool(
        ToolSpec(
            slug="osm_gis.display_on_geotiff",
            name="Display On GeoTIFF",
            description=(
                "Render one or more GeoPackage vector layers (with feature names) directly "
                "over a given GeoTIFF, saving a georeferenced RGB overlay raster with the same "
                "CRS and resolution as the input GeoTIFF."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                    "layers": {"description": "Layer name (string) or list of layer names to overlay."},
                    "geotiff": {"type": "string", "description": "Path to the base GeoTIFF."},
                    "show_names": {"type": "boolean", "description": "Draw feature name labels (default true)."},
                },
                "required": ["gpkg", "layers", "geotiff"]
            },
            requires_connection=False
        ),
        display_on_geotiff_handler,
    )

    # 9. AddIndexLayer (local raster substitute for GEE; STAC fetch via stac_basic)
    registrar.tool(
        ToolSpec(
            slug="osm_gis.add_index_layer",
            name="Add Index Layer",
            description=(
                "Compute a spectral index (NDVI/NDBI/NBR) over the area in a GeoPackage for "
                "a given year (and optional month) and save it as a layer, which can then be "
                "previewed (show_index_layer) or differenced (compute_index_change). "
                "Imagery for {year, month} is fetched from public Sentinel-2 archives "
                "(stac_basic); alternatively, pass two local Sentinel-2 bands directly via "
                "band_a_path/band_b_path (NDVI=NIR/Red, NDBI=SWIR1/NIR, NBR=NIR/SWIR2)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpkg": {"type": "string", "description": "Path to the GeoPackage (from get_area_boundary)."},
                    "index_type": {"type": "string", "enum": ["NDVI", "NDBI", "NBR"], "description": "Spectral index to compute."},
                    "layer_name": {"type": "string", "description": "Output raster layer name to save into the GeoPackage."},
                    "year": {"type": "integer", "description": "Year to composite imagery for (e.g. 2022)."},
                    "month": {"type": "integer", "minimum": 1, "maximum": 12, "description": "Optional month (1-12); whole year if omitted."},
                    "band_a_path": {"type": "string", "description": "Optional first band raster (NDVI:NIR, NDBI:SWIR1, NBR:NIR)."},
                    "band_b_path": {"type": "string", "description": "Optional second band raster (NDVI:Red, NDBI:NIR, NBR:SWIR2)."},
                },
                "required": ["gpkg", "index_type", "layer_name"]
            },
            requires_connection=False
        ),
        add_index_layer_handler,
    )
