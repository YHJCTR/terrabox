"""
OSM GIS Toolkit
---------------
Geographic data acquisition from OpenStreetMap (OSM) and raster utilities.
Inspired by OpenEarthAgent's GIS tools (GetAreaBoundary, AddPoisLayer, ComputeDistance).

Key features:
1. Fetch geographic boundaries from place names or bounding boxes.
2. Query Points of Interest (POIs) from OSM.
3. Compute road-network distances and estimated travel times.
4. Extract bounding box info from GeoTIFF files.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional
from ..core.registry import ToolSpec

logger = logging.getLogger(__name__)


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


def _lazy_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("Missing rasterio. Install: pip install rasterio")


# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------

def get_area_boundary_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Fetch the geographic boundary of a place from OpenStreetMap by place name
    or a bounding box. Returns GeoJSON FeatureCollection.
    """
    ox = _lazy_osmnx()

    place_name = arguments.get("place_name")
    bbox = arguments.get("bbox")  # [west, south, east, north]
    output_path = arguments.get("output_path")

    if place_name:
        try:
            gdf = ox.geocode_to_gdf(place_name)
        except Exception as e:
            return {"status": "error", "message": f"Failed to geocode '{place_name}': {e}"}
    elif bbox:
        try:
            import geopandas as gpd
            from shapely.geometry import box as shapely_box
            west, south, east, north = [float(v) for v in bbox]
            geom = shapely_box(west, south, east, north)
            gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
        except Exception as e:
            return {"status": "error", "message": f"Failed to create bbox geometry: {e}"}
    else:
        raise ValueError("Provide either 'place_name' or 'bbox'")

    geojson = json.loads(gdf.to_crs("EPSG:4326").to_json())

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)
        return {"status": "success", "output_path": output_path, "feature_count": len(geojson.get("features", []))}

    return {
        "status": "success",
        "geojson": geojson,
        "feature_count": len(geojson.get("features", [])),
    }


def add_pois_layer_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Query Points of Interest (POI) from OpenStreetMap within a place or bounding box.
    Returns GeoJSON FeatureCollection of matching POIs.

    Common amenity tags: school, hospital, restaurant, bank, fuel, pharmacy, hotel, etc.
    Common tags for building types: residential, commercial, industrial.
    """
    ox = _lazy_osmnx()

    place_name = arguments.get("place_name")
    bbox = arguments.get("bbox")  # [west, south, east, north]
    tags = arguments.get("tags", {"amenity": True})
    output_path = arguments.get("output_path")
    max_results = int(arguments.get("max_results", 500))

    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = {"amenity": tags}

    try:
        if place_name:
            gdf = ox.features_from_place(place_name, tags=tags)
        elif bbox:
            west, south, east, north = [float(v) for v in bbox]
            gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
        else:
            raise ValueError("Provide either 'place_name' or 'bbox'")
    except Exception as e:
        return {"status": "error", "message": f"OSM query failed: {e}"}

    if len(gdf) > max_results:
        gdf = gdf.head(max_results)

    # Keep only Point geometries for cleaner output
    points_gdf = gdf[gdf.geometry.geom_type == "Point"].copy() if len(gdf) > 0 else gdf
    geojson = json.loads(points_gdf.to_crs("EPSG:4326")[["geometry", "name"] if "name" in points_gdf.columns else ["geometry"]].to_json())

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)
        return {"status": "success", "output_path": output_path, "poi_count": len(points_gdf)}

    return {
        "status": "success",
        "geojson": geojson,
        "poi_count": len(points_gdf),
        "total_features": len(gdf),
    }


def compute_route_dist_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute road-network distance and estimated travel time between two geographic points.
    Uses OpenStreetMap road network via osmnx + networkx.

    Returns shortest path distance in meters and estimated travel time in minutes.
    """
    ox = _lazy_osmnx()

    origin = arguments.get("origin")       # [lon, lat]
    destination = arguments.get("destination")  # [lon, lat]
    travel_mode = arguments.get("travel_mode", "drive")  # drive / walk / bike

    if not origin or not destination:
        raise ValueError("Provide 'origin' and 'destination' as [lon, lat] pairs")

    orig_lon, orig_lat = float(origin[0]), float(origin[1])
    dest_lon, dest_lat = float(destination[0]), float(destination[1])

    # Download local graph around the route
    center_lat = (orig_lat + dest_lat) / 2
    center_lon = (orig_lon + dest_lon) / 2
    # Estimate needed radius (rough haversine)
    import math
    dlat = abs(orig_lat - dest_lat)
    dlon = abs(orig_lon - dest_lon)
    dist_deg = math.sqrt(dlat**2 + dlon**2)
    radius_m = max(int(dist_deg * 111320 * 1.5), 2000)  # at least 2km, 1.5x buffer

    try:
        G = ox.graph_from_point((center_lat, center_lon), dist=radius_m, network_type=travel_mode)
        orig_node = ox.nearest_nodes(G, X=orig_lon, Y=orig_lat)
        dest_node = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)

        import networkx as nx
        path_length = nx.shortest_path_length(G, orig_node, dest_node, weight="length")

        # Estimate travel time: drive=50km/h, walk=5km/h, bike=15km/h
        speeds = {"drive": 50_000 / 60, "walk": 5_000 / 60, "bike": 15_000 / 60}
        speed = speeds.get(travel_mode, 50_000 / 60)
        travel_time_min = path_length / speed

        return {
            "status": "success",
            "distance_m": round(path_length, 1),
            "distance_km": round(path_length / 1000, 3),
            "travel_time_minutes": round(travel_time_min, 1),
            "travel_mode": travel_mode,
            "origin": [orig_lon, orig_lat],
            "destination": [dest_lon, dest_lat],
        }
    except Exception as e:
        return {"status": "error", "message": f"Route computation failed: {e}"}


def get_bbox_from_raster_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Extract bounding box, CRS, resolution and basic metadata from a GeoTIFF file.
    """
    rasterio = _lazy_rasterio()
    input_path = arguments["input_path"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raster file not found: {input_path}")

    with rasterio.open(input_path) as src:
        bounds = src.bounds
        crs = src.crs.to_string() if src.crs else None
        transform = list(src.transform)[:6]
        width, height = src.width, src.height
        count = src.count
        dtype = str(src.dtypes[0])
        res = src.res  # (x_res, y_res) in CRS units

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
# Registration
# ------------------------------------------------------------------------------

def setup(registrar):
    """Register all OSM GIS tools."""
    registrar.toolkit(
        name="osm_gis",
        description="Geospatial data acquisition from OpenStreetMap: area boundaries, POI layers, road-network distances, and raster metadata extraction.",
        version="1.0.0"
    )

    # 1. Get area boundary
    registrar.tool(
        ToolSpec(
            slug="osm_gis.get_area_boundary",
            name="Get Area Boundary",
            description="Fetch the geographic boundary of a place from OpenStreetMap by place name or bounding box. Returns GeoJSON polygon.",
            parameters={
                "type": "object",
                "properties": {
                    "place_name": {
                        "type": "string",
                        "description": "Place name to geocode and fetch boundary for (e.g., 'Beijing, China', 'Central Park, New York')."
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Bounding box as [west, south, east, north] in WGS84 degrees. Used if place_name is not provided."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save the result as a GeoJSON file."
                    }
                },
                "required": []
            },
            requires_connection=False
        ),
        get_area_boundary_handler,
    )

    # 2. Add POIs layer
    registrar.tool(
        ToolSpec(
            slug="osm_gis.add_pois_layer",
            name="Add POIs Layer",
            description="Query Points of Interest (POIs) from OpenStreetMap within a place or bounding box. Supports amenities like schools, hospitals, restaurants, banks, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "place_name": {
                        "type": "string",
                        "description": "Place name to search POIs within (e.g., 'Nanjing, China')."
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Bounding box [west, south, east, north] in WGS84. Used if place_name is not provided."
                    },
                    "tags": {
                        "description": "OSM tags to filter POIs. Can be a dict like {\"amenity\": \"school\"} or {\"amenity\": true} for all amenities. Defaults to all amenities.",
                        "default": {"amenity": True}
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 500,
                        "description": "Maximum number of POIs to return."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save the result as a GeoJSON file."
                    }
                },
                "required": []
            },
            requires_connection=False
        ),
        add_pois_layer_handler,
    )

    # 3. Compute route distance
    registrar.tool(
        ToolSpec(
            slug="osm_gis.compute_route_dist",
            name="Compute Road Network Distance",
            description="Compute shortest road-network distance (meters) and estimated travel time (minutes) between two geographic points using OpenStreetMap.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Origin point as [longitude, latitude] in WGS84."
                    },
                    "destination": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Destination point as [longitude, latitude] in WGS84."
                    },
                    "travel_mode": {
                        "type": "string",
                        "enum": ["drive", "walk", "bike"],
                        "default": "drive",
                        "description": "Transportation mode: 'drive' (50km/h), 'walk' (5km/h), or 'bike' (15km/h)."
                    }
                },
                "required": ["origin", "destination"]
            },
            requires_connection=False
        ),
        compute_route_dist_handler,
    )

    # 4. Get bbox from raster
    registrar.tool(
        ToolSpec(
            slug="osm_gis.get_bbox_from_raster",
            name="Get Bounding Box from Raster",
            description="Extract bounding box, CRS, spatial resolution, and basic metadata from a GeoTIFF file.",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the GeoTIFF raster file."
                    }
                },
                "required": ["input_path"]
            },
            requires_connection=False
        ),
        get_bbox_from_raster_handler,
    )
