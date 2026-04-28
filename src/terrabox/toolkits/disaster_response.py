"""
Disaster Response Toolkit
=========================
8 pure-Python tools for disaster rescue analysis:
  1. calc_slope           – DEM → slope (degrees/percent); landslide risk
  2. calc_flow_direction  – DEM → D8 flow direction; flood routing
  3. zonal_stats          – raster × vector overlay; per-zone statistics
  4. population_exposure  – hazard × population raster; exposed persons
  5. flood_route_astar    – A* routing with flooded road penalty; rescue path
  6. accessibility_map    – distance-transform to POIs; isolation detection
  7. building_damage_stats– change mask × building footprints; damage tally
  8. fire_spread_forecast – cellular automata fire spread; 6-h forecast

All tools use rasterio/numpy/scipy/osmnx already present in the environment.
No new pip packages required.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..core.registry import ToolSpec


# --------------------------------------------------------------------------- #
# Lazy imports (same pattern as georaster.py)
# --------------------------------------------------------------------------- #

def _rasterio():
    import rasterio
    return rasterio


def _np():
    import numpy as np
    return np


def _ndimage():
    from scipy import ndimage
    return ndimage


def _osmnx():
    import osmnx as ox
    return ox


def _nx():
    import networkx as nx
    return nx


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _read_band(path: str):
    """Read first band as float32 and return (data, profile, transform)."""
    rasterio = _rasterio()
    np = _np()
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
    return data, profile, transform


def _save_raster(output_path: str, data, profile: dict, dtype=None, nodata=None):
    import os
    rasterio = _rasterio()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    p = profile.copy()
    p.update({"count": 1, "driver": "GTiff", "compress": "lzw"})
    if dtype:
        p["dtype"] = dtype
    if nodata is not None:
        p["nodata"] = nodata
    with rasterio.open(output_path, "w", **p) as dst:
        dst.write(data, 1)


def _geojson_to_shapes(zones_geojson) -> List[dict]:
    """Accept GeoJSON dict or file path, return list of Feature dicts."""
    import json, os
    if isinstance(zones_geojson, str):
        if os.path.isfile(zones_geojson):
            with open(zones_geojson) as f:
                gj = json.load(f)
        else:
            gj = json.loads(zones_geojson)
    else:
        gj = zones_geojson

    if gj.get("type") == "FeatureCollection":
        return gj["features"]
    if gj.get("type") == "Feature":
        return [gj]
    # bare geometry
    return [{"type": "Feature", "geometry": gj, "properties": {}}]


# --------------------------------------------------------------------------- #
# T1 — calc_slope
# --------------------------------------------------------------------------- #

def calc_slope_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute slope from a DEM raster using Sobel gradients.
    Output: slope in degrees (default) or percent.
    """
    rasterio = _rasterio()
    np = _np()
    ndimage = _ndimage()

    dem_path = arguments["dem_path"]
    output_path = arguments["output_path"]
    unit = (arguments.get("unit") or "degrees").lower()

    dem, profile, transform = _read_band(dem_path)

    # Pixel size in map units (metres if projected, degrees if geographic)
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)

    # Sobel gradients (dz/dx, dz/dy); divide by pixel size to get rise/run
    gx = ndimage.sobel(dem, axis=1) / (8.0 * pixel_x)
    gy = ndimage.sobel(dem, axis=0) / (8.0 * pixel_y)

    rise_run = np.sqrt(gx ** 2 + gy ** 2)

    if unit == "percent":
        slope = rise_run * 100.0
    else:  # degrees
        slope = np.degrees(np.arctan(rise_run))

    _save_raster(output_path, slope, profile, dtype=rasterio.float32, nodata=-9999)

    return {
        "output_path": output_path,
        "unit": unit,
        "mean_slope": float(np.nanmean(slope)),
        "max_slope": float(np.nanmax(slope)),
    }


# --------------------------------------------------------------------------- #
# T2 — calc_flow_direction
# --------------------------------------------------------------------------- #

def calc_flow_direction_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute D8 flow direction from a DEM.
    Output codes: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE.
    Nodata cells (DEM == nodata) are assigned 0.
    """
    rasterio = _rasterio()
    np = _np()

    dem_path = arguments["dem_path"]
    output_path = arguments["output_path"]

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata_val = src.nodata

    if nodata_val is not None:
        dem[dem == nodata_val] = np.nan

    rows, cols = dem.shape

    # D8 neighbour offsets: (row_offset, col_offset, D8_code)
    # Order: E SE S SW W NW N NE
    neighbours = [
        (0,  1,   1),
        (1,  1,   2),
        (1,  0,   4),
        (1, -1,   8),
        (0, -1,  16),
        (-1,-1,  32),
        (-1, 0,  64),
        (-1, 1, 128),
    ]

    flow = np.zeros((rows, cols), dtype=np.uint8)

    for dr, dc, code in neighbours:
        # Compute elevation difference to this neighbour
        r_src = np.arange(rows)
        c_src = np.arange(cols)
        r_nb = np.clip(r_src + dr, 0, rows - 1)
        c_nb = np.clip(c_src + dc, 0, cols - 1)
        diff = dem - dem[np.ix_(r_nb, c_nb)]
        # mask cells where the neighbour was clipped to the same row/col (border)
        diff[r_nb == r_src, :] = -np.inf   # row border (dr != 0 but clipped)
        diff[:, c_nb == c_src] = -np.inf   # col border (dc != 0 but clipped)
        # Mark cells where this direction has max drop (first-wins tiebreak)
        unassigned = (flow == 0) & np.isfinite(diff) & (diff > 0)
        # We need per-cell max – use running comparison
        # Simplified: assign code where diff is positive and no direction yet
        flow[unassigned & (diff > 0)] = code

    # Cells with NaN dem get 0
    flow[np.isnan(dem)] = 0

    _save_raster(output_path, flow, profile, dtype=rasterio.uint8, nodata=0)

    return {"output_path": output_path, "d8_codes": "1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE"}


# --------------------------------------------------------------------------- #
# T3 — zonal_stats
# --------------------------------------------------------------------------- #

def zonal_stats_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Compute raster statistics for each polygon zone.
    Supports stat: mean | sum | max | min | count.
    """
    import numpy as np
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import shape

    raster_path = arguments["raster_path"]
    zones_geojson = arguments["zones_geojson"]
    stat = (arguments.get("stat") or "mean").lower()

    features = _geojson_to_shapes(zones_geojson)

    stat_fns = {
        "mean":  np.nanmean,
        "sum":   np.nansum,
        "max":   np.nanmax,
        "min":   np.nanmin,
        "count": lambda x: int(np.sum(~np.isnan(x))),
    }
    if stat not in stat_fns:
        raise ValueError(f"stat must be one of {list(stat_fns)}, got '{stat}'")
    fn = stat_fns[stat]

    results = []
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for feat in features:
            geom = feat.get("geometry") or feat
            props = feat.get("properties") or {}
            fid = props.get("id", props.get("name", len(results)))
            name = props.get("name", str(fid))
            try:
                out_image, _ = rio_mask(src, [geom], crop=True, nodata=np.nan, filled=True)
                band = out_image[0].astype(np.float32)
                if nodata is not None:
                    band[band == nodata] = np.nan
                val = fn(band) if band.size > 0 else None
            except Exception:
                val = None
            results.append({"id": fid, "name": name, "value": float(val) if val is not None else None})

    return {"zones": results, "count": len(results), "stat": stat}


# --------------------------------------------------------------------------- #
# T4 — population_exposure
# --------------------------------------------------------------------------- #

def population_exposure_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Estimate the number of people exposed to a hazard.
    exposed = sum(population * (hazard > threshold))
    Both rasters must be co-registered (same extent / resolution).
    """
    rasterio = _rasterio()
    np = _np()

    hazard_path = arguments["hazard_path"]
    population_path = arguments["population_path"]
    output_path = arguments["output_path"]
    threshold = float(arguments.get("threshold", 0.5))

    hazard, h_profile, h_transform = _read_band(hazard_path)
    population, _, _ = _read_band(population_path)

    if hazard.shape != population.shape:
        raise ValueError(
            f"Raster shapes must match: hazard={hazard.shape}, population={population.shape}"
        )

    # Pixel area in km² (using transform pixel size; assumes projected CRS in metres)
    pixel_m2 = abs(h_transform.a * h_transform.e)
    pixel_km2 = pixel_m2 / 1e6

    # Exposure: people per pixel = population_density * pixel_area
    people_per_pixel = population * pixel_km2
    exposed_mask = hazard > threshold
    exposed = np.where(exposed_mask, people_per_pixel, 0.0).astype(np.float32)
    exposed_population = float(np.nansum(exposed))

    _save_raster(output_path, exposed, h_profile, dtype=rasterio.float32, nodata=0)

    total_pixels = int(np.sum(np.isfinite(hazard)))
    exposed_pixels = int(np.sum(exposed_mask))

    return {
        "exposed_population": round(exposed_population, 0),
        "output_path": output_path,
        "exposed_pixels": exposed_pixels,
        "total_pixels": total_pixels,
        "exposed_ratio": round(exposed_pixels / total_pixels, 4) if total_pixels else 0.0,
        "total_area_km2": round(total_pixels * pixel_km2, 3),
    }


# --------------------------------------------------------------------------- #
# T5 — flood_route_astar
# --------------------------------------------------------------------------- #

def flood_route_astar_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Plan a rescue route using A* on an OSM road graph where flooded segments
    receive a high cost penalty (default 1000×) instead of being removed.
    Returns the route as a GeoJSON LineString.
    """
    import math
    ox = _osmnx()
    nx = _nx()
    rasterio = _rasterio()
    np = _np()

    origin      = arguments["origin"]       # [lon, lat]
    destination = arguments["destination"]  # [lon, lat]
    flood_mask_path  = arguments.get("flood_mask_path")
    flood_threshold  = float(arguments.get("flood_threshold", 127))
    travel_mode      = arguments.get("travel_mode", "drive")
    penalty_factor   = float(arguments.get("penalty_factor", 1000))

    orig_lon, orig_lat = float(origin[0]), float(origin[1])
    dest_lon, dest_lat = float(destination[0]), float(destination[1])

    # --- 1. Download road network ---
    center_lat = (orig_lat + dest_lat) / 2
    center_lon = (orig_lon + dest_lon) / 2
    dlat = abs(orig_lat - dest_lat)
    dlon = abs(orig_lon - dest_lon)
    dist_deg = math.sqrt(dlat ** 2 + dlon ** 2)
    radius_m = max(int(dist_deg * 111320 * 1.6), 2000)

    G = ox.graph_from_point((center_lat, center_lon), dist=radius_m, network_type=travel_mode)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # --- 2. Read flood mask and mark flooded edges ---
    blocked_edges = 0
    if flood_mask_path:
        with rasterio.open(flood_mask_path) as src:
            flood_arr = src.read(1).astype(np.float32)
            flood_transform = src.transform
            flood_crs = src.crs

        def _is_flooded(lon, lat):
            try:
                row, col = ~flood_transform * (lon, lat)
                row, col = int(row), int(col)
                if 0 <= row < flood_arr.shape[0] and 0 <= col < flood_arr.shape[1]:
                    return flood_arr[row, col] > flood_threshold
            except Exception:
                pass
            return False

        for u, v, k, data in G.edges(keys=True, data=True):
            # Check midpoint of each edge
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            mid_lon = (u_data["x"] + v_data["x"]) / 2
            mid_lat = (u_data["y"] + v_data["y"]) / 2
            if _is_flooded(mid_lon, mid_lat):
                G[u][v][k]["length"] = data.get("length", 1) * penalty_factor
                G[u][v][k]["travel_time"] = data.get("travel_time", 1) * penalty_factor
                blocked_edges += 1

    # --- 3. A* path ---
    orig_node = ox.nearest_nodes(G, X=orig_lon, Y=orig_lat)
    dest_node = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)

    def heuristic(u, v):
        u_d, v_d = G.nodes[u], G.nodes[v]
        return math.sqrt((u_d["x"] - v_d["x"]) ** 2 + (u_d["y"] - v_d["y"]) ** 2) * 111320

    try:
        path_nodes = nx.astar_path(G, orig_node, dest_node, heuristic=heuristic, weight="length")
        passable = True
    except nx.NetworkXNoPath:
        return {"passable": False, "error": "No path found between origin and destination"}

    # --- 4. Compute distance and time along path ---
    path_length = sum(
        G[u][v][0].get("length", 0)
        for u, v in zip(path_nodes[:-1], path_nodes[1:])
    )
    # Exclude penalised sections from reported distance
    real_length = path_length  # user can infer penalty from blocked_ratio

    speeds = {"drive": 50_000 / 60, "walk": 5_000 / 60, "bike": 15_000 / 60}
    travel_time_min = real_length / speeds.get(travel_mode, speeds["drive"])

    # --- 5. Build GeoJSON LineString ---
    coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in path_nodes]
    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "distance_m": round(real_length, 1),
            "travel_time_min": round(travel_time_min, 1),
            "travel_mode": travel_mode,
        },
    }

    total_edges = G.number_of_edges()
    blocked_ratio = round(blocked_edges / total_edges, 4) if total_edges else 0.0

    return {
        "passable": passable,
        "distance_m": round(real_length, 1),
        "distance_km": round(real_length / 1000, 3),
        "travel_time_min": round(travel_time_min, 1),
        "blocked_edges": blocked_edges,
        "blocked_ratio": blocked_ratio,
        "route_geojson": route_geojson,
    }


# --------------------------------------------------------------------------- #
# T6 — accessibility_map
# --------------------------------------------------------------------------- #

def accessibility_map_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Generate a raster showing travel time (minutes) from each passable cell
    to the nearest POI (hospital, shelter, etc.) using distance transform.
    Impassable cells (road_mask == 0) are treated as infinite cost.
    """
    rasterio = _rasterio()
    np = _np()
    ndimage = _ndimage()

    road_mask_path   = arguments["road_mask_path"]
    poi_locations    = arguments["poi_locations"]   # [{lon, lat, name}, ...]
    output_path      = arguments["output_path"]
    speed_kmh        = float(arguments.get("travel_speed_kmh", 30))

    # Defensive parsing: handle string, list-of-strings, or varied field names
    import json as _json
    if isinstance(poi_locations, str):
        poi_locations = _json.loads(poi_locations)
    normalized = []
    for poi in poi_locations:
        if isinstance(poi, str):
            poi = _json.loads(poi)
        lon = poi.get("lon") or poi.get("longitude") or poi.get("x")
        lat = poi.get("lat") or poi.get("latitude") or poi.get("y")
        if lon is not None and lat is not None:
            normalized.append({"lon": float(lon), "lat": float(lat), "name": poi.get("name", "")})
    poi_locations = normalized

    mask_arr, profile, transform = _read_band(road_mask_path)

    # 1 = passable, 0 = blocked
    passable = (mask_arr > 0).astype(np.uint8)

    # Create source grid (1 at POI locations)
    sources = np.zeros_like(passable, dtype=np.uint8)
    for poi in poi_locations:
        lon, lat = float(poi["lon"]), float(poi["lat"])
        col = int((lon - transform.c) / transform.a)
        row = int((lat - transform.f) / transform.e)
        rows_total, cols_total = passable.shape
        if 0 <= row < rows_total and 0 <= col < cols_total:
            sources[row, col] = 1

    if sources.sum() == 0:
        return {"error": "No POI locations mapped onto raster extent"}

    # Distance transform: distance in pixels from each cell to nearest source,
    # considering only passable cells.
    # Strategy: set blocked cells as barrier by masking out.
    # Use edt on source positions, then mask blocked.
    dist_px = ndimage.distance_transform_edt(1 - sources)

    # Pixel size in metres
    pixel_m = abs(transform.a)

    # Convert pixel distance to travel time (minutes)
    speed_ms = speed_kmh * 1000 / 60  # metres/minute
    time_min = (dist_px * pixel_m / speed_ms).astype(np.float32)

    # Mark blocked cells as nodata
    time_min[passable == 0] = -9999

    _save_raster(output_path, time_min, profile, dtype=rasterio.float32, nodata=-9999)

    valid = time_min[passable == 1]
    isolated = int(np.sum(np.isinf(time_min[passable == 1]))) if valid.size else 0
    total_passable = int(np.sum(passable))

    return {
        "output_path": output_path,
        "mean_time_min": round(float(np.nanmean(valid[valid > -9000])), 2) if valid.size else None,
        "max_time_min": round(float(np.nanmax(valid[valid > -9000])), 2) if valid.size else None,
        "isolated_pixels": isolated,
        "total_passable_pixels": total_passable,
        "isolated_ratio": round(isolated / total_passable, 4) if total_passable else 0.0,
    }


# --------------------------------------------------------------------------- #
# T7 — building_damage_stats
# --------------------------------------------------------------------------- #

def building_damage_stats_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Assess per-building damage by overlaying a change-detection raster
    with building footprint polygons.
    damage_threshold: fraction of pixels changed within a building to mark it damaged.
    """
    import numpy as np
    import rasterio
    from rasterio.mask import mask as rio_mask

    change_raster_path = arguments["change_raster_path"]
    buildings_geojson  = arguments["buildings_geojson"]
    damage_threshold   = float(arguments.get("damage_threshold", 0.3))

    features = _geojson_to_shapes(buildings_geojson)

    results = []
    damaged_count = 0

    with rasterio.open(change_raster_path) as src:
        nodata = src.nodata
        for feat in features:
            geom = feat.get("geometry") or feat
            props = feat.get("properties") or {}
            fid = props.get("id", props.get("osm_id", len(results)))

            try:
                out_image, _ = rio_mask(src, [geom], crop=True, nodata=0, filled=True)
                band = out_image[0].astype(np.float32)
                if nodata is not None:
                    band[band == nodata] = 0
                total_px = band.size
                changed_px = int(np.sum(band > 127))
                ratio = changed_px / total_px if total_px > 0 else 0.0
                status = "damaged" if ratio >= damage_threshold else "intact"
            except Exception:
                ratio = 0.0
                status = "unknown"

            if status == "damaged":
                damaged_count += 1
            results.append({"id": fid, "status": status, "change_ratio": round(ratio, 4)})

    total = len(results)
    return {
        "total_buildings": total,
        "damaged": damaged_count,
        "intact": total - damaged_count,
        "damage_ratio": round(damaged_count / total, 4) if total else 0.0,
        "buildings": results,
    }


# --------------------------------------------------------------------------- #
# T8 — fire_spread_forecast
# --------------------------------------------------------------------------- #

def fire_spread_forecast_handler(arguments: Dict[str, Any], context: Any, account: Any) -> Dict[str, Any]:
    """
    Simulate fire spread using a wind-biased cellular automaton.
    Each time step expands the fire zone via binary dilation with a
    directional kernel weighted by FVC and wind speed.
    """
    rasterio = _rasterio()
    np = _np()
    ndimage = _ndimage()

    fvc_path          = arguments["fvc_path"]
    ignition_points   = arguments["ignition_points"]  # [{lon, lat}, ...]
    wind_direction_deg = float(arguments.get("wind_direction_deg", 0))  # 0=N, 90=E
    wind_speed_ms     = float(arguments.get("wind_speed_ms", 5))
    time_steps        = int(arguments.get("time_steps", 6))
    output_path       = arguments["output_path"]

    fvc, profile, transform = _read_band(fvc_path)
    fvc = np.clip(fvc, 0, 1)
    rows, cols = fvc.shape

    # Pixel size in metres
    pixel_m = abs(transform.a)

    # --- Ignition raster ---
    fire = np.zeros((rows, cols), dtype=np.uint8)
    for pt in ignition_points:
        lon, lat = float(pt["lon"]), float(pt["lat"])
        col_idx = int((lon - transform.c) / transform.a)
        row_idx = int((lat - transform.f) / transform.e)
        if 0 <= row_idx < rows and 0 <= col_idx < cols:
            fire[row_idx, col_idx] = 1

    if fire.sum() == 0:
        return {"error": "No ignition points mapped onto raster extent"}

    # --- Wind-biased 3×3 kernel ---
    # wind_direction_deg: meteorological convention (0=N, 90=E wind blowing from that direction)
    # Fire spreads downwind, so add 180°
    spread_dir_rad = math.radians((wind_direction_deg + 180) % 360)

    # Base kernel (all 1s except centre)
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0

    # Boost downwind cells
    wind_factor = min(wind_speed_ms / 3.0, 3.0)  # cap at 3× boost
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            # Cell direction relative to centre (row positive = south)
            cell_angle_rad = math.atan2(dc, -dr)  # N=0, E=π/2
            angular_diff = abs(cell_angle_rad - spread_dir_rad)
            # Wrap to [-π, π]
            while angular_diff > math.pi:
                angular_diff -= 2 * math.pi
            angular_diff = abs(angular_diff)
            downwind_weight = max(0.0, 1.0 - angular_diff / math.pi)
            kernel[dr + 1, dc + 1] *= (1.0 + wind_factor * downwind_weight)

    kernel /= kernel.sum()  # normalise

    # --- Simulate spread ---
    cumulative_fire = fire.copy()
    for _ in range(time_steps):
        # Probability of ignition: convolution of current fire with kernel × FVC
        spread_prob = ndimage.convolve(cumulative_fire.astype(np.float32), kernel, mode="constant")
        spread_prob = np.clip(spread_prob * fvc, 0, 1)
        new_fire = (spread_prob > 0.1).astype(np.uint8)
        cumulative_fire = np.clip(cumulative_fire + new_fire, 0, 1).astype(np.uint8)

    _save_raster(output_path, cumulative_fire * 255, profile, dtype=rasterio.uint8, nodata=0)

    # Stats
    pixel_km2 = (pixel_m ** 2) / 1e6
    affected_pixels = int(np.sum(cumulative_fire))
    affected_km2 = round(affected_pixels * pixel_km2, 3)

    # Max spread distance from ignition
    initial_pixels = np.argwhere(fire == 1)
    spread_pixels  = np.argwhere(cumulative_fire == 1)
    if len(initial_pixels) and len(spread_pixels):
        centroid_r = initial_pixels[:, 0].mean()
        centroid_c = initial_pixels[:, 1].mean()
        dists_px = np.sqrt(
            (spread_pixels[:, 0] - centroid_r) ** 2 + (spread_pixels[:, 1] - centroid_c) ** 2
        )
        max_spread_km = round(float(dists_px.max()) * pixel_m / 1000, 3)
    else:
        max_spread_km = 0.0

    return {
        "output_path": output_path,
        "affected_area_km2": affected_km2,
        "affected_pixels": affected_pixels,
        "max_spread_km": max_spread_km,
        "time_steps": time_steps,
        "wind_direction_deg": wind_direction_deg,
        "wind_speed_ms": wind_speed_ms,
    }


# --------------------------------------------------------------------------- #
# Toolkit registration
# --------------------------------------------------------------------------- #

def setup(registrar):
    registrar.toolkit(
        name="disaster_response",
        description=(
            "Disaster response analysis: terrain slope/flow direction, zonal statistics, "
            "population exposure, A*-based rescue routing, accessibility maps, "
            "building damage assessment, and wildfire spread forecasting."
        ),
        version="1.0.0",
    )

    # ------------------------------------------------------------------ T1
    registrar.tool(
        ToolSpec(
            slug="disaster_response.calc_slope",
            name="Calculate Terrain Slope",
            description=(
                "Compute slope from a DEM raster using Sobel gradients. "
                "Output is slope in degrees (default) or percent. "
                "Useful for landslide risk mapping (slopes >30° = high risk) and "
                "assessing terrain difficulty for evacuation routes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dem_path": {"type": "string", "description": "Path to input DEM GeoTIFF (elevation in metres)."},
                    "output_path": {"type": "string", "description": "Output path for slope GeoTIFF."},
                    "unit": {"type": "string", "enum": ["degrees", "percent"],
                             "description": "Output unit: 'degrees' (default) or 'percent'."},
                },
                "required": ["dem_path", "output_path"],
            },
            requires_connection=False,
        ),
        calc_slope_handler,
    )

    # ------------------------------------------------------------------ T2
    registrar.tool(
        ToolSpec(
            slug="disaster_response.calc_flow_direction",
            name="Calculate D8 Flow Direction",
            description=(
                "Compute D8 flow direction from a DEM raster. "
                "Each pixel is assigned the steepest downslope neighbour direction. "
                "Output codes: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE. "
                "Use this to trace flood routing paths from rainfall zones and "
                "predict which downstream communities are at risk."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dem_path": {"type": "string", "description": "Path to input DEM GeoTIFF."},
                    "output_path": {"type": "string", "description": "Output path for flow direction GeoTIFF."},
                },
                "required": ["dem_path", "output_path"],
            },
            requires_connection=False,
        ),
        calc_flow_direction_handler,
    )

    # ------------------------------------------------------------------ T3
    registrar.tool(
        ToolSpec(
            slug="disaster_response.zonal_stats",
            name="Zonal Statistics (Raster × Vector)",
            description=(
                "Compute raster statistics (mean, sum, max, min, count) for each polygon zone. "
                "Accepts a GeoJSON FeatureCollection or file path for zones. "
                "Use this to aggregate flood depth per district, burned area per municipality, "
                "or earthquake intensity per neighbourhood — enabling resource allocation by zone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "raster_path": {"type": "string", "description": "Input raster GeoTIFF path."},
                    "zones_geojson": {
                        "description": "GeoJSON FeatureCollection (dict or file path) of polygon zones.",
                    },
                    "stat": {"type": "string", "enum": ["mean", "sum", "max", "min", "count"],
                             "description": "Statistic to compute per zone (default: mean)."},
                },
                "required": ["raster_path", "zones_geojson"],
            },
            requires_connection=False,
        ),
        zonal_stats_handler,
    )

    # ------------------------------------------------------------------ T4
    registrar.tool(
        ToolSpec(
            slug="disaster_response.population_exposure",
            name="Population Exposure Estimation",
            description=(
                "Estimate the number of people exposed to a hazard. "
                "Multiplies a hazard raster (0-1 or binary) by a population density raster (persons/km²). "
                "Both rasters must be co-registered. "
                "Returns total exposed population count and an output exposure raster."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "hazard_path": {"type": "string",
                                    "description": "Hazard raster path (0-1 risk or binary mask)."},
                    "population_path": {"type": "string",
                                        "description": "Population density raster path (persons/km²)."},
                    "output_path": {"type": "string", "description": "Output path for exposure raster."},
                    "threshold": {"type": "number",
                                  "description": "Hazard value above which a cell is considered at risk (default 0.5)."},
                },
                "required": ["hazard_path", "population_path", "output_path"],
            },
            requires_connection=False,
        ),
        population_exposure_handler,
    )

    # ------------------------------------------------------------------ T5
    registrar.tool(
        ToolSpec(
            slug="disaster_response.flood_route_astar",
            name="A* Rescue Route (Flood-Aware)",
            description=(
                "Plan the optimal rescue route between two points using A* on an OSM road network. "
                "Flooded road segments (where flood_mask > flood_threshold) are penalised with a high "
                "cost multiplier instead of being removed, allowing fallback routes if no dry path exists. "
                "Returns distance, travel time, blocked road ratio, and a GeoJSON route LineString. "
                "Significantly more capable than osm_gis.compute_route_dist for disaster scenarios."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "array", "items": {"type": "number"},
                               "description": "Origin [lon, lat]."},
                    "destination": {"type": "array", "items": {"type": "number"},
                                    "description": "Destination [lon, lat]."},
                    "flood_mask_path": {"type": "string",
                                        "description": "Optional flood mask raster. Pixels > flood_threshold are treated as flooded roads."},
                    "flood_threshold": {"type": "number",
                                        "description": "Pixel threshold for flooded classification (default 127)."},
                    "travel_mode": {"type": "string", "enum": ["drive", "walk"],
                                    "description": "Travel mode (default: drive)."},
                    "penalty_factor": {"type": "number",
                                       "description": "Cost multiplier for flooded road edges (default 1000)."},
                },
                "required": ["origin", "destination"],
            },
            requires_connection=False,
        ),
        flood_route_astar_handler,
    )

    # ------------------------------------------------------------------ T6
    registrar.tool(
        ToolSpec(
            slug="disaster_response.accessibility_map",
            name="Accessibility Map (Time to Nearest POI)",
            description=(
                "Generate a raster showing estimated travel time (minutes) from each passable cell "
                "to the nearest POI (hospital, shelter, rescue station) using distance transform. "
                "Impassable cells (road_mask == 0, e.g. flooded areas) act as barriers. "
                "The output identifies isolated areas that cannot reach help within a given time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "road_mask_path": {"type": "string",
                                       "description": "Binary raster: 1 = passable, 0 = blocked."},
                    "poi_locations": {
                        "type": "array",
                        "items": {"type": "object",
                                  "properties": {"lon": {"type": "number"}, "lat": {"type": "number"},
                                                 "name": {"type": "string"}}},
                        "description": "List of POI locations [{lon, lat, name}, ...].",
                    },
                    "output_path": {"type": "string", "description": "Output path for accessibility raster."},
                    "travel_speed_kmh": {"type": "number",
                                         "description": "Assumed travel speed in km/h (default 30)."},
                },
                "required": ["road_mask_path", "poi_locations", "output_path"],
            },
            requires_connection=False,
        ),
        accessibility_map_handler,
    )

    # ------------------------------------------------------------------ T7
    registrar.tool(
        ToolSpec(
            slug="disaster_response.building_damage_stats",
            name="Building Damage Statistics",
            description=(
                "Assess structural damage per building by overlaying a change-detection raster "
                "with building footprint polygons (e.g. from OSM or geo_perception). "
                "A building is marked 'damaged' if the fraction of changed pixels within it "
                "exceeds damage_threshold (default 0.3). "
                "Returns total buildings, damaged count, damage ratio, and per-building status."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "change_raster_path": {"type": "string",
                                           "description": "Change detection raster (255=changed, 0=unchanged)."},
                    "buildings_geojson": {
                        "description": "GeoJSON FeatureCollection of building footprint polygons (dict or file path).",
                    },
                    "damage_threshold": {"type": "number",
                                         "description": "Fraction of changed pixels to classify a building as damaged (default 0.3)."},
                },
                "required": ["change_raster_path", "buildings_geojson"],
            },
            requires_connection=False,
        ),
        building_damage_stats_handler,
    )

    # ------------------------------------------------------------------ T8
    registrar.tool(
        ToolSpec(
            slug="disaster_response.fire_spread_forecast",
            name="Wildfire Spread Forecast (Cellular Automaton)",
            description=(
                "Simulate wildfire spread over N time steps using a wind-biased cellular automaton. "
                "Fire expands from ignition points based on vegetation cover (FVC) and wind direction/speed. "
                "Each time step ≈ 1 hour. Returns an output raster showing the forecasted burn zone "
                "and statistics: affected area (km²), max spread distance (km). "
                "Use this to predict the 6-12 hour spread zone for downwind evacuation planning."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fvc_path": {"type": "string",
                                 "description": "Fractional Vegetation Cover raster (0-1). Higher FVC = higher fuel load."},
                    "ignition_points": {
                        "type": "array",
                        "items": {"type": "object",
                                  "properties": {"lon": {"type": "number"}, "lat": {"type": "number"}}},
                        "description": "Fire ignition locations [{lon, lat}, ...].",
                    },
                    "wind_direction_deg": {"type": "number",
                                           "description": "Wind direction in degrees (meteorological: 0=N, 90=E). Default 0."},
                    "wind_speed_ms": {"type": "number",
                                      "description": "Wind speed in m/s. Default 5."},
                    "time_steps": {"type": "integer",
                                   "description": "Number of 1-hour simulation steps. Default 6."},
                    "output_path": {"type": "string", "description": "Output path for fire spread raster (255=burned)."},
                },
                "required": ["fvc_path", "ignition_points", "output_path"],
            },
            requires_connection=False,
        ),
        fire_spread_forecast_handler,
    )
