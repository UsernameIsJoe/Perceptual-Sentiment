#!/usr/bin/env python3
"""
Pick two map points (A and B), find drivable-road intersections spread across the
A–B region (grid-like selection), and optionally download Street View at each corner.

Uses OSRM for a reference driving route, OpenStreetMap Overpass for intersection
nodes, and Google Street View Static API for images (when enabled). Outputs a manifest:
  - nodes: corner id, lat/lng, osm_node_id, street_names, grid_cell
  - route_polyline: reference OSRM route (not the corner path)
  - area_bbox: search rectangle used for intersection discovery

Independent utility (no pipeline imports).

Setup:
  1. Enable on Google Cloud: "Maps JavaScript API" (+ "Street View Static API" when fetching).
  2. Set API_KEY below or export GOOGLE_MAPS_API_KEY.
  3. python fetch_streetview_corners.py
  4. Open the URL, click point A then point B, set max corners, find intersections.
  5. python plot_corners_map.py --open  to verify spread on a map.

Notes:
  - Corners are car-drivable road intersections (residential, unclassified, service,
    primary/secondary/tertiary, etc.). Footways, cycleways, paths, and park trails excluded.
  - FETCH_STREETVIEW=False skips image download while refining intersection spread.
  - Internet required (OSRM, Overpass, Google Maps UI).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

# ============================================================================
# EDIT HERE
# ----------------------------------------------------------------------------
# Leave blank and set GOOGLE_MAPS_API_KEY, or pass --api-key. Do not commit a key.
API_KEY = ""
# Set False while refining intersection selection (no Street View downloads).
FETCH_STREETVIEW = True

MAX_CORNERS = 100
IMAGE_SIZE = "640x480"
FOV = 90
# Search area = bbox between A and B, expanded by this margin (m). Expands until
# enough spread-out intersections are found (not linear along one route).
AREA_MARGIN_START_M = 80.0
AREA_MARGIN_STEP_M = 60.0
AREA_MARGIN_MAX_M = 350.0
# Grid cells for spatial spread (sqrt(N) x sqrt(N) over the A-B box).
GRID_SPREAD = True
# Merge intersection nodes closer than this (same physical corner).
MERGE_CORNERS_M = 25.0
# ============================================================================

STREETVIEW_IMG = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_META = "https://maps.googleapis.com/maps/api/streetview/metadata"
OSRM_ROUTE = "https://router.project-osrm.org/route/v1/driving"
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACKS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
OVERPASS_HEADERS = {
    "Content-Type": "text/plain; charset=utf-8",
    "Accept": "application/json",
    "User-Agent": "0416-study-streetview-corners/1.0",
}

# OSM highway types excluded — pedestrian paths, park trails, cycle-only, etc.
_NON_DRIVABLE_HIGHWAY = frozenset({
    "footway", "pedestrian", "cycleway", "path", "steps", "bridleway",
    "corridor", "elevator", "escape", "platform", "proposed", "construction",
    "bus_guideway", "raceway", "via_ferrata", "milepost", "traffic_sign",
    "street_lamp", "crossing", "give_way", "stop", "turning_circle",
    "turning_loop", "mini_roundabout", "passing_place", "bus_stop",
})
# Node must be shared by at least this many distinct drivable-road ways.
MIN_ROAD_WAYS_AT_NODE = 2

EARTH_R = 6371000.0


# ----------------------------- geometry ------------------------------------

def _haversine(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lat2 = math.radians(p1[0]), math.radians(p2[0])
    dlon = math.radians(p2[1] - p1[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _bbox_between(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    margin_m: float,
) -> tuple[float, float, float, float]:
    """Axis-aligned bbox covering A, B, plus margin on all sides."""
    lats = [point_a[0], point_b[0]]
    lngs = [point_a[1], point_b[1]]
    mid_lat = sum(lats) / 2.0
    dlat = margin_m / 111_320.0
    dlng = margin_m / (111_320.0 * math.cos(math.radians(mid_lat)))
    return (
        min(lats) - dlat, min(lngs) - dlng,
        max(lats) + dlat, max(lngs) + dlng,
    )


def _in_bbox(lat: float, lng: float, bbox: tuple[float, float, float, float]) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lng <= east


def _point_seg_dist(
    pt: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Return (distance_m, fraction_along_segment 0..1) from a toward b."""
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    px, py = pt[1], pt[0]
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 0:
        return _haversine(pt, a), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
    proj = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return _haversine(pt, proj), t


def _route_metrics(
    pt: tuple[float, float],
    route: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return (distance_m to route, distance_along_route_m from start)."""
    if len(route) < 2:
        return _haversine(pt, route[0]), 0.0
    best_d, best_along = float("inf"), 0.0
    cum = 0.0
    for i in range(len(route) - 1):
        seg_len = _haversine(route[i], route[i + 1])
        d, t = _point_seg_dist(pt, route[i], route[i + 1])
        along = cum + t * seg_len
        if d < best_d:
            best_d, best_along = d, along
        cum += seg_len
    return best_d, best_along


# ----------------------------- route + corners -----------------------------

def _osrm_route(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]]:
    url = f"{OSRM_ROUTE}/{a[1]},{a[0]};{b[1]},{b[0]}"
    r = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM routing failed: {data.get('message', data.get('code'))}")
    coords = data["routes"][0]["geometry"]["coordinates"]
    return [(lat, lng) for lng, lat in coords]


# Node roles that are not street corners even if they sit on a main road.
_SKIP_NODE_HIGHWAY = frozenset({
    "crossing", "traffic_signals", "stop", "give_way", "motorway_junction",
    "turning_circle", "turning_loop", "mini_roundabout", "passing_place",
    "elevator", "platform", "bus_stop",
})


def _is_drivable_highway(tags: dict) -> bool:
    """True if OSM tags describe a way cars can generally drive on."""
    hw = tags.get("highway")
    if not hw or hw in _NON_DRIVABLE_HIGHWAY:
        return False
    for key in ("motor_vehicle", "vehicle", "access"):
        val = (tags.get(key) or "").split(";")[0].strip().lower()
        if val == "no":
            return False
    if hw == "track":
        mv = (tags.get("motor_vehicle") or tags.get("vehicle") or "").lower()
        return mv in {"yes", "permissive", "destination", "designated"}
    return True


def _overpass_intersections(bbox: tuple[float, float, float, float]) -> list[dict]:
    south, west, north, east = bbox
    # Fetch all highway ways; filter in Python so intersections require 2+ drivable
    # roads (not road × footway/cycleway/path).
    query = f"""[out:json][timeout:90];
way["highway"]["area"!="yes"]({south},{west},{north},{east});
out body;
>;
out skel qt;
"""
    body = query.encode("utf-8")
    last_err: Exception | None = None
    elements: list[dict] = []
    for url in OVERPASS_FALLBACKS:
        try:
            r = requests.post(url, data=body, headers=OVERPASS_HEADERS, timeout=120)
            r.raise_for_status()
            elements = r.json().get("elements", [])
            break
        except requests.RequestException as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_err}") from last_err

    drivable_ways: dict[int, dict] = {}
    nodes: dict[int, dict] = {}
    for el in elements:
        if el.get("type") == "way":
            tags = el.get("tags") or {}
            if not _is_drivable_highway(tags):
                continue
            hw = tags["highway"]
            names = [tags["name"]] if tags.get("name") else []
            drivable_ways[el["id"]] = {
                "highway": hw,
                "nodes": el.get("nodes") or [],
                "street_names": names,
            }
        elif el.get("type") == "node":
            nodes[el["id"]] = {
                "lat": el["lat"],
                "lng": el["lon"],
                "tags": el.get("tags") or {},
            }

    node_to_ways: dict[int, set[int]] = {}
    for wid, way in drivable_ways.items():
        for nid in way["nodes"]:
            node_to_ways.setdefault(nid, set()).add(wid)

    out = []
    for nid, wids in node_to_ways.items():
        if len(wids) < MIN_ROAD_WAYS_AT_NODE:
            continue
        node = nodes.get(nid)
        if not node:
            continue
        ntags = node["tags"]
        if ntags.get("highway") in _SKIP_NODE_HIGHWAY:
            continue
        if ntags.get("footway") or ntags.get("bicycle") == "designated":
            continue
        street_names: list[str] = []
        road_types: list[str] = []
        for wid in wids:
            w = drivable_ways[wid]
            road_types.append(w["highway"])
            for nm in w["street_names"]:
                if nm not in street_names:
                    street_names.append(nm)
        out.append({
            "osm_node_id": nid,
            "lat": node["lat"],
            "lng": node["lng"],
            "street_names": street_names,
            "main_way_count": len(wids),
            "main_highways": sorted(set(road_types)),
        })
    return out


def _merge_nearby(corners: list[dict], merge_m: float) -> list[dict]:
    if not corners:
        return corners
    kept: list[dict] = []
    for c in corners:
        if any(_haversine((c["lat"], c["lng"]), (k["lat"], k["lng"])) < merge_m for k in kept):
            continue
        kept.append(c)
    return kept


def _candidates_in_area(
    raw: list[dict],
    bbox: tuple[float, float, float, float],
    merge_m: float = MERGE_CORNERS_M,
) -> list[dict]:
    candidates = [dict(c) for c in raw if _in_bbox(c["lat"], c["lng"], bbox)]
    return _merge_nearby(candidates, merge_m)


def _corner_score(c: dict) -> tuple:
    return (c.get("main_way_count", 0), len(c.get("street_names", [])))


def _select_spread_corners(
    candidates: list[dict],
    max_corners: int,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> list[dict]:
    """Pick intersections spread across the A-B region (grid-like), not along one route."""
    if not candidates or max_corners <= 0:
        return []
    if len(candidates) <= max_corners:
        return candidates

    south, west, north, east = _bbox_between(point_a, point_b, 0.0)
    # Use actual candidate extent if wider than A-B line alone.
    south = min(south, min(c["lat"] for c in candidates))
    north = max(north, max(c["lat"] for c in candidates))
    west = min(west, min(c["lng"] for c in candidates))
    east = max(east, max(c["lng"] for c in candidates))

    if not GRID_SPREAD:
        pool = list(candidates)
        selected: list[dict] = []
        while len(selected) < max_corners and pool:
            if not selected:
                nxt = min(pool, key=lambda c: _haversine(point_a, (c["lat"], c["lng"])))
            else:
                nxt = max(pool, key=lambda c: min(
                    _haversine((c["lat"], c["lng"]), (s["lat"], s["lng"])) for s in selected
                ))
            selected.append(dict(nxt))
            pool = [c for c in pool if c["osm_node_id"] != nxt["osm_node_id"]]
        return selected[:max_corners]

    n_side = max(2, int(math.ceil(math.sqrt(max_corners))))
    lat_span = max(north - south, 1e-9)
    lng_span = max(east - west, 1e-9)

    by_cell: dict[tuple[int, int], list[dict]] = {}
    for c in candidates:
        row = min(n_side - 1, int((c["lat"] - south) / lat_span * n_side))
        col = min(n_side - 1, int((c["lng"] - west) / lng_span * n_side))
        by_cell.setdefault((row, col), []).append(c)

    selected: list[dict] = []
    seen_osm: set[int] = set()
    for _cell, items in sorted(by_cell.items()):
        best = max(items, key=_corner_score)
        if best["osm_node_id"] not in seen_osm:
            seen_osm.add(best["osm_node_id"])
            item = dict(best)
            item["grid_cell"] = list(_cell)
            selected.append(item)

    if len(selected) >= max_corners:
        return selected[:max_corners]

    # Farthest-point sampling to fill remaining slots (maximize spread).
    pool = [c for c in candidates if c["osm_node_id"] not in seen_osm]
    if not selected and pool:
        first = min(pool, key=lambda c: _haversine(point_a, (c["lat"], c["lng"])))
        selected.append(dict(first))
        seen_osm.add(first["osm_node_id"])
        pool = [c for c in pool if c["osm_node_id"] != first["osm_node_id"]]

    while len(selected) < max_corners and pool:
        def _min_dist(c: dict) -> float:
            p = (c["lat"], c["lng"])
            return min(_haversine(p, (s["lat"], s["lng"])) for s in selected)

        nxt = max(pool, key=_min_dist)
        selected.append(dict(nxt))
        seen_osm.add(nxt["osm_node_id"])
        pool = [c for c in pool if c["osm_node_id"] != nxt["osm_node_id"]]

    return selected[:max_corners]


def find_corners_between(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    max_corners: int,
    *,
    merge_m: float = MERGE_CORNERS_M,
) -> tuple[list[dict], list[tuple[float, float]], float, list[dict]]:
    """Return selected spread intersections, reference route, final area margin, all in area."""
    route = _osrm_route(point_a, point_b)
    raw = _overpass_intersections(_bbox_between(point_a, point_b, AREA_MARGIN_MAX_M))

    margin_m = AREA_MARGIN_START_M
    margins_tried: list[float] = []
    selected: list[dict] = []
    pool: list[dict] = []

    while margin_m <= AREA_MARGIN_MAX_M:
        margins_tried.append(margin_m)
        bbox = _bbox_between(point_a, point_b, margin_m)
        pool = _candidates_in_area(raw, bbox, merge_m)
        selected = _select_spread_corners(pool, max_corners, point_a, point_b)
        if len(selected) >= max_corners:
            break
        next_margin = margin_m + AREA_MARGIN_STEP_M
        if next_margin > AREA_MARGIN_MAX_M:
            if margin_m < AREA_MARGIN_MAX_M:
                margin_m = AREA_MARGIN_MAX_M
                continue
            break
        margin_m = next_margin

    return selected, route, margin_m, pool


def _build_edges(nodes: list[dict]) -> list[dict]:
    edges = []
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        dist = _haversine((a["lat"], a["lng"]), (b["lat"], b["lng"]))
        streets = sorted(set(a.get("street_names", []) + b.get("street_names", [])))
        edges.append({
            "from": a["id"],
            "to": b["id"],
            "distance_m": round(dist, 1),
            "street": streets[0] if streets else None,
            "streets": streets,
        })
    return edges


# ----------------------------- collect intersections -------------------------

def download_corners(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    max_corners: int,
    out_dir: str,
    api_key: str,
    *,
    size: str = IMAGE_SIZE,
    fov: int = FOV,
    pitch: int = 0,
    source: str = "outdoor",
) -> dict:
    selected, route, margin_m, pool = find_corners_between(point_a, point_b, max_corners)
    os.makedirs(out_dir, exist_ok=True)

    saved_nodes: list[dict] = []
    skipped: list[dict] = []
    seen_panos: set[str] = set()

    for i, c in enumerate(selected, start=1):
        lat, lng = c["lat"], c["lng"]
        node = {
            "id": f"corner_{i:03d}",
            "index": i,
            "lat": lat,
            "lng": lng,
            "osm_node_id": c["osm_node_id"],
            "street_names": c.get("street_names", []),
            "main_highways": c.get("main_highways", []),
            "main_way_count": c.get("main_way_count"),
            "grid_cell": c.get("grid_cell"),
        }

        if FETCH_STREETVIEW:
            # Street View fetch (disabled while refining intersection spread).
            idx = i - 1
            if idx < len(selected) - 1:
                nxt = selected[idx + 1]
                heading = _bearing((lat, lng), (nxt["lat"], nxt["lng"]))
            elif idx > 0:
                prev = selected[idx - 1]
                heading = _bearing((prev["lat"], prev["lng"]), (lat, lng))
            else:
                heading = _bearing(point_a, point_b)

            meta = requests.get(
                STREETVIEW_META,
                params={"location": f"{lat},{lng}", "key": api_key, "source": source},
                timeout=30,
            ).json()
            if meta.get("status") != "OK":
                skipped.append({
                    "osm_node_id": c["osm_node_id"], "lat": lat, "lng": lng,
                    "reason": meta.get("status", "UNKNOWN"),
                })
                continue

            pano_id = meta.get("pano_id")
            if pano_id and pano_id in seen_panos:
                skipped.append({
                    "osm_node_id": c["osm_node_id"], "lat": lat, "lng": lng,
                    "reason": "duplicate_pano", "pano_id": pano_id,
                })
                continue

            r = requests.get(
                STREETVIEW_IMG,
                params={
                    "size": size,
                    "location": f"{lat},{lng}",
                    "heading": f"{heading:.1f}",
                    "pitch": str(pitch),
                    "fov": str(fov),
                    "source": source,
                    "key": api_key,
                },
                timeout=60,
            )
            if r.status_code != 200 or not r.content:
                skipped.append({
                    "osm_node_id": c["osm_node_id"], "lat": lat, "lng": lng,
                    "reason": f"http_{r.status_code}",
                })
                continue

            fname = f"frame_{i:04d}.jpg"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(r.content)
            if pano_id:
                seen_panos.add(pano_id)
            node["heading"] = round(heading, 1)
            node["pano_id"] = pano_id
            node["image"] = fname

        saved_nodes.append(node)

    south, west, north, east = _bbox_between(point_a, point_b, margin_m)
    route_coords = [[lat, lng] for lat, lng in route]
    manifest = {
        "out_dir": os.path.abspath(out_dir),
        "mode": "streetview" if FETCH_STREETVIEW else "intersections_only",
        "selection": "grid_spread" if GRID_SPREAD else "farthest_point",
        "point_a": {"lat": point_a[0], "lng": point_a[1]},
        "point_b": {"lat": point_b[0], "lng": point_b[1]},
        "requested_max_corners": max_corners,
        "intersections_in_area": len(pool),
        "intersections_selected": len(selected),
        "saved": len(saved_nodes),
        "skipped": len(skipped),
        "skipped_duplicate_pano": sum(1 for s in skipped if s.get("reason") == "duplicate_pano"),
        "area_margin_start_m": AREA_MARGIN_START_M,
        "area_margin_step_m": AREA_MARGIN_STEP_M,
        "area_margin_max_m": AREA_MARGIN_MAX_M,
        "area_margin_final_m": round(margin_m, 1),
        "area_bbox": {"south": south, "west": west, "north": north, "east": east},
        "fetch_streetview": FETCH_STREETVIEW,
        "fov": fov,
        "pitch": pitch,
        "size": size,
        "route_polyline": route_coords,
        "nodes": saved_nodes,
        "edges": [],
        "skipped_detail": skipped,
    }
    manifest_path = os.path.join(out_dir, "corners_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ----------------------------- web UI --------------------------------------

PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Street View Corner Collector</title>
  <style>
    body { margin:0; font-family: system-ui, sans-serif; }
    #map { height: 70vh; width: 100%; }
    #panel { padding: 12px 16px; display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
    #panel label { font-size: 13px; }
    input[type=number], input[type=text] { padding:5px 7px; }
    button { padding:8px 16px; font-size:14px; cursor:pointer; }
    #status { font-size:13px; color:#333; white-space:pre-wrap; padding:0 16px 16px; }
    .hint { color:#666; font-size:12px; max-width:520px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="panel">
    <span class="hint">Click <b>point A</b> (start), then <b>point B</b> (end). Finds intersections of <b>car-drivable roads</b> spread across the A–B area (no footways, cycleways, or park paths). Street View download is off until intersection spread looks right.</span>
    <label>Max corners <input id="maxcorners" type="number" value="__MAX_CORNERS__" min="1" max="200" style="width:70px"></label>
    <label>Size <input id="size" type="text" value="__IMAGE_SIZE__" style="width:90px"></label>
    <label>FOV <input id="fov" type="number" value="__FOV__" min="20" max="120" style="width:60px"></label>
    <label>Output folder <input id="out" type="text" value="streetview_corners" style="width:180px"></label>
    <button id="go">Find spread intersections</button>
    <button id="clear">Clear</button>
  </div>
  <div id="status">Click point A, then point B, then find intersections.</div>

  <script>
    let map, routeLine, markers = [], pts = [];

    function initMap() {
      map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 42.3601, lng: -71.0589 }, zoom: 15, mapTypeId: "roadmap",
        streetViewControl: true,
      });
      routeLine = new google.maps.Polyline({ map, path: [], strokeColor:"#2563eb", strokeWeight:4 });
      map.addListener("click", (e) => addPoint(e.latLng));
    }

    function addPoint(latLng) {
      if (pts.length >= 2) {
        clearPoints();
      }
      const label = pts.length === 0 ? "A" : "B";
      const color = pts.length === 0 ? "#16a34a" : "#dc2626";
      pts.push({ lat: latLng.lat(), lng: latLng.lng() });
      markers.push(new google.maps.Marker({
        position: latLng, map,
        label: label,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 10, fillColor: color,
                fillOpacity: 1, strokeColor: "#fff", strokeWeight: 2 },
      }));
      if (pts.length === 2) {
        routeLine.setPath(pts.map(p => new google.maps.LatLng(p.lat, p.lng)));
        setStatus("A and B set. Click Download, or Clear to pick again.");
      } else {
        routeLine.setPath([]);
        setStatus("Point A set. Now click point B.");
      }
    }

    function clearPoints() {
      pts = []; routeLine.setPath([]);
      markers.forEach(m => m.setMap(null)); markers = [];
      setStatus("Cleared. Click point A, then point B.");
    }

    function setStatus(s) { document.getElementById("status").textContent = s; }

    async function go() {
      if (pts.length !== 2) { setStatus("Need exactly 2 points: A then B."); return; }
      const body = {
        point_a: [pts[0].lat, pts[0].lng],
        point_b: [pts[1].lat, pts[1].lng],
        max_corners: parseInt(document.getElementById("maxcorners").value, 10),
        size: document.getElementById("size").value,
        fov: parseInt(document.getElementById("fov").value, 10),
        out_dir: document.getElementById("out").value,
      };
      setStatus("Finding spread intersections... may take a minute.");
      try {
        const res = await fetch("/download", {
          method: "POST", headers: {"Content-Type":"application/json"},
          body: JSON.stringify(body),
        });
        const j = await res.json();
        if (j.error) { setStatus("Error: " + j.error); return; }
        setStatus(
          "Done.\\n" +
          "Selected: " + j.intersections_selected + " / requested " + j.requested_max_corners + "\\n" +
          "In search area: " + j.intersections_in_area + "\\n" +
          "Area margin final: " + j.area_margin_final_m + " m (started " + j.area_margin_start_m + " m)\\n" +
          "Mode: " + j.mode + "  (fetch_streetview=" + j.fetch_streetview + ")\\n" +
          "Skipped: " + j.skipped + "\\n" +
          "Manifest: corners_manifest.json — run plot_corners_map.py to verify spread\\n" +
          "Folder: " + j.out_dir
        );
      } catch (e) { setStatus("Request failed: " + e); }
    }

    document.addEventListener("DOMContentLoaded", () => {
      document.getElementById("go").addEventListener("click", go);
      document.getElementById("clear").addEventListener("click", clearPoints);
    });
  </script>
  <script async
    src="https://maps.googleapis.com/maps/api/js?key=__API_KEY__&callback=initMap">
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    api_key = ""
    base_out = "."

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            html = PAGE.replace("__API_KEY__", self.api_key)
            html = html.replace("__MAX_CORNERS__", str(MAX_CORNERS))
            html = html.replace("__IMAGE_SIZE__", IMAGE_SIZE)
            html = html.replace("__FOV__", str(FOV))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/download":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            pa = payload["point_a"]
            pb = payload["point_b"]
            point_a = (float(pa[0]), float(pa[1]))
            point_b = (float(pb[0]), float(pb[1]))
            out_name = str(payload.get("out_dir") or "streetview_corners")
            out_dir = out_name if os.path.isabs(out_name) else os.path.join(self.base_out, out_name)
            manifest = download_corners(
                point_a=point_a,
                point_b=point_b,
                max_corners=int(payload.get("max_corners", MAX_CORNERS)),
                out_dir=out_dir,
                api_key=self.api_key,
                size=str(payload.get("size") or IMAGE_SIZE),
                fov=int(payload.get("fov", FOV)),
            )
            self._send(200, json.dumps(manifest).encode("utf-8"), "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(200, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")

    def log_message(self, *args):
        return


def main() -> None:
    ap = argparse.ArgumentParser(description="Find spread drivable-road intersections between two map points.")
    ap.add_argument("--api-key", default=(API_KEY or os.environ.get("GOOGLE_MAPS_API_KEY", "")),
                    help="Google Maps key (Street View + map UI).")
    ap.add_argument("--port", type=int, default=8001,
                    help="Local server port (default 8001 to avoid clashing with walk script).")
    ap.add_argument("--base-out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set API_KEY at top of script, GOOGLE_MAPS_API_KEY env, or --api-key.")

    Handler.api_key = args.api_key
    Handler.base_out = args.base_out

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print("Street View corner collector running at:", url)
    print("Output base dir:", os.path.abspath(args.base_out))
    print("Click point A, then point B, set max corners, download.")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
