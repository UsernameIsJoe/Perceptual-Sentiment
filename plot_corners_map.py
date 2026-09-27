#!/usr/bin/env python3
"""
Quick map check for fetch_streetview_corners.py output.

Reads corners_manifest.json and writes a self-contained HTML map (Leaflet, no extra deps).
Open the HTML in a browser to verify corner locations, route, and A/B points.

Run:
    python plot_corners_map.py
    python plot_corners_map.py --open
"""
from __future__ import annotations

import argparse
import json
import os
import webbrowser
from pathlib import Path

_STUDY_DIR = Path(__file__).resolve().parent

# ============================================================================
# EDIT HERE — folder that contains corners_manifest.json
DATA_FOLDER: str = str(_STUDY_DIR / "streetview_corners")
# Output HTML ("" = save inside DATA_FOLDER as corners_map.html)
OUTPUT_HTML: str = ""
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot corner collection on an interactive map.")
    ap.add_argument("--data-folder", default=DATA_FOLDER)
    ap.add_argument("--output", default=OUTPUT_HTML)
    ap.add_argument("--open", action="store_true", help="Open the map in a browser.")
    args = ap.parse_args()

    folder = args.data_folder if os.path.isabs(args.data_folder) else str(_STUDY_DIR / args.data_folder)
    manifest_path = os.path.join(folder, "corners_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)

    pa, pb = m["point_a"], m["point_b"]
    nodes = m.get("nodes", [])
    route = m.get("route_polyline", [])
    edges = m.get("edges", [])
    area_bbox = m.get("area_bbox")

    node_by_id = {n["id"]: n for n in nodes}
    edge_lines = []
    for e in edges:
        a, b = node_by_id.get(e["from"]), node_by_id.get(e["to"])
        if a and b:
            edge_lines.append([[a["lat"], a["lng"]], [b["lat"], b["lng"]]])

    center_lat = sum(n["lat"] for n in nodes) / len(nodes) if nodes else pa["lat"]
    center_lng = sum(n["lng"] for n in nodes) / len(nodes) if nodes else pa["lng"]

    mode = m.get("mode", "streetview")
    margin = m.get("area_margin_final_m", m.get("corridor_m_final"))
    info = f"Mode: {mode} | Corners: {m.get('saved', len(nodes))}"
    if margin is not None:
        info += f" | Area margin: {margin} m"

    payload = json.dumps({
        "point_a": [pa["lat"], pa["lng"]],
        "point_b": [pb["lat"], pb["lng"]],
        "route": route,
        "edges": edge_lines,
        "area_bbox": area_bbox,
        "nodes": [
            {
                "lat": n["lat"], "lng": n["lng"], "id": n["id"],
                "image": n.get("image", ""),
                "grid_cell": n.get("grid_cell"),
                "streets": ", ".join(n.get("street_names", [])[:2]),
            }
            for n in nodes
        ],
        "center": [center_lat, center_lng],
        "saved": m.get("saved", len(nodes)),
        "info": info,
    })

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Corner map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    #info {{
      position: absolute; top: 10px; right: 10px; z-index: 1000;
      background: white; padding: 10px 12px; border-radius: 6px;
      font: 13px system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.2);
    }}
  </style>
</head>
<body>
  <div id="info"></div>
  <div id="map"></div>
  <script>
    const DATA = {payload};
    const map = L.map("map").setView(DATA.center, 15);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap",
    }}).addTo(map);

    document.getElementById("info").textContent = DATA.info || ("Corners saved: " + DATA.saved);

    if (DATA.area_bbox) {{
      const b = DATA.area_bbox;
      L.rectangle([[b.south, b.west], [b.north, b.east]], {{
        color: "#64748b", weight: 1, fillOpacity: 0.05, dashArray: "6 4",
      }}).bindPopup("Search area").addTo(map);
    }}

    if (DATA.route.length >= 2) {{
      L.polyline(DATA.route, {{color: "#2563eb", weight: 3, opacity: 0.5, dashArray: "8 6"}})
        .bindPopup("Reference OSRM route").addTo(map);
    }}
    DATA.edges.forEach(seg => {{
      L.polyline(seg, {{color: "#94a3b8", weight: 2, dashArray: "4 6"}}).addTo(map);
    }});

    L.circleMarker(DATA.point_a, {{
      radius: 10, color: "#fff", weight: 2, fillColor: "#16a34a", fillOpacity: 1,
    }}).bindPopup("Point A").addTo(map);
    L.circleMarker(DATA.point_b, {{
      radius: 10, color: "#fff", weight: 2, fillColor: "#dc2626", fillOpacity: 1,
    }}).bindPopup("Point B").addTo(map);

    DATA.nodes.forEach((n, i) => {{
      L.circleMarker([n.lat, n.lng], {{
        radius: 7, color: "#fff", weight: 2, fillColor: "#ea580c", fillOpacity: 0.95,
      }}).bindPopup("<b>" + n.id + "</b>" + (n.streets ? "<br>" + n.streets : "") + (n.image ? "<br>" + n.image : "")).addTo(map)
        .bindTooltip(String(i + 1), {{permanent: false}});
    }});

    const bounds = [];
    DATA.nodes.forEach(n => bounds.push([n.lat, n.lng]));
    bounds.push(DATA.point_a, DATA.point_b);
    if (bounds.length) map.fitBounds(bounds, {{padding: [40, 40]}});
  </script>
</body>
</html>
"""

    out = args.output or os.path.join(folder, "corners_map.html")
    if not os.path.isabs(out):
        out = str(_STUDY_DIR / out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved:", out)
    print("Corners on map:", len(nodes))
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
