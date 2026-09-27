#!/usr/bin/env python3
"""
Sample diverse A→B driving paths through random corner subsets from corners_manifest.json.

For each path:
  1. Pick K corners from the pool (spread apart).
  2. Order them forward toward B (no backtracking).
  3. Route on OSM roads via OSRM driving (not straight lines).
  4. Validate detour / monotonic progress.
  5. Aggregate per-corner sentiment from vibe_scores_per_image.csv.

Outputs (under OUTPUT_FOLDER):
  - paths_manifest.json   full path metadata + polylines  (when ROUTES_ENABLED)
  - paths_scores.csv      one row per path with vibe aggregates
  - paths_map.html        all paths on a Leaflet map
  - category_windroses.html  radar charts of combined category scores
  - category_criteria.csv mean criteria per path category
  - nodes_map.html        corner pool colored by sentiment (when ROUTES_ENABLED is False)

Run:
    python sample_corner_paths.py
    python sample_corner_paths.py --open
    python sample_corner_paths.py --nodes-only --open
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import SCORES_META_COLUMNS, cfg

_STUDY_DIR = Path(__file__).resolve().parent

# ============================================================================
# EDIT HERE
# ----------------------------------------------------------------------------
# Corner collection manifest (from fetch_streetview_corners.py).
MANIFEST_PATH: str = str(_STUDY_DIR / "streetview_corners" / "corners_manifest.json")
# Sentiment scores folder — defaults to config OUTPUT_FOLDER.
SCORES_FOLDER: str = cfg.OUTPUT_FOLDER
SCORES_CSV_NAME: str = "vibe_scores_per_image.csv"

# How many paths to sample, and corners per path (count only — which corners is random).
NUM_PATHS: int = 50
CORNERS_PER_PATH: int = 5

# False = visualize corner pool + sentiment only (no OSRM path sampling).
# Set True again once node selection looks good.
ROUTES_ENABLED: bool = False

# Where to write results ("" = <manifest_dir>/sampled_paths).
OUTPUT_FOLDER: str = ""

# Path quality / sampling.
MAX_DETOUR_RATIO: float = 2.2       # route length / direct A→B OSRM length
MIN_CORNER_SEPARATION_M: float = 200.0
MAX_RETRIES_PER_PATH: int = 80
DEDUPE_BY_CORNER_SET: bool = True
RANDOM_SEED: int | None = 42

# Sentiment aggregation for paths_scores.csv: "weighted_mean" | "mean" | "sum"
SCORE_AGGREGATION: str = "weighted_mean"

# How route lines are colored on paths_map.html:
#   "category"     — assign one of 4 path types from full vibe profile (default)
#   "accumulated"  — single vibe with highest sum across corners
#   "single_peak"  — vibe of the single highest corner score
#   "max_min"      — vibe whose lowest corner score is still highest
#   "q1"           — vibe with highest 1st quartile across corners
#   "mean"         — vibe with highest mean across corners
ROUTE_COLOR_MODE: str = "category"

# Subtract λ × (typical path category score) so labels reflect *relative* fit, not
# global score bias (open/calm/inviting were pushing almost everything to restorative).
CATEGORY_BASELINE_LAMBDA: float = 1.0

# OSRM
OSRM_ROUTE = "https://router.project-osrm.org/route/v1/driving"
# ============================================================================

EARTH_R = 6371000.0

# Vibe colors — synced with plot_all_vibes_timeline.py
VIBE_COLORS = {
    "calm":         "#1f77b4",
    "energetic":    "#d62728",
    "open":         "#2ca02c",
    "inviting":     "#ff7f0e",
    "excluding":    "#9467bd",
    "legible":      "#8c564b",
    "disorienting": "#e377c2",
    "compressed":   "#7f7f7f",
    "exploratory":  "#bcbd22",
}
VIBE_COLOR_FALLBACK = "#6b7280"

# Map styling
MAP_ROUTE_WEIGHT = 2.5
MAP_ROUTE_OPACITY = 0.48
MAP_CORNER_RADIUS = 3
MAP_NODE_RADIUS = 5
MAP_UNKNOWN_COLOR = "#4b5563"
# Path travel animation on paths_map.html (static routes stay; movers draw on top).
MAP_ANIMATION_ENABLED = True
MAP_ANIM_DURATION_MS = 9000
MAP_ANIM_STAGGER_MS = 120
MAP_ANIM_LOOP = True
MAP_ANIM_HEAD_RADIUS = 4
CATEGORY_WINDROSE_ENABLED = True
# When cross-category min–max leaves multiple spokes at 0, floor extras to this (≥1 zero kept).
WINDROSE_ZERO_FLOOR = 0.12

ROUTE_COLOR_MODES = frozenset({
    "category", "accumulated", "single_peak", "max_min", "q1", "mean",
})
ROUTE_COLOR_MODE_LABELS = {
    "category": "path type (4 categories)",
    "accumulated": "accumulated sum",
    "single_peak": "single peak score",
    "max_min": "highest minimum",
    "q1": "highest Q1 (25th pct)",
    "mean": "highest mean",
}

# Path-type profiles: score = Σ(highlight × vibe) − Σ(suppress × vibe), then minus baseline.
# Weights tuned on the corner pool so four categories spread more evenly (not uniform).
PATH_CATEGORIES: dict[str, dict] = {
    "commute": {
        "label": "Commute / Efficient",
        "short": "Commute",
        "color": "#38bdf8",
        "highlight": {"legible": 1.5, "open": 0.5, "calm": 0.35},
        "suppress": {"disorienting": 0.85, "exploratory": 1.35, "compressed": 0.5, "energetic": 0.25},
    },
    "flaneur": {
        "label": "Flâneur / Exploratory",
        "short": "Flâneur",
        "color": "#c084fc",
        "highlight": {"exploratory": 2.0, "inviting": 0.9, "energetic": 0.5},
        "suppress": {"excluding": 0.6, "legible": 0.5, "compressed": 0.4},
    },
    "restorative": {
        "label": "Restorative / Reflective",
        "short": "Restorative",
        "color": "#6ee7b7",
        "highlight": {"calm": 0.85, "inviting": 0.45, "open": 0.2},
        "suppress": {"energetic": 1.1, "compressed": 0.5, "disorienting": 0.6},
    },
    "social": {
        "label": "Social / Urban Life",
        "short": "Social",
        "color": "#fb923c",
        "highlight": {"energetic": 1.4, "inviting": 0.75, "legible": 0.6},
        "suppress": {"excluding": 0.6, "calm": 0.4, "disorienting": 0.5},
    },
}

# Category windrose axes (combined scores per path type).
PATH_CRITERIA: list[dict] = [
    {
        "key": "detour_ratio",
        "label": "Detour Ratio",
        "short": "Detour",
        "description": "Route length / shortest path length (higher = less direct).",
    },
    {
        "key": "profile_fit",
        "label": "Profile Fit",
        "short": "Profile Fit",
        "description": "Match to path type: Σ(highlight×vibe) − Σ(suppress×vibe), baseline-adjusted.",
    },
    {
        "key": "comfort",
        "label": "Perceptual Comfort",
        "short": "Comfort",
        "description": "inviting + open + legible + calm − excluding − compressed − disorienting",
    },
    {
        "key": "friction",
        "label": "Perceptual Friction",
        "short": "Friction",
        "description": "compressed + disorienting + excluding",
    },
    {
        "key": "richness",
        "label": "Experiential Richness",
        "short": "Richness",
        "description": "exploratory + energetic + inviting",
    },
    {
        "key": "stability",
        "label": "Sentiment Stability",
        "short": "Stability",
        "description": "1 / (1 + mean per-vibe σ across corners); higher = more consistent.",
    },
]


def _vibe_color(vibe: str | None) -> str:
    if vibe and vibe in VIBE_COLORS:
        return VIBE_COLORS[vibe]
    return VIBE_COLOR_FALLBACK


def _dominant_vibe(scores: dict[str, float], vibes: list[str]) -> str | None:
    if not scores or not vibes:
        return None
    return max(vibes, key=lambda v: scores.get(v, 0.0))


def _corner_score_rows(corners: list[dict], vibes: list[str]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for c in corners:
        vs = c.get("vibe_scores") or {}
        if vs:
            rows.append({v: float(vs.get(v, 0.0)) for v in vibes})
    return rows


def _route_color_profile(
    corner_rows: list[dict[str, float]],
    vibes: list[str],
    mode: str,
) -> tuple[str | None, dict[str, float], float | None]:
    """
    Pick the vibe that colors a route from per-corner scores.
    Returns (dominant_vibe, profile per vibe, winning metric value).
    """
    if not corner_rows or not vibes:
        return None, {}, None
    if mode == "category":
        return None, {}, None
    mode = mode if mode in ROUTE_COLOR_MODES else "accumulated"

    if mode == "single_peak":
        best_v, best_s = None, float("-inf")
        for row in corner_rows:
            for v in vibes:
                s = float(row.get(v, 0.0))
                if s > best_s:
                    best_v, best_s = v, s
        profile = {v: 0.0 for v in vibes}
        if best_v is not None:
            profile[best_v] = best_s
        return best_v, profile, best_s if best_v is not None else None

    profile: dict[str, float] = {}
    for v in vibes:
        vals = np.array([float(row.get(v, 0.0)) for row in corner_rows], dtype=float)
        if mode == "accumulated":
            profile[v] = float(vals.sum())
        elif mode == "mean":
            profile[v] = float(vals.mean()) if len(vals) else 0.0
        elif mode == "max_min":
            profile[v] = float(vals.min()) if len(vals) else 0.0
        elif mode == "q1":
            profile[v] = float(np.percentile(vals, 25)) if len(vals) else 0.0
        else:
            profile[v] = float(vals.sum())

    dom = max(vibes, key=lambda v: profile.get(v, 0.0))
    return dom, profile, profile.get(dom)


def _category_match_score(vibe_scores: dict[str, float], category: dict) -> float:
    score = 0.0
    for vibe, weight in category.get("highlight", {}).items():
        score += weight * float(vibe_scores.get(vibe, 0.0))
    for vibe, weight in category.get("suppress", {}).items():
        score -= weight * float(vibe_scores.get(vibe, 0.0))
    return score


def _population_vibe_mean(scores_df: pd.DataFrame, vibes: list[str]) -> dict[str, float]:
    return {v: float(scores_df[v].mean()) for v in vibes if v in scores_df.columns}


def _category_baselines(pop_mean: dict[str, float]) -> dict[str, float]:
    return {
        cid: _category_match_score(pop_mean, PATH_CATEGORIES[cid])
        for cid in PATH_CATEGORIES
    }


def _apply_path_category(
    path: dict,
    *,
    baselines: dict[str, float] | None = None,
) -> None:
    scores = path.get("vibe_scores") or {}
    raw_scores = {
        cid: _category_match_score(scores, PATH_CATEGORIES[cid])
        for cid in PATH_CATEGORIES
    }
    if baselines and CATEGORY_BASELINE_LAMBDA:
        cat_scores = {
            cid: raw_scores[cid] - CATEGORY_BASELINE_LAMBDA * baselines[cid]
            for cid in PATH_CATEGORIES
        }
    else:
        cat_scores = raw_scores

    best = max(PATH_CATEGORIES.keys(), key=lambda c: cat_scores[c])
    path["route_color_mode"] = "category"
    path["path_category"] = best
    path["path_category_label"] = PATH_CATEGORIES[best]["label"]
    path["path_category_scores"] = {k: round(v, 8) for k, v in cat_scores.items()}
    path["path_category_scores_raw"] = {k: round(v, 8) for k, v in raw_scores.items()}
    path["route_color_metric"] = round(cat_scores[best], 8)
    path["dominant_vibe"] = None


def _path_sentiment_volatility(corners: list[dict], vibes: list[str]) -> float:
    rows = _corner_score_rows(corners, vibes)
    if len(rows) < 2:
        return 0.0
    stds = [float(np.std([r[v] for r in rows], ddof=0)) for v in vibes]
    return float(np.mean(stds)) if stds else 0.0


def _path_criteria_metrics(
    path: dict,
    vibes: list[str],
    *,
    baselines: dict[str, float] | None = None,
) -> dict[str, float]:
    vs = path.get("vibe_scores") or {}
    cat = path.get("path_category")
    if cat and cat in PATH_CATEGORIES:
        raw_fit = _category_match_score(vs, PATH_CATEGORIES[cat])
        if baselines and CATEGORY_BASELINE_LAMBDA:
            profile_fit = raw_fit - CATEGORY_BASELINE_LAMBDA * baselines.get(cat, 0.0)
        else:
            profile_fit = raw_fit
    else:
        profile_fit = 0.0

    comfort = (
        float(vs.get("inviting", 0.0))
        + float(vs.get("open", 0.0))
        + float(vs.get("legible", 0.0))
        + float(vs.get("calm", 0.0))
        - float(vs.get("excluding", 0.0))
        - float(vs.get("compressed", 0.0))
        - float(vs.get("disorienting", 0.0))
    )
    friction = (
        float(vs.get("compressed", 0.0))
        + float(vs.get("disorienting", 0.0))
        + float(vs.get("excluding", 0.0))
    )
    richness = (
        float(vs.get("exploratory", 0.0))
        + float(vs.get("energetic", 0.0))
        + float(vs.get("inviting", 0.0))
    )
    volatility = _path_sentiment_volatility(path.get("corners", []), vibes)
    stability = 1.0 / (1.0 + volatility)

    return {
        "detour_ratio": float(path.get("detour_ratio", 1.0)),
        "profile_fit": profile_fit,
        "comfort": comfort,
        "friction": friction,
        "richness": richness,
        "volatility": volatility,
        "stability": stability,
    }


def _normalize_criteria_for_windrose(
    raw_by_category: dict[str, dict[str, float]],
    *,
    zero_floor: float = WINDROSE_ZERO_FLOOR,
) -> dict[str, dict[str, float]]:
    """Min–max each criterion across categories, then allow at most one 0 per windrose.

    Cross-category scaling keeps category profiles distinct; tied-at-min spokes
    after that step are lifted to ``zero_floor`` except the single weakest raw
    criterion in that category.
    """
    keys = [c["key"] for c in PATH_CRITERIA]
    normalized: dict[str, dict[str, float]] = {cid: {} for cid in PATH_CATEGORIES}

    for key in keys:
        vals = {cid: raw_by_category.get(cid, {}).get(key, 0.0) for cid in PATH_CATEGORIES}
        vmin, vmax = min(vals.values()), max(vals.values())
        span = vmax - vmin
        for cid in PATH_CATEGORIES:
            if span > 0:
                normalized[cid][key] = (vals[cid] - vmin) / span
            else:
                normalized[cid][key] = 0.5

    eps = 1e-9
    for cid in PATH_CATEGORIES:
        norms = normalized[cid]
        raw = raw_by_category.get(cid, {})
        at_zero = [k for k in keys if norms[k] <= eps]
        if len(at_zero) <= 1:
            continue
        # Keep the criterion with the lowest raw mean at 0; floor the rest.
        at_zero.sort(key=lambda k: raw.get(k, 0.0))
        for k in at_zero[1:]:
            norms[k] = zero_floor

    return normalized


def _build_category_criteria_report(
    paths: list[dict],
    vibes: list[str],
    *,
    baselines: dict[str, float] | None = None,
    scores_by_image: pd.DataFrame | None = None,
    route_color_mode: str = ROUTE_COLOR_MODE,
) -> dict:
    if scores_by_image is not None and vibes:
        _enrich_paths_for_map(
            paths, vibes, scores_by_image,
            route_color_mode=route_color_mode, baselines=baselines,
        )
    elif route_color_mode == "category":
        for p in paths:
            if not p.get("path_category"):
                _apply_path_category(p, baselines=baselines)

    per_path: list[dict] = []
    buckets: dict[str, list[dict[str, float]]] = {cid: [] for cid in PATH_CATEGORIES}
    for p in paths:
        metrics = _path_criteria_metrics(p, vibes, baselines=baselines)
        cat = p.get("path_category")
        rec = {"path_id": p.get("path_id"), "path_category": cat, **metrics}
        per_path.append(rec)
        if cat in buckets:
            buckets[cat].append(metrics)

    raw_by_category: dict[str, dict[str, float]] = {}
    categories_out = []
    for cid in PATH_CATEGORIES:
        items = buckets[cid]
        if items:
            raw = {
                key: float(np.mean([m[key] for m in items]))
                for key in [c["key"] for c in PATH_CRITERIA] + ["volatility"]
            }
        else:
            raw = {c["key"]: 0.0 for c in PATH_CRITERIA}
            raw["volatility"] = 0.0
        raw_by_category[cid] = raw
        categories_out.append({
            "id": cid,
            "label": PATH_CATEGORIES[cid]["label"],
            "short": PATH_CATEGORIES[cid]["short"],
            "color": PATH_CATEGORIES[cid]["color"],
            "n_paths": len(items),
            "raw": {k: round(v, 8) for k, v in raw.items()},
        })

    normalized = _normalize_criteria_for_windrose(raw_by_category)
    for cat in categories_out:
        cid = cat["id"]
        cat["normalized"] = [
            round(normalized[cid][c["key"]], 6) for c in PATH_CRITERIA
        ]

    return {
        "criteria": PATH_CRITERIA,
        "categories": categories_out,
        "per_path": per_path,
        "info": (
            f"{len(paths)} paths · combined mean per category · "
            f"each axis min–max across categories (≤1 zero per windrose)"
        ),
    }


def _write_category_criteria_csv(report: dict, out_path: str) -> None:
    rows = []
    for cat in report["categories"]:
        row = {
            "path_category": cat["id"],
            "path_category_label": cat["label"],
            "n_paths": cat["n_paths"],
        }
        row.update(cat["raw"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_category_windroses(
    data: dict,
    out_path: str,
    *,
    scores_by_image: pd.DataFrame | None = None,
    scores_df: pd.DataFrame | None = None,
    route_color_mode: str | None = None,
    baselines: dict[str, float] | None = None,
) -> dict:
    paths = data["paths"]
    vibes = data.get("vibes", [])
    mode = route_color_mode or data.get("route_color_mode", ROUTE_COLOR_MODE)
    if baselines is None and mode == "category":
        baselines = data.get("category_baselines")
        if baselines is None and scores_df is not None:
            baselines = _category_baselines(_population_vibe_mean(scores_df, vibes))

    report = _build_category_criteria_report(
        paths, vibes,
        baselines=baselines,
        scores_by_image=scores_by_image,
        route_color_mode=mode if mode == "category" else "category",
    )
    payload = json.dumps({
        "criteria": report["criteria"],
        "categories": report["categories"],
        "info": report["info"],
    })

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Category windroses</title>
  <style>
    html, body {{
      margin: 0; min-height: 100%;
      background: #0d1117; color: #d1d5db;
      font: 12px/1.45 system-ui, -apple-system, sans-serif;
    }}
    #header {{
      padding: 16px 20px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    #header h1 {{
      margin: 0 0 6px; font-size: 18px; font-weight: 600; color: #f3f4f6;
    }}
    #header p {{ margin: 0; color: #9ca3af; max-width: 720px; }}
    #grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 16px;
      padding: 16px 20px 24px;
      max-width: 1100px;
    }}
    .panel {{
      background: rgba(13, 17, 23, 0.88);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      padding: 12px 12px 8px;
      backdrop-filter: blur(6px);
    }}
    .panel h2 {{
      margin: 0 0 2px; font-size: 14px; font-weight: 600;
    }}
    .panel .meta {{
      color: #6b7280; font-size: 11px; margin-bottom: 6px;
    }}
    .windrose-wrap {{
      display: flex; justify-content: center; align-items: center;
      min-height: 340px;
    }}
    canvas.windrose {{
      display: block;
      width: 320px;
      height: 320px;
      max-width: 100%;
      aspect-ratio: 1 / 1;
    }}
    #criteria {{
      margin: 0 20px 24px;
      max-width: 1100px;
      background: rgba(13, 17, 23, 0.88);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      padding: 12px 14px;
    }}
    #criteria h3 {{
      margin: 0 0 10px; font-size: 11px; font-weight: 600;
      color: #6b7280; letter-spacing: 0.04em; text-transform: uppercase;
    }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 11px;
    }}
    th, td {{
      text-align: left; padding: 6px 8px;
      border-top: 1px solid rgba(255,255,255,0.05);
      vertical-align: top;
    }}
    th {{ color: #9ca3af; font-weight: 600; }}
    .raw-val {{ color: #e5e7eb; font-variant-numeric: tabular-nums; }}
    #detail {{
      position: fixed; top: 12px; right: 12px; z-index: 10;
      background: rgba(13, 17, 23, 0.92); color: #d1d5db;
      padding: 10px 14px; border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.06);
      max-width: 260px; backdrop-filter: blur(6px);
      font-size: 11px; line-height: 1.5;
    }}
    @media (max-width: 760px) {{
      #grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div id="header">
    <h1>Path category windroses</h1>
    <p id="info"></p>
  </div>
  <div id="grid"></div>
  <div id="criteria">
    <h3>Criteria</h3>
    <table>
      <thead>
        <tr><th>Axis</th><th>What it measures</th><th>Formula</th></tr>
      </thead>
      <tbody id="criteria-body"></tbody>
    </table>
  </div>
  <div id="detail">Hover a windrose for raw category means.</div>
  <script>
    const DATA = {payload};
    document.getElementById("info").textContent = DATA.info;

    const tbody = document.getElementById("criteria-body");
    DATA.criteria.forEach(c => {{
      const tr = document.createElement("tr");
      tr.innerHTML =
        '<td><b>' + c.short + '</b></td>' +
        '<td>' + c.description + '</td>' +
        '<td style="color:#9ca3af">' + c.label + '</td>';
      tbody.appendChild(tr);
    }});

    function drawWindrose(canvas, cat, criteria) {{
      const size = 320;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
      canvas.style.width = size + "px";
      canvas.style.height = size + "px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const cx = size / 2, cy = size / 2;
      const maxR = size * 0.34;
      const n = criteria.length;
      const start = -Math.PI / 2;

      ctx.clearRect(0, 0, size, size);

      for (let ring = 1; ring <= 4; ring++) {{
        const r = maxR * ring / 4;
        ctx.beginPath();
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }}

      for (let i = 0; i < n; i++) {{
        const ang = start + (i * 2 * Math.PI / n);
        const x = cx + Math.cos(ang) * maxR;
        const y = cy + Math.sin(ang) * maxR;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        ctx.stroke();

        const lx = cx + Math.cos(ang) * (maxR + 22);
        const ly = cy + Math.sin(ang) * (maxR + 22);
        ctx.fillStyle = "#9ca3af";
        ctx.font = "11px system-ui, -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(criteria[i].short, lx, ly);
      }}

      const vals = cat.normalized || [];
      ctx.beginPath();
      for (let i = 0; i < n; i++) {{
        const ang = start + (i * 2 * Math.PI / n);
        const r = maxR * Math.max(0, Math.min(1, vals[i] || 0));
        const x = cx + Math.cos(ang) * r;
        const y = cy + Math.sin(ang) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.closePath();
      ctx.fillStyle = cat.color + "55";
      ctx.fill();
      ctx.strokeStyle = cat.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      for (let i = 0; i < n; i++) {{
        const ang = start + (i * 2 * Math.PI / n);
        const r = maxR * Math.max(0, Math.min(1, vals[i] || 0));
        const x = cx + Math.cos(ang) * r;
        const y = cy + Math.sin(ang) * r;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = cat.color;
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }}
    }}

    function formatDetail(cat) {{
      if (!cat.n_paths) return cat.short + ": no paths in this category.";
      const lines = [
        '<b style="color:' + cat.color + '">' + cat.label + '</b>',
        cat.n_paths + ' paths (category mean)',
      ];
      DATA.criteria.forEach(c => {{
        const raw = cat.raw[c.key];
        const norm = cat.normalized[DATA.criteria.indexOf(c)];
        lines.push(
          c.short + ': <span class="raw-val">' + (raw != null ? raw.toFixed(5) : '—') +
          '</span> · norm ' + (norm != null ? norm.toFixed(2) : '—')
        );
      }});
      if (cat.raw.volatility != null) {{
        lines.push('Volatility σ: <span class="raw-val">' + cat.raw.volatility.toFixed(5) + '</span>');
      }}
      return lines.join('<br>');
    }}

    const grid = document.getElementById("grid");
    const detail = document.getElementById("detail");
    DATA.categories.forEach(cat => {{
      const panel = document.createElement("div");
      panel.className = "panel";
      panel.innerHTML =
        '<h2 style="color:' + cat.color + '">' + cat.short + '</h2>' +
        '<div class="meta">' + cat.n_paths + ' routes · ' + cat.label + '</div>';
      const wrap = document.createElement("div");
      wrap.className = "windrose-wrap";
      const canvas = document.createElement("canvas");
      canvas.className = "windrose";
      wrap.appendChild(canvas);
      panel.appendChild(wrap);
      grid.appendChild(panel);
      drawWindrose(canvas, cat, DATA.criteria);
      panel.addEventListener("mouseenter", () => {{
        detail.innerHTML = formatDetail(cat);
      }});
    }});
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report


def _apply_route_color_mode(
    path: dict,
    vibes: list[str],
    mode: str,
    *,
    baselines: dict[str, float] | None = None,
) -> None:
    if mode == "category":
        _apply_path_category(path, baselines=baselines)
        return
    rows = _corner_score_rows(path.get("corners", []), vibes)
    dom, profile, metric = _route_color_profile(rows, vibes, mode)
    path["route_color_mode"] = mode
    path["dominant_vibe"] = dom
    path["path_category"] = None
    path["path_category_label"] = None
    path["path_category_scores"] = None
    path["route_color_profile"] = {v: round(profile.get(v, 0.0), 8) for v in vibes}
    path["route_color_metric"] = round(metric, 8) if metric is not None else None


def _corner_vibe_details(
    corner: dict,
    vibes: list[str],
    scores_by_image: pd.DataFrame,
) -> dict:
    out = {
        "id": corner["id"],
        "lat": corner["lat"],
        "lng": corner["lng"],
        "image": corner.get("image"),
        "street_names": corner.get("street_names", []),
    }
    img = corner.get("image")
    if img and img in scores_by_image.index:
        row = scores_by_image.loc[img]
        c_scores = {v: round(float(row[v]), 8) for v in vibes}
        out["vibe_scores"] = c_scores
        out["dominant_vibe"] = _dominant_vibe(c_scores, vibes)
    else:
        out["vibe_scores"] = {}
        out["dominant_vibe"] = None
    return out


def _enrich_paths_for_map(
    paths: list[dict],
    vibes: list[str],
    scores_by_image: pd.DataFrame,
    route_color_mode: str = ROUTE_COLOR_MODE,
    baselines: dict[str, float] | None = None,
) -> None:
    for p in paths:
        enriched = []
        for c in p.get("corners", []):
            if c.get("dominant_vibe") is not None and c.get("vibe_scores"):
                enriched.append(c)
            else:
                enriched.append(_corner_vibe_details(c, vibes, scores_by_image))
        p["corners"] = enriched
        _apply_route_color_mode(p, vibes, route_color_mode, baselines=baselines)


def _haversine(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _progress_along_ab(
    pt: tuple[float, float],
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Scalar progress along A→B axis (0 at A, 1 at B; can exceed range)."""
    ax, ay = point_a[0], point_a[1]
    bx, by = point_b[0], point_b[1]
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 <= 0:
        return 0.0
    px, py = pt[0] - ax, pt[1] - ay
    return (px * dx + py * dy) / len2


def _osrm_route_waypoints(
    waypoints: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], float]:
    """Return (polyline lat/lng, distance_m) for ordered waypoints."""
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")
    coord_str = ";".join(f"{lng},{lat}" for lat, lng in waypoints)
    url = f"{OSRM_ROUTE}/{coord_str}"
    r = requests.get(
        url,
        params={"overview": "full", "geometries": "geojson", "steps": "false"},
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM failed: {data.get('message', data.get('code'))}")
    route = data["routes"][0]
    coords = route["geometry"]["coordinates"]
    polyline = [(lat, lng) for lng, lat in coords]
    return polyline, float(route["distance"])


def _route_length(polyline: list[tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return 0.0
    return sum(_haversine(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1))


def _subsample_polyline(polyline: list[tuple[float, float]], max_pts: int = 40) -> list[tuple[float, float]]:
    if len(polyline) <= max_pts:
        return polyline
    step = max(1, len(polyline) // max_pts)
    out = [polyline[i] for i in range(0, len(polyline), step)]
    if out[-1] != polyline[-1]:
        out.append(polyline[-1])
    return out


def _is_forward_route(
    polyline: list[tuple[float, float]],
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    max_backtrack: float = 0.02,
) -> bool:
    """Reject routes that move backward along A→B for more than a small fraction."""
    samples = _subsample_polyline(polyline, 50)
    progress = [_progress_along_ab(p, point_a, point_b) for p in samples]
    if progress[-1] < progress[0]:
        return False
    back = 0
    for i in range(1, len(progress)):
        if progress[i] + max_backtrack < progress[i - 1]:
            back += 1
    return back <= max(1, len(progress) // 20)


def _order_by_projection(
    corners: list[dict],
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> list[dict]:
    return sorted(
        corners,
        key=lambda c: _progress_along_ab((c["lat"], c["lng"]), point_a, point_b),
    )


def _order_greedy_forward(
    corners: list[dict],
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> list[dict]:
    remaining = list(corners)
    ordered: list[dict] = []
    current = point_a
    while remaining:
        def _score(c: dict) -> tuple[float, float]:
            p = (c["lat"], c["lng"])
            prog = _progress_along_ab(p, point_a, point_b)
            dist = _haversine(current, p)
            return (prog, -dist)

        nxt = max(remaining, key=_score)
        ordered.append(nxt)
        remaining.remove(nxt)
        current = (nxt["lat"], nxt["lng"])
    return ordered


def _corner_projections_increasing(
    ordered: list[dict],
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    epsilon: float = -0.01,
) -> bool:
    prev = _progress_along_ab(point_a, point_a, point_b)
    for c in ordered:
        p = _progress_along_ab((c["lat"], c["lng"]), point_a, point_b)
        if p + epsilon < prev:
            return False
        prev = p
    return True


def _sample_corners_spread(
    pool: list[dict],
    k: int,
    min_sep_m: float,
    rng: random.Random,
) -> list[dict] | None:
    if k > len(pool):
        return None
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked: list[dict] = []
    for c in shuffled:
        if any(_haversine((c["lat"], c["lng"]), (p["lat"], p["lng"])) < min_sep_m for p in picked):
            continue
        picked.append(c)
        if len(picked) == k:
            return picked
    return None


def _load_scores(scores_folder: str) -> tuple[pd.DataFrame, list[str]]:
    path = os.path.join(scores_folder, SCORES_CSV_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scores CSV not found: {path}")
    df = pd.read_csv(path)
    vibes = [c for c in df.columns if c not in SCORES_META_COLUMNS]
    if not vibes:
        raise ValueError(f"No vibe columns in {path}")
    return df, vibes


def _aggregate_scores(
    corner_rows: pd.DataFrame,
    vibes: list[str],
    mode: str,
) -> dict[str, float]:
    if corner_rows.empty:
        return {v: 0.0 for v in vibes}
    if mode == "sum":
        return {v: float(corner_rows[v].sum()) for v in vibes}
    if mode == "mean":
        return {v: float(corner_rows[v].mean()) for v in vibes}
    # weighted_mean (default)
    if "confidence" not in corner_rows.columns:
        return {v: float(corner_rows[v].mean()) for v in vibes}
    w = corner_rows["confidence"].astype(float).clip(lower=0)
    total = w.sum()
    if total <= 0:
        return {v: float(corner_rows[v].mean()) for v in vibes}
    out = {}
    for v in vibes:
        out[v] = float((corner_rows[v].astype(float) * w).sum() / total)
    return out


def visualize_nodes(
    manifest_path: str,
    scores_folder: str,
    out_dir: str,
) -> dict:
    """Map the full corner pool colored by each node's dominant vibe (no routes)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    pool = manifest.get("nodes", [])
    if not pool:
        raise ValueError("No nodes in corners manifest.")

    scores_df, vibes = _load_scores(scores_folder)
    scores_by_image = scores_df.set_index("image")
    os.makedirs(out_dir, exist_ok=True)

    nodes: list[dict] = []
    missing_scores = 0
    vibe_counts: dict[str, int] = {}

    for c in pool:
        detail = _corner_vibe_details(c, vibes, scores_by_image)
        if not detail.get("vibe_scores"):
            missing_scores += 1
        dom = detail.get("dominant_vibe")
        if dom:
            vibe_counts[dom] = vibe_counts.get(dom, 0) + 1
        nodes.append(detail)

    result = {
        "mode": "nodes_only",
        "manifest_path": os.path.abspath(manifest_path),
        "scores_folder": os.path.abspath(scores_folder),
        "point_a": manifest["point_a"],
        "point_b": manifest["point_b"],
        "area_bbox": manifest.get("area_bbox"),
        "vibes": vibes,
        "nodes_total": len(nodes),
        "nodes_with_scores": len(nodes) - missing_scores,
        "nodes_missing_scores": missing_scores,
        "dominant_vibe_counts": vibe_counts,
        "nodes": nodes,
    }

    manifest_out = os.path.join(out_dir, "nodes_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    rows = []
    for n in nodes:
        row = {
            "id": n["id"],
            "image": n.get("image"),
            "lat": n["lat"],
            "lng": n["lng"],
            "dominant_vibe": n.get("dominant_vibe"),
            "confidence": None,
        }
        img = n.get("image")
        if img and img in scores_by_image.index and "confidence" in scores_by_image.columns:
            row["confidence"] = float(scores_by_image.loc[img, "confidence"])
        row.update(n.get("vibe_scores") or {})
        rows.append(row)
    scores_out = os.path.join(out_dir, "nodes_scores.csv")
    pd.DataFrame(rows).to_csv(scores_out, index=False)

    map_out = os.path.join(out_dir, "nodes_map.html")
    _write_nodes_map(result, map_out)

    print(f"Corner pool: {len(nodes)} nodes  ({len(nodes) - missing_scores} with scores)")
    if vibe_counts:
        print("Dominant vibe counts:")
        for v in sorted(vibe_counts, key=lambda k: (-vibe_counts[k], k)):
            print(f"  {v}: {vibe_counts[v]}")
    print("Saved:", manifest_out)
    print("Saved:", scores_out)
    print("Saved:", map_out)
    return result


def _write_nodes_map(data: dict, out_path: str) -> None:
    pa, pb = data["point_a"], data["point_b"]
    nodes = data.get("nodes", [])
    vibes = data.get("vibes", [])
    vibe_counts = data.get("dominant_vibe_counts", {})

    all_lats = [pa["lat"], pb["lat"]] + [n["lat"] for n in nodes]
    all_lngs = [pa["lng"], pb["lng"]] + [n["lng"] for n in nodes]
    center = [sum(all_lats) / len(all_lats), sum(all_lngs) / len(all_lngs)]

    map_nodes = []
    vibes_used: set[str] = set()
    for n in nodes:
        dom = n.get("dominant_vibe")
        color = _vibe_color(dom)
        if dom:
            vibes_used.add(dom)
        scores = n.get("vibe_scores") or {}
        score_lines = "<br>".join(f"{v}: {scores.get(v, 0):.5f}" for v in vibes[:9])
        if len(vibes) > 9:
            score_lines += "<br>…"
        streets = ", ".join(n.get("street_names", [])[:3])
        popup = (
            f"<b>{n['id']}</b>"
            + (f" · <span style='color:{color}'><b>{dom}</b></span>" if dom else "")
            + (f"<br>{streets}" if streets else "")
            + (f"<br>{n.get('image', '')}" if n.get("image") else "")
            + f"<hr><b>Scores</b><br>{score_lines}"
        )
        map_nodes.append({
            "lat": n["lat"],
            "lng": n["lng"],
            "id": n["id"],
            "color": color,
            "dominant_vibe": dom,
            "popup": popup,
        })

    vibe_legend = [
        {"vibe": v, "color": VIBE_COLORS[v], "count": vibe_counts.get(v, 0)}
        for v in sorted(vibes_used)
        if v in VIBE_COLORS
    ]

    payload = json.dumps({
        "center": center,
        "point_a": [pa["lat"], pa["lng"]],
        "point_b": [pb["lat"], pb["lng"]],
        "area_bbox": data.get("area_bbox"),
        "nodes": map_nodes,
        "vibe_legend": vibe_legend,
        "node_radius": MAP_NODE_RADIUS,
        "info": (
            f"{len(nodes)} corners · {data.get('nodes_with_scores', 0)} scored · "
            f"colored by dominant vibe"
        ),
    })

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Corner nodes</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; background: #0d1117; }}
    #info {{
      position: absolute; top: 12px; right: 12px; z-index: 1000;
      background: rgba(13, 17, 23, 0.88); color: #9ca3af;
      padding: 10px 14px; border-radius: 8px;
      font: 12px/1.45 system-ui, -apple-system, sans-serif;
      border: 1px solid rgba(255,255,255,0.06);
      max-width: 280px; backdrop-filter: blur(6px);
    }}
    #legend {{
      position: absolute; bottom: 24px; left: 12px; z-index: 1000;
      background: rgba(13, 17, 23, 0.88); color: #d1d5db;
      padding: 10px 12px; border-radius: 8px;
      font: 11px system-ui, -apple-system, sans-serif;
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
    }}
    #legend h4 {{
      margin: 0 0 8px; font-size: 11px; font-weight: 600;
      color: #6b7280; letter-spacing: 0.04em; text-transform: uppercase;
    }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    .count {{ color: #6b7280; margin-left: auto; font-size: 10px; }}
    .leaflet-container {{ background: #0d1117; }}
  </style>
</head>
<body>
  <div id="info"></div>
  <div id="legend"><h4>Dominant vibe</h4></div>
  <div id="map"></div>
  <script>
    const DATA = {payload};
    const map = L.map("map", {{ zoomControl: true }}).setView(DATA.center, 14);

    L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png", {{
      maxZoom: 19, subdomains: "abcd",
      attribution: '&copy; OSM &copy; CARTO',
    }}).addTo(map);

    document.getElementById("info").textContent = DATA.info;

    if (DATA.area_bbox) {{
      const b = DATA.area_bbox;
      L.rectangle([[b.south, b.west], [b.north, b.east]], {{
        color: "#374151", weight: 1, fillOpacity: 0.04, dashArray: "6 4",
      }}).addTo(map);
    }}

    L.circleMarker(DATA.point_a, {{
      radius: 7, color: "rgba(255,255,255,0.35)", weight: 1,
      fillColor: "#22c55e", fillOpacity: 0.85,
    }}).bindPopup("Point A").addTo(map);
    L.circleMarker(DATA.point_b, {{
      radius: 7, color: "rgba(255,255,255,0.35)", weight: 1,
      fillColor: "#ef4444", fillOpacity: 0.85,
    }}).bindPopup("Point B").addTo(map);

    const legend = document.getElementById("legend");
    DATA.vibe_legend.forEach(item => {{
      const el = document.createElement("div");
      el.className = "leg-item";
      el.innerHTML =
        '<span class="swatch" style="background:' + item.color + '"></span>' +
        item.vibe + '<span class="count">' + item.count + '</span>';
      legend.appendChild(el);
    }});

    const bounds = [DATA.point_a, DATA.point_b];
    DATA.nodes.forEach(n => {{
      L.circleMarker([n.lat, n.lng], {{
        radius: DATA.node_radius,
        color: "rgba(255,255,255,0.2)",
        weight: 1,
        fillColor: n.color,
        fillOpacity: 0.88,
      }}).bindPopup(n.popup).bindTooltip(n.id, {{ permanent: false, opacity: 0.8 }}).addTo(map);
      bounds.push([n.lat, n.lng]);
    }});

    if (bounds.length) map.fitBounds(bounds, {{ padding: [48, 48] }});
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def sample_paths(
    manifest_path: str,
    scores_folder: str,
    out_dir: str,
    *,
    num_paths: int = NUM_PATHS,
    corners_per_path: int = CORNERS_PER_PATH,
    seed: int | None = RANDOM_SEED,
    route_color_mode: str = ROUTE_COLOR_MODE,
) -> dict:
    if route_color_mode not in ROUTE_COLOR_MODES:
        raise ValueError(
            f"Invalid route_color_mode {route_color_mode!r}. "
            f"Choose from: {sorted(ROUTE_COLOR_MODES)}"
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    point_a = (manifest["point_a"]["lat"], manifest["point_a"]["lng"])
    point_b = (manifest["point_b"]["lat"], manifest["point_b"]["lng"])
    pool = manifest.get("nodes", [])
    if len(pool) < corners_per_path:
        raise ValueError(f"Corner pool ({len(pool)}) smaller than CORNERS_PER_PATH ({corners_per_path})")

    scores_df, vibes = _load_scores(scores_folder)
    scores_by_image = scores_df.set_index("image")
    pop_mean = _population_vibe_mean(scores_df, vibes)
    cat_baselines = (
        _category_baselines(pop_mean) if route_color_mode == "category" else None
    )
    if cat_baselines:
        print("Category baselines (λ={}):".format(CATEGORY_BASELINE_LAMBDA))
        for cid, val in cat_baselines.items():
            print(f"  {PATH_CATEGORIES[cid]['short']:12s} {val:.6f}")

    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    # --- Route formation (disabled while refining corner/node selection) ---
    if not ROUTES_ENABLED:
        raise RuntimeError(
            "ROUTES_ENABLED is False. Run visualize_nodes() instead, or set ROUTES_ENABLED = True."
        )

    print("Direct A→B baseline route…")
    _, direct_dist_m = _osrm_route_waypoints([point_a, point_b])
    print(f"  Direct distance: {direct_dist_m:.0f} m  (max detour ratio: {MAX_DETOUR_RATIO})")

    paths: list[dict] = []
    seen_sets: set[frozenset[str]] = set()
    attempts = 0
    max_total_attempts = num_paths * MAX_RETRIES_PER_PATH

    while len(paths) < num_paths and attempts < max_total_attempts:
        attempts += 1
        picked = _sample_corners_spread(pool, corners_per_path, MIN_CORNER_SEPARATION_M, rng)
        if not picked:
            continue

        corner_ids = [c["id"] for c in picked]
        key = frozenset(corner_ids)
        if DEDUPE_BY_CORNER_SET and key in seen_sets:
            continue

        best: dict | None = None
        for order_fn in (_order_by_projection, _order_greedy_forward):
            ordered = order_fn(picked, point_a, point_b)
            if not _corner_projections_increasing(ordered, point_a, point_b):
                continue
            waypoints = [point_a] + [(c["lat"], c["lng"]) for c in ordered] + [point_b]
            try:
                polyline, dist_m = _osrm_route_waypoints(waypoints)
            except (RuntimeError, requests.RequestException):
                continue

            detour = dist_m / direct_dist_m if direct_dist_m > 0 else float("inf")
            if detour > MAX_DETOUR_RATIO:
                continue
            if not _is_forward_route(polyline, point_a, point_b):
                continue

            if best is None or dist_m < best["distance_m"]:
                best = {
                    "ordered_corners": ordered,
                    "waypoints": waypoints,
                    "polyline": polyline,
                    "distance_m": dist_m,
                    "detour_ratio": round(detour, 3),
                }

        if best is None:
            continue

        seen_sets.add(key)
        path_id = f"path_{len(paths) + 1:03d}"
        ordered = best["ordered_corners"]
        images = [c.get("image") for c in ordered if c.get("image")]
        corner_score_rows = scores_by_image.loc[
            [img for img in images if img in scores_by_image.index]
        ] if images else pd.DataFrame()

        missing = [img for img in images if img not in scores_by_image.index]
        vibe_totals = _aggregate_scores(corner_score_rows, vibes, SCORE_AGGREGATION)
        conf_mean = (
            float(corner_score_rows["confidence"].mean())
            if not corner_score_rows.empty and "confidence" in corner_score_rows.columns
            else None
        )
        corner_details = [_corner_vibe_details(c, vibes, scores_by_image) for c in ordered]

        path_rec = {
            "path_id": path_id,
            "corner_ids": [c["id"] for c in ordered],
            "corner_images": images,
            "corners": corner_details,
            "distance_m": round(best["distance_m"], 1),
            "detour_ratio": best["detour_ratio"],
            "route_polyline": [[lat, lng] for lat, lng in best["polyline"]],
            "vibe_scores": {v: round(vibe_totals[v], 8) for v in vibes},
            "score_aggregation": SCORE_AGGREGATION,
            "mean_confidence": round(conf_mean, 6) if conf_mean is not None else None,
            "missing_score_images": missing,
        }
        _apply_route_color_mode(path_rec, vibes, route_color_mode, baselines=cat_baselines)
        paths.append(path_rec)
        tag = path_rec.get("path_category_label") or path_rec.get("dominant_vibe") or "?"
        print(
            f"  [{len(paths)}/{num_paths}] {path_id}  "
            f"{best['distance_m']:.0f} m  detour={best['detour_ratio']:.2f}  "
            f"type={tag}  corners={','.join(c['id'] for c in ordered)}"
        )

    if len(paths) < num_paths:
        print(f"Warning: only found {len(paths)}/{num_paths} valid paths after {attempts} attempts.")

    if route_color_mode == "category":
        cat_counts = Counter(p.get("path_category") for p in paths)
        print("Path category counts:", dict(cat_counts))

    result = {
        "manifest_path": os.path.abspath(manifest_path),
        "scores_folder": os.path.abspath(scores_folder),
        "point_a": manifest["point_a"],
        "point_b": manifest["point_b"],
        "num_paths_requested": num_paths,
        "num_paths_found": len(paths),
        "corners_per_path": corners_per_path,
        "direct_distance_m": round(direct_dist_m, 1),
        "max_detour_ratio": MAX_DETOUR_RATIO,
        "min_corner_separation_m": MIN_CORNER_SEPARATION_M,
        "score_aggregation": SCORE_AGGREGATION,
        "route_color_mode": route_color_mode,
        "category_baseline_lambda": CATEGORY_BASELINE_LAMBDA,
        "category_baselines": (
            {k: round(v, 8) for k, v in cat_baselines.items()} if cat_baselines else None
        ),
        "population_vibe_mean": {k: round(pop_mean[k], 8) for k in vibes},
        "path_categories": {
            cid: {
                "label": PATH_CATEGORIES[cid]["label"],
                "short": PATH_CATEGORIES[cid]["short"],
                "color": PATH_CATEGORIES[cid]["color"],
                "highlight": PATH_CATEGORIES[cid]["highlight"],
                "suppress": PATH_CATEGORIES[cid]["suppress"],
            }
            for cid in PATH_CATEGORIES
        },
        "vibes": vibes,
        "paths": paths,
    }

    manifest_out = os.path.join(out_dir, "paths_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    score_rows = []
    for p in paths:
        row = {
            "path_id": p["path_id"],
            "distance_m": p["distance_m"],
            "detour_ratio": p["detour_ratio"],
            "path_category": p.get("path_category"),
            "path_category_label": p.get("path_category_label"),
            "path_category_score": p.get("route_color_metric"),
            "dominant_vibe": p.get("dominant_vibe"),
            "route_color_mode": p.get("route_color_mode"),
            "corner_ids": ";".join(p["corner_ids"]),
            "mean_confidence": p.get("mean_confidence"),
        }
        cat_scores = p.get("path_category_scores") or {}
        for cid in PATH_CATEGORIES:
            row[f"category_{cid}"] = cat_scores.get(cid)
        row.update(p["vibe_scores"])
        score_rows.append(row)
    scores_out = os.path.join(out_dir, "paths_scores.csv")
    pd.DataFrame(score_rows).to_csv(scores_out, index=False)

    map_out = os.path.join(out_dir, "paths_map.html")
    _write_paths_map(
        result, map_out,
        scores_by_image=scores_by_image,
        scores_df=scores_df,
        route_color_mode=route_color_mode,
        baselines=cat_baselines,
    )

    if CATEGORY_WINDROSE_ENABLED:
        windrose_out = os.path.join(out_dir, "category_windroses.html")
        criteria_csv = os.path.join(out_dir, "category_criteria.csv")
        report = _write_category_windroses(
            result, windrose_out,
            scores_by_image=scores_by_image,
            scores_df=scores_df,
            route_color_mode=route_color_mode,
            baselines=cat_baselines,
        )
        _write_category_criteria_csv(report, criteria_csv)

    print("Saved:", manifest_out)
    print("Saved:", scores_out)
    print("Saved:", map_out)
    if CATEGORY_WINDROSE_ENABLED:
        print("Saved:", windrose_out)
        print("Saved:", criteria_csv)
    return result


def _write_paths_map(
    data: dict,
    out_path: str,
    scores_by_image: pd.DataFrame | None = None,
    scores_df: pd.DataFrame | None = None,
    route_color_mode: str | None = None,
    baselines: dict[str, float] | None = None,
) -> None:
    pa = data["point_a"]
    pb = data["point_b"]
    paths = data["paths"]
    vibes = data.get("vibes", [])
    mode = route_color_mode or data.get("route_color_mode", ROUTE_COLOR_MODE)
    mode_label = ROUTE_COLOR_MODE_LABELS.get(mode, mode)
    if baselines is None and mode == "category":
        baselines = data.get("category_baselines")
        if baselines is None and scores_df is not None:
            baselines = _category_baselines(_population_vibe_mean(scores_df, vibes))
    if scores_by_image is not None and vibes:
        _enrich_paths_for_map(
            paths, vibes, scores_by_image,
            route_color_mode=mode, baselines=baselines,
        )
    else:
        for p in paths:
            _apply_route_color_mode(p, vibes, mode, baselines=baselines)

    category_counts = {cid: 0 for cid in PATH_CATEGORIES}
    for p in paths:
        cat = p.get("path_category")
        if cat in category_counts:
            category_counts[cat] += 1

    all_lats = [pa["lat"], pb["lat"]]
    all_lngs = [pa["lng"], pb["lng"]]
    for p in paths:
        for pt in p["route_polyline"]:
            all_lats.append(pt[0])
            all_lngs.append(pt[1])
    center = [sum(all_lats) / len(all_lats), sum(all_lngs) / len(all_lngs)]

    map_paths = []
    for p in paths:
        vibe_lines = "<br>".join(
            f"{v}: {p['vibe_scores'].get(v, 0):.5f}" for v in vibes[:8]
        )
        if len(vibes) > 8:
            vibe_lines += "<br>…"

        if mode == "category":
            cat = p.get("path_category")
            cat_def = PATH_CATEGORIES.get(cat, {}) if cat else {}
            color = cat_def.get("color", MAP_UNKNOWN_COLOR)
            cat_label = p.get("path_category_label") or cat or "unknown"
            cat_scores = p.get("path_category_scores") or {}
            cat_lines = "<br>".join(
                f"{PATH_CATEGORIES[cid]['short']}: {cat_scores.get(cid, 0):.5f}"
                for cid in PATH_CATEGORIES
            )
            color_line = (
                f"<br>Path type: <span style='color:{color}'><b>{cat_label}</b></span>"
                + (f" (score {p.get('route_color_metric', 0):.5f})" if p.get("route_color_metric") is not None else "")
            )
        else:
            dom = p.get("dominant_vibe")
            color = _vibe_color(dom)
            metric = p.get("route_color_metric")
            color_line = (
                f"<br>Route color: <span style='color:{color}'><b>{dom}</b></span>"
                + (f" ({mode_label}: {metric:.5f})" if dom and metric is not None else "")
            ) if dom else ""

        popup = (
            f"<b>{p['path_id']}</b>{color_line}"
            + f"<br>Distance: {p['distance_m']:.0f} m<br>"
            f"Detour: {p['detour_ratio']:.2f}×<br>"
            f"Corners: {', '.join(p['corner_ids'])}<br>"
            + (f"<hr><b>Category scores</b><br>{cat_lines}<br>" if mode == "category" else "")
            + f"<hr><b>Vibes ({p['score_aggregation']})</b><br>{vibe_lines}"
        )
        corners = []
        for c in p.get("corners", []):
            cdom = c.get("dominant_vibe")
            ccolor = _vibe_color(cdom)
            corners.append({
                "lat": c["lat"],
                "lng": c["lng"],
                "id": c["id"],
                "color": ccolor,
                "dominant_vibe": cdom,
                "popup": (
                    f"<b>{c['id']}</b>"
                    + (f" · <span style='color:{ccolor}'>{cdom}</span>" if cdom else "")
                    + (f"<br>{', '.join(c.get('street_names', [])[:2])}" if c.get("street_names") else "")
                ),
            })
        map_paths.append({
            "id": p["path_id"],
            "color": color,
            "path_category": p.get("path_category"),
            "path_category_label": p.get("path_category_label"),
            "dominant_vibe": p.get("dominant_vibe"),
            "polyline": p["route_polyline"],
            "corners": corners,
            "popup": popup,
        })

    category_legend = [
        {
            "id": cid,
            "label": PATH_CATEGORIES[cid]["label"],
            "short": PATH_CATEGORIES[cid]["short"],
            "color": PATH_CATEGORIES[cid]["color"],
            "count": category_counts.get(cid, 0),
        }
        for cid in PATH_CATEGORIES
    ]

    payload = json.dumps({
        "center": center,
        "point_a": [pa["lat"], pa["lng"]],
        "point_b": [pb["lat"], pb["lng"]],
        "paths": map_paths,
        "category_legend": category_legend if mode == "category" else [],
        "route_weight": MAP_ROUTE_WEIGHT,
        "route_opacity": MAP_ROUTE_OPACITY,
        "corner_radius": MAP_CORNER_RADIUS,
        "animation_enabled": MAP_ANIMATION_ENABLED,
        "animation_duration_ms": MAP_ANIM_DURATION_MS,
        "animation_stagger_ms": MAP_ANIM_STAGGER_MS,
        "animation_loop": MAP_ANIM_LOOP,
        "animation_head_radius": MAP_ANIM_HEAD_RADIUS,
        "info": (
            f"{len(paths)} paths · {data.get('corners_per_path', '?')} corners each · "
            f"routes: {mode_label} · corners: dominant vibe"
        ),
        "route_color_mode": mode,
        "route_color_mode_label": mode_label,
    })

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sampled paths</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; background: #0d1117; }}
    #info {{
      position: absolute; top: 12px; right: 12px; z-index: 1000;
      background: rgba(13, 17, 23, 0.88); color: #9ca3af;
      padding: 10px 14px; border-radius: 8px;
      font: 12px/1.45 system-ui, -apple-system, sans-serif;
      border: 1px solid rgba(255,255,255,0.06);
      max-width: 280px; backdrop-filter: blur(6px);
    }}
    #legend {{
      position: absolute; bottom: 24px; left: 12px; z-index: 1000;
      background: rgba(13, 17, 23, 0.88); color: #d1d5db;
      padding: 10px 12px; border-radius: 8px;
      font: 11px system-ui, -apple-system, sans-serif;
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
      max-width: 220px;
    }}
    #legend h4 {{
      margin: 0 0 8px; font-size: 11px; font-weight: 600;
      color: #6b7280; letter-spacing: 0.04em; text-transform: uppercase;
    }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
    .swatch {{ width: 16px; height: 3px; border-radius: 2px; flex-shrink: 0; }}
    .leaflet-container {{ background: #0d1117; }}
    #controls {{
      position: absolute; top: 12px; left: 12px; z-index: 1000;
      background: rgba(13, 17, 23, 0.88); color: #d1d5db;
      padding: 10px 12px; border-radius: 8px;
      font: 12px system-ui, -apple-system, sans-serif;
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
      display: flex; gap: 8px; align-items: center;
    }}
    #controls button {{
      background: rgba(255,255,255,0.08); color: #e5e7eb;
      border: 1px solid rgba(255,255,255,0.12); border-radius: 6px;
      padding: 5px 10px; font: inherit; cursor: pointer;
    }}
    #controls button:hover {{ background: rgba(255,255,255,0.14); }}
    #controls button.active {{ background: rgba(56,189,248,0.25); border-color: #38bdf8; }}
  </style>
</head>
<body>
  <div id="controls">
    <button id="btn-play" type="button">▶ Play</button>
    <button id="btn-pause" type="button">⏸ Pause</button>
    <button id="btn-restart" type="button">↺ Restart</button>
  </div>
  <div id="info"></div>
  <div id="legend"><h4 id="legend-title">Path type</h4></div>
  <div id="map"></div>
  <script>
    const DATA = {payload};
    const map = L.map("map", {{ zoomControl: true, attributionControl: true }}).setView(DATA.center, 14);

    L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png", {{
      maxZoom: 19,
      subdomains: "abcd",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    }}).addTo(map);

    document.getElementById("info").textContent = DATA.info;

    L.circleMarker(DATA.point_a, {{
      radius: 7, color: "rgba(255,255,255,0.35)", weight: 1,
      fillColor: "#22c55e", fillOpacity: 0.85,
    }}).bindPopup("Point A").addTo(map);
    L.circleMarker(DATA.point_b, {{
      radius: 7, color: "rgba(255,255,255,0.35)", weight: 1,
      fillColor: "#ef4444", fillOpacity: 0.85,
    }}).bindPopup("Point B").addTo(map);

    const bounds = [DATA.point_a, DATA.point_b];
    const legend = document.getElementById("legend");
    const legendTitle = document.getElementById("legend-title");

    if (DATA.category_legend && DATA.category_legend.length) {{
      legendTitle.textContent = "Path type";
      DATA.category_legend.forEach(item => {{
        const el = document.createElement("div");
        el.className = "leg-item";
        el.innerHTML =
          '<span class="swatch" style="background:' + item.color + '"></span>' +
          item.short + ' <span style="color:#6b7280">(' + item.count + ')</span>';
        legend.appendChild(el);
      }});
    }}

    DATA.paths.forEach(p => {{
      L.polyline(p.polyline, {{
        color: p.color,
        weight: DATA.route_weight,
        opacity: DATA.route_opacity,
        lineCap: "round",
        lineJoin: "round",
      }}).bindPopup(p.popup).addTo(map);

      p.corners.forEach(c => {{
        L.circleMarker([c.lat, c.lng], {{
          radius: DATA.corner_radius,
          color: "rgba(255,255,255,0.25)",
          weight: 1,
          fillColor: c.color,
          fillOpacity: 0.82,
        }}).bindPopup(c.popup).bindTooltip(c.id, {{ permanent: false, opacity: 0.75 }}).addTo(map);
        bounds.push([c.lat, c.lng]);
      }});
      p.polyline.forEach(pt => bounds.push(pt));
    }});

    if (bounds.length) map.fitBounds(bounds, {{ padding: [48, 48] }});

    // --- Travel animation along each path (A → B) ---
    if (DATA.animation_enabled) {{
      const animPaths = [];

      function segLen(a, b) {{
        const R = 6371000;
        const lat1 = a[0] * Math.PI / 180, lat2 = b[0] * Math.PI / 180;
        const dlat = lat2 - lat1, dlon = (b[1] - a[1]) * Math.PI / 180;
        const x = Math.sin(dlat / 2) ** 2 +
          Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;
        return 2 * R * Math.asin(Math.sqrt(x));
      }}

      function buildCumulative(points) {{
        const cum = [0];
        for (let i = 1; i < points.length; i++) {{
          cum.push(cum[i - 1] + segLen(points[i - 1], points[i]));
        }}
        return {{ cum, total: cum[cum.length - 1] || 0 }};
      }}

      function sliceToDistance(points, cum, dist) {{
        if (!points.length) return [];
        if (dist <= 0) return [points[0]];
        if (dist >= cum[cum.length - 1]) return points.slice();
        for (let i = 1; i < points.length; i++) {{
          if (cum[i] >= dist) {{
            const seg = cum[i] - cum[i - 1];
            const t = seg > 0 ? (dist - cum[i - 1]) / seg : 0;
            const lat = points[i - 1][0] + t * (points[i][0] - points[i - 1][0]);
            const lng = points[i - 1][1] + t * (points[i][1] - points[i - 1][1]);
            return points.slice(0, i).concat([[lat, lng]]);
          }}
        }}
        return points.slice();
      }}

      DATA.paths.forEach((p, idx) => {{
        const pts = p.polyline;
        if (!pts || pts.length < 2) return;
        const {{ cum, total }} = buildCumulative(pts);
        const progLine = L.polyline([pts[0]], {{
          color: p.color,
          weight: DATA.route_weight + 0.5,
          opacity: Math.min(0.95, DATA.route_opacity + 0.42),
          lineCap: "round",
          lineJoin: "round",
        }}).addTo(map);
        const head = L.circleMarker(pts[0], {{
          radius: DATA.animation_head_radius,
          color: "rgba(255,255,255,0.45)",
          weight: 1,
          fillColor: p.color,
          fillOpacity: 0.95,
        }}).bindTooltip(p.id, {{ permanent: false, opacity: 0.85 }}).addTo(map);
        animPaths.push({{
          id: p.id,
          points: pts,
          cum,
          total,
          progLine,
          head,
          phase: idx * DATA.animation_stagger_ms,
        }});
      }});

      let running = true;
      let startTs = null;
      let pausedAt = 0;

      function setRunning(on) {{
        running = on;
        document.getElementById("btn-play").classList.toggle("active", on);
        document.getElementById("btn-pause").classList.toggle("active", !on);
        if (on) {{
          startTs = performance.now() - pausedAt;
          requestAnimationFrame(tick);
        }}
      }}

      function restartAnim() {{
        pausedAt = 0;
        startTs = performance.now();
        animPaths.forEach(a => {{
          a.progLine.setLatLngs([a.points[0]]);
          a.head.setLatLng(a.points[0]);
        }});
        if (!running) setRunning(true);
        else requestAnimationFrame(tick);
      }}

      function tick(now) {{
        if (!running) return;
        if (startTs === null) startTs = now;
        const elapsed = now - startTs;
        animPaths.forEach(a => {{
          const local = Math.max(0, elapsed - a.phase);
          let frac = local / DATA.animation_duration_ms;
          if (DATA.animation_loop) frac = frac % 1;
          else frac = Math.min(1, frac);
          const dist = frac * a.total;
          const seg = sliceToDistance(a.points, a.cum, dist);
          a.progLine.setLatLngs(seg);
          if (seg.length) a.head.setLatLng(seg[seg.length - 1]);
        }});
        requestAnimationFrame(tick);
      }}

      document.getElementById("btn-play").addEventListener("click", () => setRunning(true));
      document.getElementById("btn-pause").addEventListener("click", () => {{
        if (running) {{
          pausedAt = performance.now() - (startTs || 0);
          running = false;
          document.getElementById("btn-play").classList.remove("active");
          document.getElementById("btn-pause").classList.add("active");
        }}
      }});
      document.getElementById("btn-restart").addEventListener("click", restartAnim);

      setRunning(true);
    }} else {{
      document.getElementById("controls").style.display = "none";
    }}
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample A→B paths through random corner subsets.")
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--scores-folder", default=SCORES_FOLDER)
    ap.add_argument("--output", default=OUTPUT_FOLDER)
    ap.add_argument("--num-paths", type=int, default=NUM_PATHS)
    ap.add_argument("--corners-per-path", type=int, default=CORNERS_PER_PATH)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument(
        "--route-color-mode",
        default=ROUTE_COLOR_MODE,
        choices=sorted(ROUTE_COLOR_MODES),
        help="Route color: 'category' (4 path types) or legacy single-vibe modes.",
    )
    ap.add_argument("--open", action="store_true", help="Open the map in a browser.")
    ap.add_argument(
        "--nodes-only",
        action="store_true",
        help="Visualize corner pool + sentiment only (no path sampling). Default when ROUTES_ENABLED is False.",
    )
    ap.add_argument(
        "--map-only",
        action="store_true",
        help="Regenerate paths_map.html from existing paths_manifest.json (requires ROUTES_ENABLED).",
    )
    ap.add_argument(
        "--routes",
        action="store_true",
        help="Force path sampling even if ROUTES_ENABLED is False.",
    )
    args = ap.parse_args()

    manifest = args.manifest if os.path.isabs(args.manifest) else str(_STUDY_DIR / args.manifest)
    scores = args.scores_folder if os.path.isabs(args.scores_folder) else str(_STUDY_DIR / args.scores_folder)
    if args.output:
        out = args.output if os.path.isabs(args.output) else str(_STUDY_DIR / args.output)
    else:
        out = str(Path(manifest).parent / "sampled_paths")

    nodes_mode = args.nodes_only or (not ROUTES_ENABLED and not args.routes)

    if nodes_mode:
        visualize_nodes(manifest, scores, out)
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(os.path.join(out, 'nodes_map.html'))}")
        return

    if args.map_only:
        manifest_path = os.path.join(out, "paths_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Not found: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        scores_df, vibes_loaded = _load_scores(scores)
        _write_paths_map(
            data,
            os.path.join(out, "paths_map.html"),
            scores_by_image=scores_df.set_index("image"),
            scores_df=scores_df,
            route_color_mode=args.route_color_mode,
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        cat_counts = Counter(p.get("path_category") for p in data.get("paths", []))
        print("Path category counts:", dict(cat_counts))
        print("Regenerated:", os.path.join(out, "paths_map.html"))
        if CATEGORY_WINDROSE_ENABLED:
            windrose_out = os.path.join(out, "category_windroses.html")
            criteria_csv = os.path.join(out, "category_criteria.csv")
            report = _write_category_windroses(
                data, windrose_out,
                scores_by_image=scores_df.set_index("image"),
                scores_df=scores_df,
                route_color_mode=args.route_color_mode,
            )
            _write_category_criteria_csv(report, criteria_csv)
            print("Regenerated:", windrose_out)
            print("Regenerated:", criteria_csv)
        if args.open:
            webbrowser.open(f"file://{os.path.abspath(os.path.join(out, 'paths_map.html'))}")
        return

    sample_paths(
        manifest,
        scores,
        out,
        num_paths=args.num_paths,
        corners_per_path=args.corners_per_path,
        seed=args.seed,
        route_color_mode=args.route_color_mode,
    )
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(os.path.join(out, 'paths_map.html'))}")


if __name__ == "__main__":
    main()
