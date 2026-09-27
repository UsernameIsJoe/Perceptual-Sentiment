"""
Configuration for the street-corner perceptual sentiment pipeline.

- IMAGE_FOLDER: input frames (read-only).
- OUTPUT_FOLDER: generated CSVs, vibe heatmaps, previews, and report PNGs.
- PROMPT_LIBRARY_CSV: prompt-to-vibe weights used by scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
STUDY_DIR = Path(__file__).resolve().parent


def resolve_data_path(path: str) -> str:
    """Resolve a path stored relative to this repo. Absolute paths are unchanged."""
    p = Path(path)
    if not p.is_absolute():
        p = STUDY_DIR / p
    return str(p)


@dataclass
class CFG:
    MODEL_ID: str = "CIDAS/clipseg-rd64-refined"
    DEVICE: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    # Input images only (read). All CSVs, maps, previews, and reports go to OUTPUT_FOLDER.
    IMAGE_FOLDER: str = str(STUDY_DIR / "streetview_corners")
    OUTPUT_FOLDER: str = str(STUDY_DIR / "outputs")
    K_TOP: int = 12
    PROMPT_CONTRIB_CAP: float = 0.002  # set None to disable, or tune 0.02–0.10

    # speed/quality
    SCORE_RES: int = 352  # scoring resolution (lower = faster)
    MAX_PROMPTS_PER_BATCH: int = 16

    # scoring rule (legacy field; prominence path uses TOP_PCT_5 / TOP_PCT_20)
    TOP_PCT: float = 0.05  # use mean of top 5% pixels from the relevance map

    # Prompt ↔ vibe library (v06, the library this scored run was built with)
    PROMPT_LIBRARY_CSV: str = str(STUDY_DIR / "data" / "prompt_vibe_library_v06.csv")


cfg = CFG()
cfg.DEVICE = "cpu"

# === Pipeline update config (multi-quantile + softmax + confidence) ===
cfg.TOP_PCT_5 = 0.05
cfg.TOP_PCT_20 = 0.20
cfg.EPS_PROM = 1e-6
cfg.ALPHA_PROM = 1.5
cfg.SOFTMAX_T = 1.0
cfg.ENT_TOPK = 10

# ---- Runner / export toggles (from notebook) ----
SAVE_VIBE_TOP1_MAPS = True
VIBE_MAPS_DIRNAME = "clipseg_vibe_top1_maps"
VIBE_MAP_SCORE_RES = getattr(cfg, "SCORE_RES", 352)
VIBE_MAP_DPI = 150  # not used for maps, kept for compatibility

PREVIEW_DIRNAME = "previews_clipseg"  # overlays use HEAT_CMAP / HEAT_ALPHA / HEAT_OVERLAY_* (see visualization.export_prompt_previews_cached)
REPORTS_DIRNAME = "per_image_reports"
DISTRICT_TIMELINE_PNG = "district_vibes_timeline.png"
TOP_CONF_HIGHLIGHT_PNG = "top_confidence_spaced_highlight.png"

# SegFormer (notebook cell)
SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# District plot (notebook cell 29)
SMOOTH_WINDOW = 7
USE_NORMALIZED = True
USE_WEIGHTED_SMOOTH = True
COLOR_SCHEME = [
    "#66c2a5",
    "#8da0cb",
    "#fc8d62",
    "#a6d854",
    "#80b1d3",
    "#b3b3b3",
    "#ffd92f",
    "#bebada",
    "#e78ac3",
]

# Fixed hue per vibe for timeline plots (same soft palette tone as COLOR_SCHEME).
# District-selected vibes keep their established slots; remaining vibes use slots 5–8.
TIMELINE_VIBE_COLORS: dict[str, str] = {
    "energetic":    COLOR_SCHEME[0],
    "calm":         COLOR_SCHEME[1],
    "open":         COLOR_SCHEME[2],
    "disorienting": COLOR_SCHEME[3],
    "legible":      COLOR_SCHEME[4],
    "compressed":   COLOR_SCHEME[5],
    "excluding":    COLOR_SCHEME[6],
    "exploratory":  COLOR_SCHEME[7],
    "inviting":     COLOR_SCHEME[8],
}

# Per-image reports (notebook cell 31)
TOP_PROMPTS_N = 5
WRAP_WIDTH = 62
HEAT_ALPHA = 0.65
HEAT_CMAP = "plasma"
# Per-image reports: remap saved vibe maps before colormap — suppress weak pixels, sharpen peaks (layout-free).
# Grayscale PNGs are min–max normalized per frame in pipeline.save maps; tuning here avoids re-running CLIPSeg.
HEAT_OVERLAY_MASK_BELOW_QUANTILE = 0.7  # 0 disables. Zero out intensities below this frame-wise quantile, then rescale remainder to [0,1].
HEAT_OVERLAY_POWER = 1.25  # >1 emphasizes strongest areas after masking; 1.0 = linear.
CHANNEL_MAX_COLS = 5
FIG_W = 18
FIG_H = 10
REPORT_DPI = 200

MIN_VIBES = 5

# Non-vibe columns in vibe_scores_per_image.csv / district_selected_vibes.csv exports.
SCORES_META_COLUMNS = frozenset({
    "image", "image_path", "confidence",
    "lat", "lng", "heading", "osm_node_id",
})
