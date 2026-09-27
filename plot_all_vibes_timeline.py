#!/usr/bin/env python3
"""
Plot a timeline of ALL vibes from a vibe_scores_per_image.csv (independent of config.py).

Unlike the district timeline (which plots only the selected vibes), this draws every vibe column,
using a FIXED color per vibe so the same vibe (e.g. "calm") is always the same color across
different datasets/runs. Set the data FOLDER at the top of this file (like config.OUTPUT_FOLDER);
the script reads vibe_scores_per_image.csv from inside it.

Run from the 0416 Study folder:
    python plot_all_vibes_timeline.py
    python plot_all_vibes_timeline.py --show
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_STUDY_DIR = Path(__file__).resolve().parent

# ============================================================================
# EDIT HERE
# ----------------------------------------------------------------------------
# Folder where the run's outputs live (same style as config.OUTPUT_FOLDER).
# The script reads "<DATA_FOLDER>/vibe_scores_per_image.csv".

DATA_FOLDER: str = str(_STUDY_DIR / "outputs")
# Name of the per-image scores CSV inside DATA_FOLDER.
SCORES_CSV_NAME: str = "vibe_scores_per_image.csv"
# Where to save the PNG. Leave "" to save inside DATA_FOLDER as all_vibes_timeline.png.
OUTPUT_PNG = ""
# Smoothing / weighting (mirrors the district timeline defaults).
SMOOTH_WINDOW = 10
USE_NORMALIZED = True       # plot each vibe's confidence-weighted proportion (sums to 1 per frame)
USE_WEIGHTED_SMOOTH = True  # confidence-weighted rolling average
# ============================================================================

# Fixed color per vibe — soft district palette (config.TIMELINE_VIBE_COLORS); same tone for all timelines.
from config import SCORES_META_COLUMNS, TIMELINE_VIBE_COLORS, COLOR_SCHEME

META_COLS = SCORES_META_COLUMNS


def _color_for(vibe: str, fallback_idx: int) -> str:
    if vibe in TIMELINE_VIBE_COLORS:
        return TIMELINE_VIBE_COLORS[vibe]
    return COLOR_SCHEME[fallback_idx % len(COLOR_SCHEME)]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Plot all vibes over the frame sequence with fixed colors.")
    ap.add_argument("--data-folder", default=DATA_FOLDER, help="Run output folder (default: DATA_FOLDER).")
    ap.add_argument("--output", default=OUTPUT_PNG, help="Output PNG path (default: inside data folder).")
    ap.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    args = ap.parse_args()

    data_folder = args.data_folder if os.path.isabs(args.data_folder) else os.path.join(here, args.data_folder)
    data_path = os.path.join(data_folder, SCORES_CSV_NAME)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Scores CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    vibes = [c for c in df.columns if c not in META_COLS]
    if not vibes:
        raise ValueError("No vibe columns found (need columns beyond image/image_path/confidence).")
    vibes = sorted(vibes)  # stable order → stable fallback colors and legend

    if "confidence" in df.columns:
        conf = df["confidence"].astype(float)
    else:
        conf = pd.Series(np.ones(len(df)), index=df.index)

    x = np.arange(len(df))
    pivot = df[vibes].astype(float)
    weighted = pivot.mul(conf, axis=0)

    if USE_NORMALIZED:
        denom = weighted.sum(axis=1).replace(0, np.nan)
        base = (weighted.T / denom).T.fillna(0.0)
        y_label = "Confidence-weighted vibe proportion"
    else:
        base = weighted
        y_label = "Confidence-weighted vibe score"

    if USE_WEIGHTED_SMOOTH:
        conf_roll = conf.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).sum().replace(0, np.nan)
        smooth = pd.DataFrame(index=base.index, columns=base.columns, dtype=float)
        for v in base.columns:
            num = (base[v] * conf).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).sum()
            smooth[v] = (num / conf_roll).fillna(0.0)
    else:
        smooth = base.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    for i, v in enumerate(vibes):
        color = _color_for(v, i)
        plt.plot(x, base[v].values, color=color, alpha=0.18, linewidth=1)
        plt.plot(x, smooth[v].values, color=color, linewidth=2.6, label=v)

    plt.title(f"All Vibes Timeline — {os.path.basename(os.path.normpath(data_folder))}")
    plt.xlabel("Image index (sequence order)")
    plt.ylabel(y_label)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=9)
    plt.tight_layout()

    out_png = args.output or os.path.join(data_folder, "all_vibes_timeline.png")
    if not os.path.isabs(out_png):
        out_png = os.path.join(here, out_png)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Saved:", out_png)
    print("Vibes plotted:", vibes)
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
