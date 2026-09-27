#!/usr/bin/env python3
"""
Rebuild ``OUTPUT_FOLDER/<DISTRICT_TIMELINE_PNG>`` using existing CSV exports only.

Reads the same files as ``visualization.run_district_vibe_timeseries_plot``: no CLIPSeg, no scoring.
Default output stays under ``cfg.OUTPUT_FOLDER`` (or ``--output-dir``).

Prerequisites inside the chosen folder (matching ``run_study`` outputs):
  - district_selected_vibes.csv
  - vibe_scores_per_image.csv
"""

from __future__ import annotations

import argparse

from config import cfg, DISTRICT_TIMELINE_PNG
from visualization import run_district_vibe_timeseries_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate district vibe timeline PNG from saved CSV outputs (no inference).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Study output directory (default: cfg.OUTPUT_FOLDER = {cfg.OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=f"Show an interactive matplotlib window after saving ``{DISTRICT_TIMELINE_PNG}``.",
    )
    args = parser.parse_args()
    root = args.output_dir or cfg.OUTPUT_FOLDER
    run_district_vibe_timeseries_plot(root, show=args.show)


if __name__ == "__main__":
    main()
