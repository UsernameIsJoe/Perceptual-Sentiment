#!/usr/bin/env python3
"""
Rebuild validation PNGs under ``OUTPUT_FOLDER/per_image_reports`` using existing artefacts only.

Reads the same CSVs + image paths + optional vibe overlay maps as
``visualization.generate_per_image_reports``: no CLIPSeg scoring, no change to segmentation,
outputs still go under ``cfg.OUTPUT_FOLDER/per_image_reports`` by default.

Prerequisites inside the chosen folder (matching ``run_study`` outputs):
  - district_selected_vibes.csv
  - clipseg_topk_prompts.csv
  - clipseg_topk_per_vibe.csv
  - valid ``image_path`` rows (PNG/JPG readable)
  - optional ``clipseg_vibe_top1_maps/<stem>/<vibe>.png`` overlays (heatmap strip still works without them).

Optional: ``PROMPT_LIBRARY_CSV`` from config is used only to hue ranked prompts by primary vibe so
those colours stay aligned with ``district_vibes_timeline.png`` when the vibe is in the district set.
"""

from __future__ import annotations

import argparse

from config import cfg
from visualization import generate_per_image_reports


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate CLIPSeg per-image report PNGs from saved CSV outputs (no inference).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Study output directory (default: cfg.OUTPUT_FOLDER = {cfg.OUTPUT_FOLDER})",
    )
    args = parser.parse_args()
    generate_per_image_reports(args.output_dir or cfg.OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
