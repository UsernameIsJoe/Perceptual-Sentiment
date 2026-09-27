#!/usr/bin/env python3
"""
Write ``OUTPUT_FOLDER/top_confidence_spaced_highlight.png`` from saved CSVs + frames on disk.

Uses ``clipseg_topk_prompts.csv`` via ``pipeline.build_results_by_image_from_export_csvs`` and
frame order from ``vibe_scores_per_image.csv`` (same row order as the pipeline export). Runs CLIPSeg
for overlays only.

Does not run ``run_study`` or the full scoring loop.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from config import cfg
from pipeline import build_results_by_image_from_export_csvs
from visualization import export_top_confidence_spaced_highlight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top confidence frames with spaced indices + CLIPSeg top-contrib overlays (one PNG).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Study output directory (default: cfg.OUTPUT_FOLDER = {cfg.OUTPUT_FOLDER})",
    )
    parser.add_argument("--n-frames", type=int, default=5, help="How many rows to try to select.")
    parser.add_argument(
        "--min-separation",
        type=int,
        default=5,
        help="Minimum index gap between any two selected frames (sequence order).",
    )
    parser.add_argument(
        "--top-overlays",
        type=int,
        default=3,
        help="CLIPSeg panels per row (top contrib prompts).",
    )
    parser.add_argument(
        "--score-res",
        type=int,
        default=None,
        help="CLIPSeg internal resolution edge (default: cfg.SCORE_RES).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = parser.parse_args()
    root = args.output_dir or cfg.OUTPUT_FOLDER
    vibes_csv = os.path.join(root, "vibe_scores_per_image.csv")
    if not os.path.isfile(vibes_csv):
        print(f"[run_top_confidence_highlight] missing {vibes_csv}")
        return

    df = pd.read_csv(vibes_csv)
    if "image_path" not in df.columns:
        print("[run_top_confidence_highlight] vibe_scores_per_image.csv needs image_path column.")
        return
    image_paths = df["image_path"].astype(str).tolist()

    results_by_image, all_prompts = build_results_by_image_from_export_csvs(root)
    if not results_by_image:
        print("[run_top_confidence_highlight] no score rows loaded from clipseg CSV; aborting.")
        return

    export_top_confidence_spaced_highlight(
        results_by_image,
        image_paths,
        all_prompts,
        output_dir=root,
        n_frames=int(args.n_frames),
        min_index_separation=int(args.min_separation),
        top_overlays=int(args.top_overlays),
        score_res=args.score_res if args.score_res is not None else getattr(cfg, "SCORE_RES", 352),
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
