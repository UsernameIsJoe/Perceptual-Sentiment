#!/usr/bin/env python3
"""
Regenerate ``OUTPUT_FOLDER/previews_clipseg/*.png`` from saved CSVs + frames on disk.

Restores contrib / softmax / raw vectors from ``clipseg_topk_prompts.csv`` and ``image_path`` from
``vibe_scores_per_image.csv`` (same ordering as ``cfg.PROMPT_LIBRARY_CSV``). Then runs CLIPSeg on the
top contributing prompts per image to draw overlays — identical flow to ``run_study``, without the
full scoring loop or vibe CSV churn.

Outputs still respect ``HEAT_CMAP`` / ``HEAT_ALPHA`` / ``HEAT_OVERLAY_*`` in ``config.py``.
"""

from __future__ import annotations

import argparse
import os

from config import cfg, PREVIEW_DIRNAME
from pipeline import build_results_by_image_from_export_csvs
from visualization import export_prompt_previews_cached


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate previews_clipseg PNGs using CSV restores + CLIPSeg overlays.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Study output directory (default: cfg.OUTPUT_FOLDER = {cfg.OUTPUT_FOLDER})",
    )
    parser.add_argument("--top-m", type=int, default=6, help="How many prompts to panel per preview.")
    parser.add_argument(
        "--score-res",
        type=int,
        default=None,
        help="CLIPSeg internal resolution edge (default: cfg.SCORE_RES).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Saved preview PNG DPI.")
    args = parser.parse_args()
    root = args.output_dir or cfg.OUTPUT_FOLDER
    results_by_image, all_prompts = build_results_by_image_from_export_csvs(root)
    if not results_by_image:
        print(
            "[run_previews_clipseg] No images loaded "
            "(empty clipseg CSV or missing image_path merges). Aborting.",
        )
        return

    preview_dir = os.path.join(root, PREVIEW_DIRNAME)
    export_prompt_previews_cached(
        results_by_image,
        all_prompts,
        preview_dir,
        top_m=int(args.top_m),
        score_res=args.score_res if args.score_res is not None else getattr(cfg, "SCORE_RES", 352),
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
