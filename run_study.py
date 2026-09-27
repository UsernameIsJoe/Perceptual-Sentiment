#!/usr/bin/env python3
"""
Run the full study in the same order as Prompt Scoring clean up.ipynb:
  pipeline (score → CSVs → district selection) → sanity print → previews →
  CSV summary plots → district time-series plot → per-image validation reports.

You can regenerate artefacts alone once the CSV exports exist:

- ``python run_per_image_reports.py`` — validation PNGs under ``per_image_reports/``
- ``python run_previews_clipseg.py`` — ``previews_clipseg/`` (CSV restore + CLIPSeg overlays only)
- ``python run_district_timeline.py`` — ``district_vibes_timeline.png``
- ``python run_top_confidence_highlight.py`` — ``top_confidence_spaced_highlight.png``

Edit paths in config.py: IMAGE_FOLDER (inputs), PROMPT_LIBRARY_CSV, OUTPUT_FOLDER (all generated files).

If IMAGE_FOLDER contains walk_manifest.json or corners_manifest.json, vibe_scores_per_image.csv
and district_selected_vibes.csv also include lat/lng (and heading / osm_node_id when available).
"""
from __future__ import annotations

import argparse
import os

from pipeline import run_full_pipeline, sanity_check_sample
from visualization import (
    export_prompt_previews_cached,
    generate_per_image_reports,
    run_csv_summary_plots,
    run_district_vibe_timeseries_plot,
)
from config import cfg, PREVIEW_DIRNAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib interactive figures (still saves preview PNGs and report PNGs).",
    )
    args = parser.parse_args()

    results_by_image, _image_paths, _prompt_ids, all_prompts = run_full_pipeline()

    sanity_check_sample(results_by_image, all_prompts)

    preview_dir = os.path.join(cfg.OUTPUT_FOLDER, PREVIEW_DIRNAME)
    export_prompt_previews_cached(
        results_by_image,
        all_prompts,
        preview_dir,
        top_m=6,
        score_res=getattr(cfg, "SCORE_RES", 352),
    )

    if not args.skip_plots:
        run_csv_summary_plots(cfg.OUTPUT_FOLDER)
        run_district_vibe_timeseries_plot(cfg.OUTPUT_FOLDER)

    generate_per_image_reports(cfg.OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
