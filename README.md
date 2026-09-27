# Perceptual Sentimen

Street-corner perceptual sentiment scoring. CLIPSeg scores 45 prompts, those prompts map onto 9 vibes, and the result is one vibe profile per image.

This public repo contains the pipeline, the prompt library, and the Cambridge / Somerville corner locations. It does not contain Street View frames or the vibe scores computed from them.

## Layout

| Path | What it is |
|---|---|
| `streetview_corners/corners_manifest.json` | 100 intersections: lat/lng, heading, OSM id, street names, Street View panorama id |
| `streetview_corners/corners_map.html` | Map of those corners (OpenStreetMap tiles) |
| `data/prompt_vibe_library_v06.csv` | Prompt text and vibe weights |
| `config.py` | Image folder, output folder, and prompt library |

The corners lie between 42.36008, −71.08752 and 42.37549, −71.11432.

## Run

Put your own frames in `streetview_corners/` as `frame_0001.jpg` onward (names in the manifest), then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_study.py
```

Scoring uses CLIPSeg (`CIDAS/clipseg-rd64-refined`) and SegFormer ADE20K for sky/ground masks. `config.py` forces CPU. Results are written to `outputs/`.

Other entry points:

- `python plot_corners_map.py` — corner locations
- `python plot_all_vibes_timeline.py` — all nine vibes, after a scoring run
- `python sample_corner_paths.py` — path or node maps from saved scores
- `python run_district_timeline.py` — district timeline PNG only
- `python overall_performance.py` — score diagnostics

`fetch_streetview_corners.py` can download a new corner set. Set `GOOGLE_MAPS_API_KEY` or pass `--api-key`. Do not commit a key, and do not commit Street View images.
