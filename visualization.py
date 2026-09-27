"""
Figures and image exports from Prompt Scoring clean up.ipynb (same layout / style).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec

import matplotlib.colors as mcolors
from PIL import Image
from textwrap import wrap

from study_progress import tqdm

from config import (
    cfg,
    PREVIEW_DIRNAME,
    REPORTS_DIRNAME,
    DISTRICT_TIMELINE_PNG,
    TOP_CONF_HIGHLIGHT_PNG,
    VIBE_MAPS_DIRNAME,
    SMOOTH_WINDOW,
    USE_NORMALIZED,
    USE_WEIGHTED_SMOOTH,
    COLOR_SCHEME,
    TOP_PROMPTS_N,
    TIMELINE_VIBE_COLORS,
    WRAP_WIDTH,
    HEAT_ALPHA,
    HEAT_CMAP,
    HEAT_OVERLAY_MASK_BELOW_QUANTILE,
    HEAT_OVERLAY_POWER,
    FIG_W,
    FIG_H,
    REPORT_DPI,
    SCORES_META_COLUMNS,
)
import pipeline as pl
from pipeline import load_rgb, get_mask_for_prompt, _normalize_map

# Sentinels / report-text helpers -------------------------------------------------
_SEGMENT_GAP = object()


def _vibe_colors_for_timeline(selected_vibes: list[str]) -> dict[str, str]:
    """Fixed soft-palette hue per vibe name (config.TIMELINE_VIBE_COLORS)."""
    return {
        v: TIMELINE_VIBE_COLORS.get(v, COLOR_SCHEME[i % len(COLOR_SCHEME)])
        for i, v in enumerate(selected_vibes)
    }


def _load_primary_vibes_by_prompt_library(library_path: str) -> dict[str, str]:
    """prompt_id → vibe with highest (w_1, w_2, w_3) in PROMPT_LIBRARY_CSV."""
    df = pd.read_csv(library_path)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["prompt_id"])
        wmap = {
            str(row["vibe_1"]): float(row["w_1"]),
            str(row["vibe_2"]): float(row["w_2"]),
            str(row["vibe_3"]): float(row["w_3"]),
        }
        out[pid] = max(wmap.items(), key=lambda kv: kv[1])[0]
    return out


def _meta_line_hex(fill_hex: str) -> str:
    """Slightly subdued line for caption under a vibe heading (readable on white)."""
    r, g, b = mcolors.to_rgb(fill_hex)
    r = r * 0.82 + 0.18 * 0.29
    g = g * 0.82 + 0.18 * 0.29
    b = b * 0.82 + 0.18 * 0.29
    return mcolors.to_hex((min(1.0, r), min(1.0, g), min(1.0, b)))


def _prep_report_heatmap_overlay(raw_01: np.ndarray) -> np.ndarray:
    """Emphasise strong CLIPSeg activations ([0,1] maps).

    Same remap as per-image reports: respects ``HEAT_OVERLAY_*`` / ``HEAT_CMAP`` / ``HEAT_ALPHA`` in callers.
    """
    v = np.asarray(raw_01, dtype=np.float32)
    q = HEAT_OVERLAY_MASK_BELOW_QUANTILE
    if q > 0.0:
        thr = float(np.quantile(v, q))
        v = np.clip((v - thr) / np.maximum(1.0 - thr, 1e-6), 0.0, 1.0)
    pwr = HEAT_OVERLAY_POWER
    if pwr != 1.0:
        v = np.power(v, np.float32(pwr))
    return v


def _flush_report_segments(ax, segments: list, *, wrap_width_chars: int) -> None:
    """Render stacked left-aligned prose in axes coordinates.

    Each entry is ``_SEGMENT_GAP`` or
    ``(text, fontsize_pt, weight, hex_color, indent_axes_frac)``.
    """
    y = 1.01
    base_x_margin = 0.02

    def _advance_line(ft: float):
        nonlocal y
        dy = max(2.05e-3, ft * 1.95e-3)
        y -= dy

    for segment in segments:
        if segment is _SEGMENT_GAP:
            y -= 0.011
            continue
        text_s, fz, wt, clr, indent = segment
        if text_s.strip() == "":
            continue
        x = base_x_margin + indent
        for raw in text_s.split("\n"):
            chunks = wrap(raw, width=max(26, wrap_width_chars)) or [""]
            for ln in chunks:
                ax.text(
                    x,
                    y,
                    ln,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=fz,
                    fontweight=wt,
                    color=clr,
                    family="sans-serif",
                    clip_on=False,
                )
                _advance_line(fz)
        y -= 0.0045


def visualize_topk(pil_img, top_list, show_overlays=3):
    """
    Notebook cell 15.
    top_list: list of (prompt, score) sorted desc
    """
    pl.init_clipseg()
    img_np = np.array(pil_img)
    K = len(top_list)

    fig = plt.figure(figsize=(14, 7))

    # Left: image
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.imshow(img_np)
    ax0.set_title("Input image")
    ax0.axis("off")

    # Right: bar chart
    ax1 = fig.add_subplot(1, 2, 2)
    labels = [p for p, _ in top_list][::-1]
    vals = [s for _, s in top_list][::-1]
    ax1.barh(range(K), vals)
    ax1.set_yticks(range(K))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_title("Top prompts (CLIPSeg prominence score)")
    ax1.set_xlabel("score (mean of top pixel pct)")
    plt.tight_layout()
    plt.show()

    # Overlays for top N
    N = min(show_overlays, len(top_list))
    fig2 = plt.figure(figsize=(14, 4*N))
    for i in range(N):
        prompt, score = top_list[i]
        mask = get_mask_for_prompt(pil_img, prompt, out_size=pil_img.size)

        ax = fig2.add_subplot(N, 2, 2*i+1)
        ax.imshow(img_np)
        ax.set_title(f"Prompt #{i+1}\n{prompt}\nscore={score:.4f}")
        ax.axis("off")

        axm = fig2.add_subplot(N, 2, 2*i+2)
        axm.imshow(img_np)
        axm.imshow(mask, alpha=0.55)  # default colormap is fine
        axm.set_title("Overlay")
        axm.axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def export_prompt_previews_cached(results_by_image, all_prompts, out_dir, top_m=6, score_res=None, dpi=150):
    """CLIPSeg overlays for top-contrib prompts per frame (notebook cell 21).

    Caller often passes ``results_by_image`` from ``run_scoring_loop`` or rebuilt via
    ``pipeline.build_results_by_image_from_export_csvs`` when using ``run_previews_clipseg.py``.
    """
    pl.init_clipseg()
    os.makedirs(out_dir, exist_ok=True)
    if score_res is None:
        score_res = getattr(cfg, "SCORE_RES", 352)

    for img_name, rec in tqdm(
        results_by_image.items(),
        total=len(results_by_image),
        desc="Preview PNGs",
        unit="img",
        dynamic_ncols=True,
    ):
        pil_img = load_rgb(rec["image_path"])
        conf = rec["confidence"]
        contrib = rec["contrib"]
        raw_scores = rec["raw_scores"]
        softmax_p = rec["softmax_p"]

        idx = np.argsort(-contrib)[:min(int(top_m), len(all_prompts))]
        top_prompts = [all_prompts[i] for i in idx]

        # resize for CLIPSeg inference overlays only
        w, h = pil_img.size
        scale = score_res / max(w, h)
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_small = pil_img.resize((new_w, new_h))
        else:
            img_small = pil_img

        inputs = pl.processor(
            text=top_prompts,
            images=[img_small] * len(top_prompts),
            padding=True,
            return_tensors="pt"
        ).to(cfg.DEVICE)

        outputs = pl.model(**inputs)
        logits = outputs.logits  # (M, Hm, Wm)

        # upsample to original size
        logits_up = F.interpolate(
            logits.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        cols = 1 + len(idx)
        fig = plt.figure(figsize=(3.2 * cols, 3.6), dpi=dpi)

        ax0 = fig.add_subplot(1, cols, 1)
        ax0.imshow(pil_img)
        ax0.axis("off")
        ax0.set_title(f"{img_name}\nconf={float(conf):.3f}", fontsize=9)

        for j, i in enumerate(idx):
            ax = fig.add_subplot(1, cols, 2 + j)
            ax.imshow(pil_img)

            heat = logits_up[j].detach().float().cpu().numpy()
            heat_n = np.asarray(_normalize_map(heat), dtype=np.float32)
            heat_n = _prep_report_heatmap_overlay(heat_n)
            ax.imshow(heat_n, cmap=HEAT_CMAP, alpha=HEAT_ALPHA, vmin=0.0, vmax=1.0)
            ax.axis("off")

            title = (
                f"{j+1:02d} contrib={float(contrib[i]):.4f}\n"
                f"p={float(softmax_p[i]):.4f} raw={float(raw_scores[i]):.2f}\n"
                f"{all_prompts[i]}"
            )
            ax.set_title(title, fontsize=7)

        out_path = os.path.join(out_dir, img_name.rsplit(".", 1)[0] + "_preview.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    print("Saved previews to:", out_dir)


def _pick_high_conf_frames_min_spacing(confidences: list[float], n_pick: int, min_sep: int) -> list[int]:
    """Greedily pick indices (sequence order) by descending confidence with pairwise |i-j| >= min_sep."""
    arr = np.asarray(confidences, dtype=np.float64)
    order = list(np.argsort(-arr))
    picked: list[int] = []
    for idx in order:
        if arr[idx] < 0:
            continue
        if len(picked) >= n_pick:
            break
        if any(abs(int(idx) - j) < min_sep for j in picked):
            continue
        picked.append(int(idx))
    picked.sort(key=lambda i: arr[i], reverse=True)
    return picked


@torch.no_grad()
def export_top_confidence_spaced_highlight(
    results_by_image,
    image_paths,
    all_prompts,
    output_dir=None,
    *,
    n_frames: int = 5,
    min_index_separation: int = 5,
    top_overlays: int = 3,
    score_res=None,
    dpi: int = 150,
):
    """One summary figure: spaced high-confidence frames, each row = frame + top-`top_overlays` CLIPSeg maps.

    Frame choice: greedy on ``confidence`` over ``image_paths`` order; any two picked indices satisfy
    ``|i-j| >= min_index_separation``. Overlay styling matches previews (HEAT_CMAP / HEAT_ALPHA /
    HEAT_OVERLAY_*).
    """
    pl.init_clipseg()
    root = output_dir if output_dir is not None else cfg.OUTPUT_FOLDER
    os.makedirs(root, exist_ok=True)
    if score_res is None:
        score_res = getattr(cfg, "SCORE_RES", 352)

    confidences: list[float] = []
    for p in image_paths:
        stem = os.path.basename(p)
        rec = results_by_image.get(stem)
        confidences.append(float(rec["confidence"]) if rec else -1.0)

    indices = _pick_high_conf_frames_min_spacing(
        confidences, n_pick=max(1, int(n_frames)), min_sep=max(1, int(min_index_separation)),
    )
    if len(indices) < n_frames:
        print(
            f"[top-confidence highlight] only {len(indices)} frame(s) meet spacing="
            f"{min_index_separation} (wanted {n_frames}); saving partial rows.",
        )
    if not indices:
        print("[top-confidence highlight] no frames selected; skipping file.")
        return None

    n_rows = len(indices)
    n_cols = 1 + max(1, int(top_overlays))
    fig = plt.figure(figsize=(3.5 * n_cols, 3.4 * n_rows), dpi=int(dpi))
    fig.subplots_adjust(top=0.98, left=0.02, right=0.98, bottom=0.02, hspace=0.16, wspace=0.05)

    tm = max(1, min(int(top_overlays), len(all_prompts)))

    for rrow, seq_ix in enumerate(indices):
        img_path = image_paths[seq_ix]
        img_name = os.path.basename(img_path)
        rec = results_by_image.get(img_name)
        if rec is None:
            continue
        pil_img = load_rgb(rec["image_path"])
        conf_f = float(rec["confidence"])
        contrib = rec["contrib"]
        raw_scores = rec["raw_scores"]
        softmax_p = rec["softmax_p"]

        p_ix = np.argsort(-contrib)[:tm]
        top_prompts = [all_prompts[i] for i in p_ix]

        w, h = pil_img.size
        scale = score_res / max(w, h)
        if scale < 1.0:
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img_small = pil_img.resize((nw, nh))
        else:
            img_small = pil_img

        inputs = pl.processor(
            text=top_prompts,
            images=[img_small] * len(top_prompts),
            padding=True,
            return_tensors="pt",
        ).to(cfg.DEVICE)
        logits = pl.model(**inputs).logits
        logits_up = F.interpolate(
            logits.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        ax0 = fig.add_subplot(n_rows, n_cols, rrow * n_cols + 1)
        ax0.imshow(pil_img)
        ax0.axis("off")
        ax0.set_title(
            f"{img_name}\nconf={conf_f:.4f}  seq#{seq_ix}",
            fontsize=8,
        )

        for j, pi in enumerate(p_ix):
            ax = fig.add_subplot(n_rows, n_cols, rrow * n_cols + 2 + j)
            ax.imshow(pil_img)
            heat = logits_up[j].detach().float().cpu().numpy()
            heat_n = np.asarray(_normalize_map(heat), dtype=np.float32)
            heat_n = _prep_report_heatmap_overlay(heat_n)
            ax.imshow(heat_n, cmap=HEAT_CMAP, alpha=HEAT_ALPHA, vmin=0.0, vmax=1.0)
            ax.axis("off")
            title = (
                f"{j+1:02d} contrib={float(contrib[pi]):.4f}\n"
                f"p={float(softmax_p[pi]):.4f} raw={float(raw_scores[pi]):.2f}\n"
                f"{all_prompts[pi]}"
            )
            ax.set_title(title, fontsize=6)

    out_path = os.path.join(root, TOP_CONF_HIGHLIGHT_PNG)
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved top-confidence highlight:", out_path)
    return out_path


def run_csv_summary_plots(image_folder: str):
    """Notebook cell 24 (VISUALIZATION CELL)."""
    prompts_csv = os.path.join(image_folder, "clipseg_topk_prompts.csv")
    pervibe_csv = os.path.join(image_folder, "clipseg_topk_per_vibe.csv")

    df_p = pd.read_csv(prompts_csv)
    df_v = pd.read_csv(pervibe_csv)

    print("Loaded:", prompts_csv, "rows:", len(df_p))
    print("Loaded:", pervibe_csv, "rows:", len(df_v))

    # ---------------------------------------
    # 1) Frequency of prompts getting selected
    # ---------------------------------------
    freq_all = (
        df_v.groupby(["prompt_id", "prompt_text"])
            .size()
            .reset_index(name="selected_count")
            .sort_values("selected_count", ascending=False)
    )

    TOP_N_FREQ = 40
    freq_plot = freq_all.head(TOP_N_FREQ)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(freq_plot)), freq_plot["selected_count"].values)
    plt.xticks(range(len(freq_plot)), freq_plot["prompt_text"].values, rotation=75, ha="right", fontsize=8)
    plt.title(f"Prompt selection frequency (Top {TOP_N_FREQ})")
    plt.ylabel("Count (times selected in per-vibe topK)")
    plt.tight_layout()
    plt.show()

    print(freq_all.head(25))

    # ---------------------------------------
    # 2) Top prompt for each vibe
    # ---------------------------------------
    top_prompt_each_vibe = (
        df_v.groupby(["vibe", "prompt_id", "prompt_text"])["weighted_score"]
            .sum()
            .reset_index()
            .sort_values(["vibe", "weighted_score"], ascending=[True, False])
    )

    top1_each_vibe = top_prompt_each_vibe.groupby("vibe").head(1).reset_index(drop=True)

    print("\nTop prompt for each vibe (by sum of weighted_score across images):")
    print(top1_each_vibe.sort_values("weighted_score", ascending=False))

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(top1_each_vibe)), top1_each_vibe["weighted_score"].values)
    plt.xticks(range(len(top1_each_vibe)), top1_each_vibe["vibe"].values, rotation=45, ha="right", fontsize=9)
    plt.title("Top prompt per vibe (winner score = sum weighted_score)")
    plt.ylabel("Sum weighted_score")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 3) Top 25 prompts and its score
    # ---------------------------------------
    global_prompt_scores = (
        df_p.groupby(["prompt_id", "prompt_text"])
            .agg(
                score_contrib_sum=("contrib", "sum"),
                score_contrib_mean=("contrib", "mean"),
                softmax_p_mean=("softmax_p", "mean"),
                raw_score_mean=("raw_score", "mean"),
                n_images=("image", "nunique"),
            )
            .reset_index()
            .sort_values("score_contrib_sum", ascending=False)
    )

    TOP_K = 25
    top25 = global_prompt_scores.head(TOP_K)

    print(f"\nTop {TOP_K} prompts (ranked by sum(contrib) across images):")
    print(top25)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(top25)), top25["score_contrib_sum"].values)
    plt.xticks(range(len(top25)), top25["prompt_text"].values, rotation=75, ha="right", fontsize=8)
    plt.title(f"Top {TOP_K} prompts by global score (sum contrib)")
    plt.ylabel("Sum contrib")
    plt.tight_layout()
    plt.show()


def run_district_vibe_timeseries_plot(image_folder: str, show: bool = True):
    """Notebook cell 29. Reads CSVs only; no CLIPSeg. Saves PNG to image_folder."""
    selected_csv_path = os.path.join(image_folder, "district_selected_vibes.csv")
    df_sel = pd.read_csv(selected_csv_path)

    select_vibes = [c for c in df_sel.columns if c not in SCORES_META_COLUMNS]

    vibe_csv_path = os.path.join(image_folder, "vibe_scores_per_image.csv")
    df_all = pd.read_csv(vibe_csv_path)

    if "confidence" not in df_all.columns:
        raise ValueError("vibe_scores_per_image.csv must contain a 'confidence' column.")

    if "image" in df_sel.columns and "image" in df_all.columns:
        df_merged = df_sel.merge(df_all[["image", "confidence"]], on="image", how="left", suffixes=("", "_all"))
    elif "image_path" in df_sel.columns and "image_path" in df_all.columns:
        df_merged = df_sel.merge(df_all[["image_path", "confidence"]], on="image_path", how="left", suffixes=("", "_all"))
    else:
        raise ValueError("Could not merge confidence. Need common key: 'image' or 'image_path' in both CSVs.")

    if df_merged["confidence"].isna().any():
        missing = df_merged[df_merged["confidence"].isna()][["image", "image_path"]].head(10)
        raise ValueError(
            "Some images have no confidence after merge. Example missing rows:\n"
            f"{missing}"
        )

    images = df_merged["image"].tolist()
    x = np.arange(len(images))

    pivot = df_merged.set_index("image")[select_vibes]

    conf = df_merged.set_index("image")["confidence"].reindex(pivot.index).astype(float)

    pivot_weighted = pivot.mul(conf, axis=0)

    if USE_NORMALIZED:
        denom = pivot_weighted.sum(axis=1).replace(0, np.nan)
        pivot_base = (pivot_weighted.T / denom).T.fillna(0.0)
        y_label = "Confidence-weighted vibe proportion"
    else:
        pivot_base = pivot_weighted
        y_label = "Confidence-weighted vibe score"

    if USE_WEIGHTED_SMOOTH:
        conf_roll = conf.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).sum().replace(0, np.nan)
        pivot_smooth = pd.DataFrame(index=pivot_base.index, columns=pivot_base.columns, dtype=float)
        for vibe in pivot_base.columns:
            num = (pivot_base[vibe] * conf).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).sum()
            pivot_smooth[vibe] = (num / conf_roll).fillna(0.0)
    else:
        pivot_smooth = pivot_base.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

    plt.figure(figsize=(14, 6))

    for i, vibe in enumerate(select_vibes):
        color = TIMELINE_VIBE_COLORS.get(vibe, COLOR_SCHEME[i % len(COLOR_SCHEME)])

        plt.plot(
            x,
            pivot_base[vibe].values,
            color=color,
            alpha=0.18,
            linewidth=1
        )

        plt.plot(
            x,
            pivot_smooth[vibe].values,
            color=color,
            linewidth=3,
            label=vibe
        )

    plt.title("District Selected Vibes (Confidence-weighted, Base + Smoothed)")
    plt.xlabel("Image index (sequence order)")
    plt.ylabel(y_label)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=9
    )

    plt.tight_layout()
    fig = plt.gcf()
    timeline_path = os.path.join(image_folder, DISTRICT_TIMELINE_PNG)
    fig.savefig(timeline_path, dpi=REPORT_DPI, bbox_inches="tight")
    print("Saved district timeline:", timeline_path)
    if show:
        plt.show()
    plt.close(fig)

    print("Selected district vibes:", select_vibes)


def generate_per_image_reports(image_folder: str | None = None):
    """Per-image PNG reports: vibes + overlays + hierarchically typed right-hand analytical column.

    Vibes use the same ``COLOR_SCHEME`` assignment as ``run_district_vibe_timeseries_plot`` (district
    CSV column order). Prompt-channel lines use that hue when the prompt’s primary library vibe matches
    a selected district vibe (see ``district_vibes_timeline.png``). Otherwise neutral gray.

    Implemented in ``run_per_image_reports.py`` as an optional rerender pass with no inference.
    """
    root = image_folder or cfg.OUTPUT_FOLDER
    reports_dir = os.path.join(root, REPORTS_DIRNAME)
    maps_root = os.path.join(root, VIBE_MAPS_DIRNAME)
    os.makedirs(reports_dir, exist_ok=True)

    selected_csv_path = os.path.join(root, "district_selected_vibes.csv")
    prompts_csv_path = os.path.join(root, "clipseg_topk_prompts.csv")
    per_vibe_csv_path = os.path.join(root, "clipseg_topk_per_vibe.csv")

    df_sel = pd.read_csv(selected_csv_path)
    df_prompts = pd.read_csv(prompts_csv_path)
    df_pervibe = pd.read_csv(per_vibe_csv_path)

    selected_vibes = [c for c in df_sel.columns if c not in SCORES_META_COLUMNS]
    if "confidence" not in df_sel.columns:
        vibe_scores_path = os.path.join(root, "vibe_scores_per_image.csv")
        df_conf = pd.read_csv(vibe_scores_path)[["image", "confidence"]]
        df_sel = df_sel.merge(df_conf, on="image", how="left")

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    COL_IMG_P = _pick_col(df_prompts, ["image"])
    COL_PROMPT_TEXT = _pick_col(df_prompts, ["prompt_text", "prompt", "text"])
    COL_PROMPT_CONTRIB = _pick_col(df_prompts, ["contrib", "weighted_contrib", "softmax_x_conf"])
    COL_PROMPT_PROB = _pick_col(df_prompts, ["softmax_p", "prob", "p"])
    COL_PROMPT_ID = _pick_col(df_prompts, ["prompt_id"])
    COL_PROMPT_RAW = _pick_col(df_prompts, ["raw_score", "score", "s_prom", "prom_score"])

    COL_IMG_V = _pick_col(df_pervibe, ["image"])
    COL_VIBE = _pick_col(df_pervibe, ["vibe"])
    COL_VIBE_SCORE = _pick_col(df_pervibe, ["vibe_score", "score"])
    COL_RANK_IN_VIBE = _pick_col(df_pervibe, ["rank_in_vibe", "rank"])
    COL_TOP_PROMPT = _pick_col(df_pervibe, ["prompt_text", "prompt", "text"])
    COL_TOP_CONTRIB = _pick_col(df_pervibe, ["contrib", "weighted_score", "prompt_contrib", "softmax_x_conf"])

    if COL_PROMPT_TEXT is None or COL_IMG_P is None:
        raise RuntimeError("clipseg_topk_prompts.csv missing required columns (image, prompt_text).")
    if COL_IMG_V is None or COL_VIBE is None or COL_RANK_IN_VIBE is None or COL_TOP_PROMPT is None:
        raise RuntimeError("clipseg_topk_per_vibe.csv missing required columns (image, vibe, rank_in_vibe, prompt_text).")

    def _load_rgb(path):
        return load_rgb(path)

    def _safe_stem(img_name):
        return os.path.splitext(os.path.basename(img_name))[0]

    def _vibe_top1_prompt(img_name, vibe):
        d = df_pervibe[(df_pervibe[COL_IMG_V] == img_name) & (df_pervibe[COL_VIBE] == vibe)].copy()
        if len(d) == 0:
            return None, None
        d = d.sort_values(COL_RANK_IN_VIBE, ascending=True)
        top = d.iloc[0]
        ptxt = str(top[COL_TOP_PROMPT])
        val = None
        if COL_TOP_CONTRIB is not None and COL_TOP_CONTRIB in d.columns:
            try:
                val = float(top[COL_TOP_CONTRIB])
            except Exception:
                val = None
        return ptxt, val

    timeline_colors = _vibe_colors_for_timeline(selected_vibes)
    primary_by_prompt: dict[str, str] = {}
    library_csv_for_hue = getattr(cfg, "PROMPT_LIBRARY_CSV", None)
    if library_csv_for_hue and os.path.isfile(str(library_csv_for_hue)):
        try:
            primary_by_prompt = _load_primary_vibes_by_prompt_library(str(library_csv_for_hue))
        except Exception as ex:
            print("per-image reports: prompt library unreadable — prompt-line hues degraded:", ex)
    prompt_hue_fallback = "#5f5f5f"

    def _hue_for_prompt_id(pid_like) -> str:
        """Timeline color when primary-library vibe is in district; else neutral gray."""
        if COL_PROMPT_ID is None:
            return prompt_hue_fallback
        pv = primary_by_prompt.get(str(pid_like))
        if pv and pv in timeline_colors:
            return timeline_colors[pv]
        return prompt_hue_fallback

    def _append_prompt_channel_sections(img_name, segments_out):
        rows = df_prompts[df_prompts[COL_IMG_P] == img_name].copy()
        segments_out.append(("II · Prompt channel (CLIPSeg)", 12.4, "bold", "#292929", 0.0))
        segments_out.append(
            ("Ranked by confidence-weighted contribution (global softmax). "
             "Colours: primary library vibe when it matches a district-series hue.",
             8.9,
             "normal",
             "#696969",
             0.0,
            )
        )
        segments_out.append(_SEGMENT_GAP)

        if len(rows) == 0:
            segments_out.append(("No prompt rows in clipseg_topk_prompts.csv for this image.", 9.9, "normal", "#878787", 0.03))
            return

        sort_col = None
        for cand in [COL_PROMPT_CONTRIB, COL_PROMPT_PROB, COL_PROMPT_RAW]:
            if cand is not None and cand in rows.columns:
                sort_col = cand
                break
        if sort_col is not None:
            rows = rows.sort_values(sort_col, ascending=False)
        rows = rows.head(TOP_PROMPTS_N)

        for ri, (_, pr) in enumerate(rows.iterrows(), start=1):
            ptxt = str(pr[COL_PROMPT_TEXT])
            pid = str(pr[COL_PROMPT_ID]) if COL_PROMPT_ID is not None and COL_PROMPT_ID in rows.columns else None
            lc = _hue_for_prompt_id(pid) if pid else prompt_hue_fallback
            segments_out.append((f"{ri}. {ptxt.strip()}", 10.2, "bold", lc, 0.03))
            parts = []
            if COL_PROMPT_CONTRIB is not None and COL_PROMPT_CONTRIB in rows.columns:
                try:
                    parts.append(f"c={float(pr[COL_PROMPT_CONTRIB]):.4f}")
                except Exception:
                    pass
            if COL_PROMPT_PROB is not None and COL_PROMPT_PROB in rows.columns:
                try:
                    parts.append(f"p={float(pr[COL_PROMPT_PROB]):.4f}")
                except Exception:
                    pass
            if COL_PROMPT_RAW is not None and COL_PROMPT_RAW in rows.columns:
                try:
                    parts.append(f"s={float(pr[COL_PROMPT_RAW]):.3f}")
                except Exception:
                    pass
            if parts:
                segments_out.append(("     " + "  ·  ".join(parts), 8.7, "normal", _meta_line_hex(lc), 0.045))
            segments_out.append(_SEGMENT_GAP)

    if COL_PROMPT_ID is None:
        print("per-image reports: clipseg_topk_prompts.csv has no prompt_id column — prompt hues use fallback gray.")

    n_total = len(df_sel)
    print("Generating per-image reports:", n_total)
    print("Selected vibes:", selected_vibes)
    print("Reports folder:", reports_dir)

    for idx, row in tqdm(
        df_sel.iterrows(),
        total=len(df_sel),
        desc="Per-image reports",
        unit="img",
        dynamic_ncols=True,
    ):
        img_name = row["image"]
        img_path = row["image_path"]
        conf = float(row["confidence"]) if "confidence" in df_sel.columns and not pd.isna(row["confidence"]) else None

        try:
            img = _load_rgb(img_path)
        except Exception as e:
            print(f"[skip] cannot open image: {img_path} ({e})")
            continue

        W, H = img.size
        stem = _safe_stem(img_name)
        map_dir = os.path.join(maps_root, stem)

        vibes_to_show = selected_vibes.copy()
        n_v = len(vibes_to_show)
        # Bottom-left overlays: fixed 2 columns when multiple vibes (larger thumbnails, fills height).
        heatmap_cols = 2 if n_v >= 2 else 1
        rows_channels = int(np.ceil(n_v / heatmap_cols))

        img_row_ratio = 2.35
        heat_row_ratio = 1.12
        # Nested layout: tweak heatmap-row gaps independently of hero↔heatmap gap.
        # - hspace_outer: gap between large top image and the heat thumbnail block only.
        # - hspace_heat_rows: gaps only between heatmap rows inside that block.
        # - hm_heat_col_gap_frac: horizontal gutter inviting|…|excluding vs energetic|… (fraction of row width).
        hspace_outer = 0.13
        hspace_heat_rows = 0.02
        hm_heat_col_gap_frac = 0.015
        fig_w_r = FIG_W + 2.4
        fig_h_r = max(FIG_H + 1.0, FIG_H + 1.25 * rows_channels)
        fig = plt.figure(figsize=(fig_w_r, fig_h_r), dpi=REPORT_DPI)
        gs_outer = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            height_ratios=[img_row_ratio, heat_row_ratio * rows_channels],
            width_ratios=[2.02, 1.22],
            hspace=hspace_outer,
            wspace=0.07,
        )
        gs_heat = GridSpecFromSubplotSpec(
            rows_channels,
            1,
            subplot_spec=gs_outer[1, 0],
            height_ratios=[heat_row_ratio] * rows_channels,
            hspace=hspace_heat_rows,
        )

        ax_img = fig.add_subplot(gs_outer[0, 0])
        ax_img.imshow(img)
        ax_img.set_axis_off()
        title = f"{img_name}"
        if conf is not None:
            title += f"   |   confidence={conf:.3f}"
        ax_img.set_title(title, fontsize=10.5, pad=4)

        ax_txt = fig.add_subplot(gs_outer[:, 1])
        ax_txt.set_axis_off()

        segs = []
        segs.append(("CLIPSeg validation · analytical report", 14.2, "bold", "#111111", 0.0))
        segs.append((stem, 10.6, "semibold", "#3a3a3a", 0.0))
        if conf is not None:
            segs.append((f"District confidence: {conf:.3f}", 10.3, "semibold", "#2c2c2c", 0.0))

        segs.append(_SEGMENT_GAP)
        segs.append(("I · Vibe decomposition", 12.4, "bold", "#222222", 0.0))
        segs.append(
            (
                "Hue matches district_vibes_timeline.png — selected vibe columns use COLOR_SCHEME "
                "in the same district CSV column order.",
                8.7,
                "normal",
                "#6a6a6a",
                0.0,
            ),
        )
        segs.append(_SEGMENT_GAP)

        for v in vibes_to_show:
            vch = timeline_colors[v]
            try:
                s_val = float(row[v])
            except Exception:
                s_val = 0.0
            ptxt, pv = _vibe_top1_prompt(img_name, v)
            headline = f"{v} · aggregate {s_val:.4f}"
            if pv is not None:
                headline += f" · top-1 contrib {pv:.4f}"
            segs.append((headline, 11.5, "bold", vch, 0.032))
            if ptxt:
                segs.append(("Evidence (per-vibe strongest prompt)", 9.6, "semibold", _meta_line_hex(vch), 0.048))
                segs.append((ptxt.strip(), 9.15, "normal", _meta_line_hex(vch), 0.052))
            segs.append(_SEGMENT_GAP)

        _append_prompt_channel_sections(img_name, segs)
        _flush_report_segments(
            ax_txt,
            segs,
            wrap_width_chars=max(38, int(WRAP_WIDTH * 1.05)),
        )

        # Finalise layout so heatmap row bboxes match left column width (flush with top image).
        fig.canvas.draw()
        _hm_top_title_room = 0.12
        _hm_bottom_pad = 0.02

        for r in range(rows_channels):
            bb = gs_heat[r, 0].get_position(fig)
            nc = heatmap_cols
            if nc > 1:
                gap_px = hm_heat_col_gap_frac * bb.width
                thumb_w = (bb.width - gap_px) / 2.0
            else:
                gap_px = 0.0
                thumb_w = bb.width

            ax_h = bb.height * (1.0 - _hm_top_title_room - _hm_bottom_pad)
            ax_y = bb.y0 + bb.height * _hm_bottom_pad

            for c in range(nc):
                idx = r * nc + c
                if idx >= n_v:
                    x0_blank = bb.x0 + c * (thumb_w + gap_px)
                    ax_blank = fig.add_axes([x0_blank, ax_y, thumb_w, ax_h])
                    ax_blank.set_axis_off()
                    continue
                vibe = vibes_to_show[idx]
                x0 = bb.x0 + c * (thumb_w + gap_px)

                ax_hm = fig.add_axes([x0, ax_y, thumb_w, ax_h])
                ax_hm.imshow(img)
                ax_hm.set_axis_off()
                vch = timeline_colors.get(vibe, "#333333")

                map_path = os.path.join(map_dir, f"{vibe}.png")
                if os.path.exists(map_path):
                    heat = Image.open(map_path).convert("L")
                    heat_np = np.array(heat).astype(np.float32) / 255.0
                    heat_np = _prep_report_heatmap_overlay(heat_np)

                    ax_hm.imshow(heat_np, cmap=HEAT_CMAP, alpha=HEAT_ALPHA, vmin=0.0, vmax=1.0)
                    ax_hm.set_title(vibe, fontsize=11.0, fontweight="bold", color=vch, pad=3)
                else:
                    ax_hm.set_title(f"{vibe} (missing map)", fontsize=10.0, fontweight="bold", color=vch, pad=3)

        out_path = os.path.join(reports_dir, f"{stem}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print("Done. Wrote per-image report PNGs to:", reports_dir)


def run_previews_and_reports(results_by_image, all_prompts, image_folder: str | None = None):
    root = image_folder or cfg.OUTPUT_FOLDER
    preview_dir = os.path.join(root, PREVIEW_DIRNAME)
    export_prompt_previews_cached(
        results_by_image, all_prompts, preview_dir,
        top_m=6, score_res=getattr(cfg, "SCORE_RES", 352),
    )
    generate_per_image_reports(root)
