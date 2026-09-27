#!/usr/bin/env python3
"""
Overall performance diagnostics from pipeline CSVs in OUTPUT_FOLDER:

- Vibe correlation matrix (raw + per-image z-score across vibes)
- Prompt correlation (across images)
- Prompt score distributions (raw / softmax / contrib)
- Per-image prompt entropy (full softmax) distribution
- Vibe score distributions (per vibe)
- Per-image vibe mixture entropy distribution
- Scatter: each vibe positioned by (cross-image entropy, score strength)
- Per-image: prompt entropy vs vibe mixture entropy
- Per-prompt: dominance (top-1 rate) vs mean confidence on winning frames
- Per-vibe: dominance (argmax win rate) vs prominence (conf-weighted mean score)
- Prompt activation + prompt→vibe accrual (four PNGs: bars + three heatmaps, pipeline mapping)
- Per-prompt softmax boxplots (colors = primary vibe from library + legend)

Run after `run_study.py` (or any run that wrote the CSVs).

Requires ``PROMPT_LIBRARY_CSV`` (same schema as the pipeline: prompt_id + vibe_1/w_1/…) for vibe-accrual figures.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from config import cfg, COLOR_SCHEME

PERFORMANCE_SUBDIR = "performance_overview"
SHOW_FIGURES = False  # True = call plt.show() when running from the IDE


def _shannon_entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > 1e-15]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def _entropy_across_images(values: np.ndarray) -> float:
    """Entropy of normalized positive mass across images (one scalar per vibe)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s < 1e-15:
        return float("nan")
    p = v / s
    return _shannon_entropy_from_probs(p)


def _vibe_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("image", "image_path", "confidence")]


def load_tables(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    v_path = os.path.join(output_dir, "vibe_scores_per_image.csv")
    p_path = os.path.join(output_dir, "clipseg_topk_prompts.csv")
    if not os.path.isfile(v_path):
        raise FileNotFoundError(f"Missing {v_path}. Run the pipeline first.")
    if not os.path.isfile(p_path):
        raise FileNotFoundError(f"Missing {p_path}. Run the pipeline first.")
    df_v = pd.read_csv(v_path)
    df_p = pd.read_csv(p_path)
    return df_v, df_p


def load_prompt_vibe_map(library_csv: str | os.PathLike) -> dict[str, dict[str, float]]:
    """
    Same mapping as ``pipeline.load_prompt_library``: each prompt_id → {vibe: weight}.
    """
    path = Path(library_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt library not found: {path}")
    df_lib = pd.read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for _, row in df_lib.iterrows():
        pid = str(row["prompt_id"])
        out[pid] = {
            str(row["vibe_1"]): float(row["w_1"]),
            str(row["vibe_2"]): float(row["w_2"]),
            str(row["vibe_3"]): float(row["w_3"]),
        }
    return out


def primary_vibe_from_map(vibe_map: dict[str, dict[str, float]], pid: str) -> str | None:
    """Vibe with highest library weight for this prompt; None if missing or empty."""
    wmap = vibe_map.get(str(pid))
    if not wmap:
        return None
    return max(wmap.items(), key=lambda kv: float(kv[1]))[0]


def accumulate_vibe_prompt_mass(
    df_p: pd.DataFrame,
    vibe_map: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """
    Per pipeline: each row has ``contrib`` = p×conf; vibe cell gets ``contrib * weight``.

    Returns:
        M: shape (n_vibes, n_prompts), summed over all images
        vibes: row labels (sorted unique vibes from the library)
        prompts: column labels (sorted unique prompt_id in df_p)
        prompt_totals: sum of M over vibes per prompt (= Σ_image contrib if weights sum to 1 per prompt)
    """
    pids = sorted(df_p["prompt_id"].astype(str).unique())
    vibes = sorted(
        {v for pid in pids for v in vibe_map.get(pid, {}).keys()}
    )
    if not vibes or not pids:
        return np.zeros((0, 0)), vibes, pids, np.zeros(0)
    vi = {v: i for i, v in enumerate(vibes)}
    pj = {p: j for j, p in enumerate(pids)}
    m = np.zeros((len(vibes), len(pids)), dtype=np.float64)
    for _, row in df_p.iterrows():
        pid = str(row["prompt_id"])
        c = float(row["contrib"])
        if not np.isfinite(c) or c <= 0:
            continue
        wmap = vibe_map.get(pid)
        if not wmap:
            continue
        j = pj.get(pid)
        if j is None:
            continue
        for v, w in wmap.items():
            if v not in vi:
                continue
            m[vi[v], j] += c * float(w)
    prompt_totals = m.sum(axis=0)
    return m, vibes, pids, prompt_totals


def _per_image_row_zscore(X: pd.DataFrame) -> pd.DataFrame:
    """For each row (image), center and scale vibe columns → mean 0, std 1 across vibes."""
    Z = X.astype(np.float64)
    mu = Z.mean(axis=1)
    sd = Z.std(axis=1, ddof=0).replace(0.0, np.nan)
    out = Z.sub(mu, axis=0).div(sd, axis=0)
    return out.fillna(0.0)


def _save_vibe_corr_heatmap(
    corr: pd.DataFrame,
    out_dir: Path,
    filename: str,
    title: str,
    show: bool,
) -> None:
    cols = corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0.0, vmin=-1, vmax=1, square=True)
    else:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticklabels(cols)
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def figure_vibe_correlation(df_v: pd.DataFrame, out_dir: Path, show: bool) -> None:
    cols = _vibe_columns(df_v)
    if len(cols) < 2:
        print("Skipping vibe correlation: need at least 2 vibe columns.")
        return
    raw = df_v[cols]

    corr_raw = raw.corr()
    _save_vibe_corr_heatmap(
        corr_raw,
        out_dir,
        "01_vibe_correlation.png",
        "Vibe correlation — Pearson (raw scores per image; not normalized)",
        show,
    )
    Z = _per_image_row_zscore(raw)
    corr_z = Z.corr()
    _save_vibe_corr_heatmap(
        corr_z,
        out_dir,
        "01_vibe_correlation_per_image_z.png",
        "Vibe correlation — Pearson after per-image z-score across vibes",
        show,
    )


def figure_prompt_correlation(df_p: pd.DataFrame, out_dir: Path, show: bool) -> None:
    # Wide: images x prompts — use contrib (confidence-weighted mass) for stability
    pivot = df_p.pivot_table(
        index="image",
        columns="prompt_id",
        values="contrib",
        aggfunc="mean",
    )
    pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if pivot.shape[1] < 2 or pivot.shape[0] < 3:
        print("Skipping prompt correlation: not enough complete rows/columns.")
        return
    corr = pivot.corr()
    n = corr.shape[0]
    fig_w = max(10, min(24, 0.35 * n + 6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.9))
    if sns is not None:
        sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0.0, vmin=-1, vmax=1, square=True)
    else:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Prompt correlation (Pearson on contrib across images)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_prompt_correlation.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def figure_prompt_score_distributions(df_p: pd.DataFrame, out_dir: Path, show: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes,
        ["raw_score", "softmax_p", "contrib"],
        ["Raw CLIPSeg score", "Softmax p (per image)", "Contrib (p × conf)"],
    ):
        if col not in df_p.columns:
            continue
        x = df_p[col].astype(float).values
        x = x[np.isfinite(x)]
        ax.hist(x, bins=60, color=COLOR_SCHEME[1], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(title)
        ax.set_ylabel("Count")
    fig.suptitle("Prompt-level score distributions (all image×prompt rows)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "03_prompt_score_distributions.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def _short_prompt_label(s: str, max_len: int = 44) -> str:
    s = str(s).strip().replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _full_prompt_line(s: str) -> str:
    """Normalize whitespace; does not truncate (for heatmap axes)."""
    return str(s).strip().replace("\n", " ")


def figure_prompt_activation_and_vibe_accrual(
    df_p: pd.DataFrame,
    df_v: pd.DataFrame,
    out_dir: Path,
    library_csv: str | Path,
    show: bool,
) -> None:
    """
    Per-prompt activation: total Σ contrib over images (conf-weighted prompt mass).

    Vibe accrual: same weighting as the pipeline — each cell accumulates contrib × weight(vibe|prompt).

    Writes four PNG files: activation bars; log absolute mass heatmap; per-prompt vibe-split heatmap;
    per-vibe prompt-mix heatmap.
    """
    if "contrib" not in df_p.columns:
        print("Skipping prompt activation / vibe accrual: need contrib column.")
        return
    try:
        vibe_map = load_prompt_vibe_map(library_csv)
    except FileNotFoundError as e:
        print(f"Skipping prompt activation / vibe accrual: {e}")
        return

    raw_by_p = df_p.groupby(df_p["prompt_id"].astype(str))["contrib"].sum()
    raw_by_p = raw_by_p[np.isfinite(raw_by_p) & (raw_by_p >= 0)]
    if raw_by_p.size < 1:
        return

    M, vibes, pids_cols, _ = accumulate_vibe_prompt_mass(df_p, vibe_map)
    if M.size == 0 or not vibes:
        print("Skipping prompt activation / vibe accrual: empty mass matrix.")
        return

    id_to_text = (
        df_p.drop_duplicates(subset=["prompt_id"])[["prompt_id", "prompt_text"]]
        .assign(prompt_id=lambda d: d["prompt_id"].astype(str))
        .set_index("prompt_id")["prompt_text"]
        .astype(str)
        .to_dict()
    )

    order = raw_by_p.sort_values(ascending=False).index.tolist()
    pj = {p: j for j, p in enumerate(pids_cols)}
    idx = [pj[p] for p in order if p in pj]
    order = [p for p in order if p in pj]
    M_ord = M[:, idx]

    n_p = len(order)
    n_v = len(vibes)
    h_prompt = max(7.0, min(28.0, 0.22 * n_p + 3.0))
    w_heat = max(14.0, min(24.0, 8.0 + 0.28 * n_v))
    w_mix = max(16.0, min(48.0, 10.0 + 0.4 * n_p))

    col_sums = np.maximum(M_ord.sum(axis=0), 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        col_norm = M_ord / col_sums[np.newaxis, :]
    col_norm = np.nan_to_num(col_norm, nan=0.0)

    row_sums = np.maximum(M_ord.sum(axis=1), 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = M_ord / row_sums[:, np.newaxis]
    row_norm = np.nan_to_num(row_norm, nan=0.0)

    y = np.arange(n_p)

    fig_b, ax_bar = plt.subplots(figsize=(9.0, h_prompt))
    ax_bar.barh(y, raw_by_p.reindex(order).fillna(0.0).values, color=COLOR_SCHEME[0], alpha=0.9)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([_short_prompt_label(id_to_text.get(p, p)) for p in order], fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Σ contrib over images\n(conf-weighted prompt mass)")
    ax_bar.set_title("Prompt activation (stronger = more mass)")
    ax_bar.grid(True, axis="x", alpha=0.25)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "11a_prompt_activation_bars.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_b)

    fig_log, ax_abs = plt.subplots(figsize=(w_heat, h_prompt))
    im_data = np.log1p(M_ord.T)
    im0 = ax_abs.imshow(im_data, aspect="auto", cmap="magma", interpolation="nearest")
    ax_abs.set_yticks(np.arange(n_p))
    ax_abs.set_yticklabels([_full_prompt_line(id_to_text.get(p, p)) for p in order], fontsize=8)
    ax_abs.set_xticks(np.arange(n_v))
    ax_abs.set_xticklabels(vibes, rotation=45, ha="right", fontsize=8)
    ax_abs.set_title(
        "log(1 + mass to vibe) (contrib × weight, summed over images)\nprompt activation · library mapping",
        fontsize=10,
    )
    fig_log.colorbar(im0, ax=ax_abs, fraction=0.046, pad=0.02)
    fig_log.tight_layout()
    fig_log.savefig(out_dir / "11b_mass_to_vibe_log_heatmap.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_log)

    fig_split, ax_col = plt.subplots(figsize=(w_heat, h_prompt))
    im1 = ax_col.imshow(col_norm.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax_col.set_yticks(np.arange(n_p))
    ax_col.set_yticklabels([_full_prompt_line(id_to_text.get(p, p)) for p in order], fontsize=8)
    ax_col.set_xticks(np.arange(n_v))
    ax_col.set_xticklabels(vibes, rotation=45, ha="right", fontsize=8)
    ax_col.set_title(
        "Per-prompt split → vibes — each mapped-mass row sums to 1\nprompt activation · library mapping",
        fontsize=10,
    )
    fig_split.colorbar(im1, ax=ax_col, fraction=0.046, pad=0.02)
    fig_split.tight_layout()
    fig_split.savefig(out_dir / "11c_prompt_split_across_vibes.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_split)

    fig_mix, ax_row = plt.subplots(figsize=(w_mix, max(7.0, n_v * 0.75 + 1.5)))
    im2 = ax_row.imshow(row_norm, aspect="auto", cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
    ax_row.set_xticks(np.arange(n_p))
    ax_row.set_xticklabels(
        [_full_prompt_line(id_to_text.get(p, p)) for p in order],
        rotation=90,
        fontsize=8,
    )
    ax_row.set_yticks(np.arange(n_v))
    ax_row.set_yticklabels(vibes, fontsize=8)
    ax_row.set_title(
        "Per-vibe mix ← prompts — each vibe row sums to 1\nprompt activation · library mapping",
        fontsize=10,
    )
    fig_mix.colorbar(im2, ax=ax_row, fraction=0.046, pad=0.02)
    fig_mix.tight_layout()
    fig_mix.savefig(out_dir / "11d_vibe_mix_from_prompts.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_mix)

    vibe_tot_m = pd.Series(M.sum(axis=1), index=vibes)
    have = [c for c in vibes if c in df_v.columns]
    if have:
        table = pd.DataFrame({
            "from_matrix": vibe_tot_m.reindex(have).fillna(0.0),
            "from_vibe_csv_sum": df_v[have].astype(float).sum(axis=0),
        })
        table["abs_diff"] = (table["from_matrix"] - table["from_vibe_csv_sum"]).abs()
        table.to_csv(out_dir / "summary_vibe_mass_matrix_vs_csv.csv")

    long_rows = []
    for vi, v in enumerate(vibes):
        for pj_, p in enumerate(order):
            long_rows.append({
                "vibe": v,
                "prompt_id": p,
                "weighted_mass": float(M_ord[vi, pj_]),
            })
    pd.DataFrame(long_rows).to_csv(out_dir / "summary_prompt_vibe_weighted_mass_long.csv", index=False)

    pd.DataFrame({
        "prompt_id": order,
        "total_contrib_sum_over_images": raw_by_p.reindex(order).values,
    }).to_csv(out_dir / "summary_prompt_activation_totals.csv", index=False)


def figure_prompt_softmax_distribution_by_prompt(
    df_p: pd.DataFrame,
    out_dir: Path,
    show: bool,
    library_csv: str | os.PathLike | None = None,
    *,
    top_n: int = 18,
) -> None:
    """
    Boxplots of per-image softmax_p for prompts with highest mean softmax.

    When ``library_csv`` is readable, each box is colored by **primary vibe**: the vibe with the
    largest weight for that prompt in the library. A legend maps color → vibe. Scatter points on
    each column are **outliers** (Tukey fliers: softmax_p outside the whiskers for that prompt).
    """
    if "softmax_p" not in df_p.columns:
        print("Skipping per-prompt softmax boxplots: need softmax_p.")
        return
    vibe_map: dict[str, dict[str, float]] | None = None
    if library_csv is not None:
        try:
            vibe_map = load_prompt_vibe_map(library_csv)
        except FileNotFoundError:
            print("Softmax boxplots: prompt library not found — using default cycle colors (no vibe legend).")

    g = df_p.groupby(df_p["prompt_id"].astype(str))["softmax_p"].mean().sort_values(ascending=False)
    if g.size < 1:
        return
    take = g.head(int(min(top_n, len(g)))).index.tolist()
    sub = df_p[df_p["prompt_id"].astype(str).isin(take)].copy()
    sub["prompt_id"] = sub["prompt_id"].astype(str)
    id_to_text = (
        sub.drop_duplicates(subset=["prompt_id"])[["prompt_id", "prompt_text"]]
        .set_index("prompt_id")["prompt_text"]
        .astype(str)
        .to_dict()
    )
    order_cat = [p for p in take if p in sub["prompt_id"].unique()]
    data = [sub.loc[sub["prompt_id"] == p, "softmax_p"].dropna().astype(float).values for p in order_cat]
    prompt_xlabels = [_full_prompt_line(id_to_text.get(p, p)) for p in order_cat]
    fig_w = max(16, min(32, 0.75 * len(order_cat) + 10))
    fig_h = max(10.0, min(22.0, 6.5 + 0.35 * len(order_cat)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bp = ax.boxplot(data, tick_labels=prompt_xlabels, patch_artist=True)

    unknown_color = "#b3b3b3"
    if vibe_map:
        primaries: list[str | None] = [
            primary_vibe_from_map(vibe_map, p) for p in order_cat
        ]
        distinct_vibes = sorted({v for v in primaries if v is not None})
        vibe_color: dict[str | None, str] = {
            v: COLOR_SCHEME[i % len(COLOR_SCHEME)] for i, v in enumerate(distinct_vibes)
        }
        if any(v is None for v in primaries):
            vibe_color[None] = unknown_color
        for i, box in enumerate(bp["boxes"]):
            fc = vibe_color.get(primaries[i], unknown_color)
            box.set(facecolor=fc, alpha=0.65, edgecolor="white", linewidth=0.6)
        for i, line in enumerate(bp.get("fliers", []) or []):
            fc = vibe_color.get(primaries[i], unknown_color)
            line.set_markerfacecolor(fc)
            line.set_markeredgecolor(fc)
            line.set_alpha(0.85)
        handles: list[Patch] = []
        for v in distinct_vibes:
            handles.append(
                Patch(
                    facecolor=vibe_color[v],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.65,
                    label=v,
                )
            )
        if None in primaries:
            handles.append(
                Patch(facecolor=unknown_color, edgecolor="white", linewidth=0.5, alpha=0.65, label="(not in library)")
            )
        ax.legend(
            handles=handles,
            title="Primary vibe (argmax weight)",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )
    else:
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=COLOR_SCHEME[i % len(COLOR_SCHEME)], alpha=0.65)

    ax.set_xticklabels(prompt_xlabels, rotation=55, ha="right", fontsize=12)
    ax.set_ylabel("softmax_p (per image)")
    ax.set_xlabel("")
    ax.set_title(
        f"Softmax distribution by prompt (top {len(order_cat)} by mean softmax)\n(circles = outliers: softmax_p outside whiskers)",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "12_prompt_softmax_by_prompt_boxes.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def compute_prompt_entropy_per_image(df_p: pd.DataFrame) -> pd.Series:
    ent = {}
    for img, g in df_p.groupby("image"):
        p = g["softmax_p"].astype(float).values
        ent[img] = _shannon_entropy_from_probs(p)
    return pd.Series(ent, name="prompt_entropy")


def figure_prompt_entropy(df_p: pd.DataFrame, out_dir: Path, show: bool) -> None:
    s = compute_prompt_entropy_per_image(df_p)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s.values, bins=40, color=COLOR_SCHEME[2], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(s.mean(), color="k", linestyle="--", label=f"mean={s.mean():.3f}")
    ax.set_title("Per-image prompt entropy (full softmax over prompts)")
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Number of images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04_prompt_entropy_per_image.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def compute_vibe_mix_entropy_per_image(df_v: pd.DataFrame, vibe_cols: list[str]) -> pd.Series:
    ent = {}
    for _, row in df_v.iterrows():
        v = row[vibe_cols].astype(float).values
        v = np.clip(v, 0.0, None)
        ent[row["image"]] = _shannon_entropy_from_probs(v)
    return pd.Series(ent, name="vibe_mix_entropy")


def figure_vibe_distributions(df_v: pd.DataFrame, out_dir: Path, show: bool) -> None:
    cols = _vibe_columns(df_v)
    if not cols:
        return
    n = len(cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(cols):
        ax = axes[i]
        x = df_v[c].astype(float).values
        ax.hist(x, bins=35, color=COLOR_SCHEME[i % len(COLOR_SCHEME)], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("Count")
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Vibe score distributions (across images)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "05_vibe_distributions.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def figure_vibe_entropy(df_v: pd.DataFrame, out_dir: Path, show: bool) -> None:
    cols = _vibe_columns(df_v)
    if len(cols) < 2:
        print("Skipping vibe mixture entropy: need ≥2 vibes.")
        return
    s = compute_vibe_mix_entropy_per_image(df_v, cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s.values, bins=40, color=COLOR_SCHEME[4], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(s.mean(), color="k", linestyle="--", label=f"mean={s.mean():.3f}")
    ax.set_title("Per-image vibe mixture entropy (normalize vibe scores to sum 1)")
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Number of images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_vibe_entropy_per_image.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def figure_vibe_entropy_vs_strength(df_v: pd.DataFrame, out_dir: Path, show: bool) -> None:
    cols = _vibe_columns(df_v)
    if not cols:
        return
    conf = df_v["confidence"].astype(float).values if "confidence" in df_v.columns else None

    xs = []
    ys = []
    labels = []
    for c in cols:
        vals = df_v[c].astype(float).values
        ent = _entropy_across_images(vals)
        if conf is not None and conf.sum() > 1e-12:
            strength = float((vals * conf).sum() / conf.sum())
        else:
            strength = float(np.mean(vals))
        xs.append(ent)
        ys.append(strength)
        labels.append(c)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, lab in enumerate(labels):
        if not np.isfinite(xs[i]) or not np.isfinite(ys[i]):
            continue
        ax.scatter(
            xs[i], ys[i],
            s=120,
            c=[COLOR_SCHEME[i % len(COLOR_SCHEME)]],
            edgecolors="white",
            linewidths=1,
            zorder=3,
        )
        ax.annotate(lab, (xs[i], ys[i]), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Cross-image entropy (nats)\nhow spread this vibe’s mass is over frames")
    ax.set_ylabel("Score strength\nconf-weighted mean vibe score (or plain mean)")
    ax.set_title("Vibe map: entropy vs strength")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "07_vibe_entropy_vs_strength.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def figure_prompt_vs_vibe_entropy_per_image(
    df_v: pd.DataFrame,
    df_p: pd.DataFrame,
    out_dir: Path,
    show: bool,
) -> None:
    """Scatter: each image — prompt entropy (softmax over prompts) vs vibe mixture entropy."""
    cols = _vibe_columns(df_v)
    if len(cols) < 1:
        print("Skipping prompt vs vibe entropy: no vibe columns.")
        return
    pe_df = compute_prompt_entropy_per_image(df_p).rename_axis("image").reset_index()
    ve_df = (
        compute_vibe_mix_entropy_per_image(df_v, cols)
        .rename_axis("image")
        .reset_index()
        .rename(columns={"vibe_mix_entropy": "vibe_entropy"})
    )
    merged = pe_df.merge(ve_df, on="image", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    if len(merged) < 2:
        print("Skipping prompt vs vibe entropy: not enough aligned rows.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        merged["prompt_entropy"].values,
        merged["vibe_entropy"].values,
        alpha=0.55,
        s=36,
        c=COLOR_SCHEME[0],
        edgecolors="white",
        linewidths=0.4,
    )
    lim_hi = max(merged["prompt_entropy"].max(), merged["vibe_entropy"].max()) * 1.05
    ax.plot([0, lim_hi], [0, lim_hi], "k--", alpha=0.35, linewidth=1, label="y = x")
    ax.set_xlabel("Prompt entropy (nats)\nuncertainty over prompts (full softmax)")
    ax.set_ylabel("Vibe entropy (nats)\nuncertainty over vibes (normalized scores)")
    ax.set_title("Per image: prompt entropy vs vibe mixture entropy")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "08_prompt_vs_vibe_entropy_per_image.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def figure_prompt_dominance_and_confidence(df_p: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """
    Per prompt: dominance = fraction of images where this prompt is top-1 by softmax_p;
    confidence = mean image-level confidence on those winning frames (fallback: global mean conf).

    Scatter and legend include every prompt in the dataframe (dominance 0 = never top-1).
    """
    if "softmax_p" not in df_p.columns or "confidence" not in df_p.columns:
        print("Skipping prompt dominance/confidence: need softmax_p and confidence.")
        return
    n_img = df_p["image"].nunique()
    if n_img < 1:
        return
    win_idx = df_p.groupby("image")["softmax_p"].idxmax()
    winners = df_p.loc[win_idx, ["prompt_id", "confidence"]].copy()
    dom = winners["prompt_id"].value_counts() / float(n_img)
    conf_on_wins = winners.groupby("prompt_id")["confidence"].mean()
    global_mean_conf = float(df_p["confidence"].astype(float).mean())

    pids = sorted(df_p["prompt_id"].astype(str).unique())
    x = dom.reindex(pids, fill_value=0.0).values.astype(float)
    y = conf_on_wins.reindex(pids).astype(float).values
    y = np.where(np.isfinite(y), y, global_mean_conf)

    id_to_text = (
        df_p.drop_duplicates(subset=["prompt_id"])[["prompt_id", "prompt_text"]]
        .assign(prompt_id=lambda d: d["prompt_id"].astype(str))
        .set_index("prompt_id")["prompt_text"]
        .astype(str)
        .to_dict()
    )

    def _short_legend_prompt(s: str, max_len: int = 52) -> str:
        s = str(s).strip().replace("\n", " ")
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    fig_w = 10 if len(pids) <= 15 else 11
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    for i, pid in enumerate(pids):
        ptxt = id_to_text.get(pid, "") or str(pid)
        leg = _short_legend_prompt(ptxt)
        ax.scatter(
            x[i],
            y[i],
            s=100,
            c=[COLOR_SCHEME[i % len(COLOR_SCHEME)]],
            edgecolors="white",
            linewidths=1,
            zorder=3,
            label=leg,
        )

    ax.set_xlabel("Dominance\nfraction of images where prompt is top-1 (softmax)")
    ax.set_ylabel("Mean confidence\n(on images where this prompt wins; else global mean)")
    ax.set_title("Per-prompt: dominance vs confidence (on wins)")
    ax.set_xlim(-0.02, max(1.02, float(np.nanmax(x)) * 1.05 + 0.02) if x.size else 1.02)
    ax.grid(True, alpha=0.25)
    n_legend = len(pids)
    if n_legend > 0:
        leg_ncol = 3 if n_legend > 20 else (2 if n_legend > 14 else 1)
        ax.legend(
            title="prompts (all)",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=7,
            ncol=leg_ncol,
            frameon=True,
            markerscale=1.2,
        )
    fig.savefig(out_dir / "09_prompt_dominance_vs_confidence.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    out_csv = out_dir / "summary_prompt_dominance_confidence.csv"
    pd.DataFrame({
        "prompt_id": pids,
        "dominance_top1_rate": x,
        "mean_confidence_on_wins": y,
    }).to_csv(out_csv, index=False)


def figure_vibe_dominance_and_prominence(df_v: pd.DataFrame, out_dir: Path, show: bool) -> None:
    """
    Per vibe: dominance = fraction of images where this vibe has the highest (clipped ≥0) score;
    prominence = confidence-weighted mean vibe score.
    """
    cols = _vibe_columns(df_v)
    if not cols:
        return
    mat = df_v[cols].astype(float).values
    mat = np.clip(mat, 0.0, None)
    win_idx = np.argmax(mat, axis=1)
    conf = df_v["confidence"].astype(float).values if "confidence" in df_v.columns else np.ones(len(df_v))

    dom = []
    prom = []
    for j, c in enumerate(cols):
        dom.append(float(np.mean(win_idx == j)))
        vcol = df_v[c].astype(float).values
        if conf.sum() > 1e-12:
            prom.append(float((vcol * conf).sum() / conf.sum()))
        else:
            prom.append(float(np.mean(vcol)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for j, c in enumerate(cols):
        ax.scatter(
            dom[j],
            prom[j],
            s=130,
            c=[COLOR_SCHEME[j % len(COLOR_SCHEME)]],
            edgecolors="white",
            linewidths=1,
            zorder=3,
            label=str(c),
        )

    ax.set_xlabel("Dominance\nfraction of images where vibe has highest score (≥0)")
    ax.set_ylabel("Prominence\nconfidence-weighted mean vibe score")
    ax.set_title("Per-vibe: dominance vs prominence")
    ax.grid(True, alpha=0.25)
    leg_ncol = 2 if len(cols) > 8 else 1
    ax.legend(
        title="vibe",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=9,
        ncol=leg_ncol,
        frameon=True,
        markerscale=1.1,
    )
    fig.savefig(out_dir / "10_vibe_dominance_vs_prominence.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    pd.DataFrame({
        "vibe": cols,
        "dominance_argmax_rate": dom,
        "prominence_conf_weighted_mean": prom,
    }).to_csv(out_dir / "summary_vibe_dominance_prominence.csv", index=False)


def export_summary_csv(
    df_v: pd.DataFrame,
    df_p: pd.DataFrame,
    out_dir: Path,
) -> None:
    cols = _vibe_columns(df_v)
    prompt_ent = compute_prompt_entropy_per_image(df_p)
    vibe_mix = compute_vibe_mix_entropy_per_image(df_v, cols) if len(cols) >= 1 else None

    rows = []
    conf = df_v["confidence"].astype(float).values if "confidence" in df_v.columns else None
    for c in cols:
        vals = df_v[c].astype(float).values
        ent_cross = _entropy_across_images(vals)
        if conf is not None and conf.sum() > 1e-12:
            strength = float((vals * conf).sum() / conf.sum())
        else:
            strength = float(np.mean(vals))
        rows.append({"vibe": c, "cross_image_entropy": ent_cross, "score_strength": strength})

    pd.DataFrame(rows).to_csv(out_dir / "summary_vibes_entropy_strength.csv", index=False)

    summary_img = prompt_ent.rename_axis("image").reset_index()
    if vibe_mix is not None:
        summary_img = summary_img.merge(
            vibe_mix.rename_axis("image").reset_index(),
            on="image",
            how="outer",
        )
    summary_img.to_csv(out_dir / "summary_per_image_entropies.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Overall performance plots from pipeline CSVs.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Folder with vibe_scores_per_image.csv etc. (default: cfg.OUTPUT_FOLDER = {cfg.OUTPUT_FOLDER})",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (only save PNGs).")
    parser.add_argument(
        "--library-csv",
        default=None,
        help=(
            "Prompt ↔ vibe library CSV (same schema as pipeline; "
            f"default: cfg.PROMPT_LIBRARY_CSV = {cfg.PROMPT_LIBRARY_CSV})"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or cfg.OUTPUT_FOLDER
    fig_dir = Path(output_dir) / PERFORMANCE_SUBDIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_v, df_p = load_tables(output_dir)
    show = SHOW_FIGURES and not args.no_show
    library_csv = args.library_csv or cfg.PROMPT_LIBRARY_CSV

    print("Loaded vibe table:", df_v.shape, "| prompt table:", df_p.shape)
    print("Writing figures to:", fig_dir)

    figure_vibe_correlation(df_v, fig_dir, show)
    figure_prompt_correlation(df_p, fig_dir, show)
    figure_prompt_score_distributions(df_p, fig_dir, show)
    figure_prompt_activation_and_vibe_accrual(df_p, df_v, fig_dir, library_csv, show)
    figure_prompt_softmax_distribution_by_prompt(df_p, fig_dir, show, library_csv)
    figure_prompt_entropy(df_p, fig_dir, show)
    figure_vibe_distributions(df_v, fig_dir, show)
    figure_vibe_entropy(df_v, fig_dir, show)
    figure_vibe_entropy_vs_strength(df_v, fig_dir, show)
    figure_prompt_vs_vibe_entropy_per_image(df_v, df_p, fig_dir, show)
    figure_prompt_dominance_and_confidence(df_p, fig_dir, show)
    figure_vibe_dominance_and_prominence(df_v, fig_dir, show)
    export_summary_csv(df_v, df_p, fig_dir)

    print("Done.")


if __name__ == "__main__":
    main()
