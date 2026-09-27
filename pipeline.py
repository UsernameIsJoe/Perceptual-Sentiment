"""
CLIPSeg scoring, vibe aggregation, CSV exports, and district selection —
extracted from Prompt Scoring clean up.ipynb (same logic, single module).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from config import (
    cfg,
    SAVE_VIBE_TOP1_MAPS,
    SEG_MODEL_NAME,
    VIBE_MAPS_DIRNAME,
    VIBE_MAP_SCORE_RES,
    MIN_VIBES,
    SCORES_META_COLUMNS,
    resolve_data_path,
)
from study_progress import tqdm

# --- CLIPSeg globals (notebook style) ---
processor = None
model = None

_seg_proc = None
_seg_model = None
_seg_id2label = None


def init_clipseg():
    global processor, model
    if processor is not None:
        return
    processor = CLIPSegProcessor.from_pretrained(cfg.MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(cfg.MODEL_ID).to(cfg.DEVICE)
    model.eval()
    model = model.float()
    print("DEVICE:", cfg.DEVICE)


def _load_seg_model():
    global _seg_proc, _seg_model, _seg_id2label
    if _seg_model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seg_proc = SegformerImageProcessor.from_pretrained(SEG_MODEL_NAME)
    _seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME).to(device)
    _seg_model.eval()
    _seg_id2label = _seg_model.config.id2label


def _label_ids_containing(substr_list):
    ids = set()
    for k, v in _seg_id2label.items():
        name = str(v).lower()
        for s in substr_list:
            if s in name:
                ids.add(int(k))
                break
    return ids


def get_sky_ground_other_masks(pil_img, out_h, out_w):
    """
    Returns boolean masks (sky, ground, other) at (out_h, out_w).
    Uses SegFormer ADE20K. Masks are used only for scoring selection rules.
    """
    _load_seg_model()
    device = next(_seg_model.parameters()).device

    inputs = _seg_proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _seg_model(**inputs)
        logits = outputs.logits  # (1, C, h, w)
        logits = F.interpolate(
            logits, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        seg = logits.argmax(dim=1)[0]  # (out_h, out_w)

    sky_ids = _label_ids_containing(["sky"])
    ground_ids = _label_ids_containing([
        "road", "street", "sidewalk", "path", "pavement",
        "ground", "earth", "floor", "dirt", "sand"
    ])

    sky = torch.zeros_like(seg, dtype=torch.bool)
    ground = torch.zeros_like(seg, dtype=torch.bool)

    for i in sky_ids:
        sky |= (seg == i)
    for i in ground_ids:
        ground |= (seg == i)

    other = ~(sky | ground)
    return sky, ground, other


# --- From notebook cell 11 (must exist before prominence helper uses it) ---

def dominance_multiplier(p_sky, p_ground, t_sky=0.30, t_ground=0.50, lam_sky=3.0, lam_ground=3.0):
    over_s = max(0.0, float(p_sky) - t_sky)
    over_g = max(0.0, float(p_ground) - t_ground)
    return math.exp(-(lam_sky * over_s + lam_ground * over_g))


def capped_topk_mean_from_logits(
    logit_map_2d,
    sky_mask_2d,
    ground_mask_2d,
    other_mask_2d,
    top_pct=0.05,
    cap_sky=0.15,
    cap_ground=0.25,
    apply_dominance_penalty=True,
    t_sky=0.30,
    t_ground=0.50,
    lam_sky=3.0,
    lam_ground=3.0,
):
    H, W = logit_map_2d.shape
    N = H * W
    K = max(1, int(N * top_pct))

    K_sky_max = int(round(K * cap_sky))
    K_ground_max = int(round(K * cap_ground))
    K_other = max(0, K - K_sky_max - K_ground_max)

    flat = logit_map_2d.reshape(-1)

    def topk_from_mask(mask2d, k):
        if k <= 0:
            return torch.empty((0,), device=flat.device, dtype=flat.dtype)
        idx = torch.where(mask2d.reshape(-1))[0]
        if idx.numel() == 0:
            return torch.empty((0,), device=flat.device, dtype=flat.dtype)
        vals = flat[idx]
        kk = min(k, vals.numel())
        topv, _ = torch.topk(vals, k=kk, largest=True, sorted=False)
        return topv

    top_sky = topk_from_mask(sky_mask_2d, K_sky_max)
    top_gnd = topk_from_mask(ground_mask_2d, K_ground_max)
    top_oth = topk_from_mask(other_mask_2d, K_other)

    selected = torch.cat([top_sky, top_gnd, top_oth], dim=0)

    # Backfill if masks are empty or tiny
    if selected.numel() < K:
        kk = min(K, flat.numel())
        selected, _ = torch.topk(flat, k=kk, largest=True, sorted=False)

    base_score = selected.mean().item()

    if not apply_dominance_penalty:
        return base_score

    denom = max(1, selected.numel())
    p_sky = top_sky.numel() / denom
    p_ground = top_gnd.numel() / denom

    mult = dominance_multiplier(
        p_sky, p_ground,
        t_sky=t_sky, t_ground=t_ground,
        lam_sky=lam_sky, lam_ground=lam_ground
    )
    return base_score * mult


# --- Notebook cell 4: multi-quantile anti-spike ---

def _capped_topk_mean_and_mult(
    logit_map_2d,
    sky_mask_2d,
    ground_mask_2d,
    other_mask_2d,
    top_pct,
    cap_sky,
    cap_ground,
    apply_dominance_penalty,
    t_sky,
    t_ground,
    lam_sky,
    lam_ground,
):
    """Same selection rules as capped_topk_mean_from_logits, but returns (mean, mult)."""
    H, W = logit_map_2d.shape
    N = H * W
    K = max(1, int(N * top_pct))

    K_sky_max = int(round(K * cap_sky))
    K_ground_max = int(round(K * cap_ground))
    K_other = max(0, K - K_sky_max - K_ground_max)

    flat = logit_map_2d.reshape(-1)

    def topk_from_mask(mask2d, k):
        if k <= 0:
            return torch.empty((0,), device=flat.device, dtype=flat.dtype)
        idx = torch.where(mask2d.reshape(-1))[0]
        if idx.numel() == 0:
            return torch.empty((0,), device=flat.device, dtype=flat.dtype)
        vals = flat[idx]
        kk = min(k, vals.numel())
        topv, _ = torch.topk(vals, k=kk, largest=True, sorted=False)
        return topv

    top_sky = topk_from_mask(sky_mask_2d, K_sky_max)
    top_gnd = topk_from_mask(ground_mask_2d, K_ground_max)
    top_oth = topk_from_mask(other_mask_2d, K_other)

    selected = torch.cat([top_sky, top_gnd, top_oth], dim=0)

    # Backfill if masks are empty or tiny
    if selected.numel() < K:
        kk = min(K, flat.numel())
        selected, _ = torch.topk(flat, k=kk, largest=True, sorted=False)
        # In backfill mode we no longer know region composition reliably, so set mult=1
        return selected.mean().item(), 1.0

    base = selected.mean().item()

    if not apply_dominance_penalty:
        return base, 1.0

    denom = max(1, selected.numel())
    p_sky = top_sky.numel() / denom
    p_ground = top_gnd.numel() / denom

    mult = dominance_multiplier(
        p_sky, p_ground,
        t_sky=t_sky, t_ground=t_ground,
        lam_sky=lam_sky, lam_ground=lam_ground
    )
    return base, mult


def capped_topk_prominence_from_logits(
    logit_map_2d,
    sky_mask_2d,
    ground_mask_2d,
    other_mask_2d,
    top_pct_5=None,
    top_pct_20=None,
    cap_sky=0.15,
    cap_ground=0.15,
    apply_dominance_penalty=True,
    t_sky=0.30,
    t_ground=0.30,
    lam_sky=3.0,
    lam_ground=3.0,
    eps=None,
    alpha=None,
):
    """
    Returns:
        S_prom = S5 * (S20 / (S5 + eps))^alpha

    S5 and S20 are computed with the SAME caps/backfill rules as before.
    """
    if top_pct_5 is None:
        top_pct_5 = getattr(cfg, "TOP_PCT_5", 0.05)
    if top_pct_20 is None:
        top_pct_20 = getattr(cfg, "TOP_PCT_20", 0.20)
    if eps is None:
        eps = getattr(cfg, "EPS_PROM", 1e-6)
    if alpha is None:
        alpha = getattr(cfg, "ALPHA_PROM", 1.5)

    s5, m5 = _capped_topk_mean_and_mult(
        logit_map_2d, sky_mask_2d, ground_mask_2d, other_mask_2d,
        top_pct=top_pct_5, cap_sky=cap_sky, cap_ground=cap_ground,
        apply_dominance_penalty=apply_dominance_penalty,
        t_sky=t_sky, t_ground=t_ground, lam_sky=lam_sky, lam_ground=lam_ground,
    )
    s20, m20 = _capped_topk_mean_and_mult(
        logit_map_2d, sky_mask_2d, ground_mask_2d, other_mask_2d,
        top_pct=top_pct_20, cap_sky=cap_sky, cap_ground=cap_ground,
        apply_dominance_penalty=apply_dominance_penalty,
        t_sky=t_sky, t_ground=t_ground, lam_sky=lam_sky, lam_ground=lam_ground,
    )

    # Apply dominance penalty per quantile (same idea as before, just per slice)
    S5 = s5 * m5
    S20 = s20 * m20

    ratio = S20 / (S5 + eps)
    # Guard: a negative base with a fractional alpha yields a complex number. A negative ratio means
    # the top-5%/top-20% slices disagree in sign (no coherent prominence), so floor it at 0.
    ratio = max(ratio, 0.0)
    S_prom = S5 * (ratio ** alpha)
    return float(S_prom)


# --- Notebook cell 5: S_prom scoring (active path; notebook cell 11 duplicate omitted) ---

@torch.no_grad()
def score_prompts_for_image(pil_img, prompts, score_res=352, batch_size=16, top_pct=0.05):
    """
    Returns:
        scores: np.ndarray shape (len(prompts),)
    Notes:
        - top_pct is kept only for backward compatibility with existing calls.
        - The actual scoring now uses cfg.TOP_PCT_5 and cfg.TOP_PCT_20 via capped_topk_prominence_from_logits.
    """
    init_clipseg()
    if not hasattr(pil_img, "size"):
        raise TypeError("score_prompts_for_image expects a PIL image (RGB).")

    # downscale for speed (scoring only)
    w, h = pil_img.size
    scale = score_res / max(w, h)
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_small = pil_img.resize((new_w, new_h), Image.BILINEAR)
    else:
        img_small = pil_img

    scores = np.zeros(len(prompts), dtype=np.float32)

    n = len(prompts)
    nb = math.ceil(n / batch_size)

    for bi in range(nb):
        start = bi * batch_size
        end = min((bi + 1) * batch_size, n)
        batch_prompts = prompts[start:end]

        inputs = processor(
            text=batch_prompts,
            images=[img_small] * len(batch_prompts),
            padding=True,
            return_tensors="pt"
        ).to(cfg.DEVICE)

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.contiguous()

        outputs = model(**inputs)
        logits = outputs.logits  # (B, Hm, Wm)
        B, Hm, Wm = logits.shape

        # semantic masks once per image at (Hm, Wm)
        sky_m, gnd_m, oth_m = get_sky_ground_other_masks(
            img_small,
            out_h=Hm,
            out_w=Wm
        )
        sky_m = sky_m.to(logits.device)
        gnd_m = gnd_m.to(logits.device)
        oth_m = oth_m.to(logits.device)

        batch_scores = []
        for i in range(B):
            sc = capped_topk_prominence_from_logits(
                logit_map_2d=logits[i],
                sky_mask_2d=sky_m,
                ground_mask_2d=gnd_m,
                other_mask_2d=oth_m,
                # keep your current caps and penalty params consistent with the working setup
                cap_sky=0.15,
                cap_ground=0.25,
                apply_dominance_penalty=True,
                t_sky=0.30,
                t_ground=0.50,
                lam_sky=3.0,
                lam_ground=3.0
            )
            batch_scores.append(sc)

        scores[start:end] = np.array(batch_scores, dtype=np.float32)

    return scores


def topk_prompts(scores, prompts, k=10):
    idx = np.argsort(-scores)[:k]
    return [(prompts[i], float(scores[i])) for i in idx], idx


def list_images(folder):
    """Notebook cell 17: sorted list from directory (not glob)."""
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts)
    ]


def list_images_glob(folder):
    """Notebook cell 11 helper (glob); kept for compatibility."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, e))
    return sorted(paths)


def load_rgb(path):
    return Image.open(resolve_data_path(path)).convert("RGB")


# --- Vibe scoring (notebook cells 12–13) ---

def _softmax_np(x, T=1.0):
    x = np.asarray(x, dtype=np.float64)
    if T is None or T <= 0:
        T = 1.0
    z = x / float(T)
    z = z - np.max(z)
    e = np.exp(z)
    s = e / np.sum(e)
    return s.astype(np.float32)


def _topk_entropy_conf(p, K):
    """
    p: np.ndarray probs summing to 1
    Returns:
      conf in [0,1] computed from top-K entropy:
        conf = 1 - H(q)/log(K), where q is topK(p) renormalized.
    """
    p = np.asarray(p, dtype=np.float64)
    N = p.size
    if N == 0:
        return 0.0
    K = int(K)
    K = max(1, min(K, N))

    # top-K probs
    idx = np.argpartition(-p, K - 1)[:K]
    top = p[idx]
    s = float(np.sum(top))
    if s <= 0:
        return 0.0
    q = top / s  # renormalize

    eps = 1e-12
    H = -float(np.sum(q * np.log(q + eps)))
    H_norm = H / float(np.log(K)) if K > 1 else 0.0
    conf = 1.0 - H_norm
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0
    return float(conf)


def score_vibes_from_prompt_scores(
    scores,
    prompt_ids,
    prompt_texts,
    prompt_id_to_vibes,
    *,
    candidate_top_m=40,     # used only for per_vibe_top evidence list (debug/preview)
    topk_per_vibe=4,        # keep top K prompt evidences per vibe (debug/preview)
    vibe_score_mode="sum"   # "sum" or "max" or "top2_mean"
):
    """
    New logic:
      1) Softmax over all prompts (per image) -> p
      2) Confidence from top-K entropy on p -> conf
      3) Prompt contribution = p_i * conf
      4) Vibe score = sum over prompts (prompt_contribution * mapping_weight)

    Returns:
      per_vibe_top: dict[vibe] -> list of dicts sorted by weighted_score desc
      final_vibe_scores: dict[vibe] -> float
    """
    scores = np.asarray(scores, dtype=np.float32)

    # 1) Per-image softmax over all prompts
    T = getattr(cfg, "SOFTMAX_T", 1.0)
    p = _softmax_np(scores, T=T)

    # 2) Confidence via top-K entropy
    K_ent = getattr(cfg, "ENT_TOPK", 10)
    conf = _topk_entropy_conf(p, K_ent)

    # 3) Confidence-weighted prompt contributions
    contrib = p * conf  # shape (num_prompts,)

    # 4) Aggregate to vibes via mapping
    vibe_accum = defaultdict(float)

    # For debug: keep prompt evidence lists per vibe
    per_vibe = defaultdict(list)

    # We still keep a candidate list for evidence display, but final accumulation uses all prompts
    cand_idx = np.argsort(-contrib)[:min(int(candidate_top_m), len(contrib))]

    for i in range(len(contrib)):
        pid = prompt_ids[i]
        wmap = prompt_id_to_vibes.get(pid, {})
        if not wmap:
            continue

        c = float(contrib[i])
        cap = getattr(cfg, "PROMPT_CONTRIB_CAP", None)
        if cap is not None:
            c = min(c, float(cap))

        if c <= 0:
            continue

        for vibe, w in wmap.items():
            vibe_accum[vibe] += c * float(w)

    # Build per_vibe evidence lists from candidate prompts only (keeps display clean)
    for i in cand_idx:
        pid = prompt_ids[i]
        ptxt = prompt_texts[i]
        wmap = prompt_id_to_vibes.get(pid, {})
        if not wmap:
            continue

        c = float(contrib[i])
        cap = getattr(cfg, "PROMPT_CONTRIB_CAP", None)
        if cap is not None:
            c = min(c, float(cap))

        if c <= 0:
            continue

        for vibe, w in wmap.items():
            per_vibe[vibe].append({
                "prompt_id": pid,
                "prompt_text": ptxt,
                "raw_score": float(scores[i]),
                "softmax_p": float(p[i]),
                "confidence": float(conf),
                "contrib": float(c),
                "weight": float(w),
                "weighted_score": float(c * float(w)),
            })

    # Finalize per_vibe_top and final_vibe_scores
    per_vibe_top = {}
    final_vibe_scores = {}

    for vibe, items in per_vibe.items():
        items_sorted = sorted(items, key=lambda d: d["weighted_score"], reverse=True)
        per_vibe_top[vibe] = items_sorted[:topk_per_vibe]

    if vibe_score_mode == "max":
        for vibe, val in vibe_accum.items():
            final_vibe_scores[vibe] = float(val)  # will be overwritten below if using max logic
        # For "max", compute max evidence instead of sum
        for vibe, items in per_vibe.items():
            if not items:
                continue
            final_vibe_scores[vibe] = float(max(d["weighted_score"] for d in items))
    elif vibe_score_mode == "top2_mean":
        for vibe, items in per_vibe.items():
            if not items:
                continue
            vals = sorted([d["weighted_score"] for d in items], reverse=True)
            final_vibe_scores[vibe] = float(np.mean(vals[:min(2, len(vals))]))
    else:
        # Default: sum of confidence-weighted contributions across all prompts
        for vibe, val in vibe_accum.items():
            final_vibe_scores[vibe] = float(val)

    return per_vibe_top, final_vibe_scores


def compute_prompt_probs_and_conf(scores):
    """
    Inputs:
      scores: array-like raw prompt scores for ONE image (already S_prom from pixel stage)

    Returns:
      p: softmax probs over all prompts, shape (N,)
      conf: float in [0,1]
      contrib: p * conf, shape (N,)
    """
    scores = np.asarray(scores, dtype=np.float32)

    T = getattr(cfg, "SOFTMAX_T", 1.0)
    p = _softmax_np(scores, T=T)

    K_ent = getattr(cfg, "ENT_TOPK", 10)
    conf = _topk_entropy_conf(p, K_ent)

    contrib = p * conf
    return p, conf, contrib


@torch.no_grad()
def get_mask_for_prompt(pil_img, prompt, out_size=None):
    """
    Returns: mask float32 in [0,1], shape (H,W) at out_size if provided else original size.
    """
    init_clipseg()
    inputs = processor(text=[prompt], images=[pil_img], padding=True, return_tensors="pt").to(cfg.DEVICE)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.contiguous()
    logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits).detach().cpu().numpy()  # (H,W)

    if out_size is not None:
        mask = Image.fromarray((probs * 255).astype(np.uint8)).resize(out_size, Image.BILINEAR)
        probs = np.array(mask).astype(np.float32) / 255.0
    return probs


def _normalize_map(a, eps=1e-6):
    a = a.astype(np.float32)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < eps:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn + eps)


def _safe_stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def _save_top1_vibe_maps_for_image(pil_img, img_name, per_vibe_top, out_root, score_res=352):
    """
    Saves one grayscale heatmap PNG per vibe using the top-1 prompt for that vibe.
    Also saves a small CSV mapping vibe -> prompt_text for this image.
    """
    init_clipseg()
    stem = _safe_stem(img_name)
    out_dir = _ensure_dir(os.path.join(out_root, stem))

    # Collect vibe -> top1 prompt
    vibe_prompt = []
    for vibe, items in per_vibe_top.items():
        if not items:
            continue
        ptxt = items[0].get("prompt_text", None)
        if ptxt:
            vibe_prompt.append((vibe, ptxt))

    if not vibe_prompt:
        return

    vibes = [vp[0] for vp in vibe_prompt]
    prompts = [vp[1] for vp in vibe_prompt]

    # Resize only for CLIPSeg inference
    w, h = pil_img.size
    scale = score_res / max(w, h)
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_small = pil_img.resize((new_w, new_h))
    else:
        img_small = pil_img

    inputs = processor(
        text=prompts,
        images=[img_small] * len(prompts),
        padding=True,
        return_tensors="pt"
    ).to(cfg.DEVICE)

    outputs = model(**inputs)
    logits = outputs.logits  # (M, Hm, Wm)

    # Upsample to original size
    logits_up = F.interpolate(
        logits.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)  # (M, h, w)

    # Save each vibe heatmap as uint8 PNG
    for j, vibe in enumerate(vibes):
        heat = logits_up[j].detach().float().cpu().numpy()
        heat_n = _normalize_map(heat)
        png = (heat_n * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(png, mode="L").save(os.path.join(out_dir, f"{vibe}.png"))

    # Save mapping CSV for this image (for later labeling without parsing notebooks)
    map_df = pd.DataFrame({
        "vibe": vibes,
        "top1_prompt": prompts,
    })
    map_df.to_csv(os.path.join(out_dir, "_vibe_to_top1_prompt.csv"), index=False)


def load_prompt_library(csv_path: str):
    df_prompts = pd.read_csv(csv_path)
    all_prompts = df_prompts["prompt_text"].tolist()
    prompt_ids = df_prompts["prompt_id"].tolist()
    prompt_id_to_vibes = {}
    for _, row in df_prompts.iterrows():
        prompt_id_to_vibes[row["prompt_id"]] = {
            row["vibe_1"]: float(row["w_1"]),
            row["vibe_2"]: float(row["w_2"]),
            row["vibe_3"]: float(row["w_3"]),
        }
    prompt_text_to_id = dict(zip(all_prompts, prompt_ids))
    return df_prompts, all_prompts, prompt_ids, prompt_id_to_vibes, prompt_text_to_id


def build_results_by_image_from_export_csvs(study_folder: str) -> tuple[dict[str, dict], list[str]]:
    """Rebuild ``results_by_image`` slices from CSVs (contrib / softmax / raw per prompt).

    Used to regenerate ``previews_clipseg`` without rerunning ``run_scoring_loop``. Overlay PNGs still
    run CLIPSeg forward passes; only the numeric vectors are loaded from disk.

    Expects ``study_folder/clipseg_topk_prompts.csv`` and ``study_folder/vibe_scores_per_image.csv``.
    Prompt order follows ``cfg.PROMPT_LIBRARY_CSV`` (same as the pipeline exports).
    """
    prompts_csv = os.path.join(study_folder, "clipseg_topk_prompts.csv")
    vibes_csv = os.path.join(study_folder, "vibe_scores_per_image.csv")
    if not os.path.isfile(prompts_csv):
        raise FileNotFoundError(prompts_csv)
    if not os.path.isfile(vibes_csv):
        raise FileNotFoundError(vibes_csv)

    _df_lib, all_prompts, prompt_ids, _pm, _tid = load_prompt_library(cfg.PROMPT_LIBRARY_CSV)
    n_prompts = len(all_prompts)
    pid_to_i = {}
    txt_to_i = {}
    for i, (pid, ptxt) in enumerate(zip(prompt_ids, all_prompts)):
        pid_to_i[str(pid)] = i
        txt_to_i[str(ptxt).strip()] = i

    df_paths = pd.read_csv(vibes_csv)
    if "image" not in df_paths.columns or "image_path" not in df_paths.columns:
        raise ValueError("vibe_scores_per_image.csv must have 'image' and 'image_path' columns.")

    paths_by_image = (
        df_paths.drop_duplicates(subset=["image"], keep="first")
        .set_index("image")["image_path"]
        .astype(str)
        .to_dict()
    )

    df_scores = pd.read_csv(prompts_csv)
    req = {"image", "prompt_id", "raw_score", "softmax_p", "confidence", "contrib"}
    missing_cols = req - set(df_scores.columns)
    if missing_cols:
        raise ValueError(f"clipseg_topk_prompts.csv missing columns: {sorted(missing_cols)}")

    results_by_image: dict[str, dict] = {}

    for img_name, grp in df_scores.groupby("image", sort=False):
        path = paths_by_image.get(img_name)
        if path is None or path == "" or path.lower() == "nan":
            print(f"[preview reload] skip {img_name!r}: no image_path in vibe_scores_per_image.csv")
            continue
        contrib = np.zeros(n_prompts, dtype=np.float32)
        raw_scores = np.zeros(n_prompts, dtype=np.float32)
        softmax_p = np.zeros(n_prompts, dtype=np.float32)

        for _, row in grp.iterrows():
            pid_raw = row["prompt_id"]
            ix = pid_to_i.get(str(pid_raw))
            if ix is None:
                ix = txt_to_i.get(str(row.get("prompt_text", "")).strip())
            if ix is None:
                print(f"[preview reload] skip unknown prompt_id={pid_raw!r} for image={img_name!r}")
                continue
            contrib[ix] = float(row["contrib"])
            raw_scores[ix] = float(row["raw_score"])
            softmax_p[ix] = float(row["softmax_p"])

        results_by_image[img_name] = {
            "image_path": path,
            "raw_scores": raw_scores,
            "softmax_p": softmax_p,
            "confidence": float(grp["confidence"].iloc[0]),
            "contrib": contrib,
        }

    return results_by_image, all_prompts


def run_scoring_loop(
    image_paths,
    all_prompts,
    prompt_ids,
    prompt_id_to_vibes,
):
    """Notebook cell 17 core loop; returns RESULTS_BY_IMAGE dict."""
    init_clipseg()
    results_by_image = {}

    vibe_maps_root = None
    if SAVE_VIBE_TOP1_MAPS:
        vibe_maps_root = _ensure_dir(os.path.join(cfg.OUTPUT_FOLDER, VIBE_MAPS_DIRNAME))
        print("Will save vibe top-1 maps to:", vibe_maps_root)

    for path in tqdm(
        image_paths,
        desc="Scoring images",
        unit="img",
        dynamic_ncols=True,
    ):
        img = load_rgb(path)
        img_name = os.path.basename(path)

        raw_scores = score_prompts_for_image(
            img,
            all_prompts,
            score_res=cfg.SCORE_RES,
            batch_size=cfg.MAX_PROMPTS_PER_BATCH,
            top_pct=getattr(cfg, "TOP_PCT", 0.05)
        )

        p, conf, contrib = compute_prompt_probs_and_conf(raw_scores)

        per_vibe_top, final_vibe_scores = score_vibes_from_prompt_scores(
            scores=raw_scores,
            prompt_ids=prompt_ids,
            prompt_texts=all_prompts,
            prompt_id_to_vibes=prompt_id_to_vibes,
            candidate_top_m=40,
            topk_per_vibe=4,
            vibe_score_mode="sum"
        )

        if SAVE_VIBE_TOP1_MAPS and vibe_maps_root is not None:
            _save_top1_vibe_maps_for_image(
                pil_img=img,
                img_name=img_name,
                per_vibe_top=per_vibe_top,
                out_root=vibe_maps_root,
                score_res=VIBE_MAP_SCORE_RES
            )

        results_by_image[img_name] = {
            "image_path": path,
            "raw_scores": raw_scores.astype(np.float32),
            "softmax_p": p.astype(np.float32),
            "confidence": float(conf),
            "contrib": contrib.astype(np.float32),
            "final_vibe_scores": dict(final_vibe_scores),
            "per_vibe_top": dict(per_vibe_top),
        }

    return results_by_image


def export_clipseg_topk_prompts_csv(results_by_image, prompt_ids, all_prompts, out_path):
    rows_prompts = []

    for img_name, rec in results_by_image.items():
        raw_scores = rec["raw_scores"]
        softmax_p = rec["softmax_p"]
        conf = rec["confidence"]
        contrib = rec["contrib"]

        for pid, ptext, sc, pi, ci in zip(prompt_ids, all_prompts, raw_scores, softmax_p, contrib):
            rows_prompts.append({
                "image": img_name,
                "prompt_id": pid,
                "prompt_text": ptext,
                "raw_score": float(sc),
                "softmax_p": float(pi),
                "confidence": float(conf),
                "contrib": float(ci),
            })

    fieldnames_prompts = [
        "image",
        "prompt_id",
        "prompt_text",
        "raw_score",
        "softmax_p",
        "confidence",
        "contrib",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_prompts)
        w.writeheader()
        for r in rows_prompts:
            w.writerow(r)

    print("Saved:", out_path)


def export_clipseg_topk_per_vibe_csv(results_by_image, out_path):
    rows = []

    for img_name, rec in results_by_image.items():
        per_vibe_top = rec["per_vibe_top"]
        final_vibe_scores = rec["final_vibe_scores"]

        for vibe, top_items in per_vibe_top.items():
            vibe_score = float(final_vibe_scores.get(vibe, 0.0))

            for rank_in_vibe, d in enumerate(top_items, start=1):
                rows.append({
                    "image": img_name,
                    "vibe": vibe,
                    "vibe_score": vibe_score,
                    "rank_in_vibe": rank_in_vibe,
                    "prompt_id": d["prompt_id"],
                    "prompt_text": d["prompt_text"],
                    "raw_score": float(d["raw_score"]),
                    "softmax_p": float(d["softmax_p"]),
                    "confidence": float(d["confidence"]),
                    "contrib": float(d["contrib"]),
                    "weight": float(d["weight"]),
                    "weighted_score": float(d["weighted_score"]),
                })

    fieldnames = [
        "image", "vibe", "vibe_score", "rank_in_vibe",
        "prompt_id", "prompt_text",
        "raw_score", "softmax_p", "confidence", "contrib",
        "weight", "weighted_score"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Saved:", out_path)


def load_image_coords(image_folder: str) -> dict[str, dict]:
    """Load lat/lng per image basename from walk_manifest.json or corners_manifest.json."""
    folder = os.path.abspath(image_folder)
    coords: dict[str, dict] = {}

    corners_path = os.path.join(folder, "corners_manifest.json")
    if os.path.isfile(corners_path):
        with open(corners_path, encoding="utf-8") as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            img = node.get("image")
            if not img or "lat" not in node or "lng" not in node:
                continue
            entry: dict = {"lat": float(node["lat"]), "lng": float(node["lng"])}
            if node.get("heading") is not None:
                entry["heading"] = float(node["heading"])
            if node.get("osm_node_id") is not None:
                entry["osm_node_id"] = int(node["osm_node_id"])
            coords[img] = entry
        if coords:
            print(f"Loaded coords for {len(coords)} images from corners_manifest.json")
            return coords

    walk_path = os.path.join(folder, "walk_manifest.json")
    if os.path.isfile(walk_path):
        with open(walk_path, encoding="utf-8") as f:
            data = json.load(f)
        for frame in data.get("frames", []):
            img = frame.get("file")
            if not img or "lat" not in frame or "lng" not in frame:
                continue
            entry = {"lat": float(frame["lat"]), "lng": float(frame["lng"])}
            if frame.get("heading") is not None:
                entry["heading"] = float(frame["heading"])
            coords[img] = entry
        if coords:
            print(f"Loaded coords for {len(coords)} images from walk_manifest.json")
            return coords

    return coords


def export_vibe_scores_per_image_csv(
    results_by_image,
    image_paths,
    out_path,
    image_coords: dict[str, dict] | None = None,
):
    all_vibes = sorted({
        vibe
        for r in results_by_image.values()
        for vibe in r["final_vibe_scores"].keys()
    })
    image_coords = image_coords or {}
    include_coords = bool(image_coords)
    extra_cols: list[str] = []
    if include_coords:
        extra_cols = ["lat", "lng"]
        if any("heading" in image_coords.get(os.path.basename(p), {}) for p in image_paths):
            extra_cols.append("heading")
        if any("osm_node_id" in image_coords.get(os.path.basename(p), {}) for p in image_paths):
            extra_cols.append("osm_node_id")

    rows = []
    for path in image_paths:
        img_name = os.path.basename(path)
        r = results_by_image.get(img_name, None)
        if r is None:
            continue

        vibe_scores = r["final_vibe_scores"]

        row = {
            "image": img_name,
            "image_path": r["image_path"],
            "confidence": r["confidence"],
        }
        if include_coords:
            c = image_coords.get(img_name, {})
            for col in extra_cols:
                row[col] = c.get(col)
        for v in all_vibes:
            row[v] = float(vibe_scores.get(v, 0.0))

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["image", "image_path", "confidence"] + extra_cols + all_vibes]

    df.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print("Rows:", len(df), "Cols:", len(df.columns))
    if include_coords:
        matched = df["lat"].notna().sum()
        print(f"Coords attached: {matched}/{len(df)} rows")
    return df


def run_district_vibe_selection(image_folder: str):
    """Notebook cell 28: reads vibe_scores_per_image.csv, writes district_selected_vibes.csv."""
    vibe_csv_path = os.path.join(image_folder, "vibe_scores_per_image.csv")
    df = pd.read_csv(vibe_csv_path)

    vibe_cols = [c for c in df.columns if c not in SCORES_META_COLUMNS]

    stats = []

    for v in vibe_cols:
        mean_val = df[v].mean()
        std_val = df[v].std()
        conf_weighted_mean = (df[v] * df["confidence"]).sum() / df["confidence"].sum()
        stats.append({
            "vibe": v,
            "mean": mean_val,
            "std": std_val,
            "conf_weighted_mean": conf_weighted_mean
        })

    stats_df = pd.DataFrame(stats)

    stats_df = stats_df.sort_values("conf_weighted_mean", ascending=False)

    top_high = stats_df.head(3)
    top_low = stats_df.tail(3)

    if top_high["conf_weighted_mean"].iloc[0] - top_high["conf_weighted_mean"].iloc[2] > \
       top_low["conf_weighted_mean"].iloc[2] - top_low["conf_weighted_mean"].iloc[0]:
        stable_selected = list(top_high["vibe"].iloc[:2]) + list(top_low["vibe"].iloc[:1])
    else:
        stable_selected = list(top_high["vibe"].iloc[:1]) + list(top_low["vibe"].iloc[:2])

    transitional_vibe = stats_df.sort_values("std", ascending=False)["vibe"].iloc[0]

    corr = df[vibe_cols].corr()
    _corr_np = np.array(corr.to_numpy(dtype=float), copy=True)
    np.fill_diagonal(_corr_np, 0)
    corr = pd.DataFrame(_corr_np, index=corr.index, columns=corr.columns)

    corr_masked = corr.copy()
    corr_masked[(corr_masked.abs() >= 0.9)] = 0

    abs_corr = corr_masked.abs()

    if abs_corr.max().max() == 0:
        tension_pair = None
    else:
        idx = abs_corr.stack().idxmax()
        tension_pair = list(idx)

    tension_for_set = tension_pair if tension_pair is not None else []
    selected_vibes = list(set(stable_selected + [transitional_vibe] + tension_for_set))

    print("Selected Stable Extremes:", stable_selected)
    print("Selected Transitional:", transitional_vibe)
    print("Selected Tension Pair:", tension_pair)
    print("Final Selected Vibes:", selected_vibes)

    if len(selected_vibes) < MIN_VIBES:
        ranked = (
            stats_df.sort_values("conf_weighted_mean", ascending=False)["vibe"]
            .tolist()
        )
        for v in ranked:
            if v not in selected_vibes:
                selected_vibes.append(v)
            if len(selected_vibes) >= MIN_VIBES:
                break

    meta_cols = [c for c in df.columns if c in SCORES_META_COLUMNS and c not in selected_vibes]
    final_df = df[meta_cols + selected_vibes]

    out_path = os.path.join(image_folder, "district_selected_vibes.csv")
    final_df.to_csv(out_path, index=False)

    print("Exported:", out_path)
    return final_df, stats_df


def print_pipeline_cfg():
    print(
        "PIPELINE CFG |",
        "TOP_PCT_5:", cfg.TOP_PCT_5,
        "TOP_PCT_20:", cfg.TOP_PCT_20,
        "ALPHA:", cfg.ALPHA_PROM,
        "T:", cfg.SOFTMAX_T,
        "ENT_TOPK:", cfg.ENT_TOPK
    )


def run_full_pipeline():
    """End-to-end: load prompts, score images, export CSVs, district selection."""
    print_pipeline_cfg()

    _, all_prompts, prompt_ids, prompt_id_to_vibes, _ = load_prompt_library(cfg.PROMPT_LIBRARY_CSV)
    print("Total prompts:", len(all_prompts))
    print("Example prompt:", prompt_ids[0], all_prompts[0], prompt_id_to_vibes[prompt_ids[0]])

    image_paths = list_images(cfg.IMAGE_FOLDER)
    print("Runner using images:", len(image_paths))

    os.makedirs(cfg.OUTPUT_FOLDER, exist_ok=True)
    print("Outputs directory:", cfg.OUTPUT_FOLDER)

    results_by_image = run_scoring_loop(
        image_paths, all_prompts, prompt_ids, prompt_id_to_vibes
    )
    print("Cached results for images:", len(results_by_image))
    print("Example keys:", list(results_by_image.keys())[:5])

    out_root = cfg.OUTPUT_FOLDER
    export_clipseg_topk_prompts_csv(
        results_by_image, prompt_ids, all_prompts,
        os.path.join(out_root, "clipseg_topk_prompts.csv"),
    )
    export_clipseg_topk_per_vibe_csv(
        results_by_image,
        os.path.join(out_root, "clipseg_topk_per_vibe.csv"),
    )
    image_coords = load_image_coords(cfg.IMAGE_FOLDER)
    export_vibe_scores_per_image_csv(
        results_by_image, image_paths,
        os.path.join(out_root, "vibe_scores_per_image.csv"),
        image_coords=image_coords,
    )

    run_district_vibe_selection(out_root)

    return results_by_image, image_paths, prompt_ids, all_prompts


def sanity_check_sample(results_by_image, all_prompts, test_img_name=None):
    """Notebook cell 20."""
    if test_img_name is None:
        test_img_name = list(results_by_image.keys())[0]
    rec = results_by_image[test_img_name]

    raw_scores = rec["raw_scores"]
    p = rec["softmax_p"]
    conf = rec["confidence"]
    contrib = rec["contrib"]
    final_vibe_scores = rec["final_vibe_scores"]
    per_vibe_top = rec["per_vibe_top"]

    print("IMAGE:", test_img_name)
    print("CONFIDENCE:", float(conf))

    topN = 10
    idx = np.argsort(-contrib)[:topN]
    print("\nTop prompts by contrib (softmax * confidence):")
    for r, i in enumerate(idx, start=1):
        print(f"{r:02d}. contrib={contrib[i]:.6f}  p={p[i]:.6f}  raw={raw_scores[i]:.4f}  | {all_prompts[i]}")

    v_items = sorted(final_vibe_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop vibes:")
    for r, (v, sc) in enumerate(v_items, start=1):
        print(f"{r:02d}. {v:15s} {sc:.6f}")

    print("\nEvidence (top 3 vibes):")
    for v, _ in v_items[:3]:
        print(f"\nVIBE: {v}")
        for d in per_vibe_top.get(v, [])[:4]:
            print(
                f"  weighted={d['weighted_score']:.6f}  contrib={d['contrib']:.6f}  "
                f"p={d['softmax_p']:.6f}  raw={d['raw_score']:.4f}  | {d['prompt_text']}"
            )
