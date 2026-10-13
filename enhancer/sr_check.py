"""Step 4: SR Check. Does this cut-out need enhancement, or is it fine as it is? One yes/no.

Same logic and thresholds as the standalone SR Check (data/sr_thresholds.json), calibrated on
50 catalogue photos with hand labels; false positives cost money, so it leans to "ok as is".

Measured inside the cut-out only. Any hard failure -> needs enhancement:
  product_size     the product's short side is under 200 px
  text_*           a WEAK text line exists. A line is weak only when 2 of 3 signals agree:
                   height < 20 px, PP-OCRv5 confidence < 0.90, letter edges wider than 12 % of
                   the height. Lines touching the cut-out edge, detections covering > 25 % of the
                   product (logos), and a lone weak fragment of <= 3 characters are ignored.
Informational only: Laplacian sharpness (tracks resolution, not blur), per-tile blur.
"""
from __future__ import annotations

import json
import statistics
import time

import cv2
import numpy as np
from PIL import Image

from .config import SR_THRESHOLDS

HARD = ("product_size", "sharpness", "text_size", "text_confidence", "text_edge")


def thresholds() -> dict:
    return {k: v for k, v in json.loads(SR_THRESHOLDS.read_text()).items() if not k.startswith("_")}


def on_white(rgba: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """-> (BGR on white, product mask)."""
    a = np.asarray(rgba.convert("RGBA"))
    alpha = a[..., 3].astype(np.float32) / 255.0
    rgb = a[..., :3].astype(np.float32) * alpha[..., None] + 255 * (1 - alpha[..., None])
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), alpha > 0.5


def _laplacian_var(gray, mask):
    inner = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    if inner.sum() < 100:
        inner = mask
    v = cv2.Laplacian(gray, cv2.CV_64F)[inner]
    return float(v.var()) if v.size else 0.0


def _tile_blur(gray, mask, tile, min_cov):
    h, w = gray.shape
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            m = mask[y:y + tile, x:x + tile]
            if m.mean() >= min_cov:
                vals.append((float(lap[y:y + tile, x:x + tile][m].var()), x, y))
    if not vals:
        return {"tiles": 0}
    med = statistics.median(v for v, _, _ in vals)
    worst = min(vals)
    return {"tiles": len(vals), "median_lap_var": round(med, 1), "min_lap_var": round(worst[0], 1),
            "min_tile_xy": [worst[1], worst[2]], "min_over_median": round(worst[0] / med, 3) if med > 0 else None}


def edge_width_px(gray_crop: np.ndarray, min_contrast: int = 40) -> float | None:
    """Median edge spread across strong horizontal transitions: contrast / peak gradient."""
    g = cv2.GaussianBlur(gray_crop.astype(np.float32), (0, 0), 0.6)
    gx = np.gradient(g, axis=1)
    peak = np.abs(gx).max()
    if peak < 4:
        return None
    widths = []
    h, w = g.shape
    for y in range(h):
        row, d = g[y], gx[y]
        ad = np.abs(d)
        for x in range(1, w - 1):
            if ad[x] < 0.3 * peak or ad[x] < ad[x - 1] or ad[x] < ad[x + 1]:
                continue
            s = np.sign(d[x])
            l = x
            while l > 0 and np.sign(d[l - 1]) == s and ad[l - 1] > 0.05 * ad[x]:
                l -= 1
            r = x
            while r < w - 1 and np.sign(d[r + 1]) == s and ad[r + 1] > 0.05 * ad[x]:
                r += 1
            contrast = abs(row[r] - row[l])
            if contrast >= min_contrast:
                widths.append(contrast / ad[x])
    return float(statistics.median(widths)) if len(widths) >= 5 else None


def _ocr_lines(ocr, bgr, min_chars):
    res = ocr.predict(bgr)
    polys = res.get("rec_polys", []) or res.get("dt_polys", []) or []
    out = []
    for poly, text, conf in zip(polys, list(res.get("rec_texts", []) or []), list(res.get("rec_scores", []) or [])):
        if len(text.strip()) < min_chars:
            continue
        a = np.asarray(poly, np.float32).reshape(-1, 2)
        hgt = float(np.linalg.norm(a[0] - a[3]) + np.linalg.norm(a[1] - a[2])) / 2
        x1, y1 = a.min(0)
        x2, y2 = a.max(0)
        out.append({"text": text, "conf": round(float(conf), 4), "height_px": round(hgt, 1),
                    "box": [int(x1), int(y1), int(x2), int(y2)], "poly": a.round().astype(int).tolist()})
    return out


def check(ocr, rgba: Image.Image, T: dict | None = None) -> tuple[dict, np.ndarray]:
    """-> (record with needs_enhancement + every metric, evidence image RGB)."""
    T = T or thresholds()
    t0 = time.perf_counter()
    bgr, mask = on_white(rgba)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    checks, warnings = {}, []

    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(ys) else None
    short = min(box[2] - box[0], box[3] - box[1]) if box else 0
    small_warn = T.get("product_small_warn_px", 500)
    checks["product_size"] = {"short_side_px": short, "threshold_min_px": T["product_min_short_side_px"],
                              "pass": short >= T["product_min_short_side_px"]}
    if short < small_warn and checks["product_size"]["pass"]:
        warnings.append(f"small product ({short} px short side): fine if its text reads; the text checks decide")

    lap = _laplacian_var(gray, mask)
    hard_sharp = T.get("sharpness_is_hard_fail", False)
    checks["sharpness"] = {"laplacian_var": round(lap, 1), "threshold_min": T["sharpness_min_laplacian_var"],
                           "pass": lap >= T["sharpness_min_laplacian_var"] or not hard_sharp}
    tb = _tile_blur(gray, mask, T["local_blur_tile_px"], T["local_blur_min_mask_coverage"])
    lb_bad = bool(tb.get("tiles", 0) >= 4 and tb.get("min_over_median") is not None
                  and tb["min_over_median"] < T["local_blur_warn_ratio"])
    checks["local_blur"] = {**tb, "flagged": lb_bad, "pass": not (lb_bad and T["local_blur_is_hard_fail"])}
    if lb_bad:
        warnings.append(f"one area is {tb['min_over_median']:.2f}x the median sharpness (warning only)")

    t_ocr = time.perf_counter()
    all_lines = _ocr_lines(ocr, bgr, T["text_min_chars"])
    ocr_s = round(time.perf_counter() - t_ocr, 2)
    prod_area = max(1, (box[2] - box[0]) * (box[3] - box[1])) if box else W * H
    ignored, lines = [], []
    for ln in all_lines:
        bx = ln["box"]
        frac = (bx[2] - bx[0]) * (bx[3] - bx[1]) / prod_area
        if frac > T.get("text_max_box_area_frac_of_product", 0.25):
            ln["ignored_reason"] = "covers too much of the product to be a text line (logo / graphic)"
            ignored.append(ln)
        else:
            lines.append(ln)
    if T.get("text_ignore_lines_touching_cutout_edge", True):
        keep = []
        for ln in lines:
            x1, y1, x2, y2 = ln["box"]
            sub = mask[max(0, y1 - 2):min(H, y2 + 2), max(0, x1 - 2):min(W, x2 + 2)]
            if sub.size and float(1 - sub.mean()) > 0.0:
                ln["ignored_reason"] = "touches the cut-out edge: partial text or a neighbouring pack"
                ignored.append(ln)
            else:
                keep.append(ln)
        lines = keep
    need = int(T.get("text_weak_needs_signals", 2))
    for ln in lines:
        x1, y1, x2, y2 = ln["box"]
        pad = max(2, int(ln["height_px"] * 0.15))
        crop = gray[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
        ew = edge_width_px(crop) if crop.size and ln["height_px"] >= 6 else None
        ln["edge_width_frac"] = round(ew / ln["height_px"], 3) if ew is not None and ln["height_px"] > 0 else None
        sig = []
        if ln["height_px"] < T["text_min_line_height_px"]:
            sig.append("text_size")
        if ln["conf"] < T["text_min_ocr_confidence"]:
            sig.append("text_confidence")
        if ln["edge_width_frac"] is not None and ln["edge_width_frac"] > T["text_max_edge_width_frac"]:
            sig.append("text_edge")
        ln["signals"], ln["weak"] = sig, len(sig) >= need
    weak = [l for l in lines if l["weak"]]
    if len(weak) == 1 and len(weak[0]["text"].strip()) <= int(T.get("text_lone_fragment_max_chars", 3)):
        weak[0]["weak"] = False
        warnings.append(f"lone weak fragment '{weak[0]['text'][:12]}' ignored: not enough evidence on its own")
        weak = []
    if not lines:
        warnings.append("no text line detected: the text checks passed trivially")
    checks["text_present"] = {"lines": len(lines), "ignored": len(ignored), "pass": True}
    for name in ("text_size", "text_confidence", "text_edge"):
        checks[name] = {"weak_lines": sum(1 for l in weak if name in l["signals"]),
                        "pass": not any(name in l["signals"] for l in weak)}
    failed = [k for k in HARD if not checks[k]["pass"]]
    if T["local_blur_is_hard_fail"] and not checks["local_blur"]["pass"]:
        failed.append("local_blur")
    rec = {"needs_enhancement": bool(failed), "failed_checks": failed, "warnings": warnings, "checks": checks,
           "lines": [{k: v for k, v in l.items() if k != "poly"} for l in lines],
           "ignored": [{k: v for k, v in l.items() if k != "poly"} for l in ignored],
           "weak_lines": [l["text"] for l in weak], "cutout_size": [W, H],
           "seconds": round(time.perf_counter() - t0, 2), "ocr_seconds": ocr_s}

    vis = bgr.copy()
    lw = max(1, W // 600)
    for ln in ignored:
        cv2.polylines(vis, [np.asarray(ln["poly"], np.int32)], True, (160, 160, 160), lw)
    for ln in lines:
        col = (40, 40, 220) if ln["weak"] else ((20, 150, 230) if ln["signals"] else (40, 170, 40))
        cv2.polylines(vis, [np.asarray(ln["poly"], np.int32)], True, col, lw)
    return rec, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
