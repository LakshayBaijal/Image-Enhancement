"""Step 3: cut the product out. SAM 3 decides WHICH object, BiRefNet decides the EDGE.

  detect   SAM 3 is asked for "<sub-category> product", "<category> product", "product package",
           "product" in that order; the first prompt with an instance wins, and among its instances
           the best score x area^0.3 is the product (one object, not the whole shelf).
           Package check: if a "product package" instance holds >= 85 % of the winner and is
           >= 1.3x larger, the winner was food PRINTED ON the pack and the package is used.
  segment  BiRefNet (1024 px) on the box grown by 15 %, AND-ed with SAM's own mask dilated by
           3 % of the box's short side (so a touching neighbour pack is excluded). Thick chunks
           SAM is sure of but BiRefNet dropped are filled back; small holes are filled; the
           component overlapping the box most is kept.
  output   RGBA: the photo's pixels with the mask as alpha, cropped to the product + 12 px.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

PACKAGE_PROMPT = "product package"
PAD, CONTEXT = 12, 0.15


def _pick(boxes, scores, h, w, min_area=0.01):
    best, best_v = None, -1.0
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(h * w)
        if area < min_area:
            continue
        v = scores[i] * area ** 0.3
        if v > best_v:
            best, best_v = i, v
    return best


def _containment(inner, outer) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    iw, ih = max(0.0, min(ix2, ox2) - max(ix1, ox1)), max(0.0, min(iy2, oy2) - max(iy1, oy1))
    return iw * ih / max(1e-6, (ix2 - ix1) * (iy2 - iy1))


def detect(sam, image: np.ndarray, embeds, prompts: list[str], score: float, mask_thr: float) -> dict:
    h, w = image.shape[:2]
    tried, found = [], None
    for prompt in prompts:
        boxes, scores, masks = sam.detect((h, w), prompt, embeds, score)
        i = _pick(boxes, scores, h, w)
        tried.append({"prompt": prompt, "instances": int(len(scores)),
                      "best_score": round(float(scores.max()), 3) if len(scores) else None})
        if i is not None:
            found = (prompt, boxes[i], float(scores[i]), masks[i])
            break
    override = None
    if found and found[0] != PACKAGE_PROMPT:
        pb, ps, pm = sam.detect((h, w), PACKAGE_PROMPT, embeds, score)
        wb = found[1]
        warea = (wb[2] - wb[0]) * (wb[3] - wb[1])
        best = None
        for j in range(len(ps)):
            parea = (pb[j][2] - pb[j][0]) * (pb[j][3] - pb[j][1])
            if _containment(wb, pb[j]) >= 0.85 and parea >= 1.3 * warea and (best is None or ps[j] > ps[best]):
                best = j
        if best is not None:
            override = f"'{found[0]}' matched something printed on the pack; the package around it is used"
            found = (PACKAGE_PROMPT, pb[best], float(ps[best]), pm[best])
    if not found:
        return {"box": None, "score": None, "prompt_used": None, "prompts_tried": tried, "package_override": None,
                "sam_mask": None}
    prompt, box, s, logit = found
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return {"box": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)], "score": round(s, 3), "prompt_used": prompt,
            "prompts_tried": tried, "package_override": override, "sam_mask": sam.mask_at(logit, h, w, mask_thr)}


def _grow(box, w, h, r=CONTEXT):
    x1, y1, x2, y2 = box
    gw, gh = (x2 - x1) * r, (y2 - y1) * r
    return int(max(0, x1 - gw)), int(max(0, y1 - gh)), int(min(w, x2 + gw)), int(min(h, y2 + gh))


def _keep_one(binary: np.ndarray, box, min_frac=0.01) -> np.ndarray:
    labeled, n = ndimage.label(binary)
    if n <= 1:
        return binary
    h, w = binary.shape
    best, best_v = None, None
    for i in range(1, n + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area < min_frac * h * w:
            continue
        if box is not None:
            x1, y1, x2, y2 = box
            v = -comp[y1:y2, x1:x2].sum() / area
        else:
            ys, xs = np.nonzero(comp)
            v = ((ys.mean() - h / 2) ** 2 + (xs.mean() - w / 2) ** 2) ** 0.5
        if best_v is None or v < best_v:
            best, best_v = i, v
    return binary if best is None else labeled == best


def segment(birefnet, image: np.ndarray, det: dict, size: int, gate_grow: float, fill_thick: float) -> tuple:
    h, w = image.shape[:2]
    box, sam_mask = det.get("box"), det.get("sam_mask")
    cx1, cy1, cx2, cy2 = _grow(box, w, h) if box else (0, 0, w, h)
    full = np.zeros((h, w), np.float32)
    full[cy1:cy2, cx1:cx2] = birefnet.prob(image[cy1:cy2, cx1:cx2], size)
    raw = full >= 0.5
    stats = {"crop": [cx1, cy1, cx2, cy2], "gated_by_sam": False}
    if sam_mask is not None and box is not None:
        r = max(5, int(round(gate_grow * min(box[2] - box[0], box[3] - box[1]))))
        gate = cv2.dilate(sam_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        sam = sam_mask > 0
        stats.update(gated_by_sam=True, removed_by_gate=round(float((raw & ~gate).sum()) / max(1, int(raw.sum())), 4))
        raw = raw & gate
        fill_px = max(4.0, fill_thick * min(box[2] - box[0], box[3] - box[1]))
        missed = (sam & ~raw).astype(np.uint8)
        filled = np.zeros_like(raw)
        n, lab = cv2.connectedComponents(missed)
        if n > 1:
            thick = ndimage.maximum(cv2.distanceTransform(missed, cv2.DIST_L2, 5), lab, index=np.arange(1, n))
            for k, t in enumerate(thick, start=1):
                if t >= fill_px:
                    filled |= lab == k
        raw = raw | filled
        stats["filled_from_sam"] = round(float(filled.sum()) / max(1, int(raw.sum())), 4)
    holes = ndimage.binary_fill_holes(raw) & ~raw
    if holes.any():
        hl, hn = ndimage.label(holes)
        sizes = ndimage.sum(holes, hl, index=np.arange(1, hn + 1))
        raw = raw | np.isin(hl, 1 + np.nonzero(sizes < 0.005 * max(1, int(raw.sum())))[0])
    mask = _keep_one(raw, box).astype(np.uint8) * 255
    stats["area_frac"] = round(float((mask > 0).mean()), 4)
    return mask, stats


def silhouette(image: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0, y1 = max(0, ys.min() - PAD), min(h, ys.max() + PAD + 1)
    x0, x1 = max(0, xs.min() - PAD), min(w, xs.max() + PAD + 1)
    return Image.fromarray(np.dstack([image, mask])[y0:y1, x0:x1], "RGBA")


def debug_overlay(image: np.ndarray, det: dict, mask: np.ndarray) -> np.ndarray:
    vis = image.copy()
    if det.get("sam_mask") is not None:
        m = det["sam_mask"] > 0
        vis[m] = (0.6 * vis[m] + 0.4 * np.array([60, 60, 200])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t = max(2, image.shape[1] // 400)
    cv2.drawContours(vis, cnts, -1, (60, 220, 60), t)
    if det.get("box"):
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 165, 0), t)
    return vis
