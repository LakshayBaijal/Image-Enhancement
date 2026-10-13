"""Step 2: remove the hand holding the product, on the full photo, before the cut-out.

Mask (same rules as the ONDC FLUX.2 hand removal):
  - SAM 3 is asked for "hand", "finger" and "arm"; every instance over 0.3 goes in the mask
    (masking only the fingers made the painter grow a new hand from what was left);
  - in "auto" mode the step runs only when a "hand" scores 0.6 or more;
  - the mask is grown by 1.2 % of the photo's long side.

Painters:
  lama  LaMa (Apache-2.0, ONNX, CPU). Runs anywhere, including free hosting. It continues the
        surrounding texture; it cannot redraw printed artwork the hand covered.
  flux  FLUX.2 Klein 4B + LanPaint on a ComfyUI server (GPU). Redraws the hidden part of the
        pack. Used when FLUX_COMFY_URL points at a running server.

Either way only pixels inside the mask change; everything outside is copied from the photo
and checked.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageFilter

MASK_PROMPTS = ("hand", "finger", "arm")
TRIGGER_PROMPT = "hand"


def hand_mask(sam, image: np.ndarray, embeds, trigger: float | None, score: float, grow: float) -> tuple:
    """-> (mask uint8 0/1 or None, info). trigger=None: mask whatever SAM 3 finds."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    found = []
    for prompt in MASK_PROMPTS:
        _, scores, logits = sam.detect((h, w), prompt, embeds, score)
        for s, logit in zip(scores, logits):
            mask |= sam.mask_at(logit, h, w)
            found.append({"prompt": prompt, "score": round(float(s), 3)})
    top = max((f["score"] for f in found if f["prompt"] == TRIGGER_PROMPT), default=0.0)
    info = {"instances": found, "top_hand_score": top, "trigger": trigger, "mask_score": score}
    if not mask.any():
        return None, dict(info, reason="SAM 3 found no hand, finger or arm")
    if trigger is not None and top < trigger:
        return None, dict(info, reason=f"no hand scored {trigger} or more (best {top})")
    r = max(3, round(grow * max(h, w)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask, dict(info, grow_px=r, coverage=round(float(mask.mean()), 4))


def compose(source: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Candidate inside the mask, feathered 2 px inward; every pixel outside is the source's."""
    support = mask > 0
    soft = np.asarray(Image.fromarray((support * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
                      dtype=np.float32) / 255
    soft[~support] = 0
    out = np.rint(source * (1 - soft[..., None]) + candidate * soft[..., None]).astype(np.uint8)
    out[~support] = source[~support]
    return out


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = (0.55 * image + 0.45 * np.array([235, 45, 45])).astype(np.uint8)
    return np.where(mask[..., None] > 0, red, image)


# ----------------------------------------------------------------------------- LaMa
def fill_lama(lama, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """LaMa works at 512 px, so it gets a crop around the hand (plus context), not the whole photo."""
    h, w = image.shape[:2]
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = int(max(32, 0.5 * max(x2 - x1, y2 - y1)))
    cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
    crop, hole = image[cy1:cy2, cx1:cx2], mask[cy1:cy2, cx1:cx2]
    filled = lama(crop, hole)
    candidate = image.copy()
    candidate[cy1:cy2, cx1:cx2] = filled
    return compose(image, candidate, mask), {"engine": "LaMa (ONNX)", "crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
                                             "working_px": 512}


# ----------------------------------------------------------------------------- FLUX.2 on ComfyUI
FLUX_VAE, FLUX_CLIP = "flux2-vae.safetensors", "qwen_3_4b.safetensors"
FLUX_MAX_EDGE, FLUX_STEPS, FLUX_LANPAINT_STEPS, FLUX_SEED = 768, 4, 5, 42


def _cond_name(prompt: str) -> str:
    """Saved-conditioning file name, same scheme as FLUX2 Hand Removal/build_conditioning.py."""
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "empty"
    return f"{slug}-{hashlib.sha1(prompt.encode()).hexdigest()[:8]}.safetensors"


def flux_available(server: str) -> bool:
    if not server:
        return False
    try:
        import requests
        return requests.get(f"{server}/system_stats", timeout=3).ok
    except Exception:
        return False


def _flux_graph(ref: str, width: int, height: int, prompt: str, unet: str, saved: set[str], prefix: str) -> dict:
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    g = {
        "1": node("UnetLoaderGGUF", unet_name=unet) if unet.endswith(".gguf") else node("UNETLoader", unet_name=unet, weight_dtype="default"),
        "3": node("VAELoader", vae_name=FLUX_VAE),
        "4": node("LoadImage", image=ref),
        "7": node("LanPaint_ImageEncode", image=["4", 0], vae=["3", 0], mask=["4", 1]),
        "10": node("CFGGuider", model=["1", 0], positive=["5", 0], negative=["6", 0], cfg=1.0),
        "11": node("KSamplerSelect", sampler_name="euler"),
        "12": node("Flux2Scheduler", steps=FLUX_STEPS, width=width, height=height),
        "13": node("RandomNoise", noise_seed=FLUX_SEED),
        "14": node("LanPaint_SamplerCustomAdvanced", noise=["13", 0], guider=["10", 0], sampler=["11", 0],
                   sigmas=["12", 0], latent_image=["7", 0], LanPaint_NumSteps=FLUX_LANPAINT_STEPS, LanPaint_Lambda=5.0,
                   LanPaint_StepSize=0.2, LanPaint_PromptMode="Image First", LanPaint_Info="Product occlusion repair"),
        "15": node("LanPaint_ImageDecode", samples=["14", 0], vae=["3", 0], image=["4", 0], mask=["4", 1], blend_overlap=9),
        "16": node("SaveImage", images=["15", 0], filename_prefix=prefix),
    }
    for key, text in (("5", prompt), ("6", "")):
        name = _cond_name(text)
        if name in saved:          # prompt encoded ahead of time: the 8 GB text encoder never loads
            g[key] = node("ConditioningLoader", conditioning_name=name)
        else:
            g["2"] = node("CLIPLoader", clip_name=FLUX_CLIP, type="flux2", device="default")
            g[key] = node("CLIPTextEncode", clip=["2", 0], text=text)
    return g


def fill_flux(server: str, unet: str, image: np.ndarray, mask: np.ndarray, prompt: str, timeout: float = 900) -> tuple:
    import requests
    h, w = image.shape[:2]
    f = min(1, FLUX_MAX_EDGE / max(w, h))
    size = tuple(max(32, round(n * f / 16) * 16) for n in (w, h))
    rgba = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    rgba.putalpha(Image.fromarray(255 - (mask > 0).astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    name = f"enh_{uuid.uuid4().hex[:12]}.png"
    t0 = time.perf_counter()
    up = requests.post(f"{server}/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                       data={"subfolder": "enhancer", "type": "input", "overwrite": "true"}, timeout=60)
    up.raise_for_status()
    u = up.json()
    ref = f"{u['subfolder']}/{u['name']}" if u.get("subfolder") else u["name"]
    try:
        info = requests.get(f"{server}/object_info/ConditioningLoader", timeout=10).json()
        saved = set(info["ConditioningLoader"]["input"]["required"]["conditioning_name"][0])
    except Exception:
        saved = set()
    graph = _flux_graph(ref, *size, prompt, unet, saved, f"enhancer/{name[:-4]}")
    r = requests.post(f"{server}/prompt", json={"prompt": graph}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"ComfyUI refused the graph: {r.text[:300]}")
    pid = r.json()["prompt_id"]
    while True:
        hist = requests.get(f"{server}/history/{pid}", timeout=30).json().get(pid)
        if hist:
            break
        if time.perf_counter() - t0 > timeout:
            requests.post(f"{server}/interrupt", timeout=30)
            raise TimeoutError(f"FLUX.2 took more than {timeout:.0f} s")
        time.sleep(1)
    if hist["status"]["status_str"] != "success":
        raise RuntimeError(f"ComfyUI failed: {str(hist['status'])[:300]}")
    out = hist["outputs"]["16"]["images"][0]
    got = requests.get(f"{server}/view", params={"filename": out["filename"], "subfolder": out["subfolder"],
                                                 "type": out["type"]}, timeout=60)
    got.raise_for_status()
    cand = np.asarray(Image.open(io.BytesIO(got.content)).convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
    used_encoder = "2" in graph
    return compose(image, cand, mask), {"engine": "FLUX.2 Klein 4B + LanPaint (ComfyUI)", "unet": unet,
                                        "working_size": list(size), "steps": FLUX_STEPS, "seed": FLUX_SEED,
                                        "prompt": prompt, "text_encoder_loaded": used_encoder}
