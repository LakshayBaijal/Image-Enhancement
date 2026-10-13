"""Fast checks that need no model download:  python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from enhancer import classify, cutout, hands, sr_check, upscale  # noqa: E402
from enhancer.config import SAM3_TEXT_CACHE  # noqa: E402


def test_prompts_order_and_fallbacks():
    p = classify.sam_prompts({"category": "Grocery", "sub_category": "Dairy & Cheese"})
    assert p == ["dairy and cheese product", "grocery product", "product package", "product"]
    assert classify.sam_prompts(None) == ["product package", "product"]


def test_every_prompt_is_cached():
    from safetensors import safe_open
    with safe_open(str(SAM3_TEXT_CACHE), "pt") as f:
        cached = set(json.loads(f.metadata()["prompts"]))
    need = set(hands.MASK_PROMPTS) | set(classify.FALLBACK_PROMPTS)
    for cat, subs in classify.load_taxonomy().items():
        for sub in subs:
            need |= set(classify.sam_prompts({"category": cat, "sub_category": sub}))
    assert need <= cached, sorted(need - cached)[:5]


def test_fill_prompt_uses_subcategory_only_when_confident():
    assert classify.fill_prompt({"sub_category": "Juices", "confidence": 0.5})[1] == "product.classify"
    assert classify.fill_prompt({"sub_category": "Juices", "confidence": 0.1})[1] == "generic"


def test_compose_never_touches_pixels_outside_mask():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)
    cand = rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)
    mask = np.zeros((64, 80), np.uint8)
    mask[20:40, 30:50] = 1
    out = hands.compose(src, cand, mask)
    assert np.array_equal(out[mask == 0], src[mask == 0])
    assert not np.array_equal(out[mask == 1], src[mask == 1])


def test_silhouette_crops_to_mask_with_padding():
    img = np.full((100, 120, 3), 200, np.uint8)
    mask = np.zeros((100, 120), np.uint8)
    mask[40:60, 50:70] = 255
    sil = cutout.silhouette(img, mask)
    assert sil.mode == "RGBA" and sil.size == (20 + 2 * cutout.PAD, 20 + 2 * cutout.PAD)
    assert cutout.silhouette(img, np.zeros_like(mask)) is None


def test_target_scale_caps_output():
    assert upscale.target_scale((300, 800), 2048) == 2048 / 800
    assert upscale.target_scale((200, 300), 2048) == 4.0


def test_edge_width_sharp_vs_blurred():
    step = np.zeros((20, 60), np.uint8)
    step[:, 30:] = 255
    import cv2
    sharp = sr_check.edge_width_px(np.tile(step, (1, 1)))
    blurred = sr_check.edge_width_px(cv2.GaussianBlur(step, (0, 0), 3))
    assert sharp is not None and blurred is not None and blurred > sharp


def test_on_white_mask_from_alpha():
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    rgba.paste((10, 20, 30, 255), (2, 2, 8, 8))
    bgr, mask = sr_check.on_white(rgba)
    assert mask.sum() == 36 and tuple(bgr[0, 0]) == (255, 255, 255)
