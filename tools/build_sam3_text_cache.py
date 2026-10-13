#!/usr/bin/env python3
"""Rebuild data/sam3_text_embeds.safetensors: SAM 3's text encoder run ONCE over every prompt the
app can ask for, so the app never loads the encoder (354 M of SAM 3's 840 M weights).

Prompts: "<sub-category> product" and "<category> product" for the whole taxonomy, the fallbacks
("product package", "product"), and "hand", "finger", "arm". Re-run after changing
data/taxonomy.json or the prompt template in enhancer/classify.py:

    python tools/build_sam3_text_cache.py          # needs HF_TOKEN with SAM 3 access
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from enhancer.classify import FALLBACK_PROMPTS, PROMPT_TEMPLATE, load_taxonomy, sam_prompts  # noqa: E402
from enhancer.config import SAM3_TEXT_CACHE  # noqa: E402
from enhancer.hands import MASK_PROMPTS  # noqa: E402
from enhancer.models import MANIFEST  # noqa: E402


def all_prompts() -> list[str]:
    out = [*FALLBACK_PROMPTS, *MASK_PROMPTS]
    for cat, subs in load_taxonomy().items():
        for sub in subs:
            out += sam_prompts({"category": cat, "sub_category": sub})
    return sorted(set(out))


def main() -> int:
    import torch
    from safetensors.torch import save_file
    from transformers import Sam3Model, Sam3Processor
    spec = MANIFEST["sam3"]
    prompts = all_prompts()
    proc = Sam3Processor.from_pretrained(spec["repo"], revision=spec["revision"])
    model = Sam3Model.from_pretrained(spec["repo"], revision=spec["revision"], dtype=torch.float32).eval()
    model.vision_encoder = None
    feats, masks = [], []
    with torch.inference_mode():
        for i in range(0, len(prompts), 32):
            txt = proc(text=prompts[i:i + 32], return_tensors="pt", padding="max_length")
            feats.append(model.get_text_features(input_ids=txt["input_ids"], attention_mask=txt["attention_mask"]).pooler_output)
            masks.append(txt["attention_mask"])
    feats, masks = torch.cat(feats), torch.cat(masks)
    save_file({"features": feats.contiguous(), "mask": masks.contiguous()}, str(SAM3_TEXT_CACHE),
              metadata={"checkpoint": spec["repo"], "revision": spec["revision"], "template": PROMPT_TEMPLATE,
                        "prompts": json.dumps(prompts)})
    print(f"{len(prompts)} prompts -> {SAM3_TEXT_CACHE} ({SAM3_TEXT_CACHE.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
