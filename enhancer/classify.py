"""Step 1: which product is this? ONDC intelligence `product.classify`, asked twice.

  1st call: the whole Grocery + Electronics taxonomy -> the category with the most probability
  2nd call: only that category's sub-categories     -> the sub-category and its confidence

The answer only steers prompts (SAM 3 is asked for "<sub-category> product", FLUX.2 is told
what to paint). If the service is down or unsure, the run continues with generic prompts.
"""
from __future__ import annotations

import io
import json
import re
import time
from collections import defaultdict

from PIL import Image

from .config import TAXONOMY

PROMPT_TEMPLATE = "{sub_category} product"
FALLBACK_PROMPTS = ("product package", "product")


def load_taxonomy() -> dict[str, list[str]]:
    data = json.loads(TAXONOMY.read_text())
    return data["categories"] if "categories" in data else data


def _post(jpeg: bytes, taxonomy: dict, url: str, timeout: float) -> dict:
    import requests
    req = {"input": {"image_refs": ["photo.jpg"], "taxonomy": taxonomy}}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, timeout=timeout, files={
            "request": ("request.json", json.dumps(req), "application/json"),
            "files": ("photo.jpg", jpeg, "image/jpeg")})
        body = resp.json()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000)}
    out = {"ok": body.get("status") == "success", "http_status": resp.status_code,
           "elapsed_ms": round((time.perf_counter() - t0) * 1000), "request_id": body.get("request_id")}
    if out["ok"]:
        out["result"] = body.get("result", {})
    else:
        out["error"] = str(body.get("error") or body.get("detail") or body)[:300]
    return out


def _candidates(result: dict) -> list[dict]:
    """Ranked [{category, sub_category, p}]. When the service abstains ("Others",
    others_reason=low_confidence) the real ranked candidates are still in `image_top`."""
    ranked = list(result.get("image_top") or [])
    if ranked:
        return ranked
    top = {"category": result.get("category"), "sub_category": result.get("sub_category"), "p": result.get("confidence")}
    return [top] + list(result.get("alternatives") or [])


def _abstained(result: dict) -> bool:
    return bool(result.get("others_reason")) or result.get("category") == "Others"


def classify(image: Image.Image, url: str, timeout: float = 45.0) -> dict:
    """-> {"category", "category_confidence", "sub_category", "confidence", ...} or {"error": ...}."""
    taxonomy = load_taxonomy()
    im = image.convert("RGB")
    if max(im.size) > 1600:                       # the service resizes anyway; keep the upload small
        im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    jpeg = buf.getvalue()
    rec: dict = {"endpoint": url}

    s1 = _post(jpeg, taxonomy, url, timeout)
    rec["step1"] = {k: v for k, v in s1.items() if k != "result"}
    if not s1["ok"]:
        rec["error"] = f"category call failed: {s1.get('error')}"
        return rec
    votes: dict[str, float] = defaultdict(float)
    for c in _candidates(s1["result"]):
        if c.get("category") in taxonomy and c.get("p") is not None:
            votes[c["category"]] += float(c["p"])
    if not votes:
        rec["error"] = "the service returned no category from the taxonomy"
        return rec
    category = max(votes, key=votes.get)
    total = sum(votes.values()) or 1.0
    rec.update(category=category, category_confidence=round(votes[category] / total, 4))

    s2 = _post(jpeg, {category: taxonomy[category]}, url, timeout)
    rec["step2"] = {k: v for k, v in s2.items() if k != "result"}
    if not s2["ok"]:
        rec["error"] = f"sub-category call failed: {s2.get('error')}"
        return rec
    r2 = s2["result"]
    ranked = _candidates(r2)
    if _abstained(r2):
        best = ranked[0] if ranked else {}
        sub, conf, alts = best.get("sub_category"), best.get("p", r2.get("image_confidence")), ranked[1:]
    else:
        sub, conf, alts = r2.get("sub_category"), r2.get("confidence"), (r2.get("alternatives") or ranked[1:])
    thr = r2.get("threshold")
    rec.update(sub_category=sub, confidence=conf, threshold=thr, service_abstained=_abstained(r2),
               low_confidence=(conf is not None and thr is not None and conf < thr),
               alternatives=[{"sub_category": a.get("sub_category"), "p": a.get("p")} for a in alts[:4]],
               elapsed_ms=s1["elapsed_ms"] + s2["elapsed_ms"])
    return rec


# ----------------------------------------------------------------------------- prompts
def _phrase(label: str) -> str:
    """'Diapers & Wipes' -> 'diapers and wipes'."""
    s = label.replace("&", " and ").replace("/", " or ")
    s = re.sub(r"[()\[\],]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def sam_prompts(rec: dict | None) -> list[str]:
    """SAM 3 prompts in order: '<sub-category> product', '<category> product', then generic."""
    out: list[str] = []
    if rec and rec.get("sub_category"):
        out.append(PROMPT_TEMPLATE.format(sub_category=_phrase(rec["sub_category"])))
        if rec.get("category"):
            out.append(f"{_phrase(rec['category'])} product")
    for p in FALLBACK_PROMPTS:
        if p not in out:
            out.append(p)
    return out


def fill_prompt(rec: dict | None, min_conf: float = 0.20) -> tuple[str, str]:
    """What the repainted area should show (FLUX.2). Never 'remove the hand': a described product
    'standing on a shelf' stops the model drawing a new hand holding it."""
    c = rec or {}
    sub, conf = c.get("sub_category"), c.get("confidence")
    if sub and conf is not None and conf > min_conf:
        return f"A {sub.lower()} product package standing on a supermarket shelf, fully visible.", "product.classify"
    return "A retail product package standing on a supermarket shelf, fully visible.", "generic"
