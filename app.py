"""Streamlit page: upload a product photo, get a catalogue-ready product image.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import streamlit as st
from PIL import Image

from enhancer import Enhancer, Settings
from enhancer.runner import STEPS, TITLES

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MAX_UPLOAD_MB = 15

st.set_page_config(page_title="Product Image Enhancement", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner=False)
def get_enhancer() -> Enhancer:
    return Enhancer(Settings())


def png_bytes(im: Image.Image, fmt: str = "PNG") -> bytes:
    b = io.BytesIO()
    im.save(b, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return b.getvalue()


def checkerboard(rgba: Image.Image, cell: int = 12) -> Image.Image:
    """Show transparency the way image editors do."""
    w, h = rgba.size
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    tile = Image.new("RGB", (cell * 2, cell * 2), (255, 255, 255))
    tile.paste((225, 225, 225), (0, 0, cell, cell))
    tile.paste((225, 225, 225), (cell, cell, cell * 2, cell * 2))
    for y in range(0, h, cell * 2):
        for x in range(0, w, cell * 2):
            bg.paste(tile, (x, y))
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def show(im: Image.Image, caption: str):
    st.image(checkerboard(im) if im.mode == "RGBA" else im, caption=caption, width="stretch")


ICON = {"done": "✅", "skipped": "⏭️", "failed": "⚠️", "pending": "⏳"}

enh = get_enhancer()
engines = enh.engines()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    hand_mode = st.radio("Hand removal", ["auto", "always", "off"], index=0,
                         help="auto: only when a hand is clearly holding the product. always: repaint every "
                              "hand, finger or arm found. off: skip.")
    enhance_mode = st.radio("Enhancement", ["auto", "always", "off"], index=0,
                            help="auto: only when SR Check finds weak text or a too-small product.")
    max_side = st.select_slider("Largest output side (px)", [1024, 1536, 2048, 3072], value=2048)
    use_classify = st.checkbox("Ask ONDC product.classify for the category", value=True)
    st.divider()
    st.caption("Running on this machine")
    st.markdown(f"- device: **{engines['device']}**\n"
                f"- hand removal: **{'FLUX.2 (GPU)' if engines['hand'] == 'flux' else 'LaMa (CPU)'}**\n"
                f"- enhancement: **{'SD x4 + TIGER (GPU)' if engines['enhance'] == 'sdx4_tiger' else 'Real-ESRGAN'}**")
    loaded = enh.models.loaded()
    st.caption("Models in memory: " + (", ".join(loaded) if loaded else "none yet (the first photo downloads "
                                                                         "and loads them, about 1–3 minutes)"))

# ----------------------------------------------------------------------------- input
st.title("Product image enhancement")
st.write("Upload a product photo. The app finds the product, removes a hand holding it, cuts it out, "
         "checks whether its text is sharp enough, enhances it if not, and gives you the result on a "
         "transparent and a white background.")

with st.expander("How it works"):
    st.markdown(
        "1. **Classify**: ONDC `product.classify` names the category and sub-category, which tells the next steps what to look for.\n"
        "2. **Hand removal**: SAM 3 finds hands, fingers and arms; the area is repainted. Pixels outside it are never changed.\n"
        "3. **Cut-out**: SAM 3 finds the product (one object, not the shelf), BiRefNet traces its edge pixel by pixel.\n"
        "4. **SR Check**: PP-OCRv5 reads every text line on the product. A line is weak only when two of three signs agree "
        "(too small, low read confidence, blurred edges). Any weak line, or a product under 200 px, means *enhance*.\n"
        "5. **Enhance**: only when step 4 says so. Upscales the cut-out (up to 4×).\n"
        "6. **Output**: transparent PNG, the same on white, and a report of every decision.\n\n"
        "On free hosting the app has no GPU, so hand removal uses LaMa (it continues the background; it cannot redraw "
        "artwork the hand covered) and enhancement uses Real-ESRGAN (sharper edges; it cannot rebuild unreadable small "
        "print). With a GPU server configured, FLUX.2 and SD x4 + TIGER are used instead.")

left, right = st.columns([2, 1])
with left:
    up = st.file_uploader("Product photo", type=["jpg", "jpeg", "png", "webp"])
with right:
    samples = sorted(p.name for p in SAMPLES.glob("*.jpg"))
    pick = st.selectbox("…or try a sample", ["(none)"] + samples)

photo, name = None, None
if up is not None:
    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please upload a photo under {MAX_UPLOAD_MB} MB.")
    else:
        try:
            photo, name = Image.open(up), Path(up.name).stem
            photo.load()
        except Exception:
            st.error("That file could not be read as an image.")
            photo = None
elif pick != "(none)":
    photo, name = Image.open(SAMPLES / pick), Path(pick).stem

if photo is not None:
    st.image(photo, caption=f"{name} ({photo.width}×{photo.height})", width=360)

run = st.button("Enhance", type="primary", disabled=photo is None)

# ----------------------------------------------------------------------------- run
if run and photo is not None:
    settings = replace(enh.settings, hand_mode=hand_mode, enhance_mode=enhance_mode, max_output_side=max_side,
                       classify_url=enh.settings.classify_url if use_classify else "")
    if enh.lock.locked():
        st.info("Another photo is being processed; yours starts as soon as it finishes.")
    status = st.status("Working…", expanded=True)
    lines = {n: status.empty() for n in STEPS}
    for n in STEPS:
        lines[n].markdown(f"{ICON['pending']} {TITLES[n]}")
    bar = status.empty()

    def on_step(r):
        lines[r.name].markdown(f"{ICON[r.status]} **{r.title}** ({r.seconds:.1f} s): {r.message}")
        bar.empty()

    def on_progress(step, frac):
        bar.progress(frac, text=f"{TITLES[step]}: {int(frac * 100)}%")

    result = enh.run(photo, settings, on_step=on_step, on_progress=on_progress)
    status.update(label=f"Finished in {result.seconds:.0f} s" if result.final else "Could not finish",
                  state="complete" if result.final else "error", expanded=False)
    st.session_state["result"], st.session_state["name"] = result, name

# ----------------------------------------------------------------------------- results
result = st.session_state.get("result")
if result is not None:
    name = st.session_state.get("name", "product")
    if result.error:
        st.error(result.error)
    if result.final is not None:
        st.subheader("Result")
        c1, c2 = st.columns(2)
        with c1:
            show(result.original, "Original photo")
        with c2:
            show(result.final, f"Result ({result.final.width}×{result.final.height})")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Transparent PNG", png_bytes(result.final), f"{name}_enhanced.png", "image/png", on_click="ignore")
        d2.download_button("On white (JPEG)", png_bytes(result.final_on_white, "JPEG"), f"{name}_on_white.jpg", "image/jpeg", on_click="ignore")
        d3.download_button("Report (JSON)", json.dumps(result.report(), indent=1, default=str), f"{name}_report.json",
                           "application/json", on_click="ignore")
        d4.download_button("Everything (ZIP)", result.zip_bytes(), f"{name}_all_steps.zip", "application/zip", on_click="ignore")

    st.subheader("Step by step")
    for s in result.steps:
        with st.expander(f"{ICON[s.status]} {s.title}: {s.message}", expanded=bool(s.images) and s.status == "done"):
            if s.engine:
                st.caption(f"Engine: {s.engine} · {s.seconds:.1f} s")
            if s.images:
                cols = st.columns(len(s.images))
                for col, (cap, im) in zip(cols, s.images.items()):
                    with col:
                        show(im, cap)
            if s.name == "sr_check" and s.details.get("lines") is not None:
                st.caption("Boxes: green = text reads fine, amber = one warning sign only (not counted), "
                           "red = weak line, grey = ignored (logo, or touching the cut-out edge).")
                rows = [{"text": l["text"], "height px": l["height_px"], "confidence": l["conf"],
                         "edge width": l.get("edge_width_frac"), "weak": l["weak"], "signals": ", ".join(l["signals"])}
                        for l in s.details["lines"]]
                if rows:
                    st.dataframe(rows, hide_index=True)
            if s.details:
                st.json({k: v for k, v in s.details.items() if k != "traceback"}, expanded=False)
            if s.status == "failed" and s.details.get("traceback"):
                st.code(s.details["traceback"])

st.divider()
st.caption("Models: SAM 3 (Meta, SAM License), BiRefNet (MIT), LaMa (Apache-2.0), Real-ESRGAN (BSD-3), "
           "PP-OCRv5 (Apache-2.0). Sample photos: Wikimedia Commons, credits in samples/CREDITS.json.")
