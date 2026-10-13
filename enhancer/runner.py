"""The six steps, in order, for one photo. Used by app.py (Streamlit) and cli.py.

Each step returns a StepResult: done / skipped / failed, which engine ran, how long it took,
the images it produced and a details dict. A failing step never stops the run when the run
can continue without it (no classification -> generic prompts; hand removal fails -> the
original photo is cut out). Only a failed cut-out ends the run, because every later step
works on the cut-out.
"""
from __future__ import annotations

import io
import json
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import classify as C
from . import cutout, hands, sr_check, upscale
from .config import Settings
from .models import Models, release

STEPS = ["classify", "hand_removal", "cutout", "sr_check", "enhance", "output"]
TITLES = {"classify": "1. Classify", "hand_removal": "2. Hand removal", "cutout": "3. Cut-out",
          "sr_check": "4. SR Check", "enhance": "5. Enhance", "output": "6. Output"}


@dataclass
class StepResult:
    name: str
    status: str = "pending"            # done | skipped | failed
    engine: str | None = None
    seconds: float = 0.0
    message: str = ""
    images: dict = field(default_factory=dict)      # caption -> PIL image
    details: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return TITLES[self.name]


@dataclass
class PipelineResult:
    steps: list[StepResult]
    original: Image.Image
    final: Image.Image | None = None             # RGBA
    final_on_white: Image.Image | None = None    # RGB
    seconds: float = 0.0
    error: str | None = None

    def step(self, name: str) -> StepResult:
        return next(s for s in self.steps if s.name == name)

    def report(self) -> dict:
        return {"seconds": round(self.seconds, 1), "error": self.error,
                "original_size": list(self.original.size),
                "final_size": list(self.final.size) if self.final else None,
                "steps": [{"step": s.name, "status": s.status, "engine": s.engine, "seconds": s.seconds,
                           "message": s.message, "details": s.details} for s in self.steps]}

    def zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            def put(name, im, fmt="PNG"):
                b = io.BytesIO()
                im.save(b, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
                z.writestr(name, b.getvalue())
            if self.final:
                put("final.png", self.final)
                put("final_on_white.jpg", self.final_on_white, "JPEG")
            for s in self.steps:
                for cap, im in s.images.items():
                    put(f"steps/{s.name}_{cap.lower().replace(' ', '_')}.png", im)
            z.writestr("report.json", json.dumps(self.report(), indent=1, ensure_ascii=False, default=str))
        return buf.getvalue()


def flatten(rgba: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    base = Image.new("RGB", rgba.size, bg)
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def prepare(image: Image.Image, max_side: int) -> Image.Image:
    im = ImageOps.exif_transpose(image).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


class Enhancer:
    """One per process. Holds the models; runs one photo at a time (a lock queues the rest)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.models = Models(self.settings)
        self.lock = threading.Lock()

    # engines actually usable on this machine
    def engines(self, s: Settings | None = None) -> dict:
        s = s or self.settings
        flux = s.hand_engine in ("auto", "flux") and hands.flux_available(s.flux_server)
        tiger = s.enhance_engine in ("auto", "sdx4_tiger") and upscale.sdx4_tiger_available(s.sdx4_tiger_dir)
        return {"device": self.models.device, "hand": "flux" if flux else "lama",
                "enhance": "sdx4_tiger" if tiger else "realesrgan"}

    def run(self, image: Image.Image, settings: Settings | None = None, on_step=None, on_progress=None) -> PipelineResult:
        """on_step(StepResult) after each step; on_progress(step_name, fraction) inside long steps."""
        with self.lock:
            return self._run(image, settings or self.settings, on_step or (lambda r: None),
                             on_progress or (lambda n, f: None))

    def _run(self, image, s: Settings, on_step, on_progress) -> PipelineResult:
        t_all = time.perf_counter()
        photo = prepare(image, s.max_input_side)
        rgb = np.asarray(photo)
        res = PipelineResult(steps=[StepResult(n) for n in STEPS], original=photo)
        engines = self.engines(s)
        state: dict = {}

        def step(name, fn):
            r = res.step(name)
            t0 = time.perf_counter()
            try:
                fn(r)
                if r.status == "pending":
                    r.status = "done"
            except Exception as exc:
                r.status, r.message = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
                r.details["traceback"] = traceback.format_exc(limit=4)
            r.seconds = round(time.perf_counter() - t0, 1)
            on_step(r)
            return r

        # 1 ------------------------------------------------------------- classify
        def do_classify(r):
            if not s.classify_url:
                r.status, r.message = "skipped", "classification switched off: generic prompts"
                return
            rec = C.classify(photo, s.classify_url)
            r.engine, r.details = "ONDC product.classify", rec
            if rec.get("error"):
                r.status, r.message = "failed", f"{rec['error']} (continuing with generic prompts)"
                return
            state["classification"] = rec
            r.message = (f"{rec['category']} / {rec['sub_category']} "
                         f"(confidence {rec['confidence']:.2f}{', low' if rec.get('low_confidence') else ''})")
        step("classify", do_classify)
        cls = state.get("classification")

        # 2 ------------------------------------------------------------- hand removal
        def do_hands(r):
            if s.hand_mode == "off":
                r.status, r.message = "skipped", "hand removal switched off"
                return
            sam = self.models.sam3()
            embeds = sam.encode(rgb)
            trigger = None if s.hand_mode == "always" else s.hand_trigger
            mask, info = hands.hand_mask(sam, rgb, embeds, trigger, s.hand_mask_score, s.hand_grow)
            r.details["mask"] = info
            if mask is None:
                state["embeds"] = embeds               # photo unchanged: step 3 reuses SAM 3's image pass
                r.status, r.message = "skipped", info["reason"]
                return
            if info["coverage"] > s.hand_max_area:
                state["embeds"] = embeds
                r.status = "skipped"
                r.message = f"the hand mask covers {100 * info['coverage']:.0f}% of the photo, too much to repaint"
                return
            del embeds
            r.images["Mask"] = Image.fromarray(hands.overlay(rgb, mask))
            if engines["hand"] == "flux":
                prompt, _ = C.fill_prompt(cls)
                fixed, run = hands.fill_flux(s.flux_server, s.flux_unet, rgb, mask, prompt)
            else:
                fixed, run = hands.fill_lama(self.models.lama(), rgb, mask)
            if not np.array_equal(fixed[mask == 0], rgb[mask == 0]):
                raise AssertionError("pixels outside the hand mask changed")
            r.engine, r.details["fill"] = run["engine"], run
            r.images["Repaired"] = Image.fromarray(fixed)
            state["photo"] = fixed
            r.message = f"repainted {100 * info['coverage']:.1f}% of the photo with {run['engine']}"
        step("hand_removal", do_hands)
        work = state.get("photo", rgb)
        if s.low_memory:
            self.models.free("lama")

        # 3 ------------------------------------------------------------- cut-out
        def do_cutout(r):
            sam = self.models.sam3()
            embeds = state.pop("embeds", None)
            if embeds is None:
                embeds = sam.encode(work)
            prompts = C.sam_prompts(cls)
            det = cutout.detect(sam, work, embeds, prompts, s.sam_score, s.sam_mask_threshold)
            del embeds
            release()
            if s.low_memory:
                self.models.free("sam3")
            mask, stats = cutout.segment(self.models.birefnet(), work, det, s.birefnet_size, s.gate_grow, s.fill_thick)
            if s.low_memory:
                self.models.free("birefnet")
            sil = cutout.silhouette(work, mask)
            r.engine = "SAM 3 + BiRefNet"
            r.details = {"prompts": prompts, "detection": {k: v for k, v in det.items() if k != "sam_mask"},
                         "segmentation": stats}
            if sil is None:
                raise RuntimeError("the cut-out came back empty")
            state["cutout"] = sil
            r.images["Found"] = Image.fromarray(cutout.debug_overlay(work, det, mask))
            r.images["Cut-out"] = sil
            how = f"SAM 3 prompt '{det['prompt_used']}' (score {det['score']})" if det.get("box") else \
                "SAM 3 found nothing; BiRefNet on the whole photo"
            r.message = f"{how}; cut-out {sil.width}x{sil.height} px"
        rc = step("cutout", do_cutout)
        if rc.status != "done":
            res.error = "the cut-out failed, so the later steps cannot run"
            for n in ("sr_check", "enhance", "output"):
                res.step(n).status, res.step(n).message = "skipped", res.error
                on_step(res.step(n))
            res.seconds = time.perf_counter() - t_all
            return res
        sil = state["cutout"]

        # 4 ------------------------------------------------------------- SR Check
        def do_sr(r):
            rec, evidence = sr_check.check(self.models.ocr(), sil)
            r.engine, r.details = "PP-OCRv5 + OpenCV", rec
            r.images["Evidence"] = Image.fromarray(evidence)
            state["needs"] = rec["needs_enhancement"]
            if rec["needs_enhancement"]:
                weak = ", ".join(f"'{t}'" for t in rec["weak_lines"][:4])
                r.message = f"needs enhancement: {', '.join(rec['failed_checks'])}" + (f" (weak text: {weak})" if weak else "")
            else:
                r.message = "OK as is: no weak text, product large enough"
        step("sr_check", do_sr)
        if s.low_memory:
            self.models.free("ocr")

        # 5 ------------------------------------------------------------- enhance
        def do_enhance(r):
            if s.enhance_mode == "off":
                r.status, r.message = "skipped", "enhancement switched off"
                return
            if s.enhance_mode == "auto" and not state.get("needs", False):
                r.status = "skipped"
                r.message = ("SR Check could not run, so the cut-out is left as it is" if "needs" not in state
                             else "SR Check says the cut-out is fine as it is")
                return
            if engines["enhance"] == "sdx4_tiger":
                big, info = upscale.upscale_sdx4_tiger(s.sdx4_tiger_dir, sil, s.max_output_side)
            else:
                big, info = upscale.upscale_realesrgan(self.models.esrgan(), sil, s.max_output_side,
                                                       lambda f: on_progress("enhance", f))
            r.engine, r.details = info["engine"], info
            if big is None:
                r.status, r.message = "skipped", info["reason"]
                return
            state["enhanced"] = big
            r.images["Enhanced"] = big
            r.message = f"{info['engine']}: {sil.width}x{sil.height} -> {big.width}x{big.height} px"
        step("enhance", do_enhance)
        if s.low_memory:
            self.models.free("esrgan")

        # 6 ------------------------------------------------------------- output
        def do_output(r):
            final = state.get("enhanced", sil)
            res.final, res.final_on_white = final, flatten(final)
            r.images["Final"] = res.final_on_white
            made = [n for n, k in (("hand removed", "photo"), ("enhanced", "enhanced")) if k in state]
            r.message = f"{final.width}x{final.height} px, transparent PNG and on white" + \
                (f" ({', '.join(made)})" if made else "")
        step("output", do_output)
        res.seconds = time.perf_counter() - t_all
        return res
