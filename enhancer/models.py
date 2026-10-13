"""Model loading, shared by every step. Each model loads on first use and stays loaded,
unless Settings.low_memory is on, in which case the runner frees it after its step.

  sam3      facebook/sam3 (gated). Its text encoder (354 M of 840 M weights) is dropped:
            every prompt the app can ask for is precomputed in data/sam3_text_embeds.safetensors.
  birefnet  ZhengPeng7/BiRefNet at 1024 px
  lama      LaMa ONNX (onnxruntime, CPU)
  esrgan    Real-ESRGAN x4plus (torch)
  ocr       PaddleOCR 3.x, PP-OCRv5 (the engine SR Check was calibrated on)
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .config import CACHE_DIR, MODELS_MANIFEST, SAM3_TEXT_CACHE, Settings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MANIFEST = json.loads(MODELS_MANIFEST.read_text())
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def release():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def pick_device(want: str) -> str:
    import torch
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str) -> Path:
    """Download one manifest file into CACHE_DIR (once) and check its sha256."""
    spec = MANIFEST[key]
    dst = CACHE_DIR / spec["file"]
    if dst.is_file() and dst.stat().st_size == spec["bytes"]:
        return dst
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if "repo" in spec:
        from huggingface_hub import hf_hub_download
        got = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"]))
        shutil.copy2(got.resolve(), dst)
    else:
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.rename(dst)
    if _sha256(dst) != spec["sha256"]:
        dst.unlink()
        raise RuntimeError(f"{spec['file']}: checksum mismatch, download removed")
    return dst


# ----------------------------------------------------------------------------- SAM 3
class Sam3:
    """SAM 3 concept detector with precomputed text features and low-memory post-processing."""

    def __init__(self, device: str, dtype):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import Sam3Model, Sam3Processor
        spec = MANIFEST["sam3"]
        token = os.environ.get("HF_TOKEN") or None
        try:
            self.proc = Sam3Processor.from_pretrained(spec["repo"], revision=spec["revision"], token=token)
            model = Sam3Model.from_pretrained(spec["repo"], revision=spec["revision"], dtype=dtype, token=token)
        except Exception as exc:
            raise RuntimeError(
                "SAM 3 could not be downloaded. It is a gated model: approve access for your Hugging Face "
                f"account at https://huggingface.co/{spec['repo']} and set HF_TOKEN. ({type(exc).__name__}: {str(exc)[:200]})"
            ) from exc
        self.device, self.dtype = device, dtype
        with safe_open(str(SAM3_TEXT_CACHE), "pt") as f:
            meta = f.metadata() or {}
        t = load_file(str(SAM3_TEXT_CACHE))
        self.text = {p: (t["features"][i:i + 1], t["mask"][i:i + 1]) for i, p in enumerate(json.loads(meta["prompts"]))}
        model.text_encoder = None          # before moving to the device, so it never takes memory there
        gc.collect()
        self.model = model.to(device).eval()
        self._torch = torch

    def encode(self, image: np.ndarray):
        pv = self.proc(images=Image.fromarray(image), return_tensors="pt")["pixel_values"]
        with self._torch.inference_mode():
            return self.model.get_vision_features(pixel_values=pv.to(self.device, self.dtype))

    def _text(self, prompt: str):
        if prompt not in self.text:
            raise KeyError(f"prompt {prompt!r} is not in {SAM3_TEXT_CACHE.name}; add it with tools/build_sam3_text_cache.py")
        f, m = self.text[prompt]
        return f.to(self.device, self.dtype), m.to(self.device)

    def detect(self, image_hw: tuple[int, int], prompt: str, embeds, score_thr: float, top_k: int = 10):
        """-> boxes [n,4] in source px, scores [n], low-res mask logits [n,h',w'] (CPU fp32)."""
        torch = self._torch
        h, w = image_hw
        feats, mask = self._text(prompt)
        with torch.inference_mode():
            out = self.model(vision_embeds=embeds, text_embeds=SimpleNamespace(pooler_output=feats), attention_mask=mask)
            scores = out.pred_logits[0].float().sigmoid()
            if getattr(out, "presence_logits", None) is not None:
                scores = scores * out.presence_logits[0].float().sigmoid()
            keep = torch.nonzero(scores > score_thr).flatten()
            keep = keep[torch.argsort(scores[keep], descending=True)[:top_k]]
            s = scores[keep].cpu().numpy().astype(np.float64)
            b = out.pred_boxes[0][keep].float().cpu().numpy().astype(np.float64)
            m = out.pred_masks[0][keep].float().cpu()
        del out
        return (b * np.array([w, h, w, h])).reshape(-1, 4), s.reshape(-1), m

    @staticmethod
    def mask_at(mask_logits, h: int, w: int, thr: float = 0.5) -> np.ndarray:
        import torch
        m = torch.nn.functional.interpolate(mask_logits[None, None], size=(h, w), mode="bilinear", align_corners=False)
        return (m[0, 0].sigmoid() > thr).numpy().astype(np.uint8)


# ----------------------------------------------------------------------------- BiRefNet
class BiRefNet:
    def __init__(self, device: str, dtype):
        import torch
        from transformers import AutoModelForImageSegmentation
        spec = MANIFEST["birefnet"]
        m = AutoModelForImageSegmentation.from_pretrained(spec["repo"], revision=spec["revision"], trust_remote_code=True)
        self.model = m.to(device=device, dtype=dtype).eval()
        self.device, self.dtype, self._torch = device, dtype, torch

    def prob(self, crop: np.ndarray, size: int) -> np.ndarray:
        import cv2
        torch = self._torch
        ch, cw = crop.shape[:2]
        x = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]).to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model(t)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        while isinstance(pred, (list, tuple)):
            pred = pred[-1]
        p = torch.sigmoid(pred.float())[0, 0].cpu().numpy()
        del out, pred, t
        return cv2.resize(p, (cw, ch), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- LaMa
class Lama:
    """LaMa (Carve ONNX export): 512x512 RGB in [0,1], mask 1 = hole."""

    def __init__(self, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(fetch("lama")), so, providers=["CPUExecutionProvider"])

    def __call__(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
        import cv2
        H, W = rgb.shape[:2]
        side = max(H, W)
        sq = np.zeros((side, side, 3), np.uint8)
        sq[:H, :W] = rgb
        sm = np.zeros((side, side), np.uint8)
        sm[:H, :W] = hole
        img = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        m = (cv2.resize(sm, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        out = self.sess.run(None, {"image": img.transpose(2, 0, 1)[None], "mask": m[None, None]})[0][0].transpose(1, 2, 0)
        out = np.clip(out, 0, 1) if out.max() <= 1.5 else np.clip(out / 255.0, 0, 1)
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)[:H, :W]
        return np.ascontiguousarray((out * 255).round().astype(np.uint8))


# ----------------------------------------------------------------------------- OCR
class Ocr:
    """PaddleOCR 3.x with the settings SR Check was calibrated with."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self.engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                                use_textline_orientation=False, lang="en")

    def predict(self, bgr: np.ndarray) -> dict:
        return self.engine.predict(bgr)[0]


# ----------------------------------------------------------------------------- registry
class Models:
    """Lazy holder for every model. Thread-safe loading; the runner serialises runs."""

    def __init__(self, settings: Settings):
        import torch
        self.settings = settings
        self.device = pick_device(settings.device)
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        torch.set_num_threads(max(1, settings.threads))
        self._lock = threading.Lock()
        self._loaded: dict[str, object] = {}
        self.load_seconds: dict[str, float] = {}

    def _get(self, name: str, make):
        with self._lock:
            if name not in self._loaded:
                t0 = time.perf_counter()
                self._loaded[name] = make()
                self.load_seconds[name] = round(time.perf_counter() - t0, 1)
            return self._loaded[name]

    def sam3(self) -> Sam3:
        return self._get("sam3", lambda: Sam3(self.device, self.dtype))

    def birefnet(self) -> BiRefNet:
        return self._get("birefnet", lambda: BiRefNet(self.device, self.dtype))

    def lama(self) -> Lama:
        return self._get("lama", lambda: Lama(self.settings.threads))

    def esrgan(self):
        from .upscale import RealESRGAN
        return self._get("esrgan", lambda: RealESRGAN(fetch("realesrgan"), self.device))

    def ocr(self) -> Ocr:
        return self._get("ocr", Ocr)

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def free(self, *names: str):
        with self._lock:
            for n in names:
                self._loaded.pop(n, None)
        release()
