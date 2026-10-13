"""Step 5: enhance the cut-out, only when SR Check asks for it.

  realesrgan  Real-ESRGAN x4plus (BSD-3, 64 MB). Runs on a CPU, so it works on free hosting.
              Sharpens edges and texture; it does NOT rebuild unreadable letters.
  sdx4_tiger  SD x4 upscaler for the picture + TIGER for the text, best source per word
              (the SDx4-TIGER-pipeline checkout). Needs a GPU and ~26 GB of weights; used when
              SDX4_TIGER_DIR points at a working checkout.

The output is capped at Settings.max_output_side on its long side: the model always runs at
x4, then the result is resized down to the cap. A cut-out already within 1.25x of the cap is
left as it is (upscaling would add nothing).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------- RRDBNet (Real-ESRGAN x4plus)
def _rrdbnet():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class RDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, nf=64, nb=23, gc=32):
            super().__init__()
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet


class RealESRGAN:
    SCALE = 4

    def __init__(self, weights: Path, device: str, tile: int = 256, pad: int = 12):
        import torch
        net = _rrdbnet()()
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        net.load_state_dict(sd.get("params_ema", sd), strict=True)
        # fp16 on mps/cuda halves memory; Real-ESRGAN is stable in half precision
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        self.net = net.to(device, self.dtype).eval()
        self.device, self.tile, self.pad, self._torch = device, tile, pad, torch

    def __call__(self, rgb: np.ndarray, progress=None) -> np.ndarray:
        """RGB uint8 (h, w) -> RGB uint8 (4h, 4w), tile by tile so memory stays flat."""
        torch = self._torch
        h, w = rgb.shape[:2]
        s, t, p = self.SCALE, self.tile, self.pad
        x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]
        out = np.zeros((h * s, w * s, 3), np.float32)
        tiles = [(y, x0) for y in range(0, h, t) for x0 in range(0, w, t)]
        for k, (y, x0) in enumerate(tiles):
            y1, x1 = min(y + t, h), min(x0 + t, w)
            py0, px0, py1, px1 = max(0, y - p), max(0, x0 - p), min(h, y1 + p), min(w, x1 + p)
            with torch.inference_mode():
                o = self.net(x[:, :, py0:py1, px0:px1].to(self.device, self.dtype)).float().clamp_(0, 1)
            o = o[0].cpu().numpy().transpose(1, 2, 0)
            oy, ox = (y - py0) * s, (x0 - px0) * s
            out[y * s:y1 * s, x0 * s:x1 * s] = o[oy:oy + (y1 - y) * s, ox:ox + (x1 - x0) * s]
            if progress:
                progress((k + 1) / len(tiles))
        return (out * 255).round().astype(np.uint8)


def target_scale(size: tuple[int, int], max_side: int) -> float:
    return min(4.0, max_side / max(size))


def upscale_realesrgan(model, rgba: Image.Image, max_side: int, progress=None) -> tuple[Image.Image | None, dict]:
    scale = target_scale(rgba.size, max_side)
    if scale < 1.25:
        return None, {"engine": "Real-ESRGAN x4plus", "skipped": True,
                      "reason": f"the cut-out is already {max(rgba.size)} px; the output cap is {max_side} px"}
    a = np.asarray(rgba.convert("RGBA"))
    up = model(np.ascontiguousarray(a[..., :3]), progress)
    w, h = rgba.size
    big = Image.fromarray(up)
    alpha = Image.fromarray(a[..., 3]).resize(big.size, Image.Resampling.LANCZOS)
    big.putalpha(alpha)
    final = (round(w * scale), round(h * scale))
    if final != big.size:
        big = big.resize(final, Image.Resampling.LANCZOS)
    return big, {"engine": "Real-ESRGAN x4plus", "skipped": False, "model_scale": 4, "output_scale": round(scale, 3),
                 "input_size": [w, h], "output_size": list(final)}


# ----------------------------------------------------------------------------- SD x4 + TIGER (external checkout)
def sdx4_tiger_available(checkout: str) -> bool:
    return bool(checkout) and (Path(checkout) / "run.py").is_file()


def upscale_sdx4_tiger(checkout: str, rgba: Image.Image, max_side: int, timeout: float = 3600) -> tuple[Image.Image, dict]:
    """Runs `python run.py <photo>` in the SDx4-TIGER-pipeline checkout (its own three
    environments, set with SDX4_PY_SD / SDX4_PY_TIGER / SDX4_PY_OCR) and reads enhanced.png."""
    root = Path(checkout)
    with tempfile.TemporaryDirectory() as tmp:
        name = f"enh_{os.getpid()}_{np.random.randint(1e9)}"
        src = Path(tmp) / f"{name}.png"
        rgba.save(src)
        r = subprocess.run([sys.executable, str(root / "run.py"), str(src)], cwd=root, capture_output=True,
                           text=True, timeout=timeout)
        out_dir = root / "output" / name
        res = out_dir / "enhanced.png"
        if r.returncode != 0 or not res.is_file():
            raise RuntimeError(f"SD x4 + TIGER failed: {(r.stdout + r.stderr)[-400:]}")
        big = Image.open(res).convert("RGBA")
        big.load()
        shutil.rmtree(out_dir, ignore_errors=True)
    scale = target_scale(rgba.size, max_side)
    final = (round(rgba.width * scale), round(rgba.height * scale))
    if final != big.size:
        big = big.resize(final, Image.Resampling.LANCZOS)
    return big, {"engine": "SD x4 + TIGER", "skipped": False, "output_scale": round(scale, 3), "output_size": list(final)}
