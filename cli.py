#!/usr/bin/env python3
"""Run the enhancement on photos from the command line (no web page).

    python cli.py samples/coconut_bar_hand.jpg                 # -> output/coconut_bar_hand/
    python cli.py photos/*.jpg --out results --device cpu
    python cli.py photo.jpg --hand always --enhance off

Per photo: final.png (transparent), final_on_white.jpg, report.json, steps/*.png.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image

from enhancer import Enhancer, Settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", default="output")
    ap.add_argument("--device", default=None, help="auto | cpu | mps | cuda")
    ap.add_argument("--hand", choices=["auto", "always", "off"], default="auto")
    ap.add_argument("--enhance", choices=["auto", "always", "off"], default="auto")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--low-memory", action="store_true", help="free each model after its step")
    ap.add_argument("--max-output-side", type=int, default=None)
    a = ap.parse_args()

    s = Settings()
    if a.device:
        s.device = a.device
    s.hand_mode, s.enhance_mode = a.hand, a.enhance
    s.low_memory = a.low_memory or s.low_memory
    if a.no_classify:
        s.classify_url = ""
    if a.max_output_side:
        s.max_output_side = a.max_output_side
    enh = Enhancer(s)
    print(f"device {enh.models.device}; engines {enh.engines()}", flush=True)
    failures = 0
    for p in map(Path, a.photos):
        out = Path(a.out) / p.stem
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print(f"--- {p.name}", flush=True)
        res = enh.run(Image.open(p), on_step=lambda r: print(f"  {r.title:16s} {r.status:7s} {r.seconds:6.1f}s  {r.message}", flush=True))
        with zipfile.ZipFile(io.BytesIO(res.zip_bytes())) as z:
            z.extractall(out)
        failures += res.final is None
        print(f"  {'done' if res.final else 'FAILED'} in {time.perf_counter() - t0:.1f}s -> {out}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
