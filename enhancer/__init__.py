"""ONDC product image enhancement: photo in, catalogue-ready product image out.

    from enhancer import Settings, Enhancer
    result = Enhancer(Settings()).run(pil_image)

Steps (enhancer/runner.py):
  1. classify      ONDC product.classify -> category, sub-category       (classify.py)
  2. hand removal  SAM 3 finds hand/finger/arm, LaMa or FLUX.2 repaints   (hands.py)
  3. cut-out       SAM 3 finds the product, BiRefNet traces its edge      (cutout.py)
  4. SR Check      does the text need enhancing? one yes/no                (sr_check.py)
  5. enhance       Real-ESRGAN, or SD x4 + TIGER on a GPU machine          (upscale.py)
  6. output        transparent PNG, on-white JPEG, JSON report            (runner.py)
"""
from .config import Settings
from .runner import Enhancer, PipelineResult, StepResult

__all__ = ["Settings", "Enhancer", "PipelineResult", "StepResult"]
