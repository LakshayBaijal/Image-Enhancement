"""Settings for one run, and where the data files live.

Every value can be set from the environment (useful on a hosted Space) or from the
Streamlit sidebar / the command line. Nothing here loads a model.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TAXONOMY = DATA / "taxonomy.json"
SAM3_TEXT_CACHE = DATA / "sam3_text_embeds.safetensors"
SR_THRESHOLDS = DATA / "sr_thresholds.json"
MODELS_MANIFEST = DATA / "models.json"
CACHE_DIR = Path(os.environ.get("ENHANCER_CACHE", Path.home() / ".cache" / "ondc-enhancer"))

DEFAULT_CLASSIFY_URL = "https://dev-digicatalog.ondc.tech/intelligence/v1/invoke"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # where the models run: auto = cuda > mps > cpu
    device: str = field(default_factory=lambda: os.environ.get("ENHANCER_DEVICE", "auto"))
    threads: int = field(default_factory=lambda: int(os.environ.get("ENHANCER_THREADS", "4")))
    # keep every model loaded between photos (fast) or free each one after its step (low memory)
    low_memory: bool = field(default_factory=lambda: os.environ.get("ENHANCER_LOW_MEMORY", "0") == "1")

    # step 1: ONDC product.classify; "" switches it off (generic prompts are used)
    classify_url: str = field(default_factory=lambda: os.environ.get("ONDC_CLASSIFY_URL", DEFAULT_CLASSIFY_URL))

    # step 2: hand removal. mode: auto = only when SAM 3 sees a hand scoring >= hand_trigger,
    # always = fill every hand/finger/arm SAM 3 finds, off = skip the step
    hand_mode: str = "auto"
    hand_trigger: float = 0.6
    hand_mask_score: float = 0.3
    hand_grow: float = 0.012
    hand_max_area: float = 0.25
    # engine: auto = FLUX.2 when a ComfyUI server is configured and reachable, else LaMa
    hand_engine: str = field(default_factory=lambda: os.environ.get("ENHANCER_HAND_ENGINE", "auto"))
    flux_server: str = field(default_factory=lambda: os.environ.get("FLUX_COMFY_URL", ""))
    flux_unet: str = field(default_factory=lambda: os.environ.get("FLUX_UNET", "flux-2-klein-4b-Q8_0.gguf"))

    # step 3: cut-out
    sam_score: float = 0.30
    sam_mask_threshold: float = 0.50
    gate_grow: float = 0.03
    fill_thick: float = 0.01
    birefnet_size: int = 1024

    # step 5: enhancement. mode: auto = only when SR Check says so, always, off
    enhance_mode: str = "auto"
    # engine: auto = SD x4 + TIGER when SDX4_TIGER_DIR points at a working checkout, else Real-ESRGAN
    enhance_engine: str = field(default_factory=lambda: os.environ.get("ENHANCER_SR_ENGINE", "auto"))
    sdx4_tiger_dir: str = field(default_factory=lambda: os.environ.get("SDX4_TIGER_DIR", ""))
    max_output_side: int = field(default_factory=lambda: int(_env_float("ENHANCER_MAX_OUTPUT_SIDE", 2048)))

    # safety: photos larger than this are scaled down before anything else
    max_input_side: int = 4000

    def to_dict(self) -> dict:
        return asdict(self)
