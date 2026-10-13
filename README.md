---
title: Product Image Enhancement
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Product photo in, clean catalogue cut-out out
---

# Product image enhancement

Upload a product photo; get back the product cut out on a transparent (and a white)
background, with the hand that held it removed and its text enhanced when it needs it.
Built for ONDC catalogue photos (Grocery and Electronics).

```
photo ─► 1 classify ─► 2 hand removal ─► 3 cut-out ─► 4 SR Check ─► 5 enhance (only if needed) ─► 6 output
```

| step | what happens | model | where the code is |
|---|---|---|---|
| 1. Classify | ONDC `product.classify` is asked twice: first the category over the whole taxonomy, then the sub-category within it | ONDC intelligence service (web call) | `enhancer/classify.py` |
| 2. Hand removal | SAM 3 finds every hand, finger and arm; the area is repainted. Runs only when a hand scores ≥ 0.6. Pixels outside the mask are never changed (checked) | SAM 3 + LaMa (CPU) or FLUX.2 (GPU) | `enhancer/hands.py` |
| 3. Cut-out | SAM 3 is asked for "<sub-category> product" and picks ONE object; BiRefNet traces its edge; SAM's mask keeps a touching neighbour out | SAM 3 + BiRefNet | `enhancer/cutout.py` |
| 4. SR Check | Yes/no: does the cut-out need enhancing? A text line is weak only when 2 of 3 signs agree (under 20 px, read confidence under 0.90, blurred edges). Any weak line, or a product under 200 px, means yes | PP-OCRv5 + OpenCV | `enhancer/sr_check.py`, `data/sr_thresholds.json` |
| 5. Enhance | Only when step 4 says yes. Upscales the cut-out, output capped at 2048 px | Real-ESRGAN (CPU) or SD x4 + TIGER (GPU) | `enhancer/upscale.py` |
| 6. Output | Transparent PNG, the same on white, a JSON report of every decision, a ZIP with every step's image | | `enhancer/runner.py` |

A step that fails does not stop the run when the run can do without it (no classification →
generic prompts; hand removal fails → the photo is cut out as it is). Only a failed cut-out ends
the run.

## Measured

On an Apple M5 (17 GB), the four photos in `samples/`:

| | per photo | peak memory |
|---|---|---|
| Mac GPU (MPS, fp16) | 6–23 s | 8.1 GB |
| CPU only, 2 threads (what free hosting gives) | 47–59 s | 8.3 GB |

The first photo after a start also downloads and loads the models (about 4 GB, 1–3 minutes).
SR Check here gives the same verdict and the same failed checks as the standalone SR Check on
all 50 calibration cut-outs.

## Run it on your machine

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...            # a Hugging Face token whose account has been approved for facebook/sam3
streamlit run app.py              # opens http://localhost:8501
python cli.py samples/*.jpg       # same pipeline without the page -> output/<photo>/
python -m pytest tests -q         # fast checks, no downloads
```

## Put it online for free: Hugging Face Spaces

Streamlit Community Cloud is free but gives an app roughly 2.7 GB of memory, and SAM 3 alone
needs more than that on a CPU, so the app cannot run there. Hugging Face Spaces runs the same
Streamlit app for free on 2 CPUs with 16 GB of memory, which is enough (peak 8.3 GB). This
README is already the Space's page; the `Dockerfile` is what the Space builds.

1. On huggingface.co: **New Space** → name it → SDK **Docker** → hardware **CPU basic (free)**.
2. In the Space: **Settings → Variables and secrets → New secret**: `HF_TOKEN` = a read token from an
   account that has access to `facebook/sam3` (request it on the model page if not).
3. Push this folder to the Space:
   ```bash
   git init -b main && git add . && git commit -m "Product image enhancement app"
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   or keep GitHub as the main copy and sync it to the Space with a GitHub Action
   ([Hugging Face guide](https://huggingface.co/docs/hub/spaces-github-actions)).
4. The build takes about 10 minutes. The first photo is slower (model download).

Free Spaces sleep after 48 hours without visitors and wake on the next visit (a couple of
minutes). Photos are processed one at a time; a second visitor waits until the first finishes.

## GPU engines (optional)

On a machine with a GPU the two CPU stand-ins can be replaced by the stronger models. Neither
fits free hosting.

| step | CPU (default) | GPU | switch it on |
|---|---|---|---|
| 2 | LaMa: continues the background; cannot redraw artwork the hand covered | FLUX.2 Klein 4B + LanPaint: redraws the hidden part of the pack | run a ComfyUI server with FLUX.2 (see the `FLUX2 Hand Removal` project) and set `FLUX_COMFY_URL=http://host:8188` |
| 5 | Real-ESRGAN x4plus: sharper edges and texture; cannot rebuild unreadable small print | SD x4 upscaler + TIGER: per word, the best-reading of three sources | set `SDX4_TIGER_DIR` to an `SDx4-TIGER-pipeline` checkout with its three environments |

The sidebar shows which engines are active.

## Settings

All optional, as environment variables (or the sidebar for the first three):

| variable | default | meaning |
|---|---|---|
| `ENHANCER_MAX_OUTPUT_SIDE` | 2048 | longest side of an enhanced image |
| `ONDC_CLASSIFY_URL` | the ONDC dev endpoint | empty = no classification, generic prompts |
| `ENHANCER_DEVICE` | auto | auto, cpu, mps or cuda |
| `ENHANCER_THREADS` | 4 | CPU threads |
| `ENHANCER_LOW_MEMORY` | 0 | 1 = free each model after its step (slower, lower peak) |
| `HF_TOKEN` | | needed to download SAM 3 |

## Layout

```
app.py                  the Streamlit page
cli.py                  the same pipeline from the command line
enhancer/               config, models (loading + downloads), one module per step, runner
data/taxonomy.json      Grocery + Electronics categories and sub-categories
data/sam3_text_embeds.safetensors   SAM 3's text encoder run once over every prompt (181), so the
                        app never loads the encoder (354 M of SAM 3's 840 M weights); rebuild with
                        tools/build_sam3_text_cache.py after changing the taxonomy
data/sr_thresholds.json SR Check thresholds (calibrated on 50 labelled photos)
data/models.json        every model file, pinned by revision and checksum
samples/                four Wikimedia Commons photos, credits in samples/CREDITS.json
Dockerfile              for Hugging Face Spaces or any Docker host
```

## Models and licences

Weights are downloaded at run time and never stored in this repository.

| model | licence | note |
|---|---|---|
| SAM 3 (`facebook/sam3`) | SAM License (Meta) | gated: each account must be approved |
| BiRefNet (`ZhengPeng7/BiRefNet`) | MIT | loaded with `trust_remote_code` at a pinned revision |
| LaMa (`Carve/LaMa-ONNX`) | Apache-2.0 | |
| Real-ESRGAN x4plus | BSD-3-Clause | |
| PP-OCRv5 (PaddleOCR) | Apache-2.0 | |
| FLUX.2 Klein 4B (GPU engine) | Apache-2.0 | |
| SD x4 upscaler / SD 3.5 Medium (GPU engine) | CreativeML Open RAIL++-M / Stability Community License | |

## Limits

- Anything a hand hid is the model's estimate. With LaMa, the part of the pack under the hand is
  usually filled with background, so the cut-out's edge there can be ragged.
- Real-ESRGAN sharpens; it does not make illegible small print legible. That needs the GPU engine,
  and text under ~8 px tall is beyond any of these models.
- Classification uses the ONDC dev endpoint; if it is unreachable the run continues with generic
  prompts.
