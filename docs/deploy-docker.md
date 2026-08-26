# Docker Deployment

GPU deployment of the AIGC Detector service. Single-stage research-instrument
image based on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`. No models are
baked in — everything downloads at runtime into the HF cache volume.

## Build

```bash
docker build -t aigc-detector:0.1.0 .
```

The dependency layer is driven by `pyproject.toml` + `uv.lock` (torch comes
from the cu124 wheel index pinned in the lock; wheels bundle their own CUDA
user-space libraries, the host only supplies the NVIDIA driver).

## Run

```bash
docker run -d --name aigc-detector \
  --gpus all \
  --shm-size 2g \
  -p 8000:8000 \
  -v aigc-hf-cache:/home/app/.cache/huggingface \
  -e HF_TOKEN=hf_xxx \
  aigc-detector:0.1.0
```

- `--gpus all` — required; the service defaults to `device=cuda`. Use
  `-e DEVICE=cpu` for a CPU smoke run (expect minutes-per-request latency).
- `HF_TOKEN` — optional; `tiiuae/falcon-7b` (Binoculars EN) is license-gated
  on HF, so pass a token that has accepted the license, or EN Binoculars will
  stay pending after its download retries fail.
- Optional local model artifacts: `-v /host/models:/app/models:ro` (trained
  LoRA/joblib artifacts — see "Known trade-offs" below).

## First-run behavior

1. **Startup (blocking)**: the language-ID model (XLM-RoBERTa based, ~1GB) is
   downloaded *before* uvicorn accepts connections. On a fresh cache, expect
   the health endpoint to come up only after that download finishes.
2. **First detection request**: base models for the requested language
   (gpt2-xl / Wenzhong-GPT2 / DeBERTa / Chinese-RoBERTa, ~4-8GB total) are
   downloaded on demand — first request per language takes minutes.
3. **Binoculars (background, non-blocking)**: ~14GB per language pair
   (~28GB for both) downloads sequentially with resume; until then the
   service runs in degraded mode (ensemble without the Binoculars axis).
   Health (`/api/v1/health`) reports loaded models and VRAM usage.

Cold start from an empty cache on a 100Mbps link: roughly 5-15 minutes to
"EN+ZH base detection usable", hours for both Binoculars pairs to land.

## Memory expectations

- Designed for a 12GB card (RTX 3060 class): 4-bit quantization for the 7B
  Binoculars pairs, 11GB VRAM budget with LRU eviction
  (`src/aigc_detector/models/manager.py`).
- Host RAM: keep ~8GB headroom for weight streaming during loads.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is preset in the image to
  reduce fragmentation under the eviction/reload cycle.

## Known trade-offs in this image

- `models/` is excluded from the build context (31+GB of checkpoints and
  experiment dumps). That also excludes the ~77MB of *trained* artifacts
  (encoder LoRA adapters, statistical/linguistic classifiers) that are not on
  HF. Consequence: EN falls back to linguistic-only (its dominant axis, 0.85
  weight — mild), but ZH loses its 0.60-weight encoder axis — detection
  quality is significantly reduced until you mount the artifacts
  (`-v .../models:/app/models:ro`) or uncomment the negation patterns at the
  bottom of `.dockerignore` and rebuild.
- The WebUI relies on an *editable* project install so `STATIC_DIR` resolves
  to `/app/static`; do not switch `uv sync` to `--no-editable`.
- GPU verification of this image is **pending** (planned for the v0.2 window).
  Only CPU-path build/run has been exercised so far.
