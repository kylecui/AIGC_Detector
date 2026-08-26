# syntax=docker/dockerfile:1
# =============================================================================
# AIGC Detector — GPU service image (single stage, research instrument)
# Target: single-tenant self-hosted, RTX 3060 12GB (4-bit + LRU by design).
# Models are NOT baked in: base models (~4-8GB) download on first use from
# HuggingFace; optional Binoculars pairs (~28GB) download in background.
#
# PATH ASSUMPTIONS (verified against src/):
#   1. STATIC_DIR (api/main.py) = 4 parents above main.py + /static. This only
#      works because `uv sync` installs the root project EDITABLE by default,
#      so aigc_detector resolves to /app/src/aigc_detector -> /app/static.
#      DO NOT add --no-editable to uv sync, or the WebUI silently disappears
#      (it would resolve to <venv>/lib/python3.12/static).
#   2. ModelManager registry_path default = "configs/models.yaml" (CWD-relative)
#      -> process MUST run with WORKDIR=/app and configs/ present.
#   3. Settings.model_dir default = CWD-relative "models/"; override at runtime
#      with -e MODEL_DIR=/home/app/data/models if you mount local artifacts
#      (see .dockerignore note about trained LoRA/joblib artifacts).
#   4. torch/torchvision wheels come from the cu124 index pinned in uv.lock and
#      bundle their own CUDA user-space libs; the CUDA 12.1.1 runtime base only
#      supplies the driver-facing userspace — driver itself comes from the host.
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="aigc-detector" \
      org.opencontainers.image.description="Bilingual (Chinese/English) AI-generated text detection service (statistical/encoder/Binoculars ensemble, FastAPI)" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/CHANGE-ME/AIGC_Detector"

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

# --- System deps: kept minimal (curl only for HEALTHCHECK) -------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- uv from the official image ----------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/uv

# Managed CPython 3.12 (ubuntu22.04 apt ships 3.10; project needs >=3.12).
# Installed to /opt/uv-python so the non-root user can execute it.
RUN uv python install 3.12

WORKDIR /app

# --- Dependency layer (cached until pyproject/lock change) -------------------
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Application --------------------------------------------------------------
# static/ and configs/ are required at runtime (WebUI + model registry).
COPY src ./src
COPY static ./static
COPY configs ./configs
RUN uv sync --frozen --no-dev

# --- Non-root user with HF cache home ----------------------------------------
RUN useradd --create-home --home-dir /home/app --shell /usr/sbin/nologin appuser \
    && mkdir -p /home/app/.cache/huggingface /home/app/data \
    && chown -R appuser:appuser /home/app /app
USER appuser

# venv first on PATH: reliable direct entrypoints, no `uv run` re-sync at start.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# First boot with an empty HF cache downloads the language-id model (~1GB)
# inside lifespan BEFORE uvicorn accepts connections -> generous start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# Suggested volumes (mount to persist caches / attach local artifacts):
#   /home/app/.cache/huggingface  — HF model cache (base + binoculars, ~36GB max)
#   /home/app/data                — optional local models; pair with
#                                   -e MODEL_DIR=/home/app/data/models
CMD ["uvicorn", "aigc_detector.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
