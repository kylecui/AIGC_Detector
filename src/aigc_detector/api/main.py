"""FastAPI application entry point for the AIGC Detector service.

Lifespan: loads the detection pipeline on startup, releases on shutdown.
Serves the API routes and static frontend files.

Usage:
    uvicorn aigc_detector.api.main:app --host 0.0.0.0 --port 8000

References:
    - DESIGN.md §5 (API design)
    - DEVPLAN.md Phase 4 task 4.7
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aigc_detector.api.middleware import log_auth_disabled_once, setup_middleware
from aigc_detector.api.routes import metrics_router, router
from aigc_detector.config import settings
from aigc_detector.detection.binoculars import BinocularsDetector
from aigc_detector.detection.ensemble import EnsembleResult
from aigc_detector.models.manager import ModelManager
from aigc_detector.utils.hf_cache import is_model_cached

logger = logging.getLogger(__name__)

_TESTING_ENV_VALUES = {"1", "true", "yes", "on"}


def testing_mode() -> bool:
    """AIGC_TESTING=1 (or true/yes/on): boot without any models/GPU."""
    return os.environ.get("AIGC_TESTING", "").strip().lower() in _TESTING_ENV_VALUES


class StubPipeline:
    """Deterministic stand-in used when AIGC_TESTING is enabled.

    Lets the real app boot in milliseconds (no models, no GPU) for
    smoke/e2e tests. Readiness checks treat the stub as ready.
    """

    is_stub: bool = True

    def detect(self, text: str) -> EnsembleResult:
        return EnsembleResult(
            predicted_label="Human-written",
            confidence=0.5,
            p_ai=0.5,
            detected_language="zh",
            stages_used=["stub"],
            breakdown={},
            processing_time_ms=1.0,
        )


# Static files directory (relative to project root)
def _resolve_static_dir() -> Path:
    """Prefer packaged assets (wheel force-include), fall back to repo layout."""
    pkg_static = Path(__file__).resolve().parent.parent / "static"  # aigc_detector/static
    if pkg_static.is_dir():
        return pkg_static
    return Path(__file__).resolve().parent.parent.parent.parent / "static"


STATIC_DIR = _resolve_static_dir()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, unload on shutdown."""
    # P0-4: logging with sanitizing guard BEFORE any request can be logged
    from aigc_detector.utils.log_hygiene import setup_service_logging

    setup_service_logging(log_dir=settings.log_dir)
    logger.info("Starting AIGC Detector service...")
    app.state.start_time = time.time()

    log_auth_disabled_once()

    # ------------------------------------------------------------------
    # AIGC_TESTING=1: skip ALL model loading; inject a deterministic stub
    # pipeline so the real app boots in milliseconds without models/GPU.
    # Readiness (/health, /ready) reports ready for the stub.
    # ------------------------------------------------------------------
    if testing_mode():
        logger.warning("AIGC_TESTING=1 active: skipping model loading, injecting stub pipeline")
        app.state.model_manager = ModelManager(max_vram_gb=settings.max_vram_gb)
        app.state.pipeline = StubPipeline()
        logger.info("AIGC Detector service started (TESTING mode, stub pipeline)")
        yield
        logger.info("Shutting down AIGC Detector service (TESTING mode)...")
        app.state.model_manager.unload_all()
        logger.info("Service stopped")
        return

    # Model manager + full assembly via the single declarative entry point
    # (v0.2a: replaces the hand-written construction; see plans/default.yaml)
    from aigc_detector.plan import PlanRunner

    bundle = PlanRunner.default().build()
    model_manager = bundle.model_manager
    app.state.model_manager = model_manager
    pipeline = bundle.pipeline
    app.state.pipeline = pipeline
    missing_binoculars = bundle.missing_binoculars

    # Start background thread to download missing Binoculars models.
    #
    # Download strategy (see DETECTOR_NOTES_2026-06.md):
    #   - Standard Python downloader (NOT hf_transfer): properly resumes from
    #     .incomplete files after service restart or network failure
    #   - Sequential (max_workers=1): each file completes fully before the next,
    #     minimizing restart blast radius
    #   - Read timeout (30s): prevents indefinite hangs on stalled connections
    #   - ignore_patterns: skip redundant *.bin/*.pt/*.onnx (safetensors only)
    #   - Retry with exponential backoff per repo
    if missing_binoculars:

        def _bg_download_binoculars():
            import time as _time

            # --- Configure for reliable resume ---
            # hf_transfer (Rust) is faster but does NOT support resume — it
            # ignores .incomplete files and restarts from scratch. Since this
            # is a background thread (non-blocking), reliability > speed.
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            # Timeout: abort download if no data received for 30 seconds
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            # Respect user-set HF_ENDPOINT (e.g. mirror) but don't default to one
            # (hf-mirror.com redirect chain breaks huggingface_hub metadata)

            from huggingface_hub import snapshot_download

            _ignore = [
                "*.bin",  # PyTorch native (duplicate of safetensors)
                "*.pt",  # PyTorch checkpoint
                "*.h5",  # TensorFlow weights
                "*.msgpack",  # Flax weights
                "*.onnx",  # ONNX inference format
                "*.gguf",  # GGUF quantized format
                "original/*",  # Pre-conversion original weights
                "tf_model*",  # TF model directory
                "flax_model*",  # Flax model directory
                "pytorch_model*",  # Legacy PyTorch model directory
            ]

            # Collect unique repos that still need downloading
            repos_needed = [
                repo_id
                for _lang, obs, perf in missing_binoculars
                for repo_id in (obs, perf)
                if not is_model_cached(repo_id)
            ]
            # Deduplicate while preserving order
            seen: set[str] = set()
            repos_needed = [r for r in repos_needed if not (r in seen or seen.add(r))]

            # Download each repo sequentially with retry
            for repo_id in repos_needed:
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        logger.info(
                            "[Binoculars BG] Downloading %s (attempt %d/%d)...",
                            repo_id,
                            attempt,
                            max_retries,
                        )
                        snapshot_download(
                            repo_id,
                            ignore_patterns=_ignore,
                            max_workers=1,  # Sequential: reliable resume
                        )
                        logger.info("[Binoculars BG] Downloaded %s", repo_id)
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            wait = 5 * (2 ** (attempt - 1))
                            logger.warning(
                                "[Binoculars BG] %s failed (attempt %d): %s — retrying in %ds",
                                repo_id,
                                attempt,
                                e,
                                wait,
                            )
                            _time.sleep(wait)
                        else:
                            logger.warning(
                                "[Binoculars BG] %s failed after %d attempts: %s",
                                repo_id,
                                max_retries,
                                e,
                            )

            # Phase 2: activate detectors for pairs where both models cached
            for lang, observer, performer in missing_binoculars:
                if not (is_model_cached(observer) and is_model_cached(performer)):
                    logger.warning("[Binoculars BG] Skipping %s: incomplete download", lang)
                    continue
                try:
                    detector = BinocularsDetector(
                        observer_name=observer,
                        performer_name=performer,
                        mode="low-fpr",
                        device=settings.device,
                        load_in_4bit=True,
                    )
                    pipeline.binoculars_detectors[lang] = detector
                    logger.info("[Binoculars BG] Binoculars now active for %s", lang)
                except Exception as e:
                    logger.warning("[Binoculars BG] Detector creation failed for %s: %s", lang, e)

        bg_thread = threading.Thread(target=_bg_download_binoculars, daemon=True, name="binoculars-dl")
        bg_thread.start()
        logger.info(
            "Binoculars background download started (%d pairs). "
            "Mode: reliable resume (sequential, no hf_transfer, 30s timeout).",
            len(missing_binoculars),
        )

    logger.info("AIGC Detector service started (device=%s)", settings.device)
    yield

    # Shutdown: unload all models
    logger.info("Shutting down AIGC Detector service...")
    model_manager.unload_all()
    logger.info("All models unloaded, service stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AIGC Detector",
        description="Bilingual (Chinese/English) AI-generated text detection API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware
    setup_middleware(app)

    # API routes
    app.include_router(router)
    app.include_router(metrics_router)

    # Static files (frontend)
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
        logger.info("Static files served from %s", STATIC_DIR)
    else:
        logger.warning("Static directory not found: %s", STATIC_DIR)

    return app


app = create_app()
