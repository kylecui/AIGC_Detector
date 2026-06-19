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

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aigc_detector.api.middleware import setup_middleware
from aigc_detector.api.routes import router
from aigc_detector.config import settings
from aigc_detector.detection.binoculars import BinocularsDetector
from aigc_detector.detection.encoder import EncoderClassifier
from aigc_detector.detection.language import LanguageRouter
from aigc_detector.detection.linguistic import LinguisticClassifier, LinguisticFeatureExtractor
from aigc_detector.detection.pipeline import DetectionPipeline
from aigc_detector.detection.statistical import StatisticalClassifier, StatisticalFeatureExtractor
from aigc_detector.models.manager import ModelManager
from aigc_detector.utils.hf_cache import is_model_cached

logger = logging.getLogger(__name__)

# Static files directory (relative to project root)
STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, unload on shutdown."""
    logger.info("Starting AIGC Detector service...")
    app.state.start_time = time.time()

    # Model manager
    model_manager = ModelManager(max_vram_gb=settings.max_vram_gb)
    app.state.model_manager = model_manager

    # Language router (always loaded — ~1 GB)
    language_router = LanguageRouter(device=settings.device)
    try:
        language_router.load()
        model_manager.load("xlm-roberta-lang-detect", language_router)
    except Exception:
        logger.warning("Language detection model failed to load, using heuristic fallback")

    # Build detector wrappers (actual model weights stay lazily loaded until first use)
    statistical_extractors = {
        "en": StatisticalFeatureExtractor(
            model_name="openai-community/gpt2-xl",
            device=settings.device,
            load_in_4bit=False,
        ),
        "zh": StatisticalFeatureExtractor(
            model_name="IDEA-CCNL/Wenzhong-GPT2-110M",
            device=settings.device,
            load_in_4bit=False,
        ),
    }
    statistical_classifiers: dict[str, StatisticalClassifier] = {}
    for lang in ("en", "zh"):
        clf_path = settings.model_dir / f"statistical-{lang}" / "classifier.joblib"
        if clf_path.exists():
            clf = StatisticalClassifier()
            clf.load(clf_path)
            cal_path = settings.model_dir / f"statistical-{lang}" / "calibration.json"
            if cal_path.exists():
                try:
                    calibration = json.loads(cal_path.read_text(encoding="utf-8"))
                    if "optimal_threshold" in calibration:
                        clf.set_threshold(float(calibration["optimal_threshold"]))
                except Exception:
                    logger.warning("Failed to load calibration for %s", lang, exc_info=True)
            statistical_classifiers[lang] = clf
        else:
            logger.warning("Statistical classifier missing for %s: %s", lang, clf_path)

    # Linguistic classifiers (CPU-only, no model_manager registration).
    # Optional: if the artifact is absent for a language, the pipeline simply
    # skips the linguistic stage for that language.
    linguistic_classifiers: dict[str, LinguisticClassifier] = {}
    for lang in ("en", "zh"):
        clf_path = settings.model_dir / f"linguistic-{lang}" / "classifier.joblib"
        if clf_path.exists():
            clf = LinguisticClassifier()
            clf.load(clf_path)
            # Linguistic has no calibration file yet, but be forward-compatible:
            cal_path = settings.model_dir / f"linguistic-{lang}" / "calibration.json"
            if cal_path.exists():
                try:
                    calibration = json.loads(cal_path.read_text(encoding="utf-8"))
                    if "optimal_threshold" in calibration:
                        clf.set_threshold(float(calibration["optimal_threshold"]))
                except Exception:
                    logger.warning("Failed to load linguistic calibration for %s", lang, exc_info=True)
            linguistic_classifiers[lang] = clf
        else:
            logger.info("Linguistic classifier not found for %s: %s (skipping)", lang, clf_path)

    # Linguistic extractors are pure-CPU and side-effect-free; instantiate one
    # per language with the default min_text_chars=200.
    linguistic_extractors: dict[str, LinguisticFeatureExtractor] = {
        "en": LinguisticFeatureExtractor(),
        "zh": LinguisticFeatureExtractor(),
    }

    encoder_classifiers = {
        "en": EncoderClassifier(
            base_model_name="microsoft/deberta-v3-large",
            adapter_path=settings.model_dir / "encoder-en",
            device=settings.device,
        ),
        "zh": EncoderClassifier(
            base_model_name="hfl/chinese-roberta-wwm-ext-large",
            adapter_path=settings.model_dir / "encoder-zh",
            device=settings.device,
        ),
    }

    # Binoculars detectors — OPTIONAL zero-shot stage.
    # Uses large 7B model pairs (~14GB per language). On startup:
    # - If models are already cached → enable immediately.
    # - If not cached → start background download thread (non-blocking),
    #   service runs without Binoculars until download completes.
    binoculars_detectors: dict[str, object] = {}

    bino_configs = {
        "en": ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),
        "zh": ("Qwen/Qwen2-7B", "Qwen/Qwen2-7B-Instruct"),
    }

    # Models that need background download
    missing_binoculars: list[tuple[str, str, str]] = []  # (lang, observer, performer)

    for lang, (observer, performer) in bino_configs.items():
        if is_model_cached(observer) and is_model_cached(performer):
            binoculars_detectors[lang] = BinocularsDetector(
                observer_name=observer,
                performer_name=performer,
                mode="low-fpr",
                device=settings.device,
                load_in_4bit=True,
            )
            logger.info("Binoculars enabled for %s (%s + %s)", lang, observer, performer)
        else:
            missing_binoculars.append((lang, observer, performer))
            logger.info(
                "Binoculars pending for %s — will download in background", lang
            )

    # Detection pipeline (detectors instantiated here, weights loaded lazily on first use)
    #
    # Language-specific ensemble weights (validated 2026-06-17 on Defactify EN + HC3 ZH):
    # - EN: encoder LoRA trained on project's original domain underperforms on modern
    #   multi-LLM text (GPT-4o/Llama/Qwen). Linguistic axis is the primary signal.
    #   See DETECTOR_NOTES_2026-06.md and scripts/tune_en_detector.py for the sweep
    #   that identified this configuration.
    # - ZH: encoder LoRA retrained 2026-06-19 with oversampled textbook data (10x).
    #   Now correctly detects modern LLM text (GPT-4/Claude) with p_ai>0.99.
    #   Statistical classifier still overfits to HC3 (low weight kept as minor signal).
    #   See DETECTOR_NOTES_2026-06.md P3 section for the diagnostic and retraining.
    en_weights = {"linguistic": 0.85, "statistical": 0.15, "encoder": 0.0, "binoculars": 0.0}
    zh_weights = {"linguistic": 0.10, "statistical": 0.10, "encoder": 0.60, "binoculars": 0.20}

    pipeline = DetectionPipeline(
        language_router=language_router,
        statistical_extractors=statistical_extractors,
        statistical_classifiers=statistical_classifiers,
        encoder_classifiers=encoder_classifiers,
        binoculars_detectors=binoculars_detectors,
        linguistic_extractors=linguistic_extractors,
        linguistic_classifiers=linguistic_classifiers,
        model_manager=model_manager,
        early_exit_threshold=0.99,  # Raised from 0.95 (was too aggressive for modern LLM text)
        ensemble_weights_by_lang={"en": en_weights, "zh": zh_weights},
    )
    app.state.pipeline = pipeline

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
                "*.bin",           # PyTorch native (duplicate of safetensors)
                "*.pt",            # PyTorch checkpoint
                "*.h5",            # TensorFlow weights
                "*.msgpack",       # Flax weights
                "*.onnx",          # ONNX inference format
                "*.gguf",          # GGUF quantized format
                "original/*",      # Pre-conversion original weights
                "tf_model*",       # TF model directory
                "flax_model*",     # Flax model directory
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
            repos_needed = [
                r for r in repos_needed
                if not (r in seen or seen.add(r))
            ]

            # Download each repo sequentially with retry
            for repo_id in repos_needed:
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        logger.info(
                            "[Binoculars BG] Downloading %s (attempt %d/%d)...",
                            repo_id, attempt, max_retries,
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
                                "[Binoculars BG] %s failed (attempt %d): %s — "
                                "retrying in %ds",
                                repo_id, attempt, e, wait,
                            )
                            _time.sleep(wait)
                        else:
                            logger.warning(
                                "[Binoculars BG] %s failed after %d attempts: %s",
                                repo_id, max_retries, e,
                            )

            # Phase 2: activate detectors for pairs where both models cached
            for lang, observer, performer in missing_binoculars:
                if not (is_model_cached(observer) and is_model_cached(performer)):
                    logger.warning(
                        "[Binoculars BG] Skipping %s: incomplete download", lang
                    )
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
                    logger.warning(
                        "[Binoculars BG] Detector creation failed for %s: %s", lang, e
                    )

        bg_thread = threading.Thread(
            target=_bg_download_binoculars, daemon=True, name="binoculars-dl"
        )
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

    # Static files (frontend)
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
        logger.info("Static files served from %s", STATIC_DIR)
    else:
        logger.warning("Static directory not found: %s", STATIC_DIR)

    return app


app = create_app()
