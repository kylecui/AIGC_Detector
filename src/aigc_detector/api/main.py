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
from aigc_detector.detection.pipeline import DetectionPipeline
from aigc_detector.detection.statistical import StatisticalClassifier, StatisticalFeatureExtractor
from aigc_detector.models.manager import ModelManager

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
            statistical_classifiers[lang] = clf
        else:
            logger.warning("Statistical classifier missing for %s: %s", lang, clf_path)

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

    binoculars_detectors = {
        "en": BinocularsDetector(
            observer_name="tiiuae/falcon-7b",
            performer_name="tiiuae/falcon-7b-instruct",
            mode="low-fpr",
            device=settings.device,
            load_in_4bit=True,
        ),
        "zh": BinocularsDetector(
            observer_name="Qwen/Qwen2-7B",
            performer_name="Qwen/Qwen2-7B-Instruct",
            mode="low-fpr",
            device=settings.device,
            load_in_4bit=True,
        ),
    }

    # Detection pipeline (detectors instantiated here, weights loaded lazily on first use)
    pipeline = DetectionPipeline(
        language_router=language_router,
        statistical_extractors=statistical_extractors,
        statistical_classifiers=statistical_classifiers,
        encoder_classifiers=encoder_classifiers,
        binoculars_detectors=binoculars_detectors,
        model_manager=model_manager,
    )
    app.state.pipeline = pipeline

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
