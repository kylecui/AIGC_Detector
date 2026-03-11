"""API route definitions for the AIGC detection service.

Endpoints:
    POST /api/v1/detect  — Run AI text detection on submitted text
    GET  /api/v1/health  — Health check with GPU status

References:
    - DESIGN.md §5 (API specification)
    - DEVPLAN.md Phase 4 task 4.6
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from aigc_detector.api.middleware import limiter
from aigc_detector.api.schemas import DetectionRequest, DetectionResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detection"])

# Concurrency semaphore: max 2 concurrent GPU inference requests
MAX_CONCURRENT_REQUESTS = 2
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
QUEUE_TIMEOUT_SECONDS = 120


@router.post("/detect", response_model=DetectionResponse)
@limiter.limit("10/minute")
async def detect_text(request: Request, data: DetectionRequest) -> DetectionResponse:
    """Detect whether submitted text is AI-generated or human-written.

    Rate limited to 10 requests per minute per IP.
    Queued with a 120-second timeout if the server is busy.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Detection pipeline not initialized")

    try:
        async with asyncio.timeout(QUEUE_TIMEOUT_SECONDS):
            async with _semaphore:
                result = await run_in_threadpool(pipeline.detect, data.text)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, please retry later")

    return DetectionResponse(
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        p_ai=result.p_ai,
        detected_language=result.detected_language,
        stages_used=result.stages_used,
        breakdown=result.breakdown,
        processing_time_ms=result.processing_time_ms,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health status and GPU memory usage."""
    start_time = getattr(request.app.state, "start_time", time.time())
    model_manager = getattr(request.app.state, "model_manager", None)

    models_loaded: list[str] = []
    gpu_used_mb = 0.0
    gpu_total_mb = 0.0

    if model_manager is not None:
        status = model_manager.status()
        models_loaded = status.get("loaded_models", [])
        gpu_used_mb = status.get("gpu_allocated_mb", 0.0)
        gpu_total_mb = status.get("gpu_total_mb", 0.0)

    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        gpu_memory_used_mb=round(gpu_used_mb, 1),
        gpu_memory_total_mb=round(gpu_total_mb, 1),
        uptime_seconds=round(time.time() - start_time, 1),
    )
