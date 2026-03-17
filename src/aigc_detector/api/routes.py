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
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from aigc_detector.api.middleware import limiter
from aigc_detector.api.schemas import DetectionRequest, DetectionResponse, HealthResponse
from aigc_detector.utils.text import split_sentences_bilingual

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detection"])

# Concurrency semaphore: max 2 concurrent GPU inference requests
MAX_CONCURRENT_REQUESTS = 2
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
QUEUE_TIMEOUT_SECONDS = 120
MIN_SEGMENT_CHARS = 80
MAX_SEGMENTS = 8


def _build_segments(text: str, min_chars: int = MIN_SEGMENT_CHARS, max_segments: int = MAX_SEGMENTS) -> list[dict]:
    """Create paragraph-like segments from bilingual sentence splitting.

    Groups adjacent sentences until a minimum character budget is reached.
    """
    sentences = split_sentences_bilingual(text)
    if not sentences:
        return []

    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence)
        if current_len >= min_chars:
            segments.append("".join(current).strip())
            current = []
            current_len = 0

    if current:
        if segments:
            segments[-1] = f"{segments[-1]} {''.join(current).strip()}".strip()
        else:
            segments.append("".join(current).strip())

    if len(segments) > max_segments:
        # Coalesce neighboring segments to bound latency.
        merged: list[str] = []
        chunk_size = max(1, (len(segments) + max_segments - 1) // max_segments)
        for i in range(0, len(segments), chunk_size):
            merged.append(" ".join(segments[i : i + chunk_size]).strip())
        segments = merged[:max_segments]

    segment_results = []
    search_start = 0
    for idx, segment_text in enumerate(segments):
        start = text.find(segment_text, search_start)
        if start < 0:
            start = search_start
        end = start + len(segment_text)
        search_start = end
        segment_results.append(
            {
                "index": idx,
                "text": segment_text,
                "char_start": start,
                "char_end": end,
            }
        )

    return segment_results


def _detect_segments(pipeline, text: str) -> tuple[list[dict], float]:
    segments = _build_segments(text)
    if not segments:
        return [], 0.0

    t0 = perf_counter()
    results: list[dict] = []
    for segment in segments:
        detected = pipeline.detect(segment["text"])
        results.append(
            {
                **segment,
                "predicted_label": detected.predicted_label,
                "confidence": detected.confidence,
                "p_ai": detected.p_ai,
                "detected_language": detected.detected_language,
                "stages_used": detected.stages_used,
                "breakdown": detected.breakdown,
                "processing_time_ms": detected.processing_time_ms,
            }
        )
    return results, (perf_counter() - t0) * 1000


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
                segments: list[dict] = []
                segment_time_ms = 0.0
                if data.include_segments:
                    segments, segment_time_ms = await run_in_threadpool(_detect_segments, pipeline, data.text)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, please retry later")

    return DetectionResponse(
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        p_ai=result.p_ai,
        detected_language=result.detected_language,
        stages_used=result.stages_used,
        breakdown=result.breakdown,
        processing_time_ms=round(result.processing_time_ms + segment_time_ms, 1),
        segments=segments,
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
