"""API route definitions for the AIGC detection service.

Endpoints:
    POST /api/v1/detect        — Run AI text detection on submitted text
    POST /api/v1/detect/file   — Upload PDF/TXT file, extract text, detect
    GET  /api/v1/health        — Health check with GPU status (503 when not ready)
    GET  /api/v1/ready         — Orchestrator readiness probe (204/503)
    GET  /metrics              — Prometheus text-format metrics

References:
    - DESIGN.md §5 (API specification)
    - DEVPLAN.md Phase 4 task 4.6
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from aigc_detector.api.metrics import metrics_registry
from aigc_detector.api.middleware import limiter
from aigc_detector.api.schemas import DetectionRequest, DetectionResponse, HealthResponse
from aigc_detector.detection.linguistic import LinguisticDiagnostics, LinguisticFeatureExtractor
from aigc_detector.detection.register import (
    EN_FORMAL_DOWNGRADE,
    FORMAL_ZH_CAVEAT,
    LITERARY_AMBIGUITY_CAVEAT,
    binoculars_floor,
    detect_literary_ambiguity,
    detect_register_en_formal,
    detect_register_zh,
    formal_temperature,
)
from aigc_detector.utils.text import split_sentences_bilingual

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detection"])
metrics_router = APIRouter(tags=["metrics"])

# Concurrency semaphore: max 2 concurrent GPU inference requests
MAX_CONCURRENT_REQUESTS = 2
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
QUEUE_TIMEOUT_SECONDS = 120
MIN_SEGMENT_CHARS = 80
MAX_SEGMENTS = 8

# File upload limits
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_EXTRACTED_TEXT_CHARS = 50000

# PDF hardening limits (decompression-bomb / oversized-doc guards)
MAX_PDF_PAGES = 150
MAX_PDF_PAGE_CHARS = 500_000


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


def _en_formal_downgrade(result, text: str) -> dict | None:
    """W16/P0-3: product-level guard on the EN formal blind spot.

    Measured basis: 71% [55%,84%] of human EN formal docs flagged as AI
    (n=35 probe). On register hit we DOWNGRADE instead of issuing a
    normal-confidence verdict: confidence is capped below the decision
    threshold (cap from models/calibration/en_register_gate.json,
    downgrade_confidence_cap, default 0.49) and a strong warning payload is
    attached. Verdict and p_ai are NOT rewritten — the score stays visible
    for ranking; we refuse to present it as a confident call. Fail-safe on
    any error.
    """
    try:
        hit, score = detect_register_en_formal(text)
    except Exception:  # noqa: BLE001
        return None
    if not hit:
        return None
    cap = 0.49
    try:
        from aigc_detector.detection.register import _calibration_dir

        meta = json.loads((_calibration_dir() / "en_register_gate.json").read_text(encoding="utf-8"))
        if isinstance(meta.get("downgrade_confidence_cap"), (int, float)):
            cap = float(meta["downgrade_confidence_cap"])
    except Exception:  # noqa: BLE001 — artifact read must never break the guard
        pass
    if result.confidence > cap:
        result.confidence = cap
    return {**EN_FORMAL_DOWNGRADE, "register_score": score}


def _literary_ambiguity_caveat(result, caveat: dict | None, text: str) -> dict | None:
    """W17: literary-ambiguity band caveat (confidence-down, never upgrade).

    Consulted only when no register caveat fired (formal_zh/formal_en own
    their registers). Fires per detect_literary_ambiguity (encoder band +
    sentence-CV); on fire, confidence is compressed toward 0.5 via the
    formal temperature when deployed (or halved toward 0.5 otherwise).
    Fail-safe: any error leaves the response untouched.
    """
    if caveat is not None:
        return None
    try:
        enc = (result.breakdown or {}).get("encoder")
        enc_p = enc.get("p_ai") if isinstance(enc, dict) else None
        if enc_p is None or not detect_literary_ambiguity(float(enc_p), text):
            return None
        # compress confidence into the low-trust zone (caveat-only rule:
        # 0.98-style confidences must not survive an acknowledged blind spot)
        if result.confidence > 0.6:
            result.confidence = 0.6
        return {**LITERARY_AMBIGUITY_CAVEAT, "encoder_p_ai": round(float(enc_p), 4)}
    except Exception:  # noqa: BLE001
        return None


def _register_caveat(text: str) -> dict | None:
    """Formal-register caveat (W3a/W3b): cheap lexical check, CPU-only.

    Returns the canonical caveat payload when the text hits the 公文体
    register, else None. Detection itself is unchanged — the caveat is the
    'entry-eligibility downgrade' (single-doc high-confidence verdict not
    suitable in this register) plus concrete action guidance (D2).
    """
    try:
        reg = detect_register_zh(text)
    except Exception:  # noqa: BLE001 — caveat must never break detection
        return None
    if not reg.is_formal_zh:
        return None
    return {
        **FORMAL_ZH_CAVEAT,
        "register_score": reg.score,
        "register_markers": reg.matched_markers[:6],
    }


def _calibrate_confidence(caveat: dict | None, confidence: float, p_ai: float) -> tuple[float, dict | None]:
    """W11-2: register-conditioned confidence calibration.

    When the formal-register caveat fired AND a fitted formal temperature is
    deployed (models/calibration/global_temperature.json applied=true), the
    displayed confidence is compressed: conf' = sigmoid(logit(conf)/T).
    Provably label-flip-free and ranking-preserving (monotone in conf).
    Non-formal texts: T=1, confidence unchanged (well-calibrated regions are
    not squashed — the single-global-T mistake W11-1 measured empirically).
    """
    if caveat is None:
        return confidence, None
    t = formal_temperature()
    if t is None or t <= 0:
        return confidence, None
    eps = 1e-6
    c = min(max(confidence, eps), 1 - eps)
    z = __import__("math").log(c / (1 - c))
    calibrated = 1 / (1 + __import__("math").exp(-z / t))
    return calibrated, {
        "method": "register-conditioned temperature scaling",
        "register": "formal_zh",
        "T": t,
        "confidence_raw": round(confidence, 4),
        "note": "置信度已按正式文书语域校准压缩；判定与排序不变",
    }


def _apply_binoculars_floor(result, caveat: dict | None, text: str, pipeline) -> dict | None:
    """W15 candidate: register-gated binoculars-floor OR-rule.

    When the formal-register caveat fired AND the floor is deployed
    (models/calibration/binoculars_floor.json enabled=true): if the
    binoculars stage did not run (early exit), force-run it via the
    pipeline; then if its p_ai >= cutoff, upgrade the verdict to
    AI-generated. This implements exactly the rule measured in
    reports/w3b_floor_analysis.json (flag if ensemble>=threshold OR
    binoculars>=cutoff). Boundary: catches raw contract generation;
    human-edited AI text (FN-1, bino 0.343) stays below every cutoff.

    Returns a provenance dict when the rule fired, else None. Fail-safe:
    any exception leaves the original verdict untouched.
    """
    if caveat is None:
        return None
    floor = binoculars_floor()
    if floor is None:
        return None
    try:
        breakdown = result.breakdown or {}
        if "binoculars" not in breakdown and text.strip():
            forced = pipeline._run_binoculars(text, getattr(result, "detected_language", "zh"))  # noqa: SLF001 — candidate bridge
            if forced:
                breakdown["binoculars"] = forced
                result.breakdown = breakdown
        bino = (result.breakdown or {}).get("binoculars") or {}
        bino_p = bino.get("p_ai") if isinstance(bino, dict) else None
        if not isinstance(bino_p, (int, float)):
            return None
        if bino_p >= floor["cutoff"] and result.predicted_label != "AI-generated":
            result.predicted_label = "AI-generated"
            result.p_ai = max(result.p_ai, float(bino_p))
            # confidence of the NEW verdict = the evidence that flipped it,
            # not max with the old human-verdict confidence (semantic fix,
            # gate-review 2026-08-21: old max() let stale human-confidence
            # endorse a thin-evidence AI flip)
            result.confidence = float(bino_p)
            return {
                "rule": "register_binoculars_floor",
                "cutoff": floor["cutoff"],
                "binoculars_p_ai": round(float(bino_p), 4),
                "note": "正式文书语域：Binoculars阶段检测到原始生成信号，判定按OR规则升级",
            }
    except Exception:  # noqa: BLE001 — floor must never break detection
        return None
    return None


def _segment_highlights(segments: list[dict], top_k: int = 3) -> dict | None:
    """Surface the strongest local AI traces as an auxiliary review signal.

    Returns {max_p_ai, top_k_segments, n_segments} or None when no segments.
    Deliberately decoupled from the document-level verdict: on mixed documents
    a single high-scoring segment may disagree with the overall label; this is
    presented to users as supporting evidence for manual review, not as a
    second verdict (see DETECTOR_NOTES_2026-08.md FN-1 / WaterSeeker precedent).
    """
    if not segments:
        return None
    scored = [
        {
            "index": s.get("index"),
            "p_ai": s.get("p_ai"),
            "text_snippet": (s.get("text") or "")[:80],
        }
        for s in segments
        if isinstance(s.get("p_ai"), (int, float))
    ]
    if not scored:
        return None
    scored.sort(key=lambda x: x["p_ai"], reverse=True)
    return {
        "max_p_ai": scored[0]["p_ai"],
        "top_k_segments": scored[:top_k],
        "n_segments": len(segments),
    }


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
    metrics_registry.inc_in_flight()
    t0 = perf_counter()
    status_code = 500
    try:
        pipeline = request.app.state.pipeline
        if pipeline is None:
            status_code = 503
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
            status_code = 503
            raise HTTPException(status_code=503, detail="Server busy, please retry later")

        # Optional linguistic-stylistic diagnostics. Computed on a fresh
        # CPU-only extractor (no shared state with the pipeline). M5/M6 require
        # per-token log-probs from a reference LM, which we don't recompute here
        # (those fields will be NaN — the diagnostics don't depend on them).
        linguistic_diagnostics: dict | None = None
        if data.include_diagnostics:
            try:
                diagnostics_extractor = LinguisticFeatureExtractor()
                features = diagnostics_extractor.extract(
                    data.text,
                    lang=result.detected_language,
                    token_log_probs=None,
                )
                diagnostics = LinguisticDiagnostics.from_features(
                    features,
                    lang=result.detected_language,
                )
                linguistic_diagnostics = dataclasses.asdict(diagnostics)
            except Exception:
                logger.warning("Linguistic diagnostics failed", exc_info=True)
                linguistic_diagnostics = None

        caveat = _register_caveat(data.text)
        en_downgrade = _en_formal_downgrade(result, data.text)
        if en_downgrade:
            caveat = en_downgrade
        if caveat is None:
            caveat = _literary_ambiguity_caveat(result, caveat, data.text)
        decision_rule = _apply_binoculars_floor(result, caveat if not en_downgrade else None, data.text, pipeline)
        confidence, calibration = _calibrate_confidence(caveat, result.confidence, result.p_ai)
        status_code = 200
        return DetectionResponse(
            predicted_label=result.predicted_label,
            confidence=round(confidence, 4),
            p_ai=result.p_ai,
            detected_language=result.detected_language,
            stages_used=result.stages_used,
            breakdown=result.breakdown,
            decision_rule=decision_rule,
            processing_time_ms=round(result.processing_time_ms + segment_time_ms, 1),
            segments=segments,
            segment_highlights=_segment_highlights(segments),
            caveat=caveat,
            calibration=calibration,
            linguistic_diagnostics=linguistic_diagnostics,
        )
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    except Exception:
        status_code = 500
        raise
    finally:
        metrics_registry.observe_seconds("detect", perf_counter() - t0)
        metrics_registry.inc_request("detect", status_code)
        metrics_registry.dec_in_flight()


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes.

    Primary: PyMuPDF (fitz) — fast, accurate text extraction.
    Fallback: pypdf — handles some edge cases PyMuPDF misses.
    Detection: scanned/image-only PDFs are identified and reported clearly.

    Raises HTTPException with actionable messages for:
    - Corrupted/invalid PDFs (both engines fail)
    - Scanned PDFs (images but no extractable text)
    - Empty PDFs (no content at all)
    - Oversized PDFs (> MAX_PDF_PAGES) and decompression bombs
      (any single page yielding > MAX_PDF_PAGE_CHARS)
    """
    text_parts: list[str] = []
    has_images = False
    pymupdf_ok = False

    # --- Primary: PyMuPDF ---
    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = doc.page_count
            if page_count > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF too large: {page_count} pages",
                )
            for page in doc:
                text = page.get_text("text")
                if not isinstance(text, str):
                    continue
                if len(text) > MAX_PDF_PAGE_CHARS:
                    raise HTTPException(
                        status_code=422,
                        detail="suspicious PDF structure",
                    )
                if text.strip():
                    text_parts.append(text)
                if page.get_images():
                    has_images = True
        pymupdf_ok = True
        logger.info(
            "PyMuPDF: extracted %d chars from %d pages (has_images=%s)",
            len("\n".join(text_parts)),
            len(text_parts),
            has_images,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("PyMuPDF failed (%s: %s), trying pypdf fallback", type(e).__name__, e)

    # --- Fallback: pypdf ---
    if not text_parts:
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF too large: {len(reader.pages)} pages",
                )
            for page in reader.pages:
                extracted = page.extract_text()
                if isinstance(extracted, str):
                    if len(extracted) > MAX_PDF_PAGE_CHARS:
                        raise HTTPException(
                            status_code=422,
                            detail="suspicious PDF structure",
                        )
                    if extracted:
                        text_parts.append(extracted)
            logger.info("pypdf fallback: extracted %d chars", len("\n".join(text_parts)))
        except HTTPException:
            raise
        except Exception as e:
            logger.error("pypdf also failed: %s: %s", type(e).__name__, e)
            if not pymupdf_ok:
                raise HTTPException(
                    status_code=422,
                    detail="PDF解析失败，文件可能已损坏或不是有效的PDF。PyMuPDF and pypdf both failed.",
                )

    raw = "\n".join(text_parts)

    # --- Diagnose empty extraction ---
    if not raw.strip():
        if has_images:
            raise HTTPException(
                status_code=422,
                detail="PDF包含图片但无可提取文本（可能是扫描件）。"
                "请先用OCR工具（如Tesseract）将图片转换为文本后重试。",
            )
        raise HTTPException(
            status_code=422,
            detail="PDF中未找到任何文本内容。",
        )

    # --- Normalize whitespace ---
    lines = [ln.strip() for ln in raw.split("\n")]
    cleaned = "\n".join(ln for ln in lines if ln)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned[:MAX_EXTRACTED_TEXT_CHARS]


def _extract_text_from_file(filename: str, content: bytes) -> str:
    """Dispatch text extraction based on file extension.

    Returns extracted plain text (UTF-8). Raises HTTPException on failure.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        # _extract_text_from_pdf handles its own HTTPException raises
        return _extract_text_from_pdf(content)
    elif ext in (".txt", ".md"):
        # Try common encodings
        for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                return content.decode(encoding)[:MAX_EXTRACTED_TEXT_CHARS]
            except (UnicodeDecodeError, ValueError):
                continue
        raise HTTPException(status_code=422, detail="Could not decode text file (tried utf-8/gbk/latin-1)")
    else:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )


@router.post("/detect/file", response_model=DetectionResponse)
@limiter.limit("10/minute")
async def detect_file(
    request: Request,
    file: UploadFile = File(...),
    include_segments: bool = True,
    include_diagnostics: bool = False,
) -> DetectionResponse:
    """Upload a PDF or text file, extract text, and run AI detection.

    Supported formats: PDF (.pdf), plain text (.txt), Markdown (.md).
    Max file size: 20 MB. Extracted text is truncated at 50,000 characters.
    PDFs over 150 pages or with bomb-like page structure are rejected (422).

    Rate limited to 10 requests per minute per IP.
    """
    metrics_registry.inc_in_flight()
    t0 = perf_counter()
    status_code = 500
    try:
        pipeline = request.app.state.pipeline
        if pipeline is None:
            status_code = 503
            raise HTTPException(status_code=503, detail="Detection pipeline not initialized")

        # Validate file extension
        filename = file.filename or "unknown"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            status_code = 415
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # Read file content with size limit
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            status_code = 413
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content) // 1024 // 1024} MB). Max: {MAX_FILE_SIZE_MB} MB.",
            )
        if len(content) == 0:
            status_code = 422
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        logger.info("File upload: '%s' (%d bytes, %s)", filename, len(content), ext)

        # Extract text (CPU-bound, run in threadpool)
        text = await run_in_threadpool(_extract_text_from_file, filename, content)

        if len(text.strip()) < 50:
            status_code = 422
            short_detail = (
                f"Extracted text too short ({len(text.strip())} chars). Need at least 50 characters for detection."
            )
            raise HTTPException(status_code=422, detail=short_detail)

        logger.info("Extracted %d characters from '%s'", len(text), filename)

        # Run detection (same pipeline as /detect)
        try:
            async with asyncio.timeout(QUEUE_TIMEOUT_SECONDS):
                async with _semaphore:
                    result = await run_in_threadpool(pipeline.detect, text)
                    segments: list[dict] = []
                    segment_time_ms = 0.0
                    if include_segments:
                        segments, segment_time_ms = await run_in_threadpool(_detect_segments, pipeline, text)
        except TimeoutError:
            status_code = 503
            raise HTTPException(status_code=503, detail="Server busy, please retry later")

        # Optional linguistic diagnostics
        linguistic_diagnostics: dict | None = None
        if include_diagnostics:
            try:
                diagnostics_extractor = LinguisticFeatureExtractor()
                features = diagnostics_extractor.extract(text, lang=result.detected_language, token_log_probs=None)
                diagnostics = LinguisticDiagnostics.from_features(features, lang=result.detected_language)
                linguistic_diagnostics = dataclasses.asdict(diagnostics)
            except Exception:
                logger.warning("Linguistic diagnostics failed", exc_info=True)
                linguistic_diagnostics = None

        caveat = _register_caveat(text)
        en_downgrade = _en_formal_downgrade(result, text)
        if en_downgrade:
            caveat = en_downgrade
        if caveat is None:
            caveat = _literary_ambiguity_caveat(result, caveat, text)
        decision_rule = _apply_binoculars_floor(result, caveat if not en_downgrade else None, text, pipeline)
        confidence, calibration = _calibrate_confidence(caveat, result.confidence, result.p_ai)
        status_code = 200
        return DetectionResponse(
            predicted_label=result.predicted_label,
            confidence=round(confidence, 4),
            p_ai=result.p_ai,
            detected_language=result.detected_language,
            stages_used=result.stages_used,
            breakdown=result.breakdown,
            decision_rule=decision_rule,
            processing_time_ms=round(result.processing_time_ms + segment_time_ms, 1),
            segments=segments,
            segment_highlights=_segment_highlights(segments),
            caveat=caveat,
            calibration=calibration,
            linguistic_diagnostics=linguistic_diagnostics,
        )
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    except Exception:
        status_code = 500
        raise
    finally:
        metrics_registry.observe_seconds("detect_file", perf_counter() - t0)
        metrics_registry.inc_request("detect_file", status_code)
        metrics_registry.dec_in_flight()


def _pipeline_ready(request: Request) -> bool:
    """Readiness: pipeline present AND its language router loaded.

    The AIGC_TESTING=1 stub pipeline (``is_stub``) counts as ready so the
    boot-check stays green in test mode even though no models are loaded.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return False
    if getattr(pipeline, "is_stub", False) is True:
        return True
    language_router = getattr(pipeline, "language_router", None)
    if language_router is None:
        return False
    is_loaded = getattr(language_router, "is_loaded", False)
    if callable(is_loaded):
        return bool(is_loaded())
    return bool(is_loaded)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request, response: Response) -> HealthResponse:
    """Return service health status and GPU memory usage.

    HTTP 503 when the detection pipeline is not ready (missing/unloaded
    language router); 200 otherwise — including AIGC_TESTING=1 stub mode.
    """
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

    ready = _pipeline_ready(request)
    if not ready:
        response.status_code = 503
    return HealthResponse(
        status="ok" if ready else "not_ready",
        models_loaded=models_loaded,
        gpu_memory_used_mb=round(gpu_used_mb, 1),
        gpu_memory_total_mb=round(gpu_total_mb, 1),
        uptime_seconds=round(time.time() - start_time, 1),
        pipeline_ready=ready,
    )


@router.get("/ready", status_code=204)
async def ready_check(request: Request) -> Response:
    """Orchestrator readiness probe: 204 when ready, 503 otherwise."""
    if not _pipeline_ready(request):
        return Response(status_code=503)
    return Response(status_code=204)


@metrics_router.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus text-format metrics (auth-exempt)."""
    return PlainTextResponse(
        metrics_registry.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
