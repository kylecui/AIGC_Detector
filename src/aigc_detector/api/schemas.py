"""Pydantic v2 schemas for the detection API.

Defines request/response models for ``POST /api/v1/detect`` and
``GET /api/v1/health``.

References:
    - DESIGN.md §5.2 (API response schema)
    - DEVPLAN.md Phase 4 task 4.4
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Request body for ``POST /api/v1/detect``."""

    text: str = Field(
        ...,
        min_length=50,
        max_length=10_000,
        description="Text to analyze (50–10,000 characters).",
    )
    models: list[str] = Field(
        default=["all"],
        description="Which detection models to use. Default 'all' runs the full pipeline.",
    )
    include_segments: bool = Field(
        default=False,
        description="If true, also return segment-level detection results.",
    )
    include_diagnostics: bool = Field(
        default=False,
        description="If true, also return linguistic-stylistic diagnostics (micro/meso/macro scores).",
    )


class StageBreakdown(BaseModel):
    """Result from a single detection stage."""

    class Config:
        extra = "allow"


class DetectionResponse(BaseModel):
    """Response body for ``POST /api/v1/detect``."""

    predicted_label: str = Field(..., description="'AI-generated' or 'Human-written'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    p_ai: float = Field(..., ge=0.0, le=1.0, description="Probability of AI generation")
    detected_language: str = Field(..., description="ISO-639 code: 'zh' or 'en'")
    stages_used: list[str] = Field(default_factory=list)
    breakdown: dict = Field(
        default_factory=dict,
        description="Per-stage result details",
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    segments: list[dict] = Field(
        default_factory=list,
        description="Optional segment-level detection results.",
    )
    segment_highlights: dict | None = Field(
        default=None,
        description=(
            "Optional auxiliary signal surfacing the strongest local AI traces: "
            "{max_p_ai, top_k_segments: [{index, p_ai, text_snippet}], n_segments}. "
            "An auxiliary review aid only — it may disagree with the document-level "
            "verdict by design (see docs/capability-statement.md)."
        ),
    )
    caveat: dict | None = Field(
        default=None,
        description=(
            "Optional register caveat: present when the text hits the formal-document "
            "register (声明/公告/承诺书…), where overall verdict reliability is "
            "reduced. Carries {code, message, action_guidance}."
        ),
    )
    calibration: dict | None = Field(
        default=None,
        description=(
            "Present when register-conditioned confidence calibration was applied "
            "(formal register only): {method, register, T, confidence_raw, note}. "
            "The displayed confidence is the calibrated value; verdict and ranking "
            "are unchanged by construction."
        ),
    )
    decision_rule: dict | None = Field(
        default=None,
        description=(
            "Present when the register-gated binoculars-floor OR-rule fired "
            "(W15 candidate, enabled=false by default): the verdict was upgraded "
            "to AI-generated because the binoculars stage exceeded its cutoff in "
            "the formal register. Carries {rule, cutoff, binoculars_p_ai, note}."
        ),
    )
    linguistic_diagnostics: dict | None = Field(
        default=None,
        description="Optional linguistic-stylistic diagnostics (only present when include_diagnostics=True).",
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /api/v1/health``."""

    status: str = Field(default="ok")
    models_loaded: list[str] = Field(default_factory=list)
    gpu_memory_used_mb: float = Field(default=0.0)
    gpu_memory_total_mb: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)
    pipeline_ready: bool = Field(
        default=False,
        description="True when the pipeline and its language router are loaded (testing-mode stubs count as ready).",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
