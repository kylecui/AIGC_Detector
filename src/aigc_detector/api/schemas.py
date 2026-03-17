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


class HealthResponse(BaseModel):
    """Response body for ``GET /api/v1/health``."""

    status: str = Field(default="ok")
    models_loaded: list[str] = Field(default_factory=list)
    gpu_memory_used_mb: float = Field(default=0.0)
    gpu_memory_total_mb: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
