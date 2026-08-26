"""Shared fixtures for AIGC Detector API tests.

Additive only — ``tests/test_api.py`` keeps its own local fixtures untouched.

Provided fixtures:
    - ``stub_pipeline``: MagicMock detection pipeline returning a fixed
      EnsembleResult (same pattern as test_api.py, exposed for reuse).
    - ``make_stub_pipeline()``: factory for customized stub pipelines.
    - ``client``: TestClient over the REAL app factory (``create_app``) with
      the production lifespan swapped for a stub injector (no GPU/model
      loading). Auth env candidates are cleared so any concurrently-added
      API-key middleware runs with auth OFF. The slowapi ``limiter``
      singleton is reset before/after each test so per-IP quota consumed
      here never leaks into other test modules (and vice versa).
    - ``tmp_log_dir``: isolated per-test log directory.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Env vars that could switch on API-key auth in concurrently-developed
# middleware. Cleared so create_app() runs with auth disabled regardless
# of the host environment.
AUTH_ENV_CANDIDATES = ("AIGC_API_KEY", "AIGC__API_KEY", "AIGC_API_KEYS", "API_KEY")


def make_stub_pipeline(
    predicted_label: str = "AI-generated",
    confidence: float = 0.87,
    p_ai: float = 0.87,
    detected_language: str = "en",
) -> MagicMock:
    """Build a stub detection pipeline returning a fixed EnsembleResult.

    Mirrors the mocking pattern used in tests/test_api.py::TestAPIRoutes.
    """
    from aigc_detector.detection.ensemble import EnsembleResult

    mock_result = EnsembleResult(
        predicted_label=predicted_label,
        confidence=confidence,
        p_ai=p_ai,
        detected_language=detected_language,
        stages_used=["statistical", "encoder"],
        breakdown={"statistical": {"p_ai": 0.82}, "encoder": {"p_ai": 0.89}},
        processing_time_ms=450.0,
    )
    pipeline = MagicMock()
    pipeline.detect.return_value = mock_result
    return pipeline


def make_stub_model_manager() -> MagicMock:
    """Build a stub ModelManager exposing a healthy GPU status dict."""
    manager = MagicMock()
    manager.status.return_value = {
        "loaded_models": ["xlm-roberta", "deberta-v3"],
        "gpu_allocated_mb": 4096.0,
        "gpu_total_mb": 12288.0,
    }
    return manager


@pytest.fixture
def stub_pipeline() -> MagicMock:
    """Default stub pipeline (AI-generated verdict, English)."""
    return make_stub_pipeline()


@pytest.fixture
def tmp_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated per-test log directory (also exported via env overrides)."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for var in ("AIGC_LOG_DIR", "LOG_DIR"):
        monkeypatch.setenv(var, str(log_dir))
    return log_dir


@pytest.fixture
def client(stub_pipeline: MagicMock, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient over create_app() with stub pipeline injected, auth OFF.

    The production lifespan (real model loading) is replaced via
    ``app.router.lifespan_context`` — the documented FastAPI testing
    pattern — so everything else the factory wires (middleware, routes,
    static mount, and any concurrently-added auth/cors/metrics layers)
    is exercised as-is.
    """
    for var in AUTH_ENV_CANDIDATES:
        monkeypatch.delenv(var, raising=False)

    from aigc_detector.api.main import create_app
    from aigc_detector.api.middleware import limiter

    app = create_app()
    stub_manager = make_stub_model_manager()

    @asynccontextmanager
    async def stub_lifespan(app: FastAPI):
        app.state.start_time = time.time()
        app.state.pipeline = stub_pipeline
        app.state.model_manager = stub_manager
        yield

    app.router.lifespan_context = stub_lifespan

    limiter.reset()  # clean per-endpoint quota state before the test
    with TestClient(app) as c:
        yield c
    limiter.reset()  # never leak consumed quota into other test modules
