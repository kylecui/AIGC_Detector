"""v0.2 hardening tests: readiness, CORS, auth, PDF guards, request-id, metrics, testing mode.

All GPU/model paths are stubbed or mocked. The shared slowapi limiter is
reset around every test so rate-limit state never leaks into other test
files (they share the module-level limiter instance).
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aigc_detector.api.metrics import LATENCY_BUCKETS, metrics_registry
from aigc_detector.api.middleware import limiter
from aigc_detector.config import settings


@pytest.fixture(autouse=True)
def _isolate_rate_limiter():
    limiter.reset()
    yield
    limiter.reset()


def _stub_result():
    from aigc_detector.detection.ensemble import EnsembleResult

    return EnsembleResult(
        predicted_label="Human-written",
        confidence=0.5,
        p_ai=0.5,
        detected_language="zh",
        stages_used=["stub"],
        breakdown={},
        processing_time_ms=1.0,
    )


def _make_app(pipeline, model_manager=None):
    """Build an app mirroring production wiring (middleware + routers)."""
    import time

    from aigc_detector.api.middleware import setup_middleware
    from aigc_detector.api.routes import metrics_router, router

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        from aigc_detector.api.middleware import log_auth_disabled_once

        app.state.start_time = time.time()
        app.state.pipeline = pipeline
        app.state.model_manager = model_manager
        log_auth_disabled_once()  # mirror production lifespan wiring
        yield

    app = FastAPI(lifespan=test_lifespan)
    setup_middleware(app)
    app.include_router(router)
    app.include_router(metrics_router)
    return app


def _mock_pipeline():
    mock = MagicMock()
    mock.detect.side_effect = lambda text: _stub_result()
    return mock


@pytest.fixture
def ready_client():
    app = _make_app(_mock_pipeline())
    with TestClient(app) as c:
        yield c


# ======================================================================
# 1. Readiness (/health + /ready)
# ======================================================================


class TestReadiness:
    def test_health_ready_200_with_pipeline_ready_field(self, ready_client):
        resp = ready_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_ready"] is True
        assert data["status"] == "ok"
        # backward-compatible fields still present
        for field in ("models_loaded", "gpu_memory_used_mb", "gpu_memory_total_mb", "uptime_seconds"):
            assert field in data

    def test_health_503_when_pipeline_missing(self):
        with TestClient(_make_app(None)) as c:
            resp = c.get("/api/v1/health")
            assert resp.status_code == 503
            assert resp.json()["pipeline_ready"] is False
            assert resp.json()["status"] == "not_ready"

    def test_health_503_when_language_router_not_loaded(self):
        from aigc_detector.detection.language import LanguageRouter

        pipeline = MagicMock()
        pipeline.is_stub = False
        pipeline.language_router = LanguageRouter(device="cpu")  # never .load()-ed
        with TestClient(_make_app(pipeline)) as c:
            assert c.get("/api/v1/health").status_code == 503
            assert c.get("/api/v1/ready").status_code == 503

    def test_health_200_when_language_router_loaded(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter(device="cpu")
        router._model = object()  # simulate loaded weights
        pipeline = MagicMock()
        pipeline.is_stub = False
        pipeline.language_router = router
        with TestClient(_make_app(pipeline)) as c:
            assert c.get("/api/v1/health").status_code == 200
            assert c.get("/api/v1/ready").status_code == 204

    def test_ready_204_and_503(self):
        with TestClient(_make_app(_mock_pipeline())) as c:
            assert c.get("/api/v1/ready").status_code == 204
        with TestClient(_make_app(None)) as c:
            resp = c.get("/api/v1/ready")
            assert resp.status_code == 503
            assert resp.text == ""  # error probes stay body-free


# ======================================================================
# 2. CORS
# ======================================================================


class TestCors:
    def test_disabled_by_default_no_cors_headers(self, ready_client):
        resp = ready_client.get("/api/v1/health", headers={"Origin": "http://evil.example"})
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    def test_enabled_origins_parsed_and_allowed(self, monkeypatch):
        monkeypatch.setattr(settings, "cors_origins", "https://a.example, https://b.example")
        app = _make_app(_mock_pipeline())
        with TestClient(app) as c:
            preflight = c.options(
                "/api/v1/detect",
                headers={
                    "Origin": "https://a.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == "https://a.example"

            # disallowed origin gets no ACAO grant
            other = c.get("/api/v1/health", headers={"Origin": "https://c.example"})
            assert "access-control-allow-origin" not in other.headers

    def test_blank_setting_disables_middleware(self, monkeypatch):
        monkeypatch.setattr(settings, "cors_origins", "  ")
        app = _make_app(_mock_pipeline())
        with TestClient(app) as c:
            resp = c.get("/api/v1/health", headers={"Origin": "https://a.example"})
            assert "access-control-allow-origin" not in resp.headers


# ======================================================================
# 3. API-key auth
# ======================================================================


class TestApiKeyAuth:
    @pytest.fixture
    def auth_client(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "sekret-key-123")
        app = _make_app(_mock_pipeline())
        with TestClient(app) as c:
            yield c

    def test_no_header_401(self, auth_client):
        resp = auth_client.post("/api/v1/detect", json={"text": "a" * 100})
        assert resp.status_code == 401
        assert resp.json() == {"detail": "invalid api key"}

    def test_wrong_key_401(self, auth_client):
        resp = auth_client.post("/api/v1/detect", json={"text": "a" * 100}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "invalid api key"

    def test_correct_key_200(self, auth_client):
        resp = auth_client.post("/api/v1/detect", json={"text": "a" * 100}, headers={"X-API-Key": "sekret-key-123"})
        assert resp.status_code == 200
        assert resp.json()["predicted_label"] == "Human-written"

    def test_detect_file_protected(self, auth_client):
        resp = auth_client.post(
            "/api/v1/detect/file",
            files={"file": ("a.txt", b"word " * 100, "text/plain")},
        )
        assert resp.status_code == 401

    def test_exempt_endpoints_open(self, auth_client):
        assert auth_client.get("/api/v1/health").status_code == 200
        assert auth_client.get("/api/v1/ready").status_code == 204
        assert auth_client.get("/metrics").status_code == 200
        assert auth_client.get("/docs").status_code == 200
        assert auth_client.get("/redoc").status_code == 200
        assert auth_client.get("/openapi.json").status_code == 200

    def test_auth_unset_means_no_auth(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "")
        with TestClient(_make_app(_mock_pipeline())) as c:
            resp = c.post("/api/v1/detect", json={"text": "a" * 100})
            assert resp.status_code == 200

    def test_auth_disabled_logged_once_at_startup(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "api_key", "")
        monkeypatch.setattr("aigc_detector.api.middleware._auth_disabled_notice_logged", False)
        with caplog.at_level(logging.WARNING, logger="aigc_detector.api.middleware"):
            with TestClient(_make_app(_mock_pipeline())):
                pass
        assert "AUTH DISABLED (api_key not set)" in caplog.text

    def test_no_auth_disabled_line_when_key_set(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "api_key", "k")
        monkeypatch.setattr("aigc_detector.api.middleware._auth_disabled_notice_logged", False)
        with caplog.at_level(logging.WARNING, logger="aigc_detector.api.middleware"):
            with TestClient(_make_app(_mock_pipeline())):
                pass
        assert "AUTH DISABLED" not in caplog.text


# ======================================================================
# 4. Upload hardening (PDF page cap + decompression-bomb guard)
# ======================================================================


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self, option: str = "text"):
        return self._text

    def get_images(self):
        return []


class _FakeDoc:
    def __init__(self, pages, page_count=None):
        self._pages = pages
        self.page_count = page_count if page_count is not None else len(pages)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._pages)


def _tiny_pdf_bytes(pages: int = 2) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        y = 72
        for _line in range(5):
            page.insert_text((72, y), f"paragraph of sample text word token {pages} ")
            y += 20
    content = doc.tobytes()
    doc.close()
    return content


class TestUploadHardening:
    def test_small_pdf_accepted(self, ready_client):
        resp = ready_client.post(
            "/api/v1/detect/file",
            files={"file": ("small.pdf", _tiny_pdf_bytes(2), "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["predicted_label"] == "Human-written"

    def test_pdf_over_150_pages_rejected(self, ready_client, monkeypatch):
        fake = _FakeDoc(pages=[], page_count=200)
        monkeypatch.setattr(fitz, "open", lambda **kwargs: fake)
        resp = ready_client.post(
            "/api/v1/detect/file",
            files={"file": ("huge.pdf", b"%PDF-1.4 minimal", "application/pdf")},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"] == "PDF too large: 200 pages"

    def test_decompression_bomb_page_rejected(self, ready_client, monkeypatch):
        bomb_page = _FakePage("x" * 500_001)
        fake = _FakeDoc(pages=[bomb_page, _FakePage("ok text")], page_count=2)
        monkeypatch.setattr(fitz, "open", lambda **kwargs: fake)
        resp = ready_client.post(
            "/api/v1/detect/file",
            files={"file": ("bomb.pdf", b"%PDF-1.4 minimal", "application/pdf")},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"] == "suspicious PDF structure"

    def test_page_at_exact_limit_passes_guard(self):
        """Exactly MAX_PDF_PAGE_CHARS must NOT trip the bomb guard (off-by-one check).

        A 422 would raise if the guard fired; instead the text flows through
        and is truncated by the ordinary 50k extraction cap.
        """
        import aigc_detector.api.routes as routes_mod
        from aigc_detector.api.routes import MAX_EXTRACTED_TEXT_CHARS, MAX_PDF_PAGE_CHARS

        assert MAX_PDF_PAGE_CHARS == 500_000
        page = _FakePage("x" * MAX_PDF_PAGE_CHARS)
        original_open = fitz.open
        fitz.open = lambda **kwargs: _FakeDoc(pages=[page])  # type: ignore[assignment]
        try:
            text = routes_mod._extract_text_from_pdf(b"ignored")
            assert len(text) == MAX_EXTRACTED_TEXT_CHARS
        finally:
            fitz.open = original_open


# ======================================================================
# 5. Request-ID middleware
# ======================================================================


class TestRequestId:
    def test_generated_when_missing(self, ready_client):
        resp = ready_client.get("/api/v1/health")
        rid = resp.headers.get("x-request-id")
        assert rid is not None
        assert len(rid) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", rid)

    def test_incoming_echoed(self, ready_client):
        resp = ready_client.get("/api/v1/health", headers={"X-Request-ID": "abc-123_def"})
        assert resp.headers["x-request-id"] == "abc-123_def"

    def test_incoming_sanitized(self, ready_client):
        resp = ready_client.get("/api/v1/health", headers={"X-Request-ID": "bad id!!;"})
        rid = resp.headers["x-request-id"]
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid)

    def test_error_responses_also_get_id(self):
        with TestClient(_make_app(None)) as c:
            resp = c.post("/api/v1/detect", json={"text": "a" * 100})
            assert resp.status_code == 503
            assert re.fullmatch(r"[0-9a-f]{12}", resp.headers["x-request-id"])

    def test_formatter_appends_rid(self):
        from aigc_detector.api.middleware import set_request_id
        from aigc_detector.utils.log_hygiene import SanitizingFormatter

        fmt = SanitizingFormatter("%(message)s")
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        token = set_request_id("deadbeef42")
        try:
            assert fmt.format(record).endswith("hello [rid=deadbeef42]")
        finally:
            from aigc_detector.api.middleware import _request_id_var

            _request_id_var.reset(token)

    def test_formatter_without_rid_unchanged(self):
        from aigc_detector.utils.log_hygiene import SanitizingFormatter

        fmt = SanitizingFormatter("%(message)s")
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="plain message",
            args=None,
            exc_info=None,
        )
        assert fmt.format(record) == "plain message"


# ======================================================================
# 6. Metrics
# ======================================================================


class TestMetrics:
    def test_exposition_format(self, ready_client):
        resp = ready_client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "# TYPE aigc_requests_total counter" in body
        assert "# TYPE aigc_detect_seconds histogram" in body
        assert "# TYPE aigc_requests_in_flight gauge" in body
        assert "aigc_requests_in_flight 0" in body
        for bound in LATENCY_BUCKETS:
            assert f'le="{bound:g}"' in body
        assert 'le="+Inf"' in body

    def test_detect_increments_counters(self):
        before_total = metrics_registry.request_count("detect", 200)
        app = _make_app(_mock_pipeline())
        with TestClient(app) as c:
            assert c.post("/api/v1/detect", json={"text": "a" * 100}).status_code == 200
        assert metrics_registry.request_count("detect", 200) == before_total + 1
        body = metrics_registry.render()
        assert 'aigc_detect_seconds_count{endpoint="detect"}' in body
        assert 'aigc_detect_seconds_bucket{endpoint="detect",le="+Inf"}' in body
        assert metrics_registry.in_flight() == 0

    def test_detect_file_increments_counters(self):
        before = metrics_registry.request_count("detect_file", 200)
        app = _make_app(_mock_pipeline())
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/detect/file",
                files={"file": ("a.txt", b"word " * 100, "text/plain")},
            )
            assert resp.status_code == 200
        assert metrics_registry.request_count("detect_file", 200) == before + 1

    def test_error_status_counted(self):
        before = metrics_registry.request_count("detect", 503)
        with TestClient(_make_app(None)) as c:
            assert c.post("/api/v1/detect", json={"text": "a" * 100}).status_code == 503
        assert metrics_registry.request_count("detect", 503) == before + 1

    def test_rate_limited_requests_counted_429(self):
        app = _make_app(_mock_pipeline())
        with TestClient(app) as c:
            statuses = [c.post("/api/v1/detect", json={"text": "a" * 100}).status_code for _ in range(11)]
        assert statuses[:10] == [200] * 10
        assert statuses[10] == 429
        assert metrics_registry.request_count("detect", 429) >= 1
        assert metrics_registry.in_flight() == 0  # 429 never entered the handler

    def test_histogram_buckets_cumulative(self):
        from aigc_detector.api.metrics import MetricsRegistry

        reg = MetricsRegistry()
        reg.observe_seconds("detect", 0.3)  # falls in 0.5 and every larger bucket
        reg.observe_seconds("detect", 7.0)  # falls in 10..120 and +Inf
        body = reg.render()
        assert 'aigc_detect_seconds_bucket{endpoint="detect",le="0.5"} 1' in body
        assert 'aigc_detect_seconds_bucket{endpoint="detect",le="1"} 1' in body
        assert 'aigc_detect_seconds_bucket{endpoint="detect",le="10"} 2' in body
        assert 'aigc_detect_seconds_bucket{endpoint="detect",le="+Inf"} 2' in body
        assert 'aigc_detect_seconds_count{endpoint="detect"} 2' in body


# ======================================================================
# 7. AIGC_TESTING=1 mode (real app, stub pipeline)
# ======================================================================


class TestTestingMode:
    def test_real_app_boots_with_stub(self, monkeypatch):
        monkeypatch.setenv("AIGC_TESTING", "1")
        from aigc_detector.api.main import StubPipeline, create_app, testing_mode

        assert testing_mode() is True
        app = create_app()
        with TestClient(app) as c:
            # health reports ready for the stub
            resp = c.get("/api/v1/health")
            assert resp.status_code == 200
            assert resp.json()["pipeline_ready"] is True
            assert c.get("/api/v1/ready").status_code == 204

            # detect returns the fixed stub verdict, no models loaded
            det = c.post("/api/v1/detect", json={"text": "a" * 100})
            assert det.status_code == 200
            body = det.json()
            assert body["predicted_label"] == "Human-written"
            assert body["p_ai"] == 0.5
            assert body["stages_used"] == ["stub"]

            # metrics surface the stub-driven request
            assert 'aigc_requests_total{endpoint="detect",status="200"}' in c.get("/metrics").text

        assert isinstance(app.state.pipeline, StubPipeline)
        assert app.state.pipeline.is_stub is True

    def test_testing_mode_off_by_default(self, monkeypatch):
        monkeypatch.delenv("AIGC_TESTING", raising=False)
        from aigc_detector.api.main import testing_mode

        assert testing_mode() is False

    def test_stub_pipeline_shape(self):
        from aigc_detector.api.main import StubPipeline

        result = StubPipeline().detect("anything " * 10)
        assert result.predicted_label == "Human-written"
        assert result.confidence == 0.5
        assert result.p_ai == 0.5
        assert result.detected_language == "zh"
        assert result.stages_used == ["stub"]
        assert result.breakdown == {}
        assert result.processing_time_ms == 1.0
