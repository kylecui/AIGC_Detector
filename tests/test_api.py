"""Tests for Phase 4 API: routes, schemas, middleware, pipeline, ensemble, manager.

Uses httpx AsyncClient with FastAPI TestClient. All GPU operations are mocked.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ======================================================================
# Schema Tests
# ======================================================================


class TestDetectionRequest:
    def test_valid_request(self):
        from aigc_detector.api.schemas import DetectionRequest

        req = DetectionRequest(text="a" * 100)
        assert len(req.text) == 100
        assert req.models == ["all"]

    def test_too_short_text(self):
        from aigc_detector.api.schemas import DetectionRequest

        with pytest.raises(Exception):
            DetectionRequest(text="short")

    def test_too_long_text(self):
        from aigc_detector.api.schemas import DetectionRequest

        with pytest.raises(Exception):
            DetectionRequest(text="a" * 10_001)

    def test_custom_models(self):
        from aigc_detector.api.schemas import DetectionRequest

        req = DetectionRequest(text="a" * 100, models=["statistical", "encoder"])
        assert req.models == ["statistical", "encoder"]


class TestDetectionResponse:
    def test_valid_response(self):
        from aigc_detector.api.schemas import DetectionResponse

        resp = DetectionResponse(
            predicted_label="AI-generated",
            confidence=0.87,
            p_ai=0.87,
            detected_language="en",
            stages_used=["statistical", "encoder"],
            breakdown={"statistical": {"p_ai": 0.82}},
            processing_time_ms=450.0,
        )
        assert resp.predicted_label == "AI-generated"
        assert resp.confidence == 0.87
        assert resp.p_ai == 0.87


class TestHealthResponse:
    def test_default_values(self):
        from aigc_detector.api.schemas import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.models_loaded == []
        assert resp.gpu_memory_used_mb == 0.0


class TestErrorResponse:
    def test_error_detail(self):
        from aigc_detector.api.schemas import ErrorResponse

        err = ErrorResponse(detail="Something went wrong")
        assert err.detail == "Something went wrong"


# ======================================================================
# Ensemble Tests
# ======================================================================


class TestEnsembleAggregator:
    def test_single_stage(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator()
        result = agg.combine(
            {"statistical": {"p_ai": 0.8, "label": "ai", "confidence": 0.8}},
            detected_language="en",
        )
        assert result.predicted_label == "AI-generated"
        assert result.p_ai == 0.8
        assert result.stages_used == ["statistical"]

    def test_two_stages_agreement(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator()
        result = agg.combine(
            {
                "statistical": {"p_ai": 0.8},
                "encoder": {"p_ai": 0.9},
            },
            detected_language="en",
        )
        assert result.predicted_label == "AI-generated"
        # Weighted: (0.2*0.8 + 0.5*0.9) / (0.2+0.5) = (0.16+0.45)/0.7 ≈ 0.871
        assert 0.85 < result.p_ai < 0.90

    def test_all_three_stages(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator()
        result = agg.combine(
            {
                "statistical": {"p_ai": 0.7},
                "encoder": {"p_ai": 0.85},
                "binoculars": {"p_ai": 0.9},
            },
            detected_language="zh",
        )
        assert result.predicted_label == "AI-generated"
        assert result.detected_language == "zh"
        assert len(result.stages_used) == 3

    def test_human_prediction(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator()
        result = agg.combine(
            {
                "statistical": {"p_ai": 0.2},
                "encoder": {"p_ai": 0.3},
            },
        )
        assert result.predicted_label == "Human-written"
        assert result.p_ai < 0.5

    def test_empty_stages(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator()
        result = agg.combine({})
        assert result.predicted_label == "Human-written"
        assert result.p_ai == 0.0
        assert result.stages_used == []

    def test_agree_all_ai(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        assert EnsembleAggregator.agree(
            {
                "statistical": {"p_ai": 0.8},
                "encoder": {"p_ai": 0.7},
            }
        )

    def test_agree_all_human(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        assert EnsembleAggregator.agree(
            {
                "statistical": {"p_ai": 0.3},
                "encoder": {"p_ai": 0.2},
            }
        )

    def test_disagree(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        assert not EnsembleAggregator.agree(
            {
                "statistical": {"p_ai": 0.8},
                "encoder": {"p_ai": 0.3},
            }
        )

    def test_single_stage_agree(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        assert EnsembleAggregator.agree({"statistical": {"p_ai": 0.9}})

    def test_to_dict(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator()
        result = agg.combine({"encoder": {"p_ai": 0.6}})
        d = result.to_dict()
        assert "predicted_label" in d
        assert "confidence" in d
        assert "p_ai" in d

    def test_custom_weights(self):
        from aigc_detector.detection.ensemble import EnsembleAggregator

        agg = EnsembleAggregator(weights={"statistical": 0.5, "encoder": 0.5})
        result = agg.combine(
            {
                "statistical": {"p_ai": 0.4},
                "encoder": {"p_ai": 0.8},
            }
        )
        # Equal weights: (0.5*0.4 + 0.5*0.8) / 1.0 = 0.6
        assert abs(result.p_ai - 0.6) < 0.01


# ======================================================================
# ModelManager Tests
# ======================================================================


class TestModelManager:
    def test_init(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        assert mgr.max_vram_gb == 11.0
        assert mgr.loaded_model_names == []
        assert mgr.used_vram_gb == 0.0

    def test_load_and_get(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        mock_instance = MagicMock()

        # Patch registry to avoid file read
        mgr._registry = {
            "test-model": MagicMock(vram_gb=2.0),
        }

        mgr.load("test-model", mock_instance)
        assert mgr.is_loaded("test-model")
        assert mgr.get("test-model") is mock_instance
        assert mgr.used_vram_gb == 2.0

    def test_unload(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        mock_instance = MagicMock()
        mgr._registry = {"test-model": MagicMock(vram_gb=3.0)}

        mgr.load("test-model", mock_instance)
        assert mgr.is_loaded("test-model")

        mgr.unload("test-model")
        assert not mgr.is_loaded("test-model")
        assert mgr.used_vram_gb == 0.0
        mock_instance.unload.assert_called_once()

    def test_lru_eviction(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=6.0)
        mock1 = MagicMock()
        mock2 = MagicMock()
        mock3 = MagicMock()

        mgr._registry = {
            "model-a": MagicMock(vram_gb=3.0),
            "model-b": MagicMock(vram_gb=3.0),
            "model-c": MagicMock(vram_gb=3.0),
        }

        mgr.load("model-a", mock1)
        mgr.load("model-b", mock2)
        # Budget full at 6.0. Loading model-c should evict model-a (LRU)
        mgr.load("model-c", mock3)

        assert not mgr.is_loaded("model-a")
        assert mgr.is_loaded("model-b")
        assert mgr.is_loaded("model-c")
        mock1.unload.assert_called_once()

    def test_load_idempotent(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        mock_instance = MagicMock()
        mgr._registry = {"m": MagicMock(vram_gb=2.0)}

        mgr.load("m", mock_instance)
        mgr.load("m", mock_instance)  # Should not add again
        assert mgr.used_vram_gb == 2.0

    def test_unload_all(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        mgr._registry = {
            "a": MagicMock(vram_gb=2.0),
            "b": MagicMock(vram_gb=3.0),
        }
        mgr.load("a", MagicMock())
        mgr.load("b", MagicMock())

        mgr.unload_all()
        assert mgr.loaded_model_names == []
        assert mgr.used_vram_gb == 0.0

    def test_unload_nonexistent(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        mgr._registry = {}
        mgr.unload("nonexistent")  # should not raise

    def test_status(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=11.0)
        mgr._registry = {"m": MagicMock(vram_gb=2.0)}
        mgr.load("m", MagicMock())

        status = mgr.status()
        assert "loaded_models" in status
        assert "m" in status["loaded_models"]
        assert status["used_vram_gb"] == 2.0
        assert status["max_vram_gb"] == 11.0

    def test_available_vram(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=10.0)
        mgr._registry = {"m": MagicMock(vram_gb=4.0)}
        mgr.load("m", MagicMock())
        assert mgr.available_vram_gb == 6.0

    def test_can_load(self):
        from aigc_detector.models.manager import ModelManager

        mgr = ModelManager(max_vram_gb=5.0)
        mgr._registry = {
            "small": MagicMock(vram_gb=2.0),
            "big": MagicMock(vram_gb=6.0),
        }
        assert mgr.can_load("small")
        assert not mgr.can_load("big")
        assert not mgr.can_load("nonexistent")


# ======================================================================
# Pipeline Tests
# ======================================================================


class TestDetectionPipeline:
    def _make_pipeline(self, stat_p_ai=0.8, enc_p_ai=0.9, bino_score=0.5):
        from aigc_detector.detection.pipeline import DetectionPipeline

        mock_router = MagicMock()
        mock_router.detect.return_value = MagicMock(lang="en", confidence=0.95)

        mock_extractor = MagicMock()
        mock_features = MagicMock()
        mock_features.to_dict.return_value = {"perplexity": 25.0}
        mock_extractor.extract.return_value = mock_features

        mock_stat_clf = MagicMock()
        mock_stat_clf.predict.return_value = {
            "label": "ai" if stat_p_ai > 0.5 else "human",
            "p_ai": stat_p_ai,
            "confidence": max(stat_p_ai, 1.0 - stat_p_ai),
        }

        mock_enc_clf = MagicMock()
        mock_enc_clf.predict.return_value = MagicMock(
            label="ai" if enc_p_ai > 0.5 else "human",
            p_ai=enc_p_ai,
            confidence=max(enc_p_ai, 1.0 - enc_p_ai),
            model_name="test-encoder",
        )

        mock_bino = MagicMock()
        mock_bino.predict.return_value = MagicMock(
            label="ai" if bino_score < 0.9 else "human",
            score=bino_score,
            threshold=0.9,
            mode="accuracy",
        )

        pipeline = DetectionPipeline(
            language_router=mock_router,
            statistical_extractors={"en": mock_extractor},
            statistical_classifiers={"en": mock_stat_clf},
            encoder_classifiers={"en": mock_enc_clf},
            binoculars_detectors={"en": mock_bino},
        )
        return pipeline, mock_bino

    def test_early_exit_high_confidence(self):
        pipeline, mock_bino = self._make_pipeline(stat_p_ai=0.98)
        result = pipeline.detect("Test text " * 10)
        assert "statistical" in result.stages_used
        # Should early-exit, so encoder not used
        assert "encoder" not in result.stages_used
        assert "binoculars" not in result.stages_used

    def test_agreement_skips_binoculars(self):
        pipeline, mock_bino = self._make_pipeline(stat_p_ai=0.8, enc_p_ai=0.85)
        result = pipeline.detect("Test text " * 10)
        assert "statistical" in result.stages_used
        assert "encoder" in result.stages_used
        # Both agree on AI → skip binoculars
        assert "binoculars" not in result.stages_used

    def test_conflict_invokes_binoculars(self):
        pipeline, mock_bino = self._make_pipeline(stat_p_ai=0.7, enc_p_ai=0.3)
        result = pipeline.detect("Test text " * 10)
        assert "statistical" in result.stages_used
        assert "encoder" in result.stages_used
        assert "binoculars" in result.stages_used

    def test_no_detectors_available(self):
        from aigc_detector.detection.pipeline import DetectionPipeline

        mock_router = MagicMock()
        mock_router.detect.return_value = MagicMock(lang="fr", confidence=0.9)

        pipeline = DetectionPipeline(language_router=mock_router)
        result = pipeline.detect("Test text " * 10)
        assert result.predicted_label == "Human-written"
        assert result.stages_used == []

    def test_processing_time_recorded(self):
        pipeline, _ = self._make_pipeline()
        result = pipeline.detect("Test text " * 10)
        assert result.processing_time_ms > 0

    def test_detected_language_passed(self):
        from aigc_detector.detection.pipeline import DetectionPipeline

        mock_router = MagicMock()
        mock_router.detect.return_value = MagicMock(lang="zh", confidence=0.99)

        pipeline = DetectionPipeline(language_router=mock_router)
        result = pipeline.detect("测试文本" * 20)
        assert result.detected_language == "zh"

    def test_binoculars_score_to_p_ai(self):
        from aigc_detector.detection.pipeline import DetectionPipeline

        # Score < threshold → AI (high p_ai)
        p_ai = DetectionPipeline._binoculars_score_to_p_ai(0.5, 0.9)
        assert p_ai > 0.5

        # Score > threshold → human (low p_ai)
        p_ai = DetectionPipeline._binoculars_score_to_p_ai(1.2, 0.9)
        assert p_ai < 0.5

        # Score == threshold → ~0.5
        p_ai = DetectionPipeline._binoculars_score_to_p_ai(0.9, 0.9)
        assert abs(p_ai - 0.5) < 0.01


# ======================================================================
# API Route Tests (using TestClient)
# ======================================================================


class TestAPIRoutes:
    @pytest.fixture
    def client(self):
        """Create a test client with mocked pipeline."""
        from aigc_detector.detection.ensemble import EnsembleResult

        mock_result = EnsembleResult(
            predicted_label="AI-generated",
            confidence=0.87,
            p_ai=0.87,
            detected_language="en",
            stages_used=["statistical", "encoder"],
            breakdown={"statistical": {"p_ai": 0.82}, "encoder": {"p_ai": 0.89}},
            processing_time_ms=450.0,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.detect.return_value = mock_result

        mock_manager = MagicMock()
        mock_manager.status.return_value = {
            "loaded_models": ["xlm-roberta", "deberta-v3"],
            "gpu_allocated_mb": 4096.0,
            "gpu_total_mb": 12288.0,
        }

        # Patch lifespan to avoid loading real models
        from contextlib import asynccontextmanager

        from fastapi import FastAPI

        from aigc_detector.api.middleware import setup_middleware
        from aigc_detector.api.routes import router

        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            app.state.start_time = time.time()
            app.state.pipeline = mock_pipeline
            app.state.model_manager = mock_manager
            yield

        app = FastAPI(lifespan=test_lifespan)
        setup_middleware(app)
        app.include_router(router)

        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "gpu_memory_used_mb" in data

    def test_detect_endpoint(self, client):
        resp = client.post(
            "/api/v1/detect",
            json={"text": "a" * 100},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_label"] == "AI-generated"
        assert data["confidence"] == 0.87
        assert data["p_ai"] == 0.87
        assert data["detected_language"] == "en"
        assert "statistical" in data["stages_used"]
        assert data["processing_time_ms"] == 450.0

    def test_detect_text_too_short(self, client):
        resp = client.post(
            "/api/v1/detect",
            json={"text": "short"},
        )
        assert resp.status_code == 422  # Validation error

    def test_detect_text_too_long(self, client):
        resp = client.post(
            "/api/v1/detect",
            json={"text": "a" * 10_001},
        )
        assert resp.status_code == 422

    def test_detect_missing_text(self, client):
        resp = client.post(
            "/api/v1/detect",
            json={},
        )
        assert resp.status_code == 422

    def test_health_no_manager(self):
        """Health endpoint works even without model manager."""
        from contextlib import asynccontextmanager

        from fastapi import FastAPI

        from aigc_detector.api.middleware import setup_middleware
        from aigc_detector.api.routes import router

        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            app.state.start_time = time.time()
            app.state.pipeline = None
            app.state.model_manager = None
            yield

        app = FastAPI(lifespan=test_lifespan)
        setup_middleware(app)
        app.include_router(router)

        with TestClient(app) as c:
            resp = c.get("/api/v1/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_detect_no_pipeline(self):
        """Detect endpoint returns 503 when pipeline is not initialized."""
        from contextlib import asynccontextmanager

        from fastapi import FastAPI

        from aigc_detector.api.middleware import setup_middleware
        from aigc_detector.api.routes import router

        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            app.state.start_time = time.time()
            app.state.pipeline = None
            app.state.model_manager = None
            yield

        app = FastAPI(lifespan=test_lifespan)
        setup_middleware(app)
        app.include_router(router)

        with TestClient(app) as c:
            resp = c.post("/api/v1/detect", json={"text": "a" * 100})
            assert resp.status_code == 503
