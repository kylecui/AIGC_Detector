"""Tests for Phase 2 detection modules: language, statistical, binoculars, evaluator, calibration.

GPU-dependent tests are mocked.  All tests here run on CPU without models downloaded.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ======================================================================
# Language Router Tests
# ======================================================================


class TestLanguageRouterHeuristic:
    """Test the heuristic fallback (no model loaded)."""

    def test_english_text(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        result = router.detect("This is a simple English sentence for testing.")
        assert result.lang == "en"
        assert result.method == "heuristic"
        assert result.confidence > 0.5

    def test_chinese_text(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        result = router.detect("这是一个简单的中文句子，用于测试语言检测功能。")
        assert result.lang == "zh"
        assert result.method == "heuristic"
        assert result.confidence > 0.5

    def test_empty_text(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        result = router.detect("")
        assert result.lang == "en"  # default
        assert result.method == "heuristic"

    def test_mixed_text_mostly_chinese(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        result = router.detect("这是中文文本，包含一些 English words 在其中。但主要是中文内容。")
        assert result.lang == "zh"

    def test_mixed_text_mostly_english(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        result = router.detect("This is English with 一点 Chinese characters mixed in.")
        assert result.lang == "en"

    def test_is_loaded_false_by_default(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        assert not router.is_loaded


class TestLanguageRouterModel:
    """Test model-based detection with mocked transformer."""

    def test_detect_with_model_english(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        # Mock the model
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "en", 1: "zh"}
        logits = torch.tensor([[2.0, 0.1]])
        mock_model.return_value = MagicMock(logits=logits)
        mock_model.device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        router._model = mock_model
        router._tokenizer = mock_tokenizer
        router.device = "cpu"

        result = router.detect("Hello world")
        assert result.lang == "en"
        assert result.method == "model"
        assert result.confidence > 0.5

    def test_detect_with_model_chinese(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "en", 1: "zh"}
        logits = torch.tensor([[0.1, 2.0]])
        mock_model.return_value = MagicMock(logits=logits)
        mock_model.device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        router._model = mock_model
        router._tokenizer = mock_tokenizer
        router.device = "cpu"

        result = router.detect("你好世界")
        assert result.lang == "zh"
        assert result.method == "model"

    def test_model_failure_falls_back_to_heuristic(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter()
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("CUDA OOM")
        mock_model.config.id2label = {0: "en"}

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

        router._model = mock_model
        router._tokenizer = mock_tokenizer
        router.device = "cpu"

        result = router.detect("English text here")
        assert result.method == "heuristic"
        assert result.lang == "en"

    def test_map_label_unsupported(self):
        from aigc_detector.detection.language import LanguageRouter

        assert LanguageRouter._map_label("fr") == "en"
        assert LanguageRouter._map_label("de") == "en"
        assert LanguageRouter._map_label("zh") == "zh"
        assert LanguageRouter._map_label("en") == "en"
        assert LanguageRouter._map_label("Chinese") == "zh"
        assert LanguageRouter._map_label("English") == "en"

    def test_unload(self):
        from aigc_detector.detection.language import LanguageRouter

        router = LanguageRouter(device="cpu")
        router._model = MagicMock()
        router._tokenizer = MagicMock()
        assert router.is_loaded
        router.unload()
        assert not router.is_loaded


# ======================================================================
# Statistical Features Tests
# ======================================================================


class TestStatisticalFeatures:
    def test_to_array(self):
        from aigc_detector.detection.statistical import StatisticalFeatures

        f = StatisticalFeatures(
            perplexity=25.0,
            avg_entropy=3.5,
            std_entropy=1.2,
            burstiness=-0.4,
            max_entropy=6.0,
            min_entropy=0.5,
        )
        arr = f.to_array()
        assert arr.shape == (6,)
        assert arr[0] == 25.0
        assert arr[3] == -0.4

    def test_to_dict(self):
        from aigc_detector.detection.statistical import StatisticalFeatures

        f = StatisticalFeatures(
            perplexity=10.0,
            avg_entropy=2.0,
            std_entropy=0.5,
            burstiness=-0.2,
            max_entropy=4.0,
            min_entropy=0.1,
        )
        d = f.to_dict()
        assert d["perplexity"] == 10.0
        assert len(d) == 6


class TestStatisticalFeatureExtractor:
    def test_not_loaded_raises(self):
        from aigc_detector.detection.statistical import StatisticalFeatureExtractor

        extractor = StatisticalFeatureExtractor("dummy-model", device="cpu")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            extractor.extract("some text")

    def test_is_loaded(self):
        from aigc_detector.detection.statistical import StatisticalFeatureExtractor

        extractor = StatisticalFeatureExtractor("dummy-model")
        assert not extractor.is_loaded

    def test_extract_with_mock(self):
        from aigc_detector.detection.statistical import StatisticalFeatureExtractor

        extractor = StatisticalFeatureExtractor("dummy-model", device="cpu")

        # Mock model and tokenizer
        seq_len = 10
        vocab_size = 100
        mock_logits = torch.randn(1, seq_len, vocab_size)
        mock_loss = torch.tensor(2.5)

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.return_value = MagicMock(logits=mock_logits, loss=mock_loss)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, vocab_size, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"

        extractor._model = mock_model
        extractor._tokenizer = mock_tokenizer

        features = extractor.extract("Test text for feature extraction.")

        assert isinstance(features.perplexity, float)
        assert isinstance(features.avg_entropy, float)
        assert isinstance(features.std_entropy, float)
        assert isinstance(features.burstiness, float)
        assert isinstance(features.max_entropy, float)
        assert isinstance(features.min_entropy, float)
        assert features.perplexity > 0
        assert features.avg_entropy >= 0

    def test_burstiness_calculation(self):
        from aigc_detector.detection.statistical import StatisticalFeatureExtractor

        # Flat entropy → burstiness near -1
        flat = torch.ones(50) * 3.0
        b_flat = StatisticalFeatureExtractor._burstiness(flat)
        assert b_flat < 0  # std ≈ 0, mean = 3 → (0-3)/(0+3) ≈ -1

        # Variable entropy → burstiness closer to 0
        variable = torch.cat([torch.ones(25) * 1.0, torch.ones(25) * 5.0])
        b_var = StatisticalFeatureExtractor._burstiness(variable)
        assert b_var > b_flat  # more variable = higher burstiness

    def test_unload(self):
        from aigc_detector.detection.statistical import StatisticalFeatureExtractor

        extractor = StatisticalFeatureExtractor("dummy-model", device="cpu")
        extractor._model = MagicMock()
        extractor._tokenizer = MagicMock()
        assert extractor.is_loaded
        extractor.unload()
        assert not extractor.is_loaded


class TestStatisticalClassifier:
    def _make_training_data(self):
        """Generate synthetic training data for classifier tests."""
        rng = np.random.RandomState(42)
        n = 200
        # Human: higher perplexity, higher entropy std, higher burstiness
        human_features = rng.randn(n // 2, 6) + np.array([50.0, 4.0, 1.5, 0.1, 6.0, 1.0])
        # AI: lower perplexity, lower entropy std, lower burstiness
        ai_features = rng.randn(n // 2, 6) + np.array([15.0, 2.5, 0.5, -0.5, 4.0, 0.5])

        x_data = np.vstack([human_features, ai_features])
        y = np.array([0] * (n // 2) + [1] * (n // 2))  # 0=human, 1=AI
        return x_data, y

    def test_xgboost_fit_predict(self):
        from aigc_detector.detection.statistical import StatisticalClassifier

        clf = StatisticalClassifier(backend="xgboost")
        x_data, y = self._make_training_data()
        stats = clf.fit(x_data, y)
        assert stats["train_accuracy"] > 0.7
        assert stats["n_samples"] == 200

        result = clf.predict(x_data[0])
        assert "label" in result
        assert "p_ai" in result
        assert "confidence" in result
        assert result["label"] in ("human", "ai")
        assert 0.0 <= result["p_ai"] <= 1.0

    def test_logistic_regression_fit_predict(self):
        from aigc_detector.detection.statistical import StatisticalClassifier

        clf = StatisticalClassifier(backend="logistic_regression")
        x_data, y = self._make_training_data()
        stats = clf.fit(x_data, y)
        assert stats["train_accuracy"] > 0.7

        result = clf.predict(x_data[0])
        assert result["label"] in ("human", "ai")

    def test_predict_batch(self):
        from aigc_detector.detection.statistical import StatisticalClassifier

        clf = StatisticalClassifier(backend="xgboost")
        x_data, y = self._make_training_data()
        clf.fit(x_data, y)

        result = clf.predict(x_data[:5])
        assert "labels" in result
        assert len(result["labels"]) == 5

    def test_predict_proba(self):
        from aigc_detector.detection.statistical import StatisticalClassifier

        clf = StatisticalClassifier(backend="xgboost")
        x_data, y = self._make_training_data()
        clf.fit(x_data, y)

        proba = clf.predict_proba(x_data[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_save_load(self, tmp_path):
        from aigc_detector.detection.statistical import StatisticalClassifier

        clf = StatisticalClassifier(backend="xgboost")
        x_data, y = self._make_training_data()
        clf.fit(x_data, y)

        save_path = tmp_path / "classifier.pkl"
        clf.save(save_path)

        clf2 = StatisticalClassifier()
        clf2.load(save_path)
        assert clf2.backend == "xgboost"

        # Predictions should match
        r1 = clf.predict(x_data[0])
        r2 = clf2.predict(x_data[0])
        assert r1["label"] == r2["label"]
        assert abs(r1["p_ai"] - r2["p_ai"]) < 1e-6

    def test_predict_from_features_dataclass(self):
        from aigc_detector.detection.statistical import StatisticalClassifier, StatisticalFeatures

        clf = StatisticalClassifier(backend="xgboost")
        x_data, y = self._make_training_data()
        clf.fit(x_data, y)

        features = StatisticalFeatures(
            perplexity=50.0,
            avg_entropy=4.0,
            std_entropy=1.5,
            burstiness=0.1,
            max_entropy=6.0,
            min_entropy=1.0,
        )
        result = clf.predict(features)
        assert result["label"] in ("human", "ai")

    def test_ensure_array_list_of_features(self):
        from aigc_detector.detection.statistical import StatisticalClassifier, StatisticalFeatures

        features = [
            StatisticalFeatures(10.0, 2.0, 0.5, -0.2, 4.0, 0.1),
            StatisticalFeatures(20.0, 3.0, 1.0, 0.0, 5.0, 0.5),
        ]
        arr = StatisticalClassifier._ensure_array(features)
        assert arr.shape == (2, 6)


# ======================================================================
# Binoculars Tests
# ======================================================================


class TestBinocularsDetector:
    def test_thresholds(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        det = BinocularsDetector(mode="accuracy")
        assert abs(det.threshold - 0.9015310749276843) < 1e-10

        det2 = BinocularsDetector(mode="low-fpr")
        assert abs(det2.threshold - 0.8536432310785527) < 1e-10

    def test_not_loaded_raises(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        det = BinocularsDetector()
        with pytest.raises(RuntimeError, match="Models not loaded"):
            det.compute_score("test text")

    def test_is_loaded(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        det = BinocularsDetector()
        assert not det.is_loaded

    def test_set_threshold(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        det = BinocularsDetector()
        det.set_threshold(0.75, mode="custom")
        assert det.threshold == 0.75
        assert det.mode == "custom"

    def test_perplexity_computation(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        seq_len = 10
        vocab_size = 50
        logits = torch.randn(1, seq_len, vocab_size)
        encodings = {
            "input_ids": torch.randint(0, vocab_size, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }

        ppl = BinocularsDetector._perplexity(encodings, logits)
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_cross_perplexity_computation(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        seq_len = 10
        vocab_size = 50
        observer_logits = torch.randn(1, seq_len, vocab_size)
        performer_logits = torch.randn(1, seq_len, vocab_size)
        encodings = {
            "input_ids": torch.randint(0, vocab_size, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }

        x_ppl = BinocularsDetector._cross_perplexity(observer_logits, performer_logits, encodings, pad_token_id=0)
        assert isinstance(x_ppl, float)
        assert x_ppl > 0

    def test_predict_with_mock(self):
        from aigc_detector.detection.binoculars import BinocularsDetector

        det = BinocularsDetector(mode="accuracy")

        seq_len = 10
        vocab_size = 50

        mock_observer = MagicMock()
        mock_observer.device = torch.device("cpu")
        mock_observer.return_value = MagicMock(logits=torch.randn(1, seq_len, vocab_size))

        mock_performer = MagicMock()
        mock_performer.device = torch.device("cpu")
        mock_performer.return_value = MagicMock(logits=torch.randn(1, seq_len, vocab_size))

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(1, vocab_size, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }

        det._observer = mock_observer
        det._performer = mock_performer
        det._tokenizer = mock_tokenizer

        result = det.predict("Test text for binoculars detection.")
        assert result.label in ("ai", "human")
        assert isinstance(result.score, float)
        assert result.threshold == det.ACCURACY_THRESHOLD

    def test_binoculars_result_dataclass(self):
        from aigc_detector.detection.binoculars import BinocularsResult

        r = BinocularsResult(score=0.85, label="ai", threshold=0.9, mode="accuracy")
        assert r.score == 0.85
        assert r.label == "ai"


# ======================================================================
# Evaluator Tests
# ======================================================================


class TestEvaluator:
    def test_binary_perfect(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator(label_names=["human", "ai"])
        y_true = ["human", "human", "ai", "ai"]
        y_pred = ["human", "human", "ai", "ai"]
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        m = ev.evaluate(y_true, y_pred, y_prob=y_prob)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.roc_auc == 1.0
        assert m.n_samples == 4

    def test_binary_mixed(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator(label_names=["human", "ai"])
        y_true = ["human", "human", "ai", "ai"]
        y_pred = ["human", "ai", "ai", "human"]

        m = ev.evaluate(y_true, y_pred)
        assert m.accuracy == 0.5
        assert m.n_samples == 4

    def test_confusion_matrix_shape(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator(label_names=["human", "ai"])
        y_true = ["human", "ai", "ai", "human"]
        y_pred = ["human", "ai", "human", "human"]

        m = ev.evaluate(y_true, y_pred)
        assert len(m.confusion) == 2
        assert len(m.confusion[0]) == 2

    def test_ternary_evaluation(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator(label_names=["human", "ai", "mixed"], pos_label="ai")
        y_true = ["human", "ai", "mixed", "ai", "human", "mixed"]
        y_pred = ["human", "ai", "ai", "ai", "human", "mixed"]

        m = ev.evaluate(y_true, y_pred)
        assert m.n_samples == 6
        assert len(m.confusion) == 3

    def test_roc_auc_none_without_probs(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator()
        m = ev.evaluate(["human", "ai"], ["human", "ai"])
        assert m.roc_auc is None

    def test_roc_curve(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator()
        y_true = ["human", "human", "ai", "ai"]
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        roc_data = ev.roc_curve(y_true, y_prob)
        assert "fpr" in roc_data
        assert "tpr" in roc_data
        assert "thresholds" in roc_data

    def test_print_report_no_crash(self):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator()
        m = ev.evaluate(["human", "ai", "ai"], ["human", "ai", "human"])
        Evaluator.print_report(m)  # should not raise

    def test_save_load_report(self, tmp_path):
        from aigc_detector.training.evaluator import Evaluator

        ev = Evaluator()
        m = ev.evaluate(
            ["human", "ai", "ai", "human"],
            ["human", "ai", "human", "human"],
            y_prob=np.array([0.1, 0.9, 0.6, 0.3]),
        )
        report_path = tmp_path / "report.json"
        Evaluator.save_report(m, report_path)

        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "roc_auc" in data

    def test_to_dict(self):
        from aigc_detector.training.evaluator import EvalMetrics

        m = EvalMetrics(accuracy=0.9, precision=0.85, recall=0.8, f1=0.82, n_samples=100)
        d = m.to_dict()
        assert d["accuracy"] == 0.9
        assert d["n_samples"] == 100


class TestEvaluatePredictionsJsonl:
    def test_evaluate_from_jsonl(self, tmp_path):
        from aigc_detector.training.evaluator import evaluate_predictions_jsonl

        records = [
            {"label": "human", "predicted_label": "human", "p_ai": 0.1},
            {"label": "ai", "predicted_label": "ai", "p_ai": 0.9},
            {"label": "ai", "predicted_label": "human", "p_ai": 0.4},
            {"label": "human", "predicted_label": "human", "p_ai": 0.2},
        ]
        path = tmp_path / "preds.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        m = evaluate_predictions_jsonl(path)
        assert m.n_samples == 4
        assert m.accuracy == 0.75
        assert m.roc_auc is not None


# ======================================================================
# Calibration Tests
# ======================================================================


class TestThresholdCalibrator:
    def test_calibrate_f1_lower_is_positive(self):
        from aigc_detector.training.calibration import ThresholdCalibrator

        cal = ThresholdCalibrator(direction="lower_is_positive")
        # AI texts have lower scores, human have higher
        y_true = np.array(["ai"] * 50 + ["human"] * 50)
        rng = np.random.RandomState(42)
        scores = np.concatenate(
            [
                rng.uniform(0.3, 0.7, 50),  # AI: lower scores
                rng.uniform(0.8, 1.2, 50),  # Human: higher scores
            ]
        )

        result = cal.calibrate_f1(y_true, scores)
        assert result.metric_name == "f1"
        assert result.metric_value > 0.5
        assert result.direction == "lower_is_positive"
        assert result.n_samples == 100

    def test_calibrate_f1_higher_is_positive(self):
        from aigc_detector.training.calibration import ThresholdCalibrator

        cal = ThresholdCalibrator(direction="higher_is_positive")
        y_true = np.array(["ai"] * 50 + ["human"] * 50)
        rng = np.random.RandomState(42)
        scores = np.concatenate(
            [
                rng.uniform(0.6, 1.0, 50),  # AI: higher scores
                rng.uniform(0.0, 0.4, 50),  # Human: lower scores
            ]
        )

        result = cal.calibrate_f1(y_true, scores)
        assert result.metric_value > 0.5
        assert result.direction == "higher_is_positive"

    def test_calibrate_fpr(self):
        from aigc_detector.training.calibration import ThresholdCalibrator

        cal = ThresholdCalibrator(direction="lower_is_positive")
        y_true = np.array(["ai"] * 100 + ["human"] * 100)
        rng = np.random.RandomState(42)
        scores = np.concatenate(
            [
                rng.uniform(0.2, 0.6, 100),
                rng.uniform(0.7, 1.1, 100),
            ]
        )

        result = cal.calibrate_fpr(y_true, scores, target_fpr=0.05)
        assert result.metric_name == "fpr"
        assert result.metric_value <= 0.10  # should be near target
        assert result.n_samples == 200

    def test_calibrate_accuracy(self):
        from aigc_detector.training.calibration import ThresholdCalibrator

        cal = ThresholdCalibrator(direction="higher_is_positive")
        y_true = np.array(["ai"] * 50 + ["human"] * 50)
        rng = np.random.RandomState(42)
        scores = np.concatenate(
            [
                rng.uniform(0.6, 1.0, 50),
                rng.uniform(0.0, 0.4, 50),
            ]
        )

        result = cal.calibrate_accuracy(y_true, scores)
        assert result.metric_name == "accuracy"
        assert result.metric_value > 0.7

    def test_invalid_direction(self):
        from aigc_detector.training.calibration import ThresholdCalibrator

        with pytest.raises(ValueError, match="direction must be"):
            ThresholdCalibrator(direction="invalid")

    def test_save_load_result(self, tmp_path):
        from aigc_detector.training.calibration import CalibrationResult, ThresholdCalibrator

        result = CalibrationResult(
            optimal_threshold=0.85,
            metric_name="f1",
            metric_value=0.92,
            direction="lower_is_positive",
            n_samples=100,
        )
        path = tmp_path / "calibration.json"
        ThresholdCalibrator.save_result(result, path)

        loaded = ThresholdCalibrator.load_result(path)
        assert loaded.optimal_threshold == 0.85
        assert loaded.metric_name == "f1"
        assert loaded.metric_value == 0.92
        assert loaded.n_samples == 100

    def test_to_dict(self):
        from aigc_detector.training.calibration import CalibrationResult

        r = CalibrationResult(
            optimal_threshold=0.9,
            metric_name="fpr",
            metric_value=0.01,
            direction="lower_is_positive",
            n_samples=200,
        )
        d = r.to_dict()
        assert d["optimal_threshold"] == 0.9
        assert d["metric_name"] == "fpr"


# ======================================================================
# Model registry integration (verify binoculars entries)
# ======================================================================


class TestModelRegistryBinoculars:
    def test_binoculars_models_exist(self):
        from aigc_detector.models.registry import load_registry

        registry = load_registry()
        assert "falcon-7b" in registry
        assert "falcon-7b-instruct" in registry
        assert "qwen2-7b" in registry
        assert "qwen2-7b-instruct" in registry

    def test_binoculars_purpose(self):
        import aigc_detector.models.registry as reg
        from aigc_detector.models.registry import get_models_by_purpose

        reg._registry = None

        bino_models = get_models_by_purpose("binoculars")
        assert len(bino_models) == 4
        names = {m.name for m in bino_models}
        assert "falcon-7b" in names
        assert "falcon-7b-instruct" in names

    def test_binoculars_languages(self):
        from aigc_detector.models.registry import load_registry

        registry = load_registry()
        assert registry["falcon-7b"].language == "en"
        assert registry["qwen2-7b"].language == "zh"


# ======================================================================
# Feature extraction from JSONL (integration test)
# ======================================================================


class TestExtractFeaturesFromJsonl:
    def test_extract_features_with_mock(self, tmp_path):
        from aigc_detector.detection.statistical import StatisticalFeatures, extract_features_from_jsonl

        records = [
            {"id": "h_001", "text": "Test text one for extraction.", "label": "human", "lang": "en"},
            {"id": "a_001", "text": "Another test text for extraction.", "label": "ai", "lang": "en"},
        ]
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        with open(input_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = StatisticalFeatures(
            perplexity=25.0,
            avg_entropy=3.5,
            std_entropy=1.0,
            burstiness=-0.3,
            max_entropy=5.5,
            min_entropy=0.8,
        )

        stats = extract_features_from_jsonl(mock_extractor, input_path, output_path)
        assert stats["processed"] == 2
        assert stats["errors"] == 0

        with open(output_path) as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 2
        assert "features" in lines[0]
        assert lines[0]["features"]["perplexity"] == 25.0
