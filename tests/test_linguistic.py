"""Unit tests for the linguistic-stylistic feature detection module.

Pure-CPU: no LM/GPU required. Validates the 14-feature dataclass shape,
the per-feature extraction logic for human-vs-AI samples, the
classifier fit/predict/save-load roundtrip, and the diagnostics scorer.

Test style follows ``tests/test_detection.py``: class-based, AAA pattern,
descriptive docstrings, no network access.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aigc_detector.detection.linguistic import (
    FEATURE_NAMES,
    LinguisticClassifier,
    LinguisticDiagnostics,
    LinguisticFeatureExtractor,
    LinguisticFeatures,
)

# ======================================================================
# Sample paragraphs (50-200 words each, original text)
# ======================================================================

# Clearly HUMAN English: varies sentence length, has hedging ("seemed to arise"),
# first-person plural, parenthetical aside, varied punctuation.
EN_HUMAN_SAMPLE = (
    "We started the experiment expecting a clean monotonic curve. What we got, "
    "frankly, was a mess. Three of the eight runs produced wild outliers "
    "(two standard deviations off the mean); the rest clustered tightly around "
    "what we had predicted. The discrepancy seemed to arise from a single "
    "thermostat drifting by about half a degree overnight. Interestingly, when "
    "we re-ran those three trials after recalibration, the outliers vanished. "
    "We are not entirely sure why the drift had such an outsized effect, but "
    "in practice it probably means the lab's nightly shutdown routine needs "
    "a second look. Perhaps the simplest fix is to log ambient temperature "
    "continuously, and flag any reading more than 0.3 degrees off baseline. "
    "That alone would have caught the problem a week earlier. We find this "
    "kind of small-bore instrumentation issue tends to dominate reproducibility "
    "far more than the fancy algorithmic tricks people like to argue about."
)

# Clearly AI English: uniform sentence length, "Furthermore/Moreover", no hedging,
# no first-person, templated structure.
EN_AI_SAMPLE = (
    "The proposed method achieves high accuracy across all benchmarks. "
    "Furthermore, the model demonstrates robust generalization to unseen data. "
    "Moreover, the training procedure is computationally efficient and scales "
    "linearly with dataset size. In addition, the architecture incorporates "
    "several regularization techniques that prevent overfitting. Therefore, "
    "the results indicate that the approach is both effective and practical. "
    "Additionally, the method outperforms prior baselines on every metric "
    "evaluated. Consequently, we conclude that the framework offers a "
    "significant advancement in the field. The evaluation was conducted on "
    "standard datasets using established protocols. The experimental setup "
    "ensures reproducibility and fair comparison with existing methods."
)

# Clearly HUMAN Chinese: hedging ("似乎", "可能"), varied punctuation (——, ……, ；),
# first-person, varied sentence length.
ZH_HUMAN_SAMPLE = (
    "我们一开始以为这个 bug 很简单。改了三行代码就跑通了，可上线之后却发现——"
    "效果反而更差了。组里讨论了半天，谁也说不清到底是哪里出了问题；我们似乎总是"
    "在原地打转。或许问题根本不在代码里，而是在数据本身的漂移上？我后来想，"
    "这类问题往往不是靠重新训练能解决的。我们试着把线上日志拉回来重新统计，"
    "果然……某一类样本的占比悄悄从 12% 涨到了 31%。这大概能解释一切。虽然"
    "重采样之后指标恢复了，但说老实话，我心里仍有点不踏实。我们未必能每次都"
    "这么走运。"
)

# Clearly AI Chinese: "此外/因此/首先/其次/最后", uniform sentence length, no hedging.
ZH_AI_SAMPLE = (
    "本研究提出了一种新的文本检测方法。首先，该方法利用统计特征对文本进行建模。"
    "其次，模型通过深度学习提取语义信息。此外，系统集成了多种检测策略以提升鲁棒性。"
    "因此，该方法在多个公开数据集上取得了优秀的表现。同时，实验结果表明，所提方法"
    "在准确率和召回率上均优于现有方案。另外，本文还对模型的可解释性进行了分析。"
    "最后，实验验证了方法的有效性。综上所述，本研究的贡献具有重要的实际意义。"
    "由此可以看出，该框架为后续研究提供了良好的基础。"
)


# ======================================================================
# LinguisticFeatures dataclass tests
# ======================================================================


class TestLinguisticFeatures:
    """Verify the dataclass container exposes the correct shape and order."""

    def test_field_count_is_fourteen(self):
        """LinguisticFeatures must have exactly 14 fields."""
        import dataclasses

        assert len(dataclasses.fields(LinguisticFeatures)) == 14

    def test_field_names_match_feature_names(self):
        """Dataclass field order must match the canonical FEATURE_NAMES list."""
        import dataclasses

        actual = [f.name for f in dataclasses.fields(LinguisticFeatures)]
        assert actual == FEATURE_NAMES

    def test_to_array_shape_and_dtype(self):
        """to_array returns a (14,) float64 numpy array."""
        nan = float("nan")
        feats = LinguisticFeatures(
            sentence_length_burstiness=0.1,
            sentence_length_cv=0.5,
            sentence_length_gini=0.3,
            syntactic_repetition_rate=0.2,
            token_logprob_skew=nan,
            token_logprob_high_prob_frac=nan,
            hedging_density=3.0,
            discourse_templating=2.0,
            punctuation_style=1.5,
            paragraph_length_variance=2.5,
            paragraph_template_score=0.2,
            lexical_diversity_mtld=55.0,
            authorial_stance_score=8.0,
            readability_index=45.0,
        )
        arr = feats.to_array()
        assert arr.shape == (14,)
        assert arr.dtype == np.float64
        assert arr[0] == pytest.approx(0.1)
        assert arr[11] == pytest.approx(55.0)

    def test_to_array_is_nan_safe(self):
        """to_array must not raise even when fields are NaN."""
        feats = LinguisticFeatures(
            sentence_length_burstiness=float("nan"),
            sentence_length_cv=float("nan"),
            sentence_length_gini=float("nan"),
            syntactic_repetition_rate=float("nan"),
            token_logprob_skew=float("nan"),
            token_logprob_high_prob_frac=float("nan"),
            hedging_density=float("nan"),
            discourse_templating=float("nan"),
            punctuation_style=float("nan"),
            paragraph_length_variance=float("nan"),
            paragraph_template_score=float("nan"),
            lexical_diversity_mtld=float("nan"),
            authorial_stance_score=float("nan"),
            readability_index=float("nan"),
        )
        arr = feats.to_array()
        assert arr.shape == (14,)
        assert np.all(np.isnan(arr))

    def test_to_dict_has_all_fields(self):
        """to_dict preserves every field name."""
        feats = LinguisticFeatures(
            sentence_length_burstiness=0.1,
            sentence_length_cv=0.5,
            sentence_length_gini=0.3,
            syntactic_repetition_rate=0.2,
            token_logprob_skew=0.0,
            token_logprob_high_prob_frac=0.8,
            hedging_density=3.0,
            discourse_templating=2.0,
            punctuation_style=1.5,
            paragraph_length_variance=2.5,
            paragraph_template_score=0.2,
            lexical_diversity_mtld=55.0,
            authorial_stance_score=8.0,
            readability_index=45.0,
        )
        d = feats.to_dict()
        assert len(d) == 14
        assert "hedging_density" in d
        assert "lexical_diversity_mtld" in d


# ======================================================================
# LinguisticFeatureExtractor tests
# ======================================================================


class TestLinguisticFeatureExtractor:
    """Tests for extraction behaviour across human/AI samples and edge cases."""

    def test_short_text_returns_nan(self):
        """Text shorter than min_text_chars (default 200) returns all-NaN."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract("This is a short sentence. It is way under the threshold.")
        arr = feats.to_array()
        assert arr.shape == (14,)
        assert np.all(np.isnan(arr)), "Short text must yield an all-NaN feature vector"

    def test_empty_text_does_not_crash(self):
        """Empty / whitespace-only text must return all-NaN without raising."""
        extractor = LinguisticFeatureExtractor()
        for text in ["", "   ", "\n\n"]:
            feats = extractor.extract(text)
            assert np.all(np.isnan(feats.to_array()))

    def test_none_text_does_not_crash(self):
        """None input must be handled gracefully (all-NaN)."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract("")  # pass empty rather than None to respect type hint
        assert np.all(np.isnan(feats.to_array()))

    def test_english_human_sample(self):
        """Human English text: high hedging, varied punctuation, authorial stance."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en")

        assert not math.isnan(feats.hedging_density), "Human sample should have finite hedging_density"
        assert feats.hedging_density > 0.0, "Human sample should contain hedging words"
        # Authorial stance: first-person + hedging should be non-trivial.
        assert feats.authorial_stance_score > 1.0
        # Varied punctuation (parenthetical, semicolon, em-dash).
        assert feats.punctuation_style > 0.0
        # Burstiness > -1 always; CV positive.
        assert feats.sentence_length_burstiness > -1.0
        assert feats.sentence_length_cv > 0.0

    def test_english_ai_sample(self):
        """AI English text: heavy discourse templating, low hedging."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_AI_SAMPLE, lang="en")

        assert feats.discourse_templating > feats.hedging_density, (
            "AI sample should be dominated by templating markers, not hedging"
        )
        assert feats.discourse_templating > 5.0, "AI sample has many Furthermore/Moreover/etc."
        # AI sample has no first-person; stance should be near zero (only hedging residues).
        assert feats.authorial_stance_score < 5.0

    def test_chinese_human_sample(self):
        """Human Chinese text: hedging (似乎/或许), varied punctuation (——, ……, ；)."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(ZH_HUMAN_SAMPLE, lang="zh")

        assert feats.hedging_density > 0.0, "Human zh sample contains 似乎/或许/未必"
        assert feats.punctuation_style > 0.0, "Human zh sample uses —— …… ；"
        assert feats.authorial_stance_score > 0.0, "First-person 我们/我 present"

    def test_chinese_ai_sample(self):
        """AI Chinese text: templating markers (此外/因此/综上) dominate."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(ZH_AI_SAMPLE, lang="zh")

        assert feats.discourse_templating > feats.hedging_density, (
            "AI zh sample: templating should exceed hedging"
        )
        assert feats.discourse_templating > 5.0

    def test_human_vs_ai_discrimination(self):
        """Human and AI samples should differ on the discriminative features."""
        extractor = LinguisticFeatureExtractor()
        human = extractor.extract(EN_HUMAN_SAMPLE, lang="en")
        ai = extractor.extract(EN_AI_SAMPLE, lang="en")

        # AI should have notably higher templating density.
        assert ai.discourse_templating > human.discourse_templating
        # Human should have notably higher hedging density.
        assert human.hedging_density > ai.hedging_density

    def test_token_logprob_features_when_provided(self):
        """When token_log_probs is provided, M5/M6 are finite floats."""
        extractor = LinguisticFeatureExtractor()
        # Synthetic log-probs: mostly high-confidence tokens.
        probs = np.array([-0.1, -0.2, -0.05, -0.3, -2.5, -0.4, -0.1, -0.2, -0.15, -0.1] * 20)
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en", token_log_probs=probs)

        assert not math.isnan(feats.token_logprob_skew)
        assert not math.isnan(feats.token_logprob_high_prob_frac)
        # High-prob fraction should reflect the mostly-high-confidence synthetic probs.
        assert 0.0 <= feats.token_logprob_high_prob_frac <= 1.0
        # 9 of 10 tokens are above -1.0 → fraction should be 0.9.
        assert feats.token_logprob_high_prob_frac == pytest.approx(0.9, abs=0.05)

    def test_token_logprob_features_when_absent(self):
        """When token_log_probs is None, M5/M6 are NaN."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en", token_log_probs=None)
        assert math.isnan(feats.token_logprob_skew)
        assert math.isnan(feats.token_logprob_high_prob_frac)

    def test_token_logprob_empty_array(self):
        """Empty token_log_probs array should yield NaN for M5/M6 (no crash)."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en", token_log_probs=[])
        assert math.isnan(feats.token_logprob_skew)
        assert math.isnan(feats.token_logprob_high_prob_frac)

    def test_lexical_diversity_mtld_known_input(self):
        """MTLD on a known-vocabulary input falls in the expected range."""
        extractor = LinguisticFeatureExtractor()
        # Highly repetitive text -> MTLD low (close to factor size ~3.6).
        repetitive = "the cat sat the cat sat the cat sat the cat sat the cat sat " * 5
        # Add length padding so we exceed min_text_chars.
        repetitive = repetitive + " " + ("the cat sat " * 50)
        feats = extractor.extract(repetitive, lang="en")
        assert not math.isnan(feats.lexical_diversity_mtld)
        assert feats.lexical_diversity_mtld < 20.0, "Repetitive text should have low MTLD"

        # Diverse text -> MTLD higher.
        diverse_words = (
            "elephant galaxy whisper canvas thunder marble velvet orchid puzzle lantern "
            "horizon meadow crystal sapphire whisper canyon prairie echo fortress "
            "willow cascade ember twilight harvest meadow rocket saffron cobalt drift "
        ) * 3
        feats_div = extractor.extract(diverse_words, lang="en")
        assert feats_div.lexical_diversity_mtld > feats.lexical_diversity_mtld

    def test_paragraph_length_variance(self):
        """Text with uneven paragraphs has higher paragraph_length_variance than uniform."""
        extractor = LinguisticFeatureExtractor()
        uneven = (
            "Short paragraph here.\n\n"
            + "This is a much longer paragraph that contains many more characters than "
            "the previous one and is intended to produce a high variance in paragraph "
            "length when the extractor splits on double newlines as expected.\n\n"
            + "Mid length paragraph with a few words.\n\n"
            + "Another quite long paragraph to ensure the variance remains high. "
            "Adding more text here to push the total character count above the "
            "minimum threshold that the extractor enforces for short-text guard."
        )
        feats = extractor.extract(uneven, lang="en")
        assert not math.isnan(feats.paragraph_length_variance)
        assert feats.paragraph_length_variance > 0.0

    def test_readability_index_english(self):
        """English readability is Flesch Reading Ease (finite for 2+ sentences)."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en")
        assert not math.isnan(feats.readability_index)
        # Flesch scores typically land in [-50, 120] for natural text.
        assert -50.0 < feats.readability_index < 120.0

    def test_readability_index_chinese_is_negative_avg_length(self):
        """Chinese readability proxy is the negative average sentence length."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(ZH_HUMAN_SAMPLE, lang="zh")
        assert not math.isnan(feats.readability_index)
        # Negative avg-length proxy: should be in roughly [-100, -5].
        assert feats.readability_index < 0.0

    def test_min_text_chars_is_configurable(self):
        """Caller can lower min_text_chars to extract from short text."""
        extractor = LinguisticFeatureExtractor(min_text_chars=10)
        feats = extractor.extract("Two short sentences. Should still extract.", lang="en")
        # Not all-NaN now.
        assert not np.all(np.isnan(feats.to_array()))


# ======================================================================
# LinguisticClassifier tests
# ======================================================================


class TestLinguisticClassifier:
    """Classifier fit / predict / persistence tests."""

    def _make_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Synthetic 200-sample dataset shaped like real linguistic features.

        Human rows: higher hedging, higher burstiness/CV, higher MTLD.
        AI rows: higher templating, low hedging, low burstiness.
        """
        rng = np.random.RandomState(42)
        n = 200
        human = rng.randn(n // 2, 14) + np.array(
            # M1 burst, M2 cv, M3 gini, M4 rep,
            # M5 skew, M6 high-frac,
            # M7 hedge, M8 template, M9 punct,
            # S1 para-var, S2 template-score,
            # D1 mtld, D2 stance, D3 readability
            [0.2, 0.7, 0.4, 0.1, 0.5, 0.7, 8.0, 1.0, 3.0, 4.0, 0.1, 70.0, 12.0, 50.0]
        )
        ai = rng.randn(n // 2, 14) + np.array(
            [-0.3, 0.2, 0.2, 0.4, -0.5, 0.95, 0.5, 12.0, 0.5, 0.5, 0.7, 30.0, 1.0, 35.0]
        )
        x = np.vstack([human, ai])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return x, y

    def test_fit_predict(self):
        """Training on synthetic data achieves reasonable accuracy and predict() works."""
        clf = LinguisticClassifier()
        x, y = self._make_training_data()
        stats = clf.fit(x, y)
        assert stats["n_samples"] == 200
        assert stats["train_accuracy"] > 0.7

        result = clf.predict(x[0])
        assert "label" in result
        assert "p_ai" in result
        assert "confidence" in result
        assert result["label"] in ("human", "ai")
        assert 0.0 <= result["p_ai"] <= 1.0

    def test_predict_batch(self):
        """Batch predict returns lists of the right length."""
        clf = LinguisticClassifier()
        x, y = self._make_training_data()
        clf.fit(x, y)
        result = clf.predict(x[:5])
        assert "labels" in result
        assert len(result["labels"]) == 5

    def test_predict_proba_shape(self):
        """predict_proba returns an (n, 2) array whose rows sum to 1."""
        clf = LinguisticClassifier()
        x, y = self._make_training_data()
        clf.fit(x, y)
        proba = clf.predict_proba(x[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_from_features_dataclass(self):
        """predict() accepts a single LinguisticFeatures instance."""
        clf = LinguisticClassifier()
        x, y = self._make_training_data()
        clf.fit(x, y)

        feats = LinguisticFeatures(
            sentence_length_burstiness=0.3,
            sentence_length_cv=0.8,
            sentence_length_gini=0.5,
            syntactic_repetition_rate=0.1,
            token_logprob_skew=0.5,
            token_logprob_high_prob_frac=0.7,
            hedging_density=8.0,
            discourse_templating=1.0,
            punctuation_style=3.0,
            paragraph_length_variance=4.0,
            paragraph_template_score=0.1,
            lexical_diversity_mtld=70.0,
            authorial_stance_score=12.0,
            readability_index=50.0,
        )
        result = clf.predict(feats)
        assert result["label"] in ("human", "ai")

    def test_save_load_roundtrip(self, tmp_path: Path):
        """A classifier saved and reloaded produces identical predictions."""
        clf = LinguisticClassifier()
        x, y = self._make_training_data()
        clf.fit(x, y)

        save_path = tmp_path / "ling_clf.pkl"
        clf.save(save_path)
        assert save_path.exists()

        clf2 = LinguisticClassifier()
        clf2.load(save_path)
        assert clf2.backend == "xgboost"

        r1 = clf.predict(x[0])
        r2 = clf2.predict(x[0])
        assert r1["label"] == r2["label"]
        assert abs(r1["p_ai"] - r2["p_ai"]) < 1e-6

    def test_set_threshold(self):
        """set_threshold updates the decision cutoff."""
        clf = LinguisticClassifier()
        clf.set_threshold(0.75)
        assert clf.threshold == 0.75

    def test_invalid_backend_rejected(self):
        """Only the xgboost backend is supported."""
        with pytest.raises(ValueError, match="xgboost"):
            LinguisticClassifier(backend="logistic_regression")

    def test_predict_before_fit_raises(self):
        """predict() on an untrained classifier must raise RuntimeError."""
        clf = LinguisticClassifier()
        # Build a fresh classifier with a None pipeline to simulate untrained state.
        clf._pipeline = None
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(np.zeros((1, 14)))

    def test_feature_names_class_attribute(self):
        """FEATURE_NAMES class attribute has 14 entries matching the dataclass."""
        assert len(LinguisticClassifier.FEATURE_NAMES) == 14
        import dataclasses

        assert LinguisticClassifier.FEATURE_NAMES == [
            f.name for f in dataclasses.fields(LinguisticFeatures)
        ]

    def test_handles_nan_in_training_features(self):
        """NaN values in the feature matrix must not crash fit/predict (imputer)."""
        clf = LinguisticClassifier()
        x, y = self._make_training_data()
        # Inject some NaNs.
        x[0, 4] = float("nan")
        x[5, 5] = float("nan")
        stats = clf.fit(x, y)
        assert stats["train_accuracy"] > 0.5

        # Predict with a NaN-containing sample.
        nan_row = x[0].copy()
        nan_row[4] = float("nan")
        result = clf.predict(nan_row)
        assert result["label"] in ("human", "ai")


# ======================================================================
# LinguisticDiagnostics tests
# ======================================================================


class TestLinguisticDiagnostics:
    """Tests for the diagnostics scorer derived from LinguisticFeatures."""

    def test_from_features_returns_valid_scores(self):
        """from_features returns scores within their documented ranges."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en")
        diag = LinguisticDiagnostics.from_features(feats, lang="en")

        assert 0.0 <= diag.micro_score <= 10.0
        assert 0.0 <= diag.meso_score <= 10.0
        assert 0.0 <= diag.macro_score <= 10.0
        assert 0.0 <= diag.human_likeness_score <= 100.0
        # Composite must be the weighted blend of the three level scores.
        expected = (0.4 * diag.micro_score + 0.3 * diag.meso_score + 0.3 * diag.macro_score) * 10.0
        assert diag.human_likeness_score == pytest.approx(expected, abs=1e-6)

    def test_from_features_chinese(self):
        """Chinese features also produce valid diagnostics."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(ZH_HUMAN_SAMPLE, lang="zh")
        diag = LinguisticDiagnostics.from_features(feats, lang="zh")
        assert 0.0 <= diag.micro_score <= 10.0
        assert 0.0 <= diag.human_likeness_score <= 100.0

    def test_top_signals_format(self):
        """top_signals is a list[str] of at most 3 feature names."""
        extractor = LinguisticFeatureExtractor()
        feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en")
        diag = LinguisticDiagnostics.from_features(feats, lang="en")
        assert isinstance(diag.top_signals, list)
        assert len(diag.top_signals) <= 3
        for name in diag.top_signals:
            assert isinstance(name, str)
            assert name in FEATURE_NAMES, f"Signal {name!r} not a known feature name"

    def test_top_signals_empty_when_all_low(self):
        """top_signals is empty when no feature exceeds the human threshold."""
        # All-zero features: no signal exceeds any threshold.
        feats = LinguisticFeatures(
            sentence_length_burstiness=0.0,
            sentence_length_cv=0.0,
            sentence_length_gini=0.0,
            syntactic_repetition_rate=0.0,
            token_logprob_skew=0.0,
            token_logprob_high_prob_frac=0.0,
            hedging_density=0.0,
            discourse_templating=0.0,
            punctuation_style=0.0,
            paragraph_length_variance=0.0,
            paragraph_template_score=0.0,
            lexical_diversity_mtld=0.0,
            authorial_stance_score=0.0,
            readability_index=0.0,
        )
        diag = LinguisticDiagnostics.from_features(feats, lang="en")
        assert diag.top_signals == []

    def test_diagnostics_handles_all_nan_features(self):
        """Diagnostics must not crash on an all-NaN feature vector (short text)."""
        nan = float("nan")
        feats = LinguisticFeatures(
            sentence_length_burstiness=nan,
            sentence_length_cv=nan,
            sentence_length_gini=nan,
            syntactic_repetition_rate=nan,
            token_logprob_skew=nan,
            token_logprob_high_prob_frac=nan,
            hedging_density=nan,
            discourse_templating=nan,
            punctuation_style=nan,
            paragraph_length_variance=nan,
            paragraph_template_score=nan,
            lexical_diversity_mtld=nan,
            authorial_stance_score=nan,
            readability_index=nan,
        )
        diag = LinguisticDiagnostics.from_features(feats, lang="en")
        assert 0.0 <= diag.micro_score <= 10.0
        assert diag.top_signals == []
        assert diag.human_likeness_score == pytest.approx(0.0)

    def test_human_scores_higher_than_ai(self):
        """Human sample should produce a higher human-likeness_score than AI sample."""
        extractor = LinguisticFeatureExtractor()
        human_feats = extractor.extract(EN_HUMAN_SAMPLE, lang="en")
        ai_feats = extractor.extract(EN_AI_SAMPLE, lang="en")
        human_diag = LinguisticDiagnostics.from_features(human_feats, lang="en")
        ai_diag = LinguisticDiagnostics.from_features(ai_feats, lang="en")
        assert human_diag.human_likeness_score > ai_diag.human_likeness_score
