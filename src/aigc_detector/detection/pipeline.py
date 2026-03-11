"""Cascading detection pipeline.

Orchestrates the three detection stages (Statistical → Encoder → Binoculars)
with early-exit logic for high-confidence results.

Flow:
    1. Language Router → determine ``zh`` or ``en``
    2. Stage 1: Statistical classifier (<200 ms)
       - If confidence > ``early_exit_threshold`` → return immediately
    3. Stage 2: Encoder classifier (300–500 ms)
       - If Stage 1 & 2 agree → weighted combine → return
    4. Stage 3: Binoculars zero-shot (1–3 s, only on conflict)
       - Final weighted combine → return

References:
    - DESIGN.md §2.1 (cascading pipeline)
    - DESIGN.md §4.4 (ensemble weights)
    - DEVPLAN.md Phase 4 task 4.3
"""

from __future__ import annotations

import logging
import time

from aigc_detector.detection.ensemble import EnsembleAggregator, EnsembleResult

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Cascading AI text detection pipeline.

    Parameters
    ----------
    language_router : object
        Language detection router (must have ``.detect(text)`` method).
    statistical_extractors : dict[str, object]
        Language → ``StatisticalFeatureExtractor`` mapping.
    statistical_classifiers : dict[str, object]
        Language → ``StatisticalClassifier`` mapping.
    encoder_classifiers : dict[str, object]
        Language → ``EncoderClassifier`` mapping.
    binoculars_detectors : dict[str, object]
        Language → ``BinocularsDetector`` mapping.
    model_manager : object or None
        VRAM lifecycle manager.
    early_exit_threshold : float
        Confidence above which Stage 1 can exit early.
    """

    def __init__(
        self,
        language_router: object,
        statistical_extractors: dict[str, object] | None = None,
        statistical_classifiers: dict[str, object] | None = None,
        encoder_classifiers: dict[str, object] | None = None,
        binoculars_detectors: dict[str, object] | None = None,
        model_manager: object | None = None,
        early_exit_threshold: float = 0.95,
    ):
        self.language_router = language_router
        self.statistical_extractors = statistical_extractors or {}
        self.statistical_classifiers = statistical_classifiers or {}
        self.encoder_classifiers = encoder_classifiers or {}
        self.binoculars_detectors = binoculars_detectors or {}
        self.model_manager = model_manager
        self.early_exit_threshold = early_exit_threshold
        self._aggregator = EnsembleAggregator()

    def detect(self, text: str) -> EnsembleResult:
        """Run the full cascading detection pipeline on *text*.

        Returns an ``EnsembleResult`` with all stage breakdown info.
        """
        t0 = time.perf_counter()
        stage_results: dict[str, dict] = {}

        # Step 0: Language detection
        lang_result = self.language_router.detect(text)
        lang = lang_result.lang
        logger.info("Detected language: %s (confidence=%.2f)", lang, lang_result.confidence)

        # Step 1: Statistical features → classifier
        stat_result = self._run_statistical(text, lang)
        if stat_result is not None:
            stage_results["statistical"] = stat_result

            # Early exit if statistical confidence is very high
            if stat_result.get("confidence", 0) > self.early_exit_threshold:
                logger.info("Stage 1 early exit: confidence=%.4f", stat_result["confidence"])
                elapsed = (time.perf_counter() - t0) * 1000
                return self._aggregator.combine(
                    stage_results,
                    detected_language=lang,
                    processing_time_ms=elapsed,
                )

        # Step 2: Encoder classifier
        encoder_result = self._run_encoder(text, lang)
        if encoder_result is not None:
            stage_results["encoder"] = encoder_result

            # If statistical and encoder agree, skip binoculars
            if stat_result is not None and self._aggregator.agree(stage_results):
                logger.info("Stage 1 & 2 agree — skipping Binoculars")
                elapsed = (time.perf_counter() - t0) * 1000
                return self._aggregator.combine(
                    stage_results,
                    detected_language=lang,
                    processing_time_ms=elapsed,
                )

        # Step 3: Binoculars (only on conflict or when previous stages missing)
        bino_result = self._run_binoculars(text, lang)
        if bino_result is not None:
            stage_results["binoculars"] = bino_result

        elapsed = (time.perf_counter() - t0) * 1000
        return self._aggregator.combine(
            stage_results,
            detected_language=lang,
            processing_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_statistical(self, text: str, lang: str) -> dict | None:
        """Run Stage 1: statistical feature extraction + classification."""
        extractor = self.statistical_extractors.get(lang)
        classifier = self.statistical_classifiers.get(lang)
        if extractor is None or classifier is None:
            logger.debug("No statistical detector for language: %s", lang)
            return None

        try:
            features = extractor.extract(text)
            result = classifier.predict(features)
            result["features"] = features.to_dict()
            return result
        except Exception:
            logger.warning("Statistical detection failed", exc_info=True)
            return None

    def _run_encoder(self, text: str, lang: str) -> dict | None:
        """Run Stage 2: encoder-based classification."""
        classifier = self.encoder_classifiers.get(lang)
        if classifier is None:
            logger.debug("No encoder classifier for language: %s", lang)
            return None

        try:
            result = classifier.predict(text)
            return {
                "label": result.label,
                "p_ai": result.p_ai,
                "confidence": result.confidence,
                "model": result.model_name,
            }
        except Exception:
            logger.warning("Encoder detection failed", exc_info=True)
            return None

    def _run_binoculars(self, text: str, lang: str) -> dict | None:
        """Run Stage 3: Binoculars zero-shot detection."""
        detector = self.binoculars_detectors.get(lang)
        if detector is None:
            logger.debug("No Binoculars detector for language: %s", lang)
            return None

        try:
            result = detector.predict(text)
            # Convert Binoculars score to p_ai (lower score = more AI)
            # Use a sigmoid-like mapping: p_ai = 1 / (1 + exp(k * (score - threshold)))
            # Simpler approach: clamp to [0, 1] with linear mapping around threshold
            p_ai = self._binoculars_score_to_p_ai(result.score, result.threshold)
            return {
                "label": result.label,
                "score": result.score,
                "threshold": result.threshold,
                "p_ai": p_ai,
                "confidence": max(p_ai, 1.0 - p_ai),
                "mode": result.mode,
            }
        except Exception:
            logger.warning("Binoculars detection failed", exc_info=True)
            return None

    @staticmethod
    def _binoculars_score_to_p_ai(score: float, threshold: float) -> float:
        """Convert a Binoculars score to a probability of AI generation.

        Binoculars: low score → AI, high score → human.
        Maps score to p_ai ∈ [0, 1] using the threshold as midpoint.
        """
        if threshold <= 0:
            return 0.5
        # Ratio: score / threshold.  ratio < 1 → AI, ratio > 1 → human
        ratio = score / threshold
        # Sigmoid-style mapping centred at ratio=1
        import math

        p_ai = 1.0 / (1.0 + math.exp(5.0 * (ratio - 1.0)))
        return max(0.0, min(1.0, p_ai))
