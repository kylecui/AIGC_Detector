"""Cascading detection pipeline.

Orchestrates the detection stages with early-exit logic for
high-confidence results.

Stages
------
- **Statistical** (Stage 1, fast): LM-probability features.
- **Linguistic** (Stage 1b, fast): CPU-only stylistic features that run
  in parallel with Stage 1. Reuses the per-token log-probs already
  computed by :class:`StatisticalFeatureExtractor` (exposed via the
  ``_last_token_log_probs`` instance attribute) to derive the M5/M6
  features without a second LM forward pass.
- **Encoder** (Stage 2): transformer-based classifier.
- **Binoculars** (Stage 3, zero-shot fallback).

Flow:
    1. Language Router → determine ``zh`` or ``en``
    2. Stage 1: Statistical + Linguistic (parallel, both "fast stage")
       - If statistical confidence > ``early_exit_threshold`` → return immediately
    3. Stage 2: Encoder classifier (300–500 ms)
       - If Stage 1 & 2 agree → weighted combine → return
    4. Stage 3: Binoculars zero-shot (1–3 s, only on conflict)
       - Final weighted combine → return

References:
    - DESIGN.md §2.1 (cascading pipeline)
    - DESIGN.md §4.4 (ensemble weights)
    - DEVPLAN.md Phase 4 task 4.3
    - .sisyphus/plans/upgrade-linguistic-detection.md (L2 linguistic axis)
"""

from __future__ import annotations

import logging
import time

from aigc_detector.detection.ensemble import EnsembleAggregator, EnsembleResult

logger = logging.getLogger(__name__)

ZH_DECISION_THRESHOLD = 0.47


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
    linguistic_extractors : dict[str, object] or None
        Language → ``LinguisticFeatureExtractor`` mapping (CPU-only, optional).
        When absent for a language, the linguistic stage is silently skipped.
    linguistic_classifiers : dict[str, object] or None
        Language → ``LinguisticClassifier`` mapping (optional).
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
        linguistic_extractors: dict[str, object] | None = None,
        linguistic_classifiers: dict[str, object] | None = None,
        model_manager: object | None = None,
        early_exit_threshold: float = 0.95,
        ensemble_weights_by_lang: dict[str, dict[str, float]] | None = None,
    ):
        self.language_router = language_router
        self.statistical_extractors = statistical_extractors or {}
        self.statistical_classifiers = statistical_classifiers or {}
        self.encoder_classifiers = encoder_classifiers or {}
        self.binoculars_detectors = binoculars_detectors or {}
        self.linguistic_extractors = linguistic_extractors or {}
        self.linguistic_classifiers = linguistic_classifiers or {}
        self.model_manager = model_manager
        self.early_exit_threshold = early_exit_threshold
        self.ensemble_weights_by_lang = ensemble_weights_by_lang or {}
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

        # Apply language-specific ensemble weights if configured
        lang_weights = self.ensemble_weights_by_lang.get(lang)
        if lang_weights:
            self._aggregator.weights = dict(lang_weights)
            logger.debug("Using lang-specific weights for %s: %s", lang, lang_weights)

        # Step 1: Statistical features → classifier. _run_statistical also
        # returns the per-token log-probs from the extractor so the
        # linguistic stage (Stage 1b) can reuse them for M5/M6.
        stat_result, token_log_probs = self._run_statistical(text, lang)
        if stat_result is not None:
            stage_results["statistical"] = stat_result

            # Stage 1b: Linguistic (CPU-only, runs alongside Stage 1). Adds
            # an orthogonal "human writing noise" evidence source. Silently
            # skipped when no extractor/classifier is registered for the lang.
            linguistic_result = self._run_linguistic(text, lang, token_log_probs)
            if linguistic_result is not None:
                stage_results["linguistic"] = linguistic_result

            # Early exit if statistical confidence is very high.
            # For Chinese, keep the encoder in the loop because the statistical
            # stage can be overconfident on fluent AI-written text and cause
            # false Human-written exits before the stronger encoder runs.
            if lang != "zh" and stat_result.get("confidence", 0) > self.early_exit_threshold:
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

            # Conservative zh arbitration: when the statistical stage says human
            # but the encoder has already crossed a moderate AI probability, the
            # encoder is usually the stronger signal on formal Chinese prose.
            if (
                lang == "zh"
                and stat_result is not None
                and stat_result.get("label") == "human"
                and encoder_result.get("p_ai", 0) >= 0.35
            ):
                logger.info(
                    "Stage 2 zh arbitration: overriding statistical human with encoder p_ai=%.4f",
                    encoder_result["p_ai"],
                )
                elapsed = (time.perf_counter() - t0) * 1000
                return self._aggregator.combine(
                    {"encoder": encoder_result},
                    detected_language=lang,
                    processing_time_ms=elapsed,
                    decision_threshold=ZH_DECISION_THRESHOLD,
                )

            # For Chinese, a strong encoder result is more reliable than the
            # statistical stage on fluent AI-written text. In that case, skip
            # the heavyweight Binoculars fallback and combine the first two
            # stages directly.
            if lang == "zh" and encoder_result.get("confidence", 0) > self.early_exit_threshold:
                logger.info("Stage 2 zh early exit: encoder confidence=%.4f", encoder_result["confidence"])
                elapsed = (time.perf_counter() - t0) * 1000
                return self._aggregator.combine(
                    stage_results,
                    detected_language=lang,
                    processing_time_ms=elapsed,
                )

            # If statistical and encoder agree, skip binoculars.
            # EXCEPT for ZH: HC3-trained models can catastrophically fail on
            # modern LLM text (p_ai as low as 0.01 for GPT-4/Claude content).
            # Always run Binoculars for ZH as a safety net.
            if (
                lang != "zh"
                and stat_result is not None
                and self._aggregator.agree(stage_results)
            ):
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
        # For ZH, apply the lower decision threshold to all exit paths
        # (not just the arbitration block) — see DETECTOR_NOTES_2026-06.md
        dt = ZH_DECISION_THRESHOLD if lang == "zh" else 0.5
        return self._aggregator.combine(
            stage_results,
            detected_language=lang,
            processing_time_ms=elapsed,
            decision_threshold=dt,
        )

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_statistical(self, text: str, lang: str) -> tuple[dict | None, list[float] | None]:
        """Run Stage 1: statistical feature extraction + classification.

        Returns a ``(result, token_log_probs)`` tuple. ``result`` is the
        classifier output dict (or ``None`` if the stage is unavailable /
        fails). ``token_log_probs`` is the per-token log-prob list cached
        on the extractor (``extractor._last_token_log_probs``) so the
        linguistic stage can reuse it for M5/M6 without a second LM
        forward pass. ``token_log_probs`` is ``None`` when extraction did
        not run or the extractor is a mock that doesn't expose the attr.
        """
        extractor = self.statistical_extractors.get(lang)
        classifier = self.statistical_classifiers.get(lang)
        if extractor is None or classifier is None:
            logger.debug("No statistical detector for language: %s", lang)
            return None, None

        try:
            if hasattr(extractor, "is_loaded") and hasattr(extractor, "load") and not extractor.is_loaded:
                extractor.load()
                if self.model_manager is not None:
                    self.model_manager.load(f"statistical-{lang}", extractor)
            features = extractor.extract(text)
            result = classifier.predict(features)
            result["features"] = features.to_dict()
            # Read the cached per-token log-probs. Defensive getattr in case
            # the extractor is a foreign/mock implementation.
            token_log_probs = getattr(extractor, "_last_token_log_probs", None)
            return result, token_log_probs
        except Exception:
            logger.warning("Statistical detection failed", exc_info=True)
            return None, None

    def _run_linguistic(
        self,
        text: str,
        lang: str,
        token_log_probs: list[float] | None,
    ) -> dict | None:
        """Run Stage 1b: linguistic-stylistic feature extraction + classification.

        Mirrors :meth:`_run_statistical` but for the CPU-only linguistic
        axis. ``token_log_probs`` (typically from Stage 1) is forwarded to
        the extractor so M5/M6 can be computed without a second LM forward
        pass; ``None`` is acceptable (M5/M6 will be NaN).

        Returns the classifier result dict with an added ``"features"``
        key, or ``None`` when the stage is unavailable / fails.
        """
        extractor = self.linguistic_extractors.get(lang)
        classifier = self.linguistic_classifiers.get(lang)
        if extractor is None or classifier is None:
            logger.debug("No linguistic detector for language: %s", lang)
            return None

        try:
            features = extractor.extract(text, lang=lang, token_log_probs=token_log_probs)
            result = classifier.predict(features)
            result["features"] = features.to_dict()
            return result
        except Exception:
            logger.warning("Linguistic detection failed", exc_info=True)
            return None

    def _run_encoder(self, text: str, lang: str) -> dict | None:
        """Run Stage 2: encoder-based classification."""
        classifier = self.encoder_classifiers.get(lang)
        if classifier is None:
            logger.debug("No encoder classifier for language: %s", lang)
            return None

        try:
            if hasattr(classifier, "is_loaded") and hasattr(classifier, "load") and not classifier.is_loaded:
                classifier.load()
                if self.model_manager is not None:
                    self.model_manager.load(f"encoder-{lang}", classifier)
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
            if hasattr(detector, "is_loaded") and hasattr(detector, "load") and not detector.is_loaded:
                detector.load()
                if self.model_manager is not None:
                    self.model_manager.load(f"binoculars-{lang}", detector)
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
