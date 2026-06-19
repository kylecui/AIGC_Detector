"""Detection subpackage — public API."""

from aigc_detector.detection.binoculars import BinocularsDetector, BinocularsResult
from aigc_detector.detection.encoder import EncoderClassifier, EncoderResult
from aigc_detector.detection.ensemble import EnsembleAggregator, EnsembleResult
from aigc_detector.detection.language import LanguageResult, LanguageRouter
from aigc_detector.detection.linguistic import (
    LinguisticClassifier,
    LinguisticDiagnostics,
    LinguisticFeatureExtractor,
    LinguisticFeatures,
)
from aigc_detector.detection.statistical import (
    StatisticalClassifier,
    StatisticalFeatureExtractor,
    StatisticalFeatures,
)

__all__ = [
    "BinocularsDetector",
    "BinocularsResult",
    "EncoderClassifier",
    "EncoderResult",
    "EnsembleAggregator",
    "EnsembleResult",
    "LanguageResult",
    "LanguageRouter",
    "LinguisticClassifier",
    "LinguisticDiagnostics",
    "LinguisticFeatureExtractor",
    "LinguisticFeatures",
    "StatisticalClassifier",
    "StatisticalFeatureExtractor",
    "StatisticalFeatures",
]
