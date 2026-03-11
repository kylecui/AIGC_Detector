"""Weighted ensemble aggregator for multi-stage detection results.

Combines scores from statistical, encoder, and binoculars detectors
using configurable weights.  Handles partial results (not all stages
may be invoked on every request due to early-exit logic).

References:
    - DESIGN.md §4.4 (weights: stat=0.2, encoder=0.5, bino=0.3)
    - DEVPLAN.md Phase 4 task 4.2
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# Default ensemble weights from DESIGN.md §4.4
DEFAULT_WEIGHTS: dict[str, float] = {
    "statistical": 0.2,
    "encoder": 0.5,
    "binoculars": 0.3,
}


@dataclass
class EnsembleResult:
    """Aggregated detection result from all stages."""

    predicted_label: str  # "AI-generated" or "Human-written"
    confidence: float
    p_ai: float
    detected_language: str
    stages_used: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class EnsembleAggregator:
    """Weighted ensemble combiner for detection stage outputs.

    Parameters
    ----------
    weights : dict[str, float] or None
        Weights per stage.  Defaults to ``DEFAULT_WEIGHTS``.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def combine(
        self,
        stage_results: dict[str, dict],
        detected_language: str = "en",
        processing_time_ms: float = 0.0,
    ) -> EnsembleResult:
        """Combine results from one or more detection stages.

        Parameters
        ----------
        stage_results : dict
            Mapping of stage name → result dict.  Each result dict must
            contain at least ``"p_ai"`` (float in [0, 1]).
        detected_language : str
            ISO-639 language code detected for the input.
        processing_time_ms : float
            Total processing time.

        Returns
        -------
        EnsembleResult
        """
        stages_used = list(stage_results.keys())

        if not stages_used:
            return EnsembleResult(
                predicted_label="Human-written",
                confidence=0.0,
                p_ai=0.0,
                detected_language=detected_language,
                stages_used=[],
                breakdown={},
                processing_time_ms=processing_time_ms,
            )

        p_ai = self._weighted_combine(stage_results)
        predicted_label = "AI-generated" if p_ai > 0.5 else "Human-written"
        confidence = max(p_ai, 1.0 - p_ai)

        return EnsembleResult(
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            p_ai=round(p_ai, 4),
            detected_language=detected_language,
            stages_used=stages_used,
            breakdown=stage_results,
            processing_time_ms=round(processing_time_ms, 1),
        )

    def _weighted_combine(self, stage_results: dict[str, dict]) -> float:
        """Compute weighted average of p_ai across available stages.

        Only stages present in ``stage_results`` contribute.  Weights are
        re-normalised to sum to 1.0 over the active stages.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for stage_name, result in stage_results.items():
            w = self.weights.get(stage_name, 0.0)
            p_ai = result.get("p_ai", 0.0)
            weighted_sum += w * p_ai
            total_weight += w

        if total_weight == 0.0:
            # Fallback: simple average
            p_values = [r.get("p_ai", 0.0) for r in stage_results.values()]
            return sum(p_values) / len(p_values) if p_values else 0.0

        return weighted_sum / total_weight

    @staticmethod
    def agree(stage_results: dict[str, dict]) -> bool:
        """Check whether all stages agree on the predicted label.

        Returns ``True`` if all stages predict the same binary outcome
        (all p_ai > 0.5 or all p_ai ≤ 0.5).
        """
        if len(stage_results) < 2:
            return True

        labels = []
        for result in stage_results.values():
            p_ai = result.get("p_ai", 0.0)
            labels.append(p_ai > 0.5)

        return len(set(labels)) == 1
