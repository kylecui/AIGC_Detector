"""Threshold calibration for detection models.

Finds optimal decision thresholds on a validation set for:
- Accuracy / F1-score optimisation
- Target false-positive-rate (FPR) control

Works with any detector that produces a continuous score where lower
values indicate AI-generated text (e.g. Binoculars) or where higher
values indicate AI-generated text (e.g. statistical classifier p_ai).

References:
    - DEVPLAN.md task 2.9
    - DESIGN.md §4.3 (threshold calibration for Binoculars)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Outcome of threshold calibration."""

    optimal_threshold: float
    metric_name: str  # "f1", "accuracy", or "fpr"
    metric_value: float
    direction: str  # "lower_is_positive" or "higher_is_positive"
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


class ThresholdCalibrator:
    """Find optimal decision thresholds on a scored validation set.

    Parameters
    ----------
    direction : str
        ``"lower_is_positive"`` — scores below threshold are positive
        (AI-generated).  Used by Binoculars.

        ``"higher_is_positive"`` — scores above threshold are positive.
        Used by statistical classifier p_ai.
    """

    def __init__(self, direction: str = "lower_is_positive"):
        if direction not in ("lower_is_positive", "higher_is_positive"):
            raise ValueError(f"direction must be 'lower_is_positive' or 'higher_is_positive', got '{direction}'")
        self.direction = direction

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    def calibrate_f1(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        pos_label: str = "ai",
        n_thresholds: int = 500,
    ) -> CalibrationResult:
        """Find the threshold that maximises binary F1-score.

        Parameters
        ----------
        y_true : array of str
            Ground-truth labels.
        scores : array of float
            Continuous scores from the detector.
        pos_label : str
            The label considered positive.
        n_thresholds : int
            Number of candidate thresholds to evaluate.
        """
        y_binary = (np.asarray(y_true) == pos_label).astype(int)
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

        best_f1 = -1.0
        best_threshold = float(thresholds[0])

        for t in thresholds:
            if self.direction == "lower_is_positive":
                preds = (scores < t).astype(int)
            else:
                preds = (scores >= t).astype(int)

            f1 = f1_score(y_binary, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        logger.info("F1-calibration: threshold=%.6f, F1=%.4f", best_threshold, best_f1)
        return CalibrationResult(
            optimal_threshold=best_threshold,
            metric_name="f1",
            metric_value=best_f1,
            direction=self.direction,
            n_samples=len(y_true),
        )

    def calibrate_fpr(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        pos_label: str = "ai",
        target_fpr: float = 0.01,
    ) -> CalibrationResult:
        """Find the threshold that achieves a target false-positive rate.

        Parameters
        ----------
        target_fpr : float
            Desired maximum false-positive rate (e.g. 0.01 for 1%).
        """
        y_binary = (np.asarray(y_true) == pos_label).astype(int)

        # For ROC curve, higher score = more positive
        # If direction is lower_is_positive, we negate scores
        if self.direction == "lower_is_positive":
            roc_scores = -scores
        else:
            roc_scores = scores

        fpr, tpr, thresholds = roc_curve(y_binary, roc_scores)

        # Find the threshold where FPR ≤ target_fpr with maximum TPR
        valid = fpr <= target_fpr
        if not valid.any():
            # Fall back to the smallest FPR available
            idx = 0
        else:
            # Among valid thresholds, pick the one with highest TPR
            valid_indices = np.where(valid)[0]
            idx = valid_indices[np.argmax(tpr[valid_indices])]

        # Convert back from ROC threshold to detector threshold
        roc_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        if self.direction == "lower_is_positive":
            optimal_threshold = -float(roc_threshold)
        else:
            optimal_threshold = float(roc_threshold)

        achieved_fpr = float(fpr[idx])
        logger.info("FPR-calibration: threshold=%.6f, achieved_fpr=%.4f", optimal_threshold, achieved_fpr)
        return CalibrationResult(
            optimal_threshold=optimal_threshold,
            metric_name="fpr",
            metric_value=achieved_fpr,
            direction=self.direction,
            n_samples=len(y_true),
        )

    def calibrate_accuracy(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        pos_label: str = "ai",
        n_thresholds: int = 500,
    ) -> CalibrationResult:
        """Find the threshold that maximises accuracy."""
        y_binary = (np.asarray(y_true) == pos_label).astype(int)
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

        best_acc = -1.0
        best_threshold = float(thresholds[0])

        for t in thresholds:
            if self.direction == "lower_is_positive":
                preds = (scores < t).astype(int)
            else:
                preds = (scores >= t).astype(int)

            acc = (preds == y_binary).mean()
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(t)

        logger.info("Accuracy-calibration: threshold=%.6f, accuracy=%.4f", best_threshold, best_acc)
        return CalibrationResult(
            optimal_threshold=best_threshold,
            metric_name="accuracy",
            metric_value=best_acc,
            direction=self.direction,
            n_samples=len(y_true),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_result(result: CalibrationResult, path: str | Path) -> None:
        """Save calibration result as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Calibration result saved to %s", path)

    @staticmethod
    def load_result(path: str | Path) -> CalibrationResult:
        """Load calibration result from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return CalibrationResult(**data)
