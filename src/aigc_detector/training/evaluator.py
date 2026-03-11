"""Evaluation framework for AI text detection.

Computes classification metrics (ROC-AUC, F1, Precision, Recall, Accuracy)
and produces human-readable reports.  Works with both binary
(human vs AI) and ternary (human / ai / mixed) classification.

References:
- DEVPLAN.md task 2.4
- DESIGN.md §4 detection engine
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float | None = None
    confusion: list[list[int]] = field(default_factory=list)
    classification_report_str: str = ""
    n_samples: int = 0
    label_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """Compute and report detection evaluation metrics.

    Parameters
    ----------
    label_names : list[str]
        Ordered class names, e.g. ``["human", "ai"]`` for binary or
        ``["human", "ai", "mixed"]`` for ternary.
    pos_label : str
        The positive class for binary metrics (default ``"ai"``).
    """

    def __init__(
        self,
        label_names: Sequence[str] = ("human", "ai"),
        pos_label: str = "ai",
    ):
        self.label_names = list(label_names)
        self.pos_label = pos_label
        self._pos_index = self.label_names.index(pos_label) if pos_label in self.label_names else 1

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: Sequence[str] | np.ndarray,
        y_pred: Sequence[str] | np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> EvalMetrics:
        """Compute metrics from ground-truth and predicted labels.

        Parameters
        ----------
        y_true : array-like of str
            Ground-truth labels (e.g. ``"human"`` / ``"ai"``).
        y_pred : array-like of str
            Predicted labels.
        y_prob : array-like, optional
            Predicted probabilities for the positive class (required for
            ROC-AUC).

        Returns
        -------
        EvalMetrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)

        is_binary = len(self.label_names) == 2
        average = "binary" if is_binary else "macro"
        pos = self.pos_label if is_binary else None

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, pos_label=pos, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, pos_label=pos, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, pos_label=pos, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=self.label_names).tolist()

        report_str = classification_report(
            y_true,
            y_pred,
            labels=self.label_names,
            target_names=self.label_names,
            zero_division=0,
        )

        # ROC-AUC (binary only, requires probabilities)
        roc_auc = None
        if y_prob is not None and is_binary:
            try:
                roc_auc = roc_auc_score((y_true == self.pos_label).astype(int), y_prob)
            except ValueError:
                logger.warning("ROC-AUC computation failed (single class in y_true?)")

        return EvalMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            confusion=cm,
            classification_report_str=report_str,
            n_samples=n,
            label_names=self.label_names,
        )

    # ------------------------------------------------------------------
    # ROC curve helper
    # ------------------------------------------------------------------

    def roc_curve(
        self,
        y_true: Sequence[str] | np.ndarray,
        y_prob: np.ndarray,
    ) -> dict:
        """Compute ROC curve data points.

        Returns dict with ``fpr``, ``tpr``, ``thresholds`` as lists.
        """
        y_binary = (np.asarray(y_true) == self.pos_label).astype(int)
        fpr, tpr, thresholds = roc_curve(y_binary, y_prob)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(metrics: EvalMetrics) -> None:
        """Pretty-print evaluation metrics to stdout."""
        print("=" * 60)
        print("  EVALUATION REPORT")
        print("=" * 60)
        print(f"  Samples : {metrics.n_samples}")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall  : {metrics.recall:.4f}")
        print(f"  F1      : {metrics.f1:.4f}")
        if metrics.roc_auc is not None:
            print(f"  ROC-AUC : {metrics.roc_auc:.4f}")
        print("-" * 60)
        print("  Confusion Matrix:")
        header = "        " + "  ".join(f"{name:>8s}" for name in metrics.label_names)
        print(header)
        for i, row in enumerate(metrics.confusion):
            row_str = "  ".join(f"{val:8d}" for val in row)
            print(f"  {metrics.label_names[i]:>6s}  {row_str}")
        print("-" * 60)
        print(metrics.classification_report_str)
        print("=" * 60)

    @staticmethod
    def save_report(metrics: EvalMetrics, path: str | Path) -> None:
        """Save evaluation metrics as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Evaluation report saved to %s", path)


# ======================================================================
# Convenience: evaluate predictions from JSONL
# ======================================================================


def evaluate_predictions_jsonl(
    predictions_path: str | Path,
    label_key: str = "label",
    pred_key: str = "predicted_label",
    prob_key: str | None = "p_ai",
    label_names: Sequence[str] = ("human", "ai"),
    pos_label: str = "ai",
) -> EvalMetrics:
    """Evaluate predictions stored in a JSONL file.

    Each line must have at least *label_key* (ground truth) and
    *pred_key* (predicted label).  Optionally *prob_key* for ROC-AUC.
    """
    predictions_path = Path(predictions_path)
    y_true: list[str] = []
    y_pred: list[str] = []
    y_prob: list[float] = []
    has_prob = False

    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            y_true.append(record[label_key])
            y_pred.append(record[pred_key])
            if prob_key and prob_key in record:
                y_prob.append(float(record[prob_key]))
                has_prob = True

    evaluator = Evaluator(label_names=label_names, pos_label=pos_label)
    prob_arr = np.array(y_prob) if has_prob and y_prob else None
    return evaluator.evaluate(y_true, y_pred, y_prob=prob_arr)
