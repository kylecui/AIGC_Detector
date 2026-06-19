"""Train a linguistic classifier from pre-extracted features.

Reads a features JSONL (output of extract_linguistic_features.py), trains a
LinguisticClassifier, calibrates the decision threshold, evaluates on a
validation set, and saves the model.

Usage:
    uv run python scripts/train_linguistic.py \\
        --lang en \\
        --features dataset/processed/train.linguistic.jsonl \\
        --val dataset/processed/val.linguistic.jsonl \\
        --output models/linguistic-en/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from aigc_detector.detection.linguistic import LinguisticClassifier
from aigc_detector.training.calibration import ThresholdCalibrator
from aigc_detector.training.evaluator import Evaluator

logger = logging.getLogger(__name__)

# Binary labels only — "mixed" is dropped (counted in `n_dropped`).
_VALID_LABELS: frozenset[str] = frozenset({"human", "ai"})


def _load_features(
    path: Path,
    feature_names: list[str],
    features_key: str,
    label_key: str,
) -> tuple[np.ndarray, list[str], int]:
    """Load features JSONL into (X_array[n,14], y_str_list, n_dropped).

    Records whose label is not in {"human","ai"} (including "mixed",
    missing, or unknown) are skipped and counted in ``n_dropped``.
    Each accepted record must have a dict at ``record[features_key]``
    with values for every name in *feature_names*.
    """
    x_rows: list[list[float]] = []
    y_str: list[str] = []
    n_dropped = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON in %s line %d, skipping", path, line_no)
                n_dropped += 1
                continue

            label = record.get(label_key)
            if label not in _VALID_LABELS:
                n_dropped += 1
                continue

            feats_dict = record.get(features_key)
            if not isinstance(feats_dict, dict):
                logger.warning("Record %d in %s missing '%s'; skipping", line_no, path, features_key)
                n_dropped += 1
                continue

            try:
                row = [float(feats_dict[name]) for name in feature_names]
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Record %d in %s has malformed features (%s); skipping", line_no, path, e)
                n_dropped += 1
                continue

            x_rows.append(row)
            y_str.append(label)

    x_arr = np.asarray(x_rows, dtype=np.float64)
    return x_arr, y_str, n_dropped


def _evaluate_split(
    clf: LinguisticClassifier,
    x: np.ndarray,
    y_str: list[str],
    threshold: float,
    pos_label: str,
    out_report: Path,
    split_name: str,
) -> dict:
    """Predict with *clf* at *threshold*, evaluate, and save a metrics JSON.

    Returns the metrics dict.
    """
    proba = clf.predict_proba(x)[:, 1]
    y_pred = ["ai" if p >= threshold else "human" for p in proba]
    evaluator = Evaluator(label_names=["human", "ai"], pos_label=pos_label)
    metrics = evaluator.evaluate(y_str, y_pred, y_prob=proba)
    Evaluator.save_report(metrics, out_report)
    logger.info(
        "%s metrics: acc=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s n=%d",
        split_name,
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "n/a",
        metrics.n_samples,
    )
    return metrics.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a LinguisticClassifier from pre-extracted linguistic features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        required=True,
        help="Language tag (informational; stored alongside the saved model).",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Training features JSONL (output of extract_linguistic_features.py).",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Validation features JSONL (used for threshold calibration).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Test features JSONL (optional, evaluated at the calibrated threshold).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (created if missing).",
    )
    parser.add_argument("--label-key", default="label", help="Record key for the ground-truth label.")
    parser.add_argument(
        "--features-key",
        default="linguistic_features",
        help="Record key under which the 14-feature dict was stored during extraction.",
    )
    parser.add_argument("--pos-label", default="ai", help="Positive label (must be one of 'human'/'ai').")
    parser.add_argument(
        "--calibration-metric",
        choices=["f1", "accuracy"],
        default="f1",
        help="Metric maximised when choosing the decision threshold on the validation set.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.pos_label not in _VALID_LABELS:
        logger.error("--pos-label must be one of %s, got %r", sorted(_VALID_LABELS), args.pos_label)
        return 1

    feature_names = list(LinguisticClassifier.FEATURE_NAMES)
    if len(feature_names) != 14:
        logger.error("LinguisticClassifier.FEATURE_NAMES has %d entries, expected 14", len(feature_names))
        return 1

    # ---- Load training data ----
    x_train, y_train_str, train_dropped = _load_features(
        args.features,
        feature_names,
        features_key=args.features_key,
        label_key=args.label_key,
    )
    if x_train.size == 0:
        logger.error("No training samples loaded from %s", args.features)
        return 1
    y_train = np.asarray([1 if s == args.pos_label else 0 for s in y_train_str], dtype=np.int64)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    logger.info(
        "Train: %d samples (pos=%s=%d, neg=%d, dropped=%d) from %s",
        len(y_train), args.pos_label, n_pos, n_neg, train_dropped, args.features,
    )

    # ---- Train ----
    clf = LinguisticClassifier()
    train_stats = clf.fit(x_train, y_train)
    logger.info(
        "Trained: backend=%s train_accuracy=%.4f n_samples=%d",
        train_stats["backend"], train_stats["train_accuracy"], train_stats["n_samples"],
    )

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Calibration on validation set ----
    threshold = clf.threshold  # default 0.5
    if args.val is not None:
        if not args.val.exists():
            logger.warning("Validation file not found: %s — skipping calibration", args.val)
        else:
            x_val, y_val_str, val_dropped = _load_features(
                args.val,
                feature_names,
                features_key=args.features_key,
                label_key=args.label_key,
            )
            if x_val.size == 0:
                logger.warning("No validation samples loaded from %s — skipping calibration", args.val)
            else:
                logger.info(
                    "Val: %d samples (dropped=%d) from %s", len(y_val_str), val_dropped, args.val,
                )
                y_val_proba = clf.predict_proba(x_val)[:, 1]

                calibrator = ThresholdCalibrator(direction="higher_is_positive")
                if args.calibration_metric == "f1":
                    cal_result = calibrator.calibrate_f1(
                        np.asarray(y_val_str), y_val_proba, pos_label=args.pos_label,
                    )
                else:
                    cal_result = calibrator.calibrate_accuracy(
                        np.asarray(y_val_str), y_val_proba, pos_label=args.pos_label,
                    )
                threshold = cal_result.optimal_threshold
                clf.set_threshold(threshold)
                logger.info(
                    "Calibrated: threshold=%.6f %s=%.4f n=%d",
                    threshold, cal_result.metric_name, cal_result.metric_value, cal_result.n_samples,
                )
                ThresholdCalibrator.save_result(cal_result, output_dir / "calibration.json")

                # Validation metrics at the calibrated threshold.
                _evaluate_split(
                    clf,
                    x_val,
                    y_val_str,
                    threshold,
                    pos_label=args.pos_label,
                    out_report=output_dir / "metrics.json",
                    split_name="val",
                )
    else:
        logger.info("No --val provided; skipping calibration (threshold stays at default %.4f).", threshold)

    # ---- Persist classifier ----
    clf_path = output_dir / "classifier.joblib"
    clf.save(clf_path)
    logger.info("Classifier saved: %s", clf_path)

    # ---- Optional test evaluation ----
    if args.test is not None:
        if not args.test.exists():
            logger.warning("Test file not found: %s — skipping test metrics", args.test)
        else:
            x_test, y_test_str, test_dropped = _load_features(
                args.test,
                feature_names,
                features_key=args.features_key,
                label_key=args.label_key,
            )
            if x_test.size == 0:
                logger.warning("No test samples loaded from %s — skipping test metrics", args.test)
            else:
                logger.info(
                    "Test: %d samples (dropped=%d) from %s", len(y_test_str), test_dropped, args.test,
                )
                _evaluate_split(
                    clf,
                    x_test,
                    y_test_str,
                    threshold,
                    pos_label=args.pos_label,
                    out_report=output_dir / "metrics_test.json",
                    split_name="test",
                )

    logger.info(
        "Summary: lang=%s threshold=%.6f train_acc=%.4f output=%s",
        args.lang, threshold, train_stats["train_accuracy"], output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
