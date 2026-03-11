"""Train the statistical (perplexity/entropy) classifier.

Steps:
  1. Load a reference language model (GPT-2-XL for EN, Wenzhong-GPT2-110M for ZH)
  2. Extract statistical features from train/val/test JSONL
  3. Train XGBoost classifier on extracted features
  4. Evaluate on validation and test sets
  5. Calibrate optimal threshold on validation set
  6. Save classifier + calibration results

Usage:
  uv run python scripts/train_statistical.py --lang en
  uv run python scripts/train_statistical.py --lang zh
  uv run python scripts/train_statistical.py --lang en --extract-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.aigc_detector.config import settings
from src.aigc_detector.detection.statistical import (
    StatisticalClassifier,
    StatisticalFeatureExtractor,
    extract_features_from_jsonl,
)
from src.aigc_detector.training.calibration import ThresholdCalibrator
from src.aigc_detector.training.evaluator import Evaluator

console = Console()

# Reference model mapping per language
REFERENCE_MODELS = {
    "en": {"model": "openai-community/gpt2-xl", "load_in_4bit": False},
    "zh": {"model": "IDEA-CCNL/Wenzhong-GPT2-110M", "load_in_4bit": False},
}

# Dataset paths
PROCESSED_DIR = settings.dataset_dir / "processed"


def _features_path(split: str, lang: str) -> Path:
    """Path for feature-augmented JSONL."""
    return PROCESSED_DIR / f"{split}_features_{lang}.jsonl"


def _filter_by_lang(input_path: Path, output_path: Path, lang: str) -> int:
    """Filter a JSONL file to keep only records matching the target language.

    Returns the count of filtered records.
    """
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("lang") == lang:
                fout.write(line)
                count += 1
    return count


def _load_features_and_labels(
    features_path: Path,
    label_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features and labels from a feature-augmented JSONL file.

    Returns (features_array, labels_array, label_strings).
    """
    if label_map is None:
        label_map = {"human": 0, "ai": 1}

    features_list: list[list[float]] = []
    labels: list[int] = []
    label_strs: list[str] = []

    with open(features_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            label_str = record.get("label", "")
            if label_str not in label_map:
                continue
            feat = record["features"]
            features_list.append(
                [
                    feat["perplexity"],
                    feat["avg_entropy"],
                    feat["std_entropy"],
                    feat["burstiness"],
                    feat["max_entropy"],
                    feat["min_entropy"],
                ]
            )
            labels.append(label_map[label_str])
            label_strs.append(label_str)

    return np.array(features_list), np.array(labels), label_strs


def step_extract_features(lang: str) -> None:
    """Extract statistical features for train/val/test splits."""
    console.rule(f"[bold]Feature Extraction ({lang.upper()})")

    model_cfg = REFERENCE_MODELS.get(lang)
    if model_cfg is None:
        console.print(f"[red]No reference model configured for lang={lang}[/]")
        return

    extractor = StatisticalFeatureExtractor(
        model_name=model_cfg["model"],
        device=settings.device,
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    for split in ["train", "val", "test"]:
        input_path = PROCESSED_DIR / f"{split}.jsonl"
        if not input_path.exists():
            console.print(f"[red]{input_path} not found. Run generate_dataset.py first.[/]")
            return

        output_path = _features_path(split, lang)

        # Filter by language first
        filtered_path = PROCESSED_DIR / f"{split}_{lang}.jsonl"
        if not filtered_path.exists():
            count = _filter_by_lang(input_path, filtered_path, lang)
            console.print(f"[dim]Filtered {split} to {count} {lang} records.[/]")
        else:
            count = sum(1 for _ in open(filtered_path, encoding="utf-8"))
            console.print(f"[dim]Using existing {filtered_path.name} ({count} records).[/]")

        if count == 0:
            console.print(f"[yellow]No {lang} records in {split}, skipping.[/]")
            continue

        # Check if features are already fully extracted (resume-aware)
        if output_path.exists():
            done_count = sum(1 for _ in open(output_path, encoding="utf-8"))
            if done_count >= count:
                console.print(f"[yellow]{output_path.name} already complete ({done_count} records), skipping.[/]")
                continue
            console.print(f"[yellow]{output_path.name} is partial ({done_count}/{count}), resuming...[/]")

        # Load extractor (lazy)
        if not extractor.is_loaded:
            console.print(f"[bold blue]Loading reference model:[/] {model_cfg['model']}")
            extractor.load()

        # Extract features
        console.print(f"[bold blue]Extracting features for {split} ({lang})...[/]")
        stats = extract_features_from_jsonl(extractor, filtered_path, output_path)
        console.print(f"[green]Done:[/] processed={stats['processed']}, errors={stats['errors']}")

    # Unload model to free VRAM
    extractor.unload()
    console.print("[dim]Reference model unloaded.[/]")


def step_train_classifier(lang: str) -> None:
    """Train XGBoost statistical classifier."""
    console.rule(f"[bold]Train Statistical Classifier ({lang.upper()})")

    train_features_path = _features_path("train", lang)
    if not train_features_path.exists():
        console.print(f"[red]{train_features_path} not found. Run --extract-only first.[/]")
        return

    # Load training data
    x_train, y_train, _ = _load_features_and_labels(train_features_path)
    console.print(f"[dim]Train set: {len(y_train)} samples (human={sum(y_train == 0)}, ai={sum(y_train == 1)})[/]")

    # Train
    clf = StatisticalClassifier(backend="xgboost")
    stats = clf.fit(x_train, y_train)
    console.print(f"[green]Training accuracy: {stats['train_accuracy']:.4f}[/]")

    # Save classifier
    clf_path = settings.model_dir / f"statistical-{lang}" / "classifier.joblib"
    clf.save(clf_path)
    console.print(f"[green]Classifier saved to {clf_path}[/]")

    # Evaluate on validation set
    val_features_path = _features_path("val", lang)
    if val_features_path.exists():
        x_val, y_val, y_val_strs = _load_features_and_labels(val_features_path)

        # Get probabilities for calibration
        proba = clf.predict_proba(x_val)
        p_ai = proba[:, 1]

        # Predict
        y_pred_labels = ["ai" if p > 0.5 else "human" for p in p_ai]

        evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
        metrics = evaluator.evaluate(y_val_strs, y_pred_labels, y_prob=p_ai)
        evaluator.print_report(metrics)

        # Save evaluation report
        eval_path = settings.model_dir / f"statistical-{lang}" / "eval_val.json"
        evaluator.save_report(metrics, eval_path)

        # Calibrate threshold on validation set
        console.print("[bold blue]Calibrating threshold...[/]")
        calibrator = ThresholdCalibrator(direction="higher_is_positive")
        cal_result = calibrator.calibrate_f1(
            y_true=np.array(y_val_strs),
            scores=p_ai,
            pos_label="ai",
        )
        console.print(
            f"[green]Optimal threshold: {cal_result.optimal_threshold:.4f} (F1={cal_result.metric_value:.4f})[/]"
        )

        cal_path = settings.model_dir / f"statistical-{lang}" / "calibration.json"
        ThresholdCalibrator.save_result(cal_result, cal_path)

    # Evaluate on test set
    test_features_path = _features_path("test", lang)
    if test_features_path.exists():
        x_test, y_test, y_test_strs = _load_features_and_labels(test_features_path)
        proba_test = clf.predict_proba(x_test)
        p_ai_test = proba_test[:, 1]
        y_pred_test = ["ai" if p > 0.5 else "human" for p in p_ai_test]

        evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
        metrics_test = evaluator.evaluate(y_test_strs, y_pred_test, y_prob=p_ai_test)

        console.rule("[bold]Test Set Results")
        evaluator.print_report(metrics_test)

        eval_test_path = settings.model_dir / f"statistical-{lang}" / "eval_test.json"
        evaluator.save_report(metrics_test, eval_test_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train statistical (perplexity/entropy) classifier for AIGC detection",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        required=True,
        help="Language to train for (en or zh)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract features, don't train classifier",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train classifier (features must already exist)",
    )

    args = parser.parse_args()

    if args.train_only:
        step_train_classifier(args.lang)
    elif args.extract_only:
        step_extract_features(args.lang)
    else:
        step_extract_features(args.lang)
        step_train_classifier(args.lang)

    return 0


if __name__ == "__main__":
    sys.exit(main())
