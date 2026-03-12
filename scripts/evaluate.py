"""Run full evaluation across all trained detectors on the test set.

Evaluates:
  1. Statistical classifier (per-language: EN with GPT-2-XL, ZH with Wenzhong-GPT2-110M)
  2. Encoder classifier (per-language: EN with DeBERTa, ZH with RoBERTa)
  3. Combined: ensemble of statistical + encoder (mimics pipeline without Binoculars)

Produces:
  - Per-detector evaluation reports (JSON)
  - Summary comparison table

Usage:
  uv run python scripts/evaluate.py
  uv run python scripts/evaluate.py --lang en
  uv run python scripts/evaluate.py --detectors statistical encoder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.aigc_detector.config import settings
from src.aigc_detector.training.evaluator import Evaluator

console = Console()

PROCESSED_DIR = settings.dataset_dir / "processed"
REPORTS_DIR = settings.model_dir / "eval_reports"


def _load_test_records(lang: str) -> list[dict]:
    """Load test records for a specific language."""
    test_path = PROCESSED_DIR / "test.jsonl"
    if not test_path.exists():
        console.print(f"[red]Test set not found: {test_path}[/]")
        return []

    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("lang") == lang and record.get("label") in ("human", "ai"):
                records.append(record)
    return records


def evaluate_statistical(lang: str) -> dict | None:
    """Evaluate the statistical classifier on test set."""
    console.rule(f"[bold]Statistical Classifier ({lang.upper()})")

    from src.aigc_detector.detection.statistical import (
        StatisticalClassifier,
        StatisticalFeatureExtractor,
    )

    # Check if classifier exists
    clf_path = settings.model_dir / f"statistical-{lang}" / "classifier.joblib"
    if not clf_path.exists():
        console.print(f"[yellow]No statistical classifier found at {clf_path}. Skipping.[/]")
        return None

    # Load classifier
    clf = StatisticalClassifier()
    clf.load(clf_path)

    # Check if features already extracted for test set
    features_path = PROCESSED_DIR / f"test_features_{lang}.jsonl"
    if not features_path.exists():
        console.print(f"[yellow]No pre-extracted features at {features_path}. Extracting now...[/]")

        # Need to extract features first
        ref_models = {
            "en": {"model": "openai-community/gpt2-xl", "load_in_4bit": False},
            "zh": {"model": "IDEA-CCNL/Wenzhong-GPT2-110M", "load_in_4bit": False},
        }
        model_cfg = ref_models[lang]

        extractor = StatisticalFeatureExtractor(
            model_name=model_cfg["model"],
            device=settings.device,
            load_in_4bit=model_cfg["load_in_4bit"],
        )
        extractor.load()

        records = _load_test_records(lang)
        console.print(f"[dim]Test records: {len(records)}[/]")

        # Extract and save
        features_path.parent.mkdir(parents=True, exist_ok=True)
        with open(features_path, "w", encoding="utf-8") as f:
            for record in records:
                try:
                    feats = extractor.extract(record["text"])
                    record["features"] = feats.to_dict()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    console.print(f"[red]Feature extraction error: {e}[/]")

        extractor.unload()

    # Load features and predict
    y_true: list[str] = []
    y_pred: list[str] = []
    y_prob: list[float] = []

    with open(features_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("label") not in ("human", "ai"):
                continue

            feat = record["features"]
            features_arr = np.array(
                [
                    feat["perplexity"],
                    feat["avg_entropy"],
                    feat["std_entropy"],
                    feat["burstiness"],
                    feat["max_entropy"],
                    feat["min_entropy"],
                ]
            ).reshape(1, -1)

            proba = clf.predict_proba(features_arr)
            p_ai = float(proba[0, 1])

            y_true.append(record["label"])
            y_pred.append("ai" if p_ai > 0.5 else "human")
            y_prob.append(p_ai)

    if not y_true:
        console.print("[red]No test samples found.[/]")
        return None

    evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
    metrics = evaluator.evaluate(y_true, y_pred, y_prob=np.array(y_prob))
    evaluator.print_report(metrics)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"statistical_{lang}_test.json"
    evaluator.save_report(metrics, report_path)

    return {
        "detector": f"statistical-{lang}",
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "n_samples": metrics.n_samples,
    }


def evaluate_encoder(lang: str) -> dict | None:
    """Evaluate the encoder classifier on test set."""
    console.rule(f"[bold]Encoder Classifier ({lang.upper()})")

    from src.aigc_detector.training.trainer import (
        TextClassificationDataset,
        load_trainer_config,
    )

    config = load_trainer_config(language=lang)
    output_dir = Path(config.output_dir)
    adapter_config = output_dir / "adapter_config.json"

    if not adapter_config.exists():
        console.print(f"[yellow]No encoder adapter found at {output_dir}. Skipping.[/]")
        return None

    # Filter test set
    filtered_test = PROCESSED_DIR / f"test_{lang}.jsonl"
    if not filtered_test.exists():
        test_path = PROCESSED_DIR / "test.jsonl"
        with open(test_path, encoding="utf-8") as fin, open(filtered_test, "w", encoding="utf-8") as fout:
            for line in fin:
                record = json.loads(line)
                if record.get("lang") == lang and record.get("label") in config.label_map:
                    fout.write(line)

    # Load model
    from peft import AutoPeftModelForSequenceClassification
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    console.print(f"[bold blue]Loading encoder + LoRA from {output_dir}...[/]")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForSequenceClassification.from_pretrained(
        str(output_dir),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to(settings.device)
    model.eval()

    dataset = TextClassificationDataset(
        path=filtered_test,
        tokenizer=tokenizer,
        max_length=config.max_length,
        text_key=config.text_key,
        label_key=config.label_key,
        label_map=config.label_map,
    )

    console.print(f"[dim]Test samples: {len(dataset)}[/]")

    inv_label_map = {v: k for k, v in config.label_map.items()}
    y_true: list[str] = []
    y_pred: list[str] = []
    y_prob: list[float] = []

    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(settings.device)
            attention_mask = batch["attention_mask"].to(settings.device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            ai_probs = probs[:, 1].cpu().numpy()

            for i in range(len(labels)):
                y_true.append(inv_label_map[labels[i].item()])
                y_pred.append(inv_label_map[preds[i]])
                y_prob.append(float(ai_probs[i]))

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()

    if not y_true:
        console.print("[red]No test samples found.[/]")
        return None

    evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
    metrics = evaluator.evaluate(y_true, y_pred, y_prob=np.array(y_prob))
    evaluator.print_report(metrics)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"encoder_{lang}_test.json"
    evaluator.save_report(metrics, report_path)

    return {
        "detector": f"encoder-{lang}",
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "n_samples": metrics.n_samples,
    }


def print_comparison(results: list[dict]) -> None:
    """Print a comparison table of all detector results."""
    console.rule("[bold green]Comparison Summary")

    table = Table(title="Detector Evaluation Results")
    table.add_column("Detector", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1", style="magenta", no_wrap=True)
    table.add_column("ROC-AUC", style="yellow")
    table.add_column("N", style="dim")

    for r in results:
        table.add_row(
            r["detector"],
            f"{r['accuracy']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "-",
            str(r["n_samples"]),
        )

    console.print(table)

    # Save comparison JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Comparison saved to {comparison_path}[/]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained detectors on test set",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default=None,
        help="Evaluate for specific language only (default: both)",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=["statistical", "encoder"],
        help="Detectors to evaluate (default: statistical encoder)",
    )

    args = parser.parse_args()
    languages = [args.lang] if args.lang else ["en", "zh"]

    all_results: list[dict] = []

    for lang in languages:
        if "statistical" in args.detectors:
            result = evaluate_statistical(lang)
            if result:
                all_results.append(result)

        if "encoder" in args.detectors:
            result = evaluate_encoder(lang)
            if result:
                all_results.append(result)

    if all_results:
        print_comparison(all_results)
    else:
        console.print("[yellow]No detectors were evaluated. Train models first.[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
