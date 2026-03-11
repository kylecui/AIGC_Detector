"""LoRA fine-tune encoder classifiers (DeBERTa-v3-large / chinese-roberta).

Steps:
  1. Load training config from configs/training.yaml
  2. Setup base model with LoRA adapters
  3. Train on train.jsonl, evaluate on val.jsonl
  4. Save LoRA adapters
  5. Evaluate on test set

Usage:
  uv run python scripts/train_encoder.py --lang en
  uv run python scripts/train_encoder.py --lang zh
  uv run python scripts/train_encoder.py --lang en --eval-only
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

from src.aigc_detector.config import settings
from src.aigc_detector.training.evaluator import Evaluator
from src.aigc_detector.training.trainer import (
    LoRATrainer,
    TextClassificationDataset,
    load_trainer_config,
)

console = Console()


def train_encoder(lang: str, config_path: str = "configs/training.yaml") -> None:
    """Train an encoder classifier for the given language."""
    console.rule(f"[bold]Train Encoder Classifier ({lang.upper()})")

    # Load config
    config = load_trainer_config(config_path=config_path, language=lang)
    console.print(f"[dim]Base model: {config.base_model}[/]")
    console.print(f"[dim]Output dir: {config.output_dir}[/]")
    console.print(f"[dim]LoRA r={config.lora_r}, alpha={config.lora_alpha}[/]")
    console.print(f"[dim]Target modules: {config.target_modules}[/]")

    # Check if already trained
    output_dir = Path(config.output_dir)
    adapter_config = output_dir / "adapter_config.json"
    if adapter_config.exists():
        console.print(f"[yellow]Adapter already exists at {output_dir}. Delete to retrain.[/]")
        return

    # Check dataset exists
    train_path = Path(config.train_path)
    val_path = Path(config.val_path)
    if not train_path.exists():
        console.print(f"[red]{train_path} not found. Run generate_dataset.py first.[/]")
        return
    if not val_path.exists():
        console.print(f"[red]{val_path} not found. Run generate_dataset.py first.[/]")
        return

    # Count samples
    def count_lang_samples(path: Path, target_lang: str, label_map: dict[str, int]) -> dict[str, int]:
        counts: dict[str, int] = {"total": 0}
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("lang") == target_lang and record.get("label") in label_map:
                    label = record["label"]
                    counts[label] = counts.get(label, 0) + 1
                    counts["total"] += 1
        return counts

    train_counts = count_lang_samples(train_path, lang, config.label_map)
    val_counts = count_lang_samples(val_path, lang, config.label_map)
    console.print(f"[dim]Train samples ({lang}): {train_counts}[/]")
    console.print(f"[dim]Val samples ({lang}): {val_counts}[/]")

    if train_counts["total"] == 0:
        console.print(f"[red]No {lang} training samples found. Cannot train.[/]")
        return

    # Filter dataset to target language before training
    # We need to create language-filtered JSONL files
    filtered_train = settings.dataset_dir / "processed" / f"train_{lang}.jsonl"
    filtered_val = settings.dataset_dir / "processed" / f"val_{lang}.jsonl"

    for src, dst in [(train_path, filtered_train), (val_path, filtered_val)]:
        if not dst.exists():
            with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                for line in fin:
                    record = json.loads(line)
                    if record.get("lang") == lang and record.get("label") in config.label_map:
                        fout.write(line)

    # Update config paths to filtered files
    config.train_path = str(filtered_train)
    config.val_path = str(filtered_val)

    # Train
    trainer = LoRATrainer(config=config, device=settings.device)
    try:
        trainer.setup()
        metrics = trainer.train()
        console.print("[bold green]Training complete.[/]")
        console.print(f"[dim]Metrics: {metrics}[/]")
    finally:
        trainer.cleanup()


def evaluate_encoder(lang: str, config_path: str = "configs/training.yaml") -> None:
    """Evaluate a trained encoder on the test set."""
    console.rule(f"[bold]Evaluate Encoder ({lang.upper()})")

    config = load_trainer_config(config_path=config_path, language=lang)
    output_dir = Path(config.output_dir)

    adapter_config = output_dir / "adapter_config.json"
    if not adapter_config.exists():
        console.print(f"[red]No trained adapter found at {output_dir}. Train first.[/]")
        return

    test_path = settings.dataset_dir / "processed" / "test.jsonl"
    if not test_path.exists():
        console.print(f"[red]{test_path} not found.[/]")
        return

    # Filter test set by language
    filtered_test = settings.dataset_dir / "processed" / f"test_{lang}.jsonl"
    if not filtered_test.exists():
        with open(test_path, encoding="utf-8") as fin, open(filtered_test, "w", encoding="utf-8") as fout:
            for line in fin:
                record = json.loads(line)
                if record.get("lang") == lang and record.get("label") in config.label_map:
                    fout.write(line)

    # Load model with LoRA adapter
    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoTokenizer

    console.print(f"[bold blue]Loading model with LoRA adapter from {output_dir}...[/]")
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

    # Load test data
    dataset = TextClassificationDataset(
        path=filtered_test,
        tokenizer=tokenizer,
        max_length=config.max_length,
        text_key=config.text_key,
        label_key=config.label_key,
        label_map=config.label_map,
    )

    console.print(f"[dim]Test samples: {len(dataset)}[/]")

    # Run inference
    inv_label_map = {v: k for k, v in config.label_map.items()}
    y_true: list[str] = []
    y_pred: list[str] = []
    y_prob: list[float] = []

    console.print("[bold blue]Running inference on test set...[/]")
    from torch.utils.data import DataLoader

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

    # Evaluate
    evaluator = Evaluator(label_names=["human", "ai"], pos_label="ai")
    metrics = evaluator.evaluate(y_true, y_pred, y_prob=np.array(y_prob))
    evaluator.print_report(metrics)

    # Save report
    eval_path = output_dir / "eval_test.json"
    evaluator.save_report(metrics, eval_path)
    console.print(f"[green]Test evaluation saved to {eval_path}[/]")

    # Cleanup
    del model
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune encoder classifiers for AIGC detection",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        required=True,
        help="Language to train/evaluate (en or zh)",
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Training config path (default: configs/training.yaml)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate (adapter must already exist)",
    )

    args = parser.parse_args()

    if args.eval_only:
        evaluate_encoder(args.lang, config_path=args.config)
    else:
        train_encoder(args.lang, config_path=args.config)
        evaluate_encoder(args.lang, config_path=args.config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
