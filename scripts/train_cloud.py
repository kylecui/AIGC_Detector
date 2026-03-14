"""Standalone LoRA fine-tuning script for cloud GPU training.

Trains DeBERTa-v3-large (EN) or chinese-roberta-wwm-ext-large (ZH)
with LoRA for binary AI text classification.

Usage:
  python train_cloud.py --lang en --data-dir /data/aigc/dataset --output-dir /data/aigc/models
  python train_cloud.py --lang zh --data-dir /data/aigc/dataset --output-dir /data/aigc/models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Configuration ----

MODEL_CONFIGS = {
    "en": {
        "base_model": "microsoft/deberta-v3-large",
        "local_path": "/root/.cache/modelscope/hub/microsoft/deberta-v3-large",
        "target_modules": ["query_proj", "value_proj"],
        "output_name": "encoder-en",
    },
    "zh": {
        "base_model": "hfl/chinese-roberta-wwm-ext-large",
        "local_path": "/root/.cache/modelscope/hub/hfl/chinese-roberta-wwm-ext-large",
        "target_modules": ["query", "value"],
        "output_name": "encoder-zh",
    },
}

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
NUM_EPOCHS = 3
BATCH_SIZE = 4  # Reduced from 16 to fit in V100 32GB (fp32 DeBERTa)
GRAD_ACCUM_STEPS = 4  # Effective batch = 4 * 4 = 16
EVAL_BATCH_SIZE = 8  # Reduced for eval too
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 512
SEED = 42
LABEL_MAP = {"human": 0, "ai": 1}


# ---- Dataset ----


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                label_str = record.get("label", "")
                if label_str not in LABEL_MAP:
                    continue
                self.texts.append(record["text"])
                self.labels.append(LABEL_MAP[label_str])

        logger.info("Loaded %d samples from %s", len(self.texts), path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---- Metrics ----


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    return {"accuracy": acc, "f1": f1}


# ---- Filter dataset by language ----


def filter_by_lang(input_path: Path, output_path: Path, lang: str) -> int:
    count = 0
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("lang") == lang and record.get("label") in LABEL_MAP:
                fout.write(line)
                count += 1
    return count


# ---- Main training ----


def train(lang: str, data_dir: str, output_dir: str):
    cfg = MODEL_CONFIGS[lang]
    data_path = Path(data_dir)
    out_path = Path(output_dir) / cfg["output_name"]

    # Check if already done
    if (out_path / "adapter_config.json").exists():
        logger.info("Adapter already exists at %s. Delete to retrain.", out_path)
        return

    # Filter data by language
    train_filtered = data_path / f"train_{lang}.jsonl"
    val_filtered = data_path / f"val_{lang}.jsonl"

    for split, filtered in [("train", train_filtered), ("val", val_filtered)]:
        if not filtered.exists():
            src = data_path / f"{split}.jsonl"
            if not src.exists():
                logger.error("%s not found!", src)
                sys.exit(1)
            count = filter_by_lang(src, filtered, lang)
            logger.info("Filtered %s to %d %s records", split, count, lang)
        else:
            count = sum(1 for _ in open(filtered, encoding="utf-8"))
            logger.info("Using existing %s (%d records)", filtered.name, count)

    # V100 does not support bf16 natively, and both EN/DeBERTa and ZH/RoBERTa
    # have shown AMP gradient-scaler instability in this environment.
    # Use pure fp32 training on cloud for reliability.
    model_dtype = torch.float32
    use_fp16 = False
    logger.info("Cloud training uses fp32 (fp16 disabled) to avoid AMP gradient issues on V100")

    # Use local model path if available (downloaded via modelscope)
    model_path = cfg.get("local_path", cfg["base_model"])
    if Path(model_path).exists():
        logger.info("Using local model path: %s", model_path)
    else:
        model_path = cfg["base_model"]
        logger.info("Local path not found, downloading from HuggingFace: %s", model_path)

    # Load tokenizer
    logger.info("Loading tokenizer: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model: %s (dtype=%s)", model_path, model_dtype)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=cfg["target_modules"],
    )
    # Enable gradient checkpointing to reduce VRAM usage
    base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total, 100 * trainable / total)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = TextClassificationDataset(train_filtered, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_filtered, tokenizer, MAX_LENGTH)

    # Training args
    out_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=use_fp16,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training: %d epochs, batch=%d, lr=%s", NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    logger.info("Training complete in %.1f minutes", elapsed / 60)
    logger.info("Train metrics: %s", train_result.metrics)

    # Save LoRA adapter
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # Rewrite adapter metadata to canonical HF model ID so the adapter remains
    # portable after downloading from cloud to local machine.
    adapter_config_path = out_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)
        adapter_config["base_model_name_or_path"] = cfg["base_model"]
        with open(adapter_config_path, "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

    logger.info("LoRA adapter saved to %s", out_path)

    # Evaluate on val
    eval_result = trainer.evaluate()
    logger.info("Val metrics: %s", eval_result)

    # Save eval results
    with open(out_path / "eval_results.json", "w") as f:
        json.dump(
            {
                "train_metrics": train_result.metrics,
                "val_metrics": eval_result,
                "training_time_minutes": elapsed / 60,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            f,
            indent=2,
        )

    # Cleanup checkpoints to save disk space (keep only final adapter)
    import shutil

    for ckpt in out_path.glob("checkpoint-*"):
        shutil.rmtree(ckpt)
        logger.info("Removed checkpoint: %s", ckpt)

    logger.info("Done! Adapter at: %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "zh"], required=True)
    parser.add_argument("--data-dir", default="/data/aigc/dataset")
    parser.add_argument("--output-dir", default="/data/aigc/models")
    args = parser.parse_args()
    train(args.lang, args.data_dir, args.output_dir)
