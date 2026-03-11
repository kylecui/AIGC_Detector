"""LoRA fine-tuning trainer for encoder-based AI text classifiers.

Wraps HuggingFace ``Trainer`` with LoRA (via ``peft``) to fine-tune
DeBERTa-v3-large (English) or chinese-roberta-wwm-ext-large (Chinese)
for binary classification (human vs AI).

Loads configuration from ``configs/training.yaml``.

References:
    - DESIGN.md §4.2 (LoRA config: r=16, alpha=32)
    - DEVPLAN.md Phase 3 tasks 3.2–3.6
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Parsed training configuration."""

    # Model
    base_model: str = "microsoft/deberta-v3-large"
    num_labels: int = 2
    output_dir: str = "models/encoder-en"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: ["query_proj", "value_proj"])

    # Training
    num_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    fp16: bool = False
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    seed: int = 42

    # Data
    train_path: str = "dataset/processed/train.jsonl"
    val_path: str = "dataset/processed/val.jsonl"
    text_key: str = "text"
    label_key: str = "label"
    label_map: dict[str, int] = field(default_factory=lambda: {"human": 0, "ai": 1})


def load_trainer_config(
    config_path: str = "configs/training.yaml",
    language: str = "en",
) -> TrainerConfig:
    """Load training configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the training YAML config.
    language : str
        ``"en"`` or ``"zh"`` — selects model-specific settings.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_cfg = raw.get("models", {}).get(language, {})
    lora_cfg = raw.get("lora", {})
    train_cfg = raw.get("training", {})
    data_cfg = raw.get("data", {})

    # Pick target modules based on model type
    if "deberta" in model_cfg.get("base_model", "").lower():
        target_modules = lora_cfg.get("target_modules_deberta", ["query_proj", "value_proj"])
    else:
        target_modules = lora_cfg.get("target_modules_roberta", ["query", "value"])

    return TrainerConfig(
        base_model=model_cfg.get("base_model", "microsoft/deberta-v3-large"),
        num_labels=model_cfg.get("num_labels", 2),
        output_dir=model_cfg.get("output_dir", f"models/encoder-{language}"),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        lora_bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        num_epochs=train_cfg.get("num_epochs", 3),
        batch_size=train_cfg.get("batch_size", 16),
        eval_batch_size=train_cfg.get("eval_batch_size", 32),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        max_length=train_cfg.get("max_length", 512),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        logging_steps=train_cfg.get("logging_steps", 50),
        eval_steps=train_cfg.get("eval_steps", 500),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "f1"),
        seed=train_cfg.get("seed", 42),
        train_path=data_cfg.get("train_path", "dataset/processed/train.jsonl"),
        val_path=data_cfg.get("val_path", "dataset/processed/val.jsonl"),
        text_key=data_cfg.get("text_key", "text"),
        label_key=data_cfg.get("label_key", "label"),
        label_map=data_cfg.get("label_map", {"human": 0, "ai": 1}),
    )


# ======================================================================
# Dataset class for JSONL files
# ======================================================================


class TextClassificationDataset(torch.utils.data.Dataset):
    """JSONL dataset for text classification.

    Each line in the JSONL file must have a text field and a label field.
    Labels are mapped to integers via ``label_map``.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        text_key: str = "text",
        label_key: str = "label",
        label_map: dict[str, int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.label_key = label_key
        self.label_map = label_map or {"human": 0, "ai": 1}

        self.texts: list[str] = []
        self.labels: list[int] = []
        self._load(path)

    def _load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                label_str = record[self.label_key]
                if label_str not in self.label_map:
                    continue  # skip labels not in map (e.g. "mixed")
                self.texts.append(record[self.text_key])
                self.labels.append(self.label_map[label_str])

        logger.info("Loaded %d samples from %s", len(self.texts), path)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
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


# ======================================================================
# Trainer wrapper
# ======================================================================


def compute_metrics(eval_pred) -> dict:
    """Compute accuracy and F1 for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    return {"accuracy": acc, "f1": f1}


class LoRATrainer:
    """LoRA fine-tuning trainer for encoder classifiers.

    Parameters
    ----------
    config : TrainerConfig
        Training configuration.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, config: TrainerConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def setup(self) -> None:
        """Initialize model, tokenizer, and LoRA configuration."""
        logger.info("Setting up LoRA trainer for %s", self.config.base_model)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Pick dtype based on precision config
        if self.config.bf16:
            model_dtype = torch.bfloat16
        elif self.config.fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=self.config.num_labels,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.SEQ_CLS,
            target_modules=self.config.target_modules,
        )

        self._model = get_peft_model(base_model, lora_config)
        trainable, total = self._model.get_nb_trainable_parameters()
        logger.info(
            "LoRA model ready: %d trainable / %d total parameters (%.2f%%)",
            trainable,
            total,
            100 * trainable / total,
        )

    def train(
        self,
        train_dataset: TextClassificationDataset | None = None,
        val_dataset: TextClassificationDataset | None = None,
    ) -> dict:
        """Run the training loop.

        Parameters
        ----------
        train_dataset : TextClassificationDataset, optional
            Training data. If ``None``, loads from config path.
        val_dataset : TextClassificationDataset, optional
            Validation data. If ``None``, loads from config path.

        Returns
        -------
        dict with training metrics.
        """
        if self._model is None or self._tokenizer is None:
            self.setup()

        assert self._tokenizer is not None
        assert self._model is not None

        if train_dataset is None:
            train_dataset = TextClassificationDataset(
                self.config.train_path,
                self._tokenizer,
                max_length=self.config.max_length,
                text_key=self.config.text_key,
                label_key=self.config.label_key,
                label_map=self.config.label_map,
            )
        if val_dataset is None:
            val_dataset = TextClassificationDataset(
                self.config.val_path,
                self._tokenizer,
                max_length=self.config.max_length,
                text_key=self.config.text_key,
                label_key=self.config.label_key,
                label_map=self.config.label_map,
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=True,
            seed=self.config.seed,
            report_to="none",
            remove_unused_columns=False,
        )

        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting training: %d epochs, batch_size=%d", self.config.num_epochs, self.config.batch_size)
        train_result = self._trainer.train()

        # Save LoRA adapters
        self.save_adapter(output_dir)

        metrics = train_result.metrics
        logger.info("Training complete: %s", metrics)
        return metrics

    def evaluate(self, dataset: TextClassificationDataset | None = None) -> dict:
        """Evaluate the model on a dataset.

        Returns
        -------
        dict with evaluation metrics (accuracy, f1).
        """
        if self._trainer is None:
            raise RuntimeError("Trainer not initialised. Call .train() first.")

        if dataset is not None:
            eval_result = self._trainer.evaluate(eval_dataset=dataset)
        else:
            eval_result = self._trainer.evaluate()

        logger.info("Evaluation: %s", eval_result)
        return eval_result

    def save_adapter(self, path: str | Path) -> None:
        """Save the LoRA adapter weights.

        Only saves the adapter parameters, not the full base model.
        """
        if self._model is None:
            raise RuntimeError("No model to save.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        logger.info("LoRA adapter saved to %s", path)

    def cleanup(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            del self._trainer
            self._model = None
            self._tokenizer = None
            self._trainer = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Trainer resources released")
