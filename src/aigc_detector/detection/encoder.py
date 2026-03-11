"""Encoder-based AI text classifier using LoRA-finetuned models.

Loads a DeBERTa-v3-large (English) or chinese-roberta-wwm-ext-large (Chinese)
model with LoRA adapters for binary classification (human vs AI).

Supports:
- Inference with pre-trained LoRA adapters
- Load/unload lifecycle (lazy loading pattern)
- Language-specific model selection

References:
    - DESIGN.md §4.2
    - DEVPLAN.md Phase 3 tasks 3.1
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EncoderResult:
    """Result of encoder-based classification."""

    label: str  # "ai" or "human"
    p_ai: float  # probability of AI-generated
    confidence: float  # max(p_ai, 1 - p_ai)
    model_name: str

    def to_dict(self) -> dict:
        return asdict(self)


class EncoderClassifier:
    """LoRA-finetuned encoder classifier for AI text detection.

    Parameters
    ----------
    base_model_name : str
        HuggingFace ID for the base encoder model.
    adapter_path : str or Path or None
        Path to the saved LoRA adapter directory. If ``None``, uses the
        base model without adapters (useful for initial testing).
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_length : int
        Maximum token length for input truncation.
    num_labels : int
        Number of classification labels (default 2: human, ai).
    """

    # Label mapping: index → name
    LABEL_MAP = {0: "human", 1: "ai"}

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-large",
        adapter_path: str | Path | None = None,
        device: str = "cuda",
        max_length: int = 512,
        num_labels: int = 2,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.max_length = max_length
        self.num_labels = num_labels

        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the base model and LoRA adapters into memory."""
        if self._model is not None:
            return

        logger.info("Loading encoder model: %s", self.base_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=self.num_labels,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        if self.adapter_path and self.adapter_path.exists():
            logger.info("Loading LoRA adapters from: %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        else:
            if self.adapter_path:
                logger.warning(
                    "Adapter path %s not found, using base model without LoRA",
                    self.adapter_path,
                )
            self._model = base_model

        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("Encoder model loaded on %s", self.device)

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Encoder model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> EncoderResult:
        """Classify a single text as AI-generated or human-written.

        Returns an ``EncoderResult`` with the predicted label, probability,
        and confidence score.

        Raises ``RuntimeError`` if model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (1, num_labels)
            probs = torch.softmax(logits, dim=-1)

        p_ai = probs[0, 1].item()
        label = "ai" if p_ai > 0.5 else "human"
        confidence = max(p_ai, 1.0 - p_ai)

        return EncoderResult(
            label=label,
            p_ai=p_ai,
            confidence=confidence,
            model_name=self.base_model_name,
        )

    def predict_batch(self, texts: list[str]) -> list[EncoderResult]:
        """Classify a batch of texts.

        Processes texts one at a time to avoid VRAM spikes.
        """
        return [self.predict(text) for text in texts]

    def predict_proba(self, text: str) -> float:
        """Return the probability that *text* is AI-generated."""
        result = self.predict(text)
        return result.p_ai

    # ------------------------------------------------------------------
    # Threshold adjustment
    # ------------------------------------------------------------------

    def predict_with_threshold(self, text: str, threshold: float = 0.5) -> EncoderResult:
        """Predict with a custom decision threshold.

        Parameters
        ----------
        text : str
            Input text.
        threshold : float
            Decision boundary for p_ai. Values above → "ai".
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        p_ai = probs[0, 1].item()
        label = "ai" if p_ai > threshold else "human"
        confidence = max(p_ai, 1.0 - p_ai)

        return EncoderResult(
            label=label,
            p_ai=p_ai,
            confidence=confidence,
            model_name=self.base_model_name,
        )
