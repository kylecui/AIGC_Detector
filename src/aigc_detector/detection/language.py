"""Language detection router using xlm-roberta-base-language-detection.

Routes input text to the appropriate language-specific detection pipeline
(Chinese or English). Falls back to heuristic detection when the model
is unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Supported language codes mapped from model output labels
_SUPPORTED_LANGUAGES = {"zh", "en"}


@dataclass
class LanguageResult:
    """Result of language detection."""

    lang: str  # "zh" or "en"
    confidence: float  # 0.0–1.0
    method: str  # "model" or "heuristic"


class LanguageRouter:
    """Detect text language and route to the correct detection pipeline.

    Uses ``papluca/xlm-roberta-base-language-detection`` when available,
    with a fast heuristic fallback based on Unicode character ratios.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "papluca/xlm-roberta-base-language-detection",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the language detection model into memory."""
        if self._model is not None:
            return
        logger.info("Loading language detection model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        ).to(self.device)
        self._model.eval()
        logger.info("Language detection model loaded on %s", self.device)

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Language detection model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, text: str, max_length: int = 512) -> LanguageResult:
        """Detect the language of *text*.

        Tries the transformer model first; if it's not loaded or fails,
        falls back to the heuristic method.

        Returns ``"zh"`` or ``"en"`` (defaults to ``"en"`` for unsupported
        languages).
        """
        if self._model is not None:
            try:
                return self._detect_with_model(text, max_length)
            except Exception:
                logger.warning("Model-based detection failed, falling back to heuristic", exc_info=True)

        return self._detect_heuristic(text)

    def _detect_with_model(self, text: str, max_length: int) -> LanguageResult:
        """Run the xlm-roberta language detection model."""
        assert self._model is not None and self._tokenizer is not None
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        predicted_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_idx].item()
        predicted_label = self._model.config.id2label[predicted_idx]

        # Map model label to our supported set
        lang = self._map_label(predicted_label)
        return LanguageResult(lang=lang, confidence=confidence, method="model")

    @staticmethod
    def _map_label(label: str) -> str:
        """Map model output label to ``"zh"`` or ``"en"``.

        The xlm-roberta-language-detection model uses ISO-639 codes
        (``"zh"``, ``"en"``, ``"fr"``, …).  Unsupported languages
        default to ``"en"``.
        """
        label_lower = label.lower()
        if label_lower in ("zh", "chinese"):
            return "zh"
        if label_lower in ("en", "english"):
            return "en"
        # Unsupported language — default to English pipeline
        return "en"

    @staticmethod
    def _detect_heuristic(text: str) -> LanguageResult:
        """Fast heuristic: ratio of CJK characters to total alphanumeric."""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total_alpha = len(re.findall(r"[\w]", text))
        if total_alpha == 0:
            return LanguageResult(lang="en", confidence=0.5, method="heuristic")

        ratio = chinese_chars / total_alpha
        if ratio > 0.3:
            return LanguageResult(lang="zh", confidence=min(ratio + 0.3, 1.0), method="heuristic")
        return LanguageResult(lang="en", confidence=min(1.0 - ratio + 0.3, 1.0), method="heuristic")
