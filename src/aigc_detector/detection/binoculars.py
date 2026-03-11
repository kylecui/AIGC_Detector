"""Binoculars zero-shot AI text detection.

Implements the Binoculars method from "Spotting LLMs With Binoculars:
Zero-Shot Detection of Machine-Generated Text" (arXiv:2401.12070).

Score formula:
    binoculars_score = PPL(performer) / X-PPL(observer, performer)

Where:
    PPL  = average per-token cross-entropy of performer logits vs true labels
    X-PPL = cross-entropy of performer logits vs observer soft predictions

Score convention: **low score = AI**, **high score = human**.
    score < threshold → AI-generated
    score ≥ threshold → human-written

References:
    - https://github.com/ahans30/Binoculars (official implementation)
    - DESIGN.md §4.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BinocularsResult:
    """Result of Binoculars detection on a single text."""

    score: float
    label: str  # "ai" or "human"
    threshold: float
    mode: str  # "accuracy" or "low-fpr"


class BinocularsDetector:
    """Binoculars zero-shot AI text detector.

    Uses an observer (base) and performer (instruct) model pair.

    Parameters
    ----------
    observer_name : str
        HuggingFace ID or path for the base (observer) model.
    performer_name : str
        HuggingFace ID or path for the instruct (performer) model.
    mode : str
        ``"accuracy"`` (F1-optimised) or ``"low-fpr"`` (0.01% FPR).
    device : str
        ``"cuda"`` or ``"cpu"``.
    load_in_4bit : bool
        Use 4-bit quantisation via bitsandbytes.
    max_length : int
        Maximum token length.
    """

    # Published thresholds from the original paper (Falcon-7B pair, bfloat16)
    ACCURACY_THRESHOLD = 0.9015310749276843
    FPR_THRESHOLD = 0.8536432310785527

    def __init__(
        self,
        observer_name: str = "tiiuae/falcon-7b",
        performer_name: str = "tiiuae/falcon-7b-instruct",
        mode: str = "low-fpr",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_length: int = 512,
    ):
        self.observer_name = observer_name
        self.performer_name = performer_name
        self.mode = mode
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length

        self.threshold = self.FPR_THRESHOLD if mode == "low-fpr" else self.ACCURACY_THRESHOLD

        self._observer = None
        self._performer = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load both observer and performer models into memory."""
        if self._observer is not None:
            return

        load_kwargs: dict = {"trust_remote_code": True}
        if self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"

        logger.info("Loading Binoculars observer: %s", self.observer_name)
        self._observer = AutoModelForCausalLM.from_pretrained(self.observer_name, **load_kwargs)
        self._observer.eval()

        logger.info("Loading Binoculars performer: %s", self.performer_name)
        self._performer = AutoModelForCausalLM.from_pretrained(self.performer_name, **load_kwargs)
        self._performer.eval()

        # Use observer tokenizer (same family, shared tokenizer)
        self._tokenizer = AutoTokenizer.from_pretrained(self.observer_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Binoculars models loaded (mode=%s, threshold=%.6f)", self.mode, self.threshold)

    def unload(self) -> None:
        """Release both models from memory."""
        del self._observer
        del self._performer
        del self._tokenizer
        self._observer = None
        self._performer = None
        self._tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Binoculars models unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._observer is not None and self._performer is not None

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_score(self, text: str) -> float:
        """Compute the Binoculars score for *text*.

        Returns a float where low values indicate AI-generated text.

        Raises ``RuntimeError`` if models are not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call .load() first.")

        assert self._tokenizer is not None
        assert self._observer is not None
        assert self._performer is not None

        encodings = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encodings = {k: v.to(self._observer.device) for k, v in encodings.items()}

        with torch.no_grad():
            observer_logits = self._observer(**encodings).logits
            performer_logits = self._performer(**encodings).logits

        ppl = self._perplexity(encodings, performer_logits)
        x_ppl = self._cross_perplexity(
            observer_logits,
            performer_logits,
            encodings,
            self._tokenizer.pad_token_id,
        )

        # Guard against division by zero
        if x_ppl < 1e-10:
            return float("inf")

        return ppl / x_ppl

    def predict(self, text: str) -> BinocularsResult:
        """Predict whether *text* is AI-generated or human-written."""
        score = self.compute_score(text)
        label = "ai" if score < self.threshold else "human"
        return BinocularsResult(
            score=score,
            label=label,
            threshold=self.threshold,
            mode=self.mode,
        )

    # ------------------------------------------------------------------
    # Metric helpers (mirrors the reference implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _perplexity(
        encodings: dict,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> float:
        """Average per-token cross-entropy of performer logits vs true labels.

        This is the standard language model perplexity (as CE, not exp).
        """
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encodings["input_ids"][..., 1:].contiguous()
        shifted_mask = encodings["attention_mask"][..., 1:].contiguous()

        ce = F.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_labels,
            reduction="none",
        )
        # Masked average
        ppl = (ce * shifted_mask).sum(1) / shifted_mask.sum(1)
        return ppl.cpu().float().item()

    @staticmethod
    def _cross_perplexity(
        observer_logits: torch.Tensor,
        performer_logits: torch.Tensor,
        encodings: dict,
        pad_token_id: int | None,
        temperature: float = 1.0,
    ) -> float:
        """Cross-entropy between observer (p) soft targets and performer (q) logits.

        H(p, q) where p = softmax(observer_logits), scored against performer logits.
        """
        vocab_size = observer_logits.shape[-1]
        total_tokens = performer_logits.shape[-2]

        p_scores = observer_logits / temperature
        q_scores = performer_logits / temperature

        p_proba = F.softmax(p_scores, dim=-1).view(-1, vocab_size)
        q_scores_flat = q_scores.view(-1, vocab_size)

        ce = F.cross_entropy(
            input=q_scores_flat,
            target=p_proba,
            reduction="none",
        ).view(-1, total_tokens)

        # Padding mask
        if pad_token_id is not None:
            padding_mask = (encodings["input_ids"] != pad_token_id).to(torch.uint8)
        else:
            padding_mask = torch.ones_like(encodings["input_ids"], dtype=torch.uint8)

        agg_ce = (ce * padding_mask).sum(1) / padding_mask.sum(1)
        return agg_ce.cpu().float().item()

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float, mode: str = "custom") -> None:
        """Override the detection threshold (e.g. after calibration)."""
        self.threshold = threshold
        self.mode = mode
        logger.info("Binoculars threshold updated: %.6f (mode=%s)", threshold, mode)
