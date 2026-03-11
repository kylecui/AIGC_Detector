"""Statistical feature extraction and classification for AI text detection.

Extracts per-token entropy and perplexity features from a reference language
model (GPT-2-XL for English, Qwen2.5-7B-Instruct for Chinese) and trains /
applies an XGBoost or Logistic Regression classifier on top.

Key features:
- perplexity, avg_entropy, std_entropy, burstiness, max_entropy, min_entropy

References:
- DESIGN.md §4.1 for the full specification
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "perplexity",
    "avg_entropy",
    "std_entropy",
    "burstiness",
    "max_entropy",
    "min_entropy",
]


@dataclass
class StatisticalFeatures:
    """Container for extracted statistical features."""

    perplexity: float
    avg_entropy: float
    std_entropy: float
    burstiness: float
    max_entropy: float
    min_entropy: float

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D numpy array in canonical order."""
        return np.array(
            [self.perplexity, self.avg_entropy, self.std_entropy, self.burstiness, self.max_entropy, self.min_entropy],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class StatisticalFeatureExtractor:
    """Extract perplexity and entropy features from text using a reference LM.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path for the reference LM.
    device : str
        ``"cuda"`` or ``"cpu"``.
    load_in_4bit : bool
        Use 4-bit quantisation (bitsandbytes). Required for large models on
        12 GB VRAM.
    max_length : int
        Maximum token length for input truncation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_length: int = 2048,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and tokenizer into memory."""
        if self._model is not None:
            return
        logger.info("Loading statistical reference model: %s (4bit=%s)", self.model_name, self.load_in_4bit)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict = {"trust_remote_code": True}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        # Resize embeddings to match tokenizer — prevents out-of-range token IDs
        # crashing CUDA with unrecoverable device-side assertion failures.
        # GPT-2 tokenizer can produce IDs beyond model's vocab_size for some
        # byte-level fallback tokens or added special tokens.
        if len(self._tokenizer) != self._model.config.vocab_size:
            logger.info(
                "Resizing embeddings: model vocab_size=%d, tokenizer len=%d",
                self._model.config.vocab_size,
                len(self._tokenizer),
            )
            self._model.resize_token_embeddings(len(self._tokenizer))

        # Cap max_length to the model's positional embedding size to prevent
        # CUDA assertion failures in nn.Embedding for position IDs.
        # GPT-2-XL has max_position_embeddings=1024 but our default is 2048.
        model_max_pos = getattr(self._model.config, "max_position_embeddings", None)
        if model_max_pos and self.max_length > model_max_pos:
            logger.info(
                "Capping max_length from %d to model's max_position_embeddings=%d",
                self.max_length,
                model_max_pos,
            )
            self.max_length = model_max_pos

        self._model.eval()
        logger.info("Statistical reference model loaded")

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass  # CUDA may be in unrecoverable state
            logger.info("Statistical reference model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract(self, text: str) -> StatisticalFeatures:
        """Extract statistical features from *text*.

        Raises ``RuntimeError`` if the model has not been loaded.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # Safety clamp: even after resize_token_embeddings, guard against any
        # edge-case where a token ID exceeds the embedding table size.
        try:
            vocab_size = self._model.config.vocab_size
            input_ids = inputs["input_ids"]
            if isinstance(vocab_size, int) and input_ids.max().item() >= vocab_size:
                inputs["input_ids"] = input_ids.clamp(max=vocab_size - 1)
        except (TypeError, AttributeError):
            pass  # skip clamp if model config is unavailable (e.g. in tests)

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction alignment
        # logits[:, :-1, :] predicts token at position 1..N
        # labels[:, 1:]     is the ground truth at position 1..N
        shifted_logits = logits[:, :-1, :].contiguous()
        target_ids = inputs["input_ids"][:, 1:].contiguous()

        # Per-token log-softmax for perplexity computation
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)

        # Per-token entropy: H = -sum(p * log(p))
        probs = torch.softmax(shifted_logits, dim=-1)
        token_entropies = -(probs * log_probs).sum(dim=-1).squeeze(0)

        # Guard against single-token inputs
        if token_entropies.numel() < 2:
            return StatisticalFeatures(
                perplexity=torch.exp(-token_log_probs.mean()).item(),
                avg_entropy=token_entropies.mean().item(),
                std_entropy=0.0,
                burstiness=0.0,
                max_entropy=token_entropies.max().item(),
                min_entropy=token_entropies.min().item(),
            )

        perplexity_val = torch.exp(-token_log_probs.mean()).item()
        avg_entropy = token_entropies.mean().item()
        std_entropy = token_entropies.std().item()
        max_entropy = token_entropies.max().item()
        min_entropy = token_entropies.min().item()
        burstiness = self._burstiness(token_entropies)

        return StatisticalFeatures(
            perplexity=perplexity_val,
            avg_entropy=avg_entropy,
            std_entropy=std_entropy,
            burstiness=burstiness,
            max_entropy=max_entropy,
            min_entropy=min_entropy,
        )

    def extract_batch(self, texts: list[str]) -> list[StatisticalFeatures]:
        """Extract features from multiple texts (sequential, no batched inference)."""
        return [self.extract(text) for text in texts]

    @staticmethod
    def _burstiness(entropies: torch.Tensor) -> float:
        """Burstiness = (std - mean) / (std + mean).

        Human text has higher burstiness (more entropy variance);
        AI text tends to be flatter.
        """
        std = entropies.std().item()
        mean = entropies.mean().item()
        return (std - mean) / (std + mean + 1e-8)


# ======================================================================
# Classifier wrapper
# ======================================================================


class StatisticalClassifier:
    """Lightweight classifier on top of statistical features.

    Supports XGBoost and Logistic Regression backends, wrapped in an
    sklearn ``Pipeline`` with ``StandardScaler``.

    Parameters
    ----------
    backend : str
        ``"xgboost"`` or ``"logistic_regression"``.
    """

    def __init__(self, backend: Literal["xgboost", "logistic_regression"] = "xgboost"):
        self.backend = backend
        self._pipeline: Pipeline | None = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        if self.backend == "xgboost":
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
            )
        else:
            clf = LogisticRegression(
                C=1.0,
                max_iter=10_000,
                penalty="l2",
                random_state=42,
            )
        self._pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray | list[StatisticalFeatures],
        labels: np.ndarray,
    ) -> dict:
        """Train the classifier.

        Parameters
        ----------
        features : array-like of shape (n_samples, 6) or list of StatisticalFeatures
        labels : array-like of shape (n_samples,)  — 0 = human, 1 = AI

        Returns
        -------
        dict with training stats.
        """
        x_train = self._ensure_array(features)
        self._pipeline.fit(x_train, labels)
        train_acc = self._pipeline.score(x_train, labels)
        logger.info("StatisticalClassifier trained: backend=%s, accuracy=%.4f", self.backend, train_acc)
        return {"backend": self.backend, "train_accuracy": train_acc, "n_samples": len(labels)}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray | StatisticalFeatures) -> dict:
        """Predict on a single sample or batch.

        Returns
        -------
        dict with ``label``, ``p_ai``, ``confidence``.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not trained. Call .fit() or .load() first.")

        x_arr = self._ensure_array(features)
        proba = self._pipeline.predict_proba(x_arr)  # (n, 2) — col 0 = human, col 1 = AI
        p_ai = float(proba[0, 1]) if x_arr.shape[0] == 1 else proba[:, 1].tolist()

        if x_arr.shape[0] == 1:
            label = "ai" if p_ai > 0.5 else "human"
            confidence = max(p_ai, 1.0 - p_ai)
            return {"label": label, "p_ai": p_ai, "confidence": confidence}

        # Batch
        labels = ["ai" if p > 0.5 else "human" for p in p_ai]
        confidences = [max(p, 1.0 - p) for p in p_ai]
        return {"labels": labels, "p_ai": p_ai, "confidences": confidences}

    def predict_proba(self, features: np.ndarray | list[StatisticalFeatures]) -> np.ndarray:
        """Return raw probability array (n_samples, 2)."""
        x_arr = self._ensure_array(features)
        return self._pipeline.predict_proba(x_arr)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self._pipeline, "backend": self.backend},
            path,
            compress=3,
        )
        logger.info("StatisticalClassifier saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a trained pipeline from disk."""
        path = Path(path)
        data = joblib.load(path)
        self._pipeline = data["pipeline"]
        self.backend = data["backend"]
        logger.info("StatisticalClassifier loaded from %s (backend=%s)", path, self.backend)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_array(features) -> np.ndarray:
        """Convert various input types to a 2-D numpy array."""
        if isinstance(features, StatisticalFeatures):
            return features.to_array().reshape(1, -1)
        if isinstance(features, list) and features and isinstance(features[0], StatisticalFeatures):
            return np.stack([f.to_array() for f in features])
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


# ======================================================================
# Convenience: extract features from JSONL and save
# ======================================================================


def extract_features_from_jsonl(
    extractor: StatisticalFeatureExtractor,
    input_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Extract features for all records in a JSONL file.

    Each output line is a JSON object with the original record fields plus
    a ``"features"`` dict containing the six statistical features.

    Supports **resume**: if *output_path* already has partial results, counting
    completed lines and skipping that many input records.

    Returns a summary dict.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count existing output lines for resume support
    already_done = 0
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            already_done = sum(1 for _ in f)
        if already_done > 0:
            logger.info("Resuming feature extraction: skipping %d already-done records", already_done)

    # Count total input lines for progress
    total = 0
    with open(input_path, encoding="utf-8") as f:
        total = sum(1 for _ in f)

    processed = already_done
    errors = 0
    consecutive_cuda_errors = 0
    max_consecutive_cuda_errors = 5  # abort if GPU is likely in unrecoverable state

    mode = "a" if already_done > 0 else "w"
    with open(input_path, encoding="utf-8") as fin, open(output_path, mode, encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            # Skip already-processed records on resume
            if i < already_done:
                continue

            record = json.loads(line)
            try:
                feats = extractor.extract(record["text"])
                record["features"] = feats.to_dict()
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                consecutive_cuda_errors = 0  # reset on success

                # Progress logging every 500 records
                if processed % 500 == 0:
                    logger.info("Feature extraction progress: %d / %d (errors=%d)", processed, total, errors)

            except RuntimeError as e:
                err_msg = str(e)
                if "CUDA" in err_msg:
                    consecutive_cuda_errors += 1
                    logger.warning(
                        "CUDA error on record %d (%s): %s",
                        i,
                        record.get("id", "?"),
                        err_msg[:120],
                    )
                    if consecutive_cuda_errors >= max_consecutive_cuda_errors:
                        logger.error(
                            "Aborting: %d consecutive CUDA errors — GPU likely in unrecoverable state. "
                            "Processed %d records before failure. Output is resumable.",
                            consecutive_cuda_errors,
                            processed,
                        )
                        break
                    # Try to recover GPU state
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    logger.warning("RuntimeError on record %d (%s): %s", i, record.get("id", "?"), err_msg[:120])
                errors += 1

            except Exception:
                logger.warning("Failed to extract features for record %d (%s)", i, record.get("id", "?"), exc_info=True)
                errors += 1

    logger.info("Feature extraction complete: processed=%d, errors=%d, total=%d", processed, errors, total)
    return {"processed": processed, "errors": errors, "output_path": str(output_path)}
