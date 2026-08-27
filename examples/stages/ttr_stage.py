"""Example third-party diagnostic stage: TTR + sentence-length variance.

This file simulates an EXTERNAL contributor: it imports nothing from the
framework except the contract module's semantics (it doesn't even need to
import it — Protocol is structural). It demonstrates the zero-core-change
integration path declared in plans/default.yaml under diagnostic_stages.

Signal basis (weak, intentionally uncalibrated): AI text tends toward
lower lexical diversity (type-token ratio) and lower sentence-length
variance than human prose — classic stylometric folklore, true on average,
unreliable per-document. It is evidence, never a verdict: the mapping below
is a smooth heuristic, NOT a fitted model, and says so in every result.

Third-party stages live outside src/ on purpose (examples/). They must be
importable from the deployment environment (repo root or installed).
"""

from __future__ import annotations

import re
from typing import Any

_SENT_SPLIT = re.compile(r"[.!?。！？\n]+")


class TTRStage:
    """Diagnostic stage: type-token ratio + sentence-length CV -> weak p_ai."""

    stage_id = "ttr"

    def __init__(self) -> None:
        self._loaded = True  # pure computation, no resources

    # ---- lifecycle (contract) ----
    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ---- computation ----
    @staticmethod
    def _tokens(text: str, language: str | None) -> list[str]:
        if language == "zh":
            return [ch for ch in text if not ch.isspace()]
        return re.findall(r"[A-Za-z']+", text.lower())

    def predict(self, text: str, language: str | None = None) -> dict[str, Any]:
        try:
            if not self.is_loaded:
                self.load()
            tokens = self._tokens(text, language)
            n = len(tokens)
            if n < 40:
                # too short for stable stylometrics — neutral, honest
                return {"p_ai": 0.5, "label": "Human-written", "confidence": 0.0,
                        "evidence": {"note": f"too few tokens ({n} < 40)", "n_tokens": n}}
            ttr = len(set(tokens)) / n
            sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
            lens = [len(self._tokens(s, language)) for s in sents]
            lens = [x for x in lens if x > 0]
            if len(lens) >= 3 and sum(lens) > 0:
                mean = sum(lens) / len(lens)
                cv = (sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5 / mean
            else:
                cv = None
            # heuristic mapping (NOT fitted): typical human TTR ~0.55-0.75,
            # AI often lower; human sentence CV ~0.5+, AI often lower.
            p_ttr = max(0.0, min(1.0, (0.62 - ttr) * 2.5 + 0.5))
            p_cv = max(0.0, min(1.0, (0.45 - (cv or 0.5)) * 2.0 + 0.5)) if cv is not None else 0.5
            p_ai = round(0.5 * p_ttr + 0.5 * p_cv, 4)
            return {
                "p_ai": p_ai,
                "label": "AI-generated" if p_ai >= 0.5 else "Human-written",
                "confidence": round(abs(p_ai - 0.5) * 2, 4),
                "model": "heuristic-ttr-cv-v0 (uncalibrated example)",
                "evidence": {"ttr": round(ttr, 4), "sentence_cv": None if cv is None else round(cv, 4),
                             "n_tokens": n, "n_sentences": len(sents),
                             "warning": "uncalibrated diagnostic — evidence only, never a verdict"},
            }
        except Exception as e:  # noqa: BLE001 — contract: degrade, never raise
            return {"p_ai": 0.5, "label": "Human-written", "confidence": 0.0,
                    "evidence": {"note": f"stage error: {type(e).__name__}: {e}"}}
