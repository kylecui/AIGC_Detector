"""Stage contract for the detection framework (v0.3).

A stage is any object satisfying this Protocol — the framework formalizes
the duck-typing the four built-in stages already follow (audit:
load/unload/is_loaded present on all; outputs normalized to
{label, p_ai, confidence} inside the pipeline).

Two integration roles exist:

* ENSEMBLE stages (the four built-ins) participate in the weighted vote.
  Adding a new ensemble stage requires pipeline+weights work — not covered
  by the contract alone; see docs/stage-contract.md "Roadmap".
* DIAGNOSTIC stages (third-party extension point) do NOT vote: their
  results are appended to ``EnsembleResult.breakdown`` under
  ``diagnostic_<id>`` as additional auditable evidence. This is the
  zero-core-change integration path demonstrated by
  examples/stages/ttr_stage.py.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StageProtocol(Protocol):
    """The stage contract: lifecycle + normalized prediction."""

    stage_id: str

    def load(self) -> None:
        """Prepare resources (model weights, tables). Idempotent."""
        ...

    def unload(self) -> None:
        """Release resources. Safe to call when not loaded."""
        ...

    @property
    def is_loaded(self) -> bool:
        """True when ready to predict."""
        ...

    def predict(self, text: str, language: str | None = None) -> dict[str, Any]:
        """Score one text.

        Returns a dict with at least:
          p_ai: float in [0,1]   — stage's probability the text is AI-generated
          label: str             — "AI-generated" | "Human-written" (stage-local threshold)
          confidence: float      — stage's confidence in its own label
        Optional keys (recommended, surfaced as evidence):
          evidence: dict         — e.g. raw features/metrics behind p_ai
          model: str             — identifier of the underlying model/table
        Must not raise on any input; return a neutral result on failure
        (p_ai=0.5) so a broken stage degrades evidence, never the verdict.
        """
        ...


def neutral_result(stage_id: str, reason: str = "error") -> dict[str, Any]:
    """The mandated failure-degradation result."""
    return {"p_ai": 0.5, "label": "Human-written", "confidence": 0.0,
            "evidence": {"note": f"stage {stage_id} returned neutral: {reason}"}}
