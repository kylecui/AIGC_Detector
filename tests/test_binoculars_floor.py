"""W15 tests: register-gated binoculars-floor OR-rule (candidate, default OFF).

Contract:
- artifact absent / enabled=false -> rule inert (no verdict change anywhere)
- formal register + floor enabled + binoculars >= cutoff -> verdict upgraded
  to AI-generated with provenance; p_ai/confidence raised accordingly
- binoculars below cutoff (incl. FN-1's 0.343) -> verdict untouched
- non-formal texts -> rule never fires
- early-exit coverage: when breakdown lacks binoculars, the pipeline is
  force-called (stubbed here) and its score decides
- fail-safe: pipeline errors leave the original verdict intact
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _apply_binoculars_floor  # noqa: E402
from aigc_detector.detection.register import binoculars_floor  # noqa: E402

CAVEAT = {"code": "formal_register_zh", "register_score": 14}


@dataclass
class StubResult:
    predicted_label: str = "Human-written"
    p_ai: float = 0.11
    confidence: float = 0.89
    detected_language: str = "zh"
    breakdown: dict = field(default_factory=dict)


class StubPipeline:
    def __init__(self, bino_p_ai: float | None = 0.7):
        self._bino = bino_p_ai
        self.calls = 0

    def _run_binoculars(self, text: str, lang: str) -> dict | None:
        self.calls += 1
        return {"p_ai": self._bino} if self._bino is not None else None


TEXT = "本公司郑重承诺：严格遵守相关法律法规。特此承诺。"


class TestFloorInert:
    def test_artifact_state_consistency(self):
        """The deployed artifact must parse and match its stated intent.

        Since the 2026-08-21 gate review the floor is DEPLOYED (enabled=true,
        cutoff 0.46). This pins the deployment state so any future silent
        flip (either direction) fails loudly here.
        """
        cfg = binoculars_floor()
        assert cfg is not None, (
            "floor artifact says enabled=true but loader returned None — "
            "artifact/loader drift")
        assert cfg["cutoff"] == 0.46
        assert cfg["register"] == "formal_zh"

    def test_no_caveat_never_fires(self, monkeypatch):
        monkeypatch.setattr(
            "aigc_detector.api.routes.binoculars_floor",
            lambda: {"cutoff": 0.46, "register": "formal_zh"},
        )
        r = StubResult(breakdown={"binoculars": {"p_ai": 0.9}})
        pipe = StubPipeline()
        assert _apply_binoculars_floor(r, None, TEXT, pipe) is None
        assert r.predicted_label == "Human-written"


class TestFloorFires:
    def test_upgrade_with_provenance(self, monkeypatch):
        monkeypatch.setattr(
            "aigc_detector.api.routes.binoculars_floor",
            lambda: {"cutoff": 0.46, "register": "formal_zh"},
        )
        r = StubResult(breakdown={"binoculars": {"p_ai": 0.62}})
        rule = _apply_binoculars_floor(r, CAVEAT, TEXT, StubPipeline())
        assert rule is not None
        assert rule["rule"] == "register_binoculars_floor"
        assert rule["cutoff"] == 0.46
        assert r.predicted_label == "AI-generated"
        assert r.p_ai >= 0.62
        # confidence of the flipped verdict = the flipping evidence itself,
        # NOT max with stale human-verdict confidence (gate-review fix)
        assert r.confidence == 0.62

    def test_forced_run_when_early_exited(self, monkeypatch):
        monkeypatch.setattr(
            "aigc_detector.api.routes.binoculars_floor",
            lambda: {"cutoff": 0.46, "register": "formal_zh"},
        )
        r = StubResult()  # no binoculars in breakdown (early exit)
        pipe = StubPipeline(bino_p_ai=0.55)
        rule = _apply_binoculars_floor(r, CAVEAT, TEXT, pipe)
        assert pipe.calls == 1
        assert rule is not None
        assert r.predicted_label == "AI-generated"

    def test_fn1_boundary_no_fire(self, monkeypatch):
        """FN-1 anchor: edited AI text sits at bino 0.343 — below every cutoff."""
        monkeypatch.setattr(
            "aigc_detector.api.routes.binoculars_floor",
            lambda: {"cutoff": 0.46, "register": "formal_zh"},
        )
        r = StubResult(breakdown={"binoculars": {"p_ai": 0.343}})
        assert _apply_binoculars_floor(r, CAVEAT, TEXT, StubPipeline()) is None
        assert r.predicted_label == "Human-written"

    def test_already_ai_no_rule_needed(self, monkeypatch):
        monkeypatch.setattr(
            "aigc_detector.api.routes.binoculars_floor",
            lambda: {"cutoff": 0.46, "register": "formal_zh"},
        )
        r = StubResult(predicted_label="AI-generated", p_ai=0.9, confidence=0.9,
                       breakdown={"binoculars": {"p_ai": 0.8}})
        assert _apply_binoculars_floor(r, CAVEAT, TEXT, StubPipeline()) is None


class TestFailSafe:
    def test_pipeline_error_leaves_verdict(self, monkeypatch):
        monkeypatch.setattr(
            "aigc_detector.api.routes.binoculars_floor",
            lambda: {"cutoff": 0.46, "register": "formal_zh"},
        )

        class Boom:
            def _run_binoculars(self, *a, **k):
                raise RuntimeError("vrsm")

        r = StubResult()
        assert _apply_binoculars_floor(r, CAVEAT, TEXT, Boom()) is None
        assert r.predicted_label == "Human-written"
