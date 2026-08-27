"""v0.3 stage-contract tests: Protocol conformance, wrapper injection, degradation."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.stages.contract import StageProtocol, neutral_result  # noqa: E402

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))  # examples/ importability (third-party layout)


def _load_ttr():
    from examples.stages.ttr_stage import TTRStage

    return TTRStage()


LONG_EN = (
    "The quick brown fox jumps over the lazy dog near the river bank every single "
    "morning. Nobody knows why. Some say it started years ago, before the bridge "
    "was even built, when the old mill still ground flour for the town below. "
    "Others tell a different story altogether, involving a bet, a borrowed coat, "
    "and one very determined animal."
)
LONG_ZH = (
    "这家店真的绝了，排队两小时但味道完全值得。豚骨拉面汤底浓郁到离谱，饺子也是满分。"
    "服务员看我们等太久还送了茶，好感度直接拉满。下次带朋友一起来，强烈推荐。"
    "不过工作日下午人稍微少一点，周末真的挤爆了。总之就是好吃，还会再来的。"
)


class TestContract:
    def test_ttr_stage_satisfies_protocol(self):
        assert isinstance(_load_ttr(), StageProtocol)

    def test_structural_no_framework_import_needed(self):
        """Third-party proof: the example imports nothing from aigc_detector."""
        src = (REPO / "examples/stages/ttr_stage.py").read_text(encoding="utf-8")
        assert "from aigc_detector" not in src and "import aigc_detector" not in src

    def test_predict_shape(self):
        out = _load_ttr().predict(LONG_EN, "en")
        for k in ("p_ai", "label", "confidence", "evidence"):
            assert k in out
        assert 0.0 <= out["p_ai"] <= 1.0

    def test_short_text_neutral(self):
        out = _load_ttr().predict("too short", "en")
        assert out["p_ai"] == 0.5 and out["confidence"] == 0.0

    def test_never_raises(self):
        s = _load_ttr()
        for bad in ("", "\x00\x01", "!!!", "a" * 100000):
            out = s.predict(bad, "zh")
            assert 0.0 <= out["p_ai"] <= 1.0

    def test_lifecycle(self):
        s = _load_ttr()
        assert s.is_loaded
        s.unload()
        assert not s.is_loaded
        s.predict(LONG_ZH, "zh")  # auto-loads on predict
        assert s.is_loaded


@dataclass
class _FakeResult:
    predicted_label: str = "Human-written"
    confidence: float = 0.9
    p_ai: float = 0.1
    detected_language: str = "en"
    stages_used: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)
    processing_time_ms: float = 1.0


class TestWrapper:
    def _wrapper(self):
        from aigc_detector.plan import _DiagnosticPipelineWrapper

        class Inner:
            def detect(self, text):
                return _FakeResult()

            binoculars_detectors = {"zh": object()}  # attr passthrough check

        return _DiagnosticPipelineWrapper(Inner(), {"ttr": _load_ttr()})

    def test_diagnostic_appended_verdict_unchanged(self):
        w = self._wrapper()
        r = w.detect(LONG_EN)
        assert r.predicted_label == "Human-written" and r.p_ai == 0.1
        assert "diagnostic_ttr" in r.breakdown
        assert r.breakdown["diagnostic_ttr"]["evidence"]["ttr"] > 0

    def test_inner_attr_passthrough(self):
        w = self._wrapper()
        assert "zh" in w.binoculars_detectors

    def test_broken_stage_degrades_not_raises(self):
        from aigc_detector.plan import _DiagnosticPipelineWrapper

        class Boom:
            stage_id = "boom"

            def predict(self, text, language=None):
                raise RuntimeError("kaboom")

        class Inner:
            def detect(self, text):
                return _FakeResult()

        w = _DiagnosticPipelineWrapper(Inner(), {"boom": Boom()})
        r = w.detect(LONG_EN)
        assert r.predicted_label == "Human-written"
        assert r.breakdown["diagnostic_boom"]["p_ai"] == 0.5

    def test_neutral_result_helper(self):
        nr = neutral_result("x", "test")
        assert nr["p_ai"] == 0.5 and "test" in nr["evidence"]["note"]
