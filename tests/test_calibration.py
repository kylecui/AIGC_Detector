"""W11-2 tests: register-conditioned confidence calibration.

Contract (plan v3.1 修订1):
- non-formal texts: confidence unchanged (T=1) — well-calibrated regions never squashed
- formal register + deployed artifact: confidence compressed, provably
  label-flip-free and ranking-preserving
- artifact absent / applied=false: behavior identical to pre-W11 (safe default)
- calibration block returned for transparency when applied
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from aigc_detector.api.routes import _calibrate_confidence  # noqa: E402
from aigc_detector.detection.register import (  # noqa: E402
    FORMAL_ZH_CAVEAT,
    detect_register_zh,
    formal_temperature,
)

CAVEAT = {**FORMAL_ZH_CAVEAT, "register_score": 14, "register_markers": ["特此声明"]}


def _expected(c: float, temp: float) -> float:
    z = math.log(c / (1 - c))
    return 1 / (1 + math.exp(-z / temp))


class TestCalibrateConfidence:
    def test_no_caveat_unchanged(self):
        conf, cal = _calibrate_confidence(None, 0.8909, 0.1091)
        assert conf == 0.8909 and cal is None

    def test_no_artifact_unchanged(self, monkeypatch):
        monkeypatch.setattr("aigc_detector.api.routes.formal_temperature", lambda: None)
        conf, cal = _calibrate_confidence(CAVEAT, 0.8909, 0.1091)
        assert conf == 0.8909 and cal is None

    def test_applied_artifact_compresses(self, monkeypatch):
        monkeypatch.setattr("aigc_detector.api.routes.formal_temperature", lambda: 5.645)
        conf, cal = _calibrate_confidence(CAVEAT, 0.8909, 0.1091)
        assert cal is not None and cal["T"] == 5.645
        assert cal["confidence_raw"] == 0.8909
        assert conf == pytest.approx(_expected(0.8909, 5.645), abs=1e-6)
        # FN-1 acceptance: high confidence must leave the >0.8 danger zone
        assert conf < 0.7

    def test_label_flip_free_boundaries(self, monkeypatch):
        """Provably no decision flips: conf'=0.5 iff conf=0.5; ordering preserved."""
        monkeypatch.setattr("aigc_detector.api.routes.formal_temperature", lambda: 8.0)
        for c in (0.5001, 0.55, 0.7, 0.89, 0.99, 0.4999, 0.3):
            conf, _ = _calibrate_confidence(CAVEAT, c, 1 - c)
            assert (conf > 0.5) == (c > 0.5), f"flip at {c}"
        # monotonicity spot-check
        prev = -1.0
        for c in (0.3, 0.5, 0.6, 0.8, 0.95):
            conf, _ = _calibrate_confidence(CAVEAT, c, 1 - c)
            assert conf > prev
            prev = conf

    def test_low_confidence_not_inflated(self, monkeypatch):
        """Compression is symmetric around 0.5: 0.5 stays 0.5; 0.4 pulls toward
        0.5 but stays below (temperature scaling moves both sides inward)."""
        monkeypatch.setattr("aigc_detector.api.routes.formal_temperature", lambda: 5.645)
        conf, _ = _calibrate_confidence(CAVEAT, 0.5, 0.5)
        assert conf == pytest.approx(0.5, abs=1e-9)
        conf, _ = _calibrate_confidence(CAVEAT, 0.4, 0.6)
        assert 0.4 < conf < 0.5, f"0.4 must stay below 0.5, got {conf}"


class TestFormalTemperatureArtifact:
    def test_reads_deployed_artifact_or_none(self):
        # current repo state: artifact exists; applied flag decides
        t = formal_temperature()
        assert t is None or (isinstance(t, float) and t > 0)

    def test_register_gate_still_independent(self):
        """W11 must not have changed the register gate itself."""
        text = "本公司郑重承诺：严格遵守相关法律法规。特此承诺。"
        assert detect_register_zh(text).is_formal_zh
