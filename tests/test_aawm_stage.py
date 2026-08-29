"""AAWM stage contract tests (neutral mode + config gating + FN-2 framing)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.stages.aawm_stage import AAWMStage  # noqa: E402


class TestContractConformance:
    def test_satisfies_protocol(self):
        from aigc_detector.stages.contract import StageProtocol

        assert isinstance(AAWMStage(), StageProtocol)

    def test_no_framework_import(self):
        """Third-party proof: structural conformance, zero framework imports."""
        src = Path("examples/stages/aawm_stage.py").read_text(encoding="utf-8")
        assert "from aigc_detector" not in src and "import aigc_detector" not in src


class TestNeutralMode:
    def test_disabled_config_returns_neutral(self):
        """Ships disabled: watermark verification unavailable, neutral evidence."""
        s = AAWMStage()
        s.load()
        out = s.predict("任何文本", "zh")
        assert out["p_ai"] == 0.5
        assert "not configured" in out["evidence"]["note"] or "unavailable" in out["evidence"]["note"]

    def test_never_raises(self):
        s = AAWMStage()
        for bad in ("", "x" * 100000):
            out = s.predict(bad, "zh")
            assert 0.0 <= out["p_ai"] <= 1.0

    def test_lifecycle(self):
        s = AAWMStage()
        assert not s.is_loaded
        s.load()
        assert s.is_loaded
        s.unload()
        assert not s.is_loaded


class TestArtifact:
    def test_ships_disabled_with_fn2_note(self):
        data = json.loads(Path("models/calibration/aawm_stage.json").read_text(encoding="utf-8"))
        assert data["enabled"] is False
        assert "FN-2" in data["_fn2_note"]


import json  # noqa: E402
