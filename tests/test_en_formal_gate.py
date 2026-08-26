"""W16/P0-3 tests: EN formal-register product-level downgrade.

Acceptance anchors:
- The EN human probe's flagged catastrophe docs (recall/correction style)
  hit the gate and get DOWNGRADED (confidence capped 0.49 + warning payload)
- Casual English (reddit/yelp style) never gated
- zh text never gated by the EN detector
- Verdict/p_ai NOT rewritten (score visible for ranking; refusal is about
  confidence presentation, not score hiding)
- Fail-safe: detector errors leave the response untouched
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _en_formal_downgrade  # noqa: E402
from aigc_detector.detection.register import detect_register_en_formal  # noqa: E402

EN_DIR = Path(__file__).parent.parent / "dataset/legal_declaration_en/human"


@dataclass
class R:
    predicted_label: str = "AI-generated"
    p_ai: float = 0.95
    confidence: float = 0.95


RECALL_SAMPLE = """FOR IMMEDIATE RELEASE
To: Consumers

NOTICE OF PRODUCT RECALL

In accordance with applicable law, this notice is issued to inform consumers
should stop using the product immediately. The company undertakes to remedy
the defect pursuant to the Consumer Product Safety Act.
Effective Date: March 1, 2025
Issued by: ExampleCorp
"""

CASUAL_SAMPLE = """ngl this ramen place is absurd. waited 2 hours but tbh the
tonkotsu literally slapped. lowkey coming back Tuesday. who's in??"""


class TestEnFormalDetector:
    def test_probe_docs_gate_rate(self):
        """Measured gate coverage on the human EN formal probe.

        Reality (2026-08-21 calibration run): 9/35 = 26% hit rate — the
        gate catches template-style formal (SEC-style commitments/
        terminations, structured corrections) but MISSES narrative formal
        (apology letters, incident statements, CPSC recall prose score
        0-2: sincere narrative carries little institutional boilerplate).
        This is the same lexical-gate limitation as the zh gate
        (template-vs-narrative), documented in capability-statement.md.
        The guard catches the worst measured sub-types; narrative formal
        remains covered only by the capability statement, not by code.
        """
        if not EN_DIR.exists():
            return  # dataset absent in some envs
        hits = total = 0
        for f in sorted(EN_DIR.glob("*.md")):
            body = f.read_text(encoding="utf-8").split("---", 2)[2]
            if len(body) < 50:
                continue
            total += 1
            hits += detect_register_en_formal(body)[0]
        assert total >= 20, f"probe too small: {total}"
        # regression anchor: current lexicon catches the template-style
        # subset; if this DROPS below 8/35 the gate regressed
        assert hits >= 8, f"gate coverage regressed: {hits}/{total}"
        # and must not balloon (false-gating casual English would show here)
        assert hits <= 14, f"gate over-firing: {hits}/{total}"

    def test_recall_notice_hits(self):
        hit, score = detect_register_en_formal(RECALL_SAMPLE)
        assert hit and score >= 5

    def test_casual_english_not_gated(self):
        hit, score = detect_register_en_formal(CASUAL_SAMPLE)
        assert not hit, f"false gate at score {score}"

    def test_zh_not_gated_by_en_detector(self):
        hit, _ = detect_register_en_formal("本公司郑重承诺：严格遵守相关法律法规。特此承诺。")
        assert not hit


class TestDowngradeBehavior:
    def test_downgrade_caps_confidence_and_attaches_payload(self):
        r = R()
        d = _en_formal_downgrade(r, RECALL_SAMPLE)
        assert d is not None
        assert d["code"] == "formal_register_en_downgrade"
        assert "71%" in d["message"]
        assert r.confidence <= 0.49          # capped below decision threshold
        assert r.predicted_label == "AI-generated"  # verdict NOT rewritten
        assert r.p_ai == 0.95                # score NOT hidden

    def test_low_confidence_unchanged(self):
        r = R(confidence=0.3, p_ai=0.3, predicted_label="Human-written")
        d = _en_formal_downgrade(r, RECALL_SAMPLE)
        assert d is not None
        assert r.confidence == 0.3           # only caps, never raises

    def test_casual_returns_none(self):
        r = R()
        assert _en_formal_downgrade(r, CASUAL_SAMPLE) is None
        assert r.confidence == 0.95

    def test_fail_safe(self, monkeypatch):
        def boom(t):
            raise RuntimeError("gate broke")

        monkeypatch.setattr(
            "aigc_detector.api.routes.detect_register_en_formal", boom
        )
        r = R()
        assert _en_formal_downgrade(r, RECALL_SAMPLE) is None
        assert r.confidence == 0.95
