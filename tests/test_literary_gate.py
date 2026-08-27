"""W17 tests: literary-ambiguity caveat rule (band + sentence-CV).

Measured anchors (dataset/literary_prose_zh, n=40 human / 270 AI):
- band [0.0047, 0.05] + cv<=0.45: fires on 26% of AI literary prose,
  0% of human literary prose, 10% of casual AI (acceptable: they receive
  a low-trust caveat, not a wrong verdict).
- FN-2 anchors: both 台风 essays have encoder ~0.008 (in band); original's
  sentence CV is high (human-like variance) so it may NOT fire — the rule
  is honest coverage (26%), not a magic catch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _literary_ambiguity_caveat  # noqa: E402
from aigc_detector.detection.register import detect_literary_ambiguity  # noqa: E402

LIT_DIR = Path(__file__).parent.parent / "dataset/literary_prose_zh"


@dataclass
class R:
    predicted_label: str = "Human-written"
    p_ai: float = 0.02
    confidence: float = 0.98
    breakdown: dict = None

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {"encoder": {"p_ai": 0.008}}


class TestBandDetector:
    def test_in_band_uniform_fires(self):
        # uniform sentence lengths (AI-like low CV)
        text = "。".join(["这是一个测试句子长度均匀" * 3] * 8) + "。"
        assert detect_literary_ambiguity(0.01, text)

    def test_in_band_high_cv_does_not_fire(self):
        # human-like variance: mixed long/short sentences
        text = "短。" + "这是一个非常非常长的句子，包含了大量的修饰成分和从属结构，" * 4 + "又如这般的长句继续延伸下去。" + "又短。" * 3 + "再一个中等长度的句子在这里出现。" + "短。"
        assert not detect_literary_ambiguity(0.01, text)

    def test_out_of_band_never_fires(self):
        text = "任意句子。" * 10
        assert not detect_literary_ambiguity(0.0001, text)
        assert not detect_literary_ambiguity(0.5, text)
        assert not detect_literary_ambiguity(None, text)

    def test_fail_safe(self):
        assert detect_literary_ambiguity(0.01, "") is False


class TestRouteCaveat:
    def test_fires_compresses_confidence(self):
        r = R(confidence=0.98)
        uniform = "。".join(["这是一个测试句子长度均匀" * 3] * 8) + "。"
        c = _literary_ambiguity_caveat(r, None, uniform)
        assert c is not None and c["code"] == "literary_ambiguity_zh"
        assert r.confidence == 0.6  # compressed from 0.98
        assert r.predicted_label == "Human-written"  # verdict untouched

    def test_skipped_when_other_caveat_present(self):
        r = R()
        assert _literary_ambiguity_caveat(r, {"code": "formal_register_zh"}, "x") is None

    def test_no_encoder_no_fire(self):
        r = R(breakdown={"statistical": {"p_ai": 0.9}})
        assert _literary_ambiguity_caveat(r, None, "some text。") is None


class TestProbeCoverageAnchors:
    def test_human_literary_zero_false_caveat(self):
        """0/40 human prose must not receive the caveat (measured anchor)."""
        import json

        evals = {json.loads(l)["id"]: json.loads(l) for l in
                 (LIT_DIR / "human_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        hits = total = 0
        for f in sorted((LIT_DIR / "human").glob("*.md")):
            e = evals.get(f.stem)
            if not e or "encoder" not in (e.get("stage_p_ai") or {}):
                continue
            total += 1
            body = f.read_text(encoding="utf-8").split("---", 2)[2]
            hits += detect_literary_ambiguity(e["stage_p_ai"]["encoder"], body)
        if total >= 30:
            assert hits == 0, f"human false-caveat regressed: {hits}/{total}"

    def test_ai_literary_coverage(self):
        """26% (69/270) AI prose receives the caveat — regression band 15-40%."""
        import json

        res_by_id = {}
        for l in (LIT_DIR / "eval_results.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                res_by_id.setdefault(r["id"], r)
        recs = [json.loads(l) for l in (LIT_DIR / "ai_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        ded = {}
        for rec in recs:
            ded.setdefault((rec["model"], rec["topic_id"], rec["seed"]), rec)
        hits = total = 0
        for rec in ded.values():
            r = res_by_id.get(rec["id"])
            if not r or "encoder" not in (r.get("stage_p_ai") or {}):
                continue
            total += 1
            hits += detect_literary_ambiguity(r["stage_p_ai"]["encoder"], rec["text"])
        if total >= 200:
            rate = hits / total
            assert 0.15 <= rate <= 0.40, f"AI coverage out of band: {rate:.0%} ({hits}/{total})"
