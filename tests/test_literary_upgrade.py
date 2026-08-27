"""W17b Variant B tests: literary upgrade rule (artifact-gated, disabled by default).

Measured anchors (dossier + probe): 26% AI-literary catch / 0% human-literary
/ 2% casual / 0% human-formal. The artifact ships enabled=false; these tests
monkeypatch the enabled config to exercise the rule, plus pin the shipped
disabled state and the probe coverage anchors (upgrade ⊆ caveat coverage:
every upgrade-hit also satisfies the caveat conditions).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _apply_literary_upgrade  # noqa: E402
from aigc_detector.detection import register as reg  # noqa: E402
from aigc_detector.detection.register import detect_literary_upgrade  # noqa: E402

LIT_DIR = Path(__file__).parent.parent / "dataset/literary_prose_zh"

ENABLED_CFG = {"band": (0.0047, 0.05), "cv_max": 0.45, "fp_min": 0.5, "img_min": 1.0}

UNIFORM_PROSAIC = (
    "我望着窗外的月光像一层薄雾，心里满是说不出的思念。我看着树叶在风中轻轻摇曳，"
    "仿佛岁月在这一刻静止了。我听见雨滴落在屋檐上的声音，像是时光温柔的叹息。"
    "我想起那些远去的回忆，泪水悄悄滑落。我依然站在夜色里，等待黎明的光。"
)  # first-person + imagery + uniform sentence lengths


@dataclass
class R:
    predicted_label: str = "Human-written"
    p_ai: float = 0.02
    confidence: float = 0.98
    breakdown: dict = field(default_factory=lambda: {"encoder": {"p_ai": 0.008}})


class TestShippedState:
    def test_artifact_ships_enabled_after_gate(self):
        """Deployed 2026-08-27 after the casual-probe FP gate (0/46 <= 5%).

        This pins the deployment state so any silent flip (either direction)
        fails loudly here.
        """
        cfg = reg.literary_upgrade_config()
        assert cfg is not None, "artifact says enabled=true but loader returned None"
        assert cfg["band"] == (0.0047, 0.05)
        assert cfg["cv_max"] == 0.45


class TestDetector:
    def test_full_conditions_fire(self, monkeypatch):
        monkeypatch.setattr(reg, "literary_upgrade_config", lambda: ENABLED_CFG)
        assert detect_literary_upgrade(0.01, UNIFORM_PROSAIC)

    def test_band_gate(self, monkeypatch):
        monkeypatch.setattr(reg, "literary_upgrade_config", lambda: ENABLED_CFG)
        assert not detect_literary_upgrade(0.001, UNIFORM_PROSAIC)
        assert not detect_literary_upgrade(0.5, UNIFORM_PROSAIC)
        assert not detect_literary_upgrade(None, UNIFORM_PROSAIC)

    def test_high_cv_blocks(self, monkeypatch):
        monkeypatch.setattr(reg, "literary_upgrade_config", lambda: ENABLED_CFG)
        varied = "短。" + "这是一个非常非常长的句子" * 6 + "。" + "又短。" * 3 + "中句。" * 2
        assert not detect_literary_upgrade(0.01, varied)

    def test_no_literary_features_blocks(self, monkeypatch):
        monkeypatch.setattr(reg, "literary_upgrade_config", lambda: ENABLED_CFG)
        plain = "今天天气不错我们出去玩吧然后吃了饭回来睡觉明天继续上班就这样结束了。"
        assert not detect_literary_upgrade(0.01, plain)

    def test_live_config_fires(self):
        """With the deployed (enabled) artifact, the calibrated sample fires."""
        assert detect_literary_upgrade(0.01, UNIFORM_PROSAIC)


class TestRouteRule:
    def test_upgrade_changes_verdict_with_provenance(self, monkeypatch):
        monkeypatch.setattr("aigc_detector.api.routes.detect_literary_upgrade",
                            lambda e, t: True)
        r = R()
        rule = _apply_literary_upgrade(r, None, UNIFORM_PROSAIC)
        assert rule is not None and rule["rule"] == "literary_upgrade_zh"
        assert r.predicted_label == "AI-generated"
        assert r.confidence == 0.008  # evidence-based, not stale-human-conf

    def test_skipped_when_caveat_present(self):
        r = R()
        assert _apply_literary_upgrade(r, {"code": "formal_register_zh"}, "x") is None
        assert r.predicted_label == "Human-written"

    def test_fail_safe(self, monkeypatch):
        monkeypatch.setattr("aigc_detector.api.routes.detect_literary_upgrade",
                            lambda e, t: (_ for _ in ()).throw(RuntimeError("boom")))
        r = R()
        assert _apply_literary_upgrade(r, None, UNIFORM_PROSAIC) is None
        assert r.predicted_label == "Human-written"


class TestProbeAnchors:
    def test_human_literary_zero_upgrades(self, monkeypatch):
        import json

        monkeypatch.setattr(reg, "literary_upgrade_config", lambda: ENABLED_CFG)
        evals = {json.loads(l)["id"]: json.loads(l) for l in
                 (LIT_DIR / "human_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        hits = total = 0
        for f in sorted((LIT_DIR / "human").glob("*.md")):
            e = evals.get(f.stem)
            if not e or "encoder" not in (e.get("stage_p_ai") or {}):
                continue
            total += 1
            body = f.read_text(encoding="utf-8").split("---", 2)[2]
            hits += detect_literary_upgrade(e["stage_p_ai"]["encoder"], body)
        if total >= 30:
            assert hits == 0, f"human upgrades regressed: {hits}/{total}"

    def test_ai_literary_coverage_band(self, monkeypatch):
        import json

        monkeypatch.setattr(reg, "literary_upgrade_config", lambda: ENABLED_CFG)
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
            hits += detect_literary_upgrade(r["stage_p_ai"]["encoder"], rec["text"])
        if total >= 200:
            rate = hits / total
            assert 0.15 <= rate <= 0.40, f"AI upgrade coverage out of band: {rate:.0%}"
