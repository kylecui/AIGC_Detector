"""W3a/W3b tests: formal-register detector + caveat wiring.

Acceptance anchors (plan v2.1):
- FN-1 fixture and clearly-formal W5 trial docs MUST hit formal_zh (caveat
  coverage target 100% on formal register).
- Normal text (chat/news/casual zh, English prose) must NOT hit (<2% false
  gating on ordinary traffic — tested here as zero on a small sample).
- Route-level caveat builder attaches the canonical payload with score.
- Schema stays backward compatible (caveat optional).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _register_caveat  # noqa: E402
from aigc_detector.api.schemas import DetectionResponse  # noqa: E402
from aigc_detector.detection.register import (  # noqa: E402
    FORMAL_ZH_CAVEAT,
    detect_register_zh,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fn1_declaration.txt"
HUMAN_DIR = Path(__file__).parent.parent / "dataset" / "legal_declaration_zh" / "human"

# Trial docs EXPECTED formal (公文体 proper — caveat must cover these).
EXPECTED_FORMAL = {
    "01-gov-samr-recall-regulation-notice.md",  # 公告如下 + 一、~七、 + 根据《中华人民共和国
    "03-company-haimo-correction.md",           # 更正公告体
    "04-company-shanghai-xinyang-apology.md",   # 致歉公告体
    "05-company-doushen-clarification.md",      # 澄清公告体
    "08-university-shanghai-statement.md",      # 情况通报
    "09-university-wuhan-statement.md",         # 情况通报
    "10-association-zibo-lawyers-statement.md", # 严正声明
}


def _body(md_text: str) -> str:
    parts = md_text.split("---", 2)
    return parts[2].strip() if len(parts) == 3 else md_text.strip()


NORMAL_TEXTS = [
    # casual chat
    "今天天气真不错，我们出去走走吧！你上次说的那家餐厅到底好不好吃啊？我一直想去试试来着。",
    # news-ish prose (no 公文 formulae)
    "记者了解到，该市轨道交通三号线已于上月底全线贯通，预计年内通车试运营。沿线居民出行时间将平均缩短四十分钟。",
    # English prose
    "Artificial intelligence has transformed many industries. Researchers continue to study how these systems behave in production environments.",
    # technical blog zh
    "这个库的API设计得很干净，初始化只需要一个配置对象，然后链式调用就行了。唯一的坑是Windows下路径要用原始字符串。",
]


class TestRegisterDetector:
    def test_fn1_fixture_hits_formal(self):
        text = FIXTURE.read_text(encoding="utf-8")
        reg = detect_register_zh(text)
        assert reg.is_formal_zh, f"fixture score {reg.score}, markers {reg.matched_markers}"

    def test_trial_formal_docs_coverage(self):
        """Acceptance: 100% caveat coverage on clearly-formal trial docs."""
        misses = []
        for name in sorted(EXPECTED_FORMAL):
            f = HUMAN_DIR / name
            if not f.exists():
                continue  # dataset not present in some envs — skip silently
            reg = detect_register_zh(_body(f.read_text(encoding="utf-8")))
            if not reg.is_formal_zh:
                misses.append(f"{name}(score={reg.score})")
        assert not misses, f"formal docs missed by register gate: {misses}"

    def test_normal_texts_not_formal(self):
        for t in NORMAL_TEXTS:
            reg = detect_register_zh(t)
            assert not reg.is_formal_zh, f"false gate (score={reg.score}): {t[:30]}"

    def test_score_is_explainable(self):
        text = "本公司郑重承诺：严格遵守相关法律法规。特此承诺。"
        reg = detect_register_zh(text)
        assert reg.score >= 6
        assert "特此承诺" in reg.matched_markers
        assert reg.matched_markers  # explainability: hits are listed


class TestRouteCaveat:
    def test_caveat_built_for_formal(self):
        caveat = _register_caveat(FIXTURE.read_text(encoding="utf-8"))
        assert caveat is not None
        assert caveat["code"] == "formal_register_zh"
        assert caveat["message"] == FORMAL_ZH_CAVEAT["message"]
        assert "人工复核" in caveat["action_guidance"] or "复核" in caveat["action_guidance"]
        assert caveat["register_score"] >= 6

    def test_caveat_none_for_normal(self):
        assert _register_caveat(NORMAL_TEXTS[0]) is None

    def test_caveat_never_raises(self):
        assert _register_caveat("") is None  # degenerate input tolerated


class TestSchemaCompat:
    def test_response_without_caveat_valid(self):
        resp = DetectionResponse(
            predicted_label="AI-generated",
            confidence=0.9,
            p_ai=0.9,
            detected_language="zh",
        )
        assert resp.caveat is None

    def test_response_with_caveat_valid(self):
        resp = DetectionResponse(
            predicted_label="Human-written",
            confidence=0.89,
            p_ai=0.11,
            detected_language="zh",
            caveat={**FORMAL_ZH_CAVEAT, "register_score": 14},
        )
        assert resp.caveat["code"] == "formal_register_zh"
