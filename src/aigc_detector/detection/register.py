"""Lexical formal-register (公文体) detector — W3a/W3b v1.

Rule-based, explainable, auditable by design (v2.1 plan: no ML register
classifier). Detects Chinese formal-document register (声明/公告/承诺书/
情况说明/通报 etc.) via weighted lexical markers + structural patterns.

Scope note (v1, 2026-08-17): tuned on the FN-1 fixture + W5 trial batch
(10 human formal docs). Simple service notices (e.g. short bank maintenance
notices without 公文 markers) may NOT hit — a known gap recorded in
DETECTOR_NOTES_2026-08.md (W5 trial section); the W9 high-confidence-error
metric catches those cases independently of this gate.

Threshold calibration is frozen pending the full W5 probe set (60-80 docs);
changing FORMULA threshold requires re-running tests/test_register_gate.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Weighted lexical markers (phrase → weight). Hit = substring occurrence.
_MARKERS_ZH: dict[str, int] = {
    # closing formulae — strongest signals
    "特此声明": 3,
    "特此承诺": 3,
    "特此公告": 3,
    "特此函告": 3,
    "特此通知": 3,
    "严正声明": 3,
    "郑重承诺": 3,
    # openings / pivot formulae
    "兹因": 2,
    "兹就": 2,
    "兹有": 2,
    "现声明如下": 3,
    "现承诺如下": 3,
    "现公告如下": 3,
    "现就有关事项公告如下": 3,
    "声明如下": 2,
    "承诺如下": 2,
    "公告如下": 2,
    # regulatory citation style
    "依据《": 2,
    "根据《中华人民共和国": 2,
    # obligation/penalty formulae
    "如有违反": 2,
    "承担相应责任": 2,
    "承担法律责任": 2,
    "严格遵守": 1,
    # address forms
    "致：": 2,
    # listed-company information-disclosure formulae (信披公告定义性套语)
    "保证信息披露": 3,
    "真实、准确、完整": 2,
    "公司及董事会": 2,
    # situational-report formulae (情况通报体)
    "通报如下": 3,
    "现将有关情况": 2,
    "说明如下": 2,
}

# Structural patterns: (compiled regex, weight, required hit count)
_STRUCTURE_ZH: list[tuple[re.Pattern[str], int, int]] = [
    # 《关于……的声明/公告/承诺书/情况说明/函》title
    (re.compile(r"《关于.{2,24}的(声明|公告|承诺书|情况说明|函|通知书?|澄清公告)》,?"), 3, 1),
    # signature-date blank line ＿＿＿＿年＿＿月＿＿日
    (re.compile(r"＿{2,}\s*年\s*＿{2,}\s*月\s*＿{2,}\s*日"), 3, 1),
    # numbered clauses 一、二、… (needs ≥2 distinct hits)
    (re.compile(r"(?m)^\s*[一二三四五六七八九十]{1,3}、"), 2, 2),
    # full-width blank signature line ＿＿＿＿＿＿＿＿ (standalone)
    (re.compile(r"(?m)^＿{4,}\s*$"), 2, 1),
    # standalone signature date line (公文落款): 2026年8月17日 on its own line
    (re.compile(r"(?m)^\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*$"), 2, 1),
]

# Decision threshold: fixture + clear trial docs score ≥6; casual text ≤2.
FORMAL_ZH_THRESHOLD = 6


@dataclass
class RegisterResult:
    """Result of the lexical formal-register check."""

    score: int
    is_formal_zh: bool
    matched_markers: list[str]
    matched_patterns: list[str]


def detect_register_zh(text: str) -> RegisterResult:
    """Score a (Chinese) text for formal-document register.

    Pure lexical/structural; O(len(text) × markers). Language-agnostic safe:
    English text simply scores ~0.
    """
    matched_markers: list[str] = []
    score = 0
    for marker, weight in _MARKERS_ZH.items():
        if marker in text:
            score += weight
            matched_markers.append(marker)

    matched_patterns: list[str] = []
    for pattern, weight, min_hits in _STRUCTURE_ZH:
        hits = len(pattern.findall(text))
        if hits >= min_hits:
            score += weight
            matched_patterns.append(f"{pattern.pattern[:24]}…×{hits}")

    return RegisterResult(
        score=score,
        is_formal_zh=score >= FORMAL_ZH_THRESHOLD,
        matched_markers=matched_markers,
        matched_patterns=matched_patterns,
    )


# Canonical caveat payload attached to responses when formal register hits
# (W3b). Wording checked for non-expert readability (D2: warning + concrete
# action guidance, no internal jargon).
FORMAL_ZH_CAVEAT: dict = {
    "code": "formal_register_zh",
    "message": (
        "该文本属于正式文书（如声明、公告、承诺书、情况说明等）。"
        "在此类文本上，系统的整体判定可靠性明显下降：AI代写的文书可能被判为人工撰写，"
        "人工撰写的文书也可能被判为AI。"
    ),
    "action_guidance": (
        "建议：① 不要单独依据整体判定做结论；"
        "② 结合“局部AI痕迹”分段证据，人工复核得分较高的段落；"
        "③ 重要决策建议采用语料级检测或创作过程留痕等旁证。"
    ),
}


def formal_temperature() -> float | None:
    """Load the fitted formal-register temperature (W11), if present.

    Artifact: models/calibration/global_temperature.json (written by
    scripts/fit_global_temperature.py; fitted on the 382-doc formal probe
    corpus, class-balanced NLL). Returns None when absent/unapplied —
    callers then skip calibration (T=1, behavior unchanged).
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "models/calibration/global_temperature.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not data.get("applied"):
        return None
    t = data.get("T")
    return float(t) if isinstance(t, (int, float)) and t > 0 else None


def binoculars_floor() -> dict | None:
    """Load the W15 register-gated binoculars-floor config, if enabled.

    Artifact: models/calibration/binoculars_floor.json
    {enabled: bool, cutoff: float, register: "formal_zh"}
    Evidence base: reports/w3b_floor_analysis.json (82-doc probe sweep;
    knee at cutoff 0.46: above-floor contract cells -> ~0% miss at +1.6pp
    point FPR; FN-1 [edited text] missed at every cutoff — the rule's
    boundary is raw-vs-edited generation). Candidate feature: ships
    DISABLED; enabling is a human deployment decision after gate review.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "models/calibration/binoculars_floor.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not data.get("enabled"):
        return None
    cutoff = data.get("cutoff")
    if not isinstance(cutoff, (int, float)) or not (0 < cutoff < 1):
        return None
    return {"cutoff": float(cutoff), "register": data.get("register", "formal_zh")}
