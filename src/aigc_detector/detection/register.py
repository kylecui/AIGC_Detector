"""Lexical formal-register (公文体) detector — W3a/W3b v1, config-driven since v0.2b.

Marker tables and thresholds live in configs/gates/{formal_zh,formal_en}.yaml
(single source of truth; force-included in the wheel). This module is the
loader + scoring logic only — editing gate behavior means editing the YAML.
The coverage anchors in tests/{test_register_gate,test_en_formal_gate}.py pin
the calibrated behavior (zh 7/7 formal + 0 casual false-fire; EN lexical
9/35 template-style): run them after any YAML edit.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def _gates_dir() -> Path:
    """configs/gates for wheel and repo layouts.

    Wheel: aigc_detector/configs/gates (force-included). Repo/editable:
    repo-root/configs/gates (parents[3] from this file).
    """
    pkg = Path(__file__).resolve().parents[1] / "configs" / "gates"
    if pkg.is_dir():
        return pkg
    return Path(__file__).resolve().parents[3] / "configs" / "gates"


@functools.lru_cache(maxsize=4)
def _load_gate(name: str) -> dict:
    path = _gates_dir() / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["_structure_compiled"] = [
        (re.compile(s["pattern"]), int(s["weight"]), int(s["min_hits"]))
        for s in data.get("structure", [])
    ]
    return data



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
    gate = _load_gate("formal_zh")
    matched_markers: list[str] = []
    score = 0
    for marker, weight in gate["markers"].items():
        if marker in text:
            score += weight
            matched_markers.append(marker)

    matched_patterns: list[str] = []
    for pattern, weight, min_hits in gate["_structure_compiled"]:
        hits = len(pattern.findall(text))
        if hits >= min_hits:
            score += weight
            matched_patterns.append(f"{pattern.pattern[:24]}…×{hits}")

    return RegisterResult(
        score=score,
        is_formal_zh=score >= int(gate["threshold"]),
        matched_markers=matched_markers,
        matched_patterns=matched_patterns,
    )


# ---- EN formal register (W16/P0-3): the EN path's measured blind spot ----
# Measured basis: 71% [55%, 84%] of human EN formal docs flagged as AI on
# the EN probe (reports/human_probe_results_en_human.json, n=35); recall-
# notice/correction/termination types 100% flagged. Product-level guard:
# on register hit the response is DOWNGRADED (strong warning), matching
# capability-statement.md — we refuse to issue a normal-confidence verdict
# where our own measurement says the verdict is worse than a coin flip.


def _en_ml_gate():
    """Load the F1 ML register gate (joblib + threshold), if present & enabled.

    Trained by scripts/train_en_register_gate.py on W4-EN arms + human
    probe: narrative-formal recall 26/26, human recall 35/35, AI-casual
    false-gate 0.9% at threshold 0.30. GATING CAVEAT: the casual negative
    class is AI-generated; human-casual false-gate rate was NOT yet
    validated at ship time (one hand-written sample fired at 0.75). The
    gate therefore ships with "enabled": false in en_register_gate.json;
    flip to true only after a human-casual validation set (>=30 real
    reddit/yelp-style comments) shows false-gate <= 5%.
    """
    import json

    import joblib

    d = _calibration_dir()
    try:
        meta = json.loads((d / "en_register_gate.json").read_text(encoding="utf-8"))
        if not meta.get("enabled", False):
            return None
        clf = joblib.load(d / "en_register_gate.joblib")
        return clf, float(meta["threshold"])
    except (OSError, ValueError, KeyError):
        return None


def detect_register_en_formal(text: str) -> tuple[bool, int]:
    """Lexical-OR-ML EN formal-register check (W16 + F1).

    Layer 1 (lexical): template formal — institutional boilerplate.
    Layer 2 (ML, when artifacts present): narrative formal — apology
    letters / incident statements the lexical layer cannot see; logistic
    gate on the 14 linguistic stylometric features. Reported score: the
    layer that fired (lexical score, or ML probability x100 as int).
    """
    gate = _load_gate("formal_en")
    score = 0
    for m, w in gate["markers"].items():
        if m.lower() in text.lower():
            score += w
    for pat, w, need in gate["_structure_compiled"]:
        if len(pat.findall(text)) >= need:
            score += w
    if score >= int(gate["threshold"]):
        return True, score
    gate = _en_ml_gate()
    if gate is not None and len(text) >= 120:
        import dataclasses

        import numpy as np

        from aigc_detector.detection.linguistic import LinguisticFeatureExtractor

        try:
            lf = LinguisticFeatureExtractor().extract(text)
            names = [f.name for f in dataclasses.fields(lf)]
            x = np.array([[getattr(lf, n) for n in names]], dtype=float)
            p = float(gate[0].predict_proba(x)[0, 1])
            if p >= gate[1]:
                return True, int(p * 100)
        except Exception:  # noqa: BLE001 — ML layer must never break the gate
            pass
    return False, score


# Canonical EN-formal downgrade payload (product-level guard, W16).
EN_FORMAL_DOWNGRADE: dict = {
    "code": "formal_register_en_downgrade",
    "message": (
        "This text is in the English formal/announcement register. On this "
        "register our measured error rate is 71% (35-document probe: most "
        "human formal documents are misclassified as AI). We do not issue "
        "normal-confidence verdicts here."
    ),
    "action_guidance": (
        "Recommendations: 1) treat any verdict on this text as unreliable; "
        "2) use human review or creation-process evidence instead; "
        "3) see docs/capability-statement.md for the full boundary list."
    ),
}


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

# ---- Literary-ambiguity band (W17, 2026-08-27) ----
# Measured basis (dataset/literary_prose_zh, n=40/270): on NON-formal zh
# text, encoder p_ai in [0.0047, 0.05] combined with LOW sentence-length CV
# (<=0.45; AI lyrical prose is more uniform than human prose, median 0.40 vs
# 0.60) marks a zone where literary-AI lives (26% of AI prose lands here,
# 0% of human prose) but verdicts are unreliable either direction. Rule
# attaches a CAVEAT (confidence-down), never upgrades the verdict.
LITERARY_BAND = (0.0047, 0.05)
LITERARY_CV_MAX = 0.45

LITERARY_AMBIGUITY_CAVEAT: dict = {
    "code": "literary_ambiguity_zh",
    "message": (
        "该文本呈现文学散文特征且处于检测器不可靠区间：系统对AI与人类散文"
        "在此区间的区分能力有限（实测AI散文检出率约10%）。"
    ),
    "action_guidance": (
        "建议：① 将本次判定视为低可信；"
        "② 如需确认，请结合创作过程留痕等旁证；"
        "③ 详见 docs/capability-statement.md 散文灰色地带说明。"
    ),
}


# Literary imagery markers shared by the upgrade rule (W17b Variant B).
_LITERARY_IMAGERY = ("像", "仿佛", "宛如", "好似", "月亮", "月光", "星空", "风", "雨", "雪",
                     "夜", "黄昏", "清晨", "光", "影", "云", "海", "山", "树", "花", "叶",
                     "梦", "泪", "心", "温柔", "寂静", "岁月", "时光", "回忆", "思念")


def _sent_cv(text: str) -> float:
    import re as _re

    sents = [s for s in _re.split(r"[。！？\n]+", text) if s.strip()]
    if not sents:
        return 1.0
    lens = [len(s) for s in sents]
    m = sum(lens) / len(lens)
    return (sum((x - m) ** 2 for x in lens) / len(lens)) ** 0.5 / m if m > 0 else 1.0


def literary_upgrade_config() -> dict | None:
    """Load the W17b upgrade-rule artifact, if enabled (models/calibration/literary_upgrade.json)."""
    import json

    try:
        data = json.loads((_calibration_dir() / "literary_upgrade.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not data.get("enabled"):
        return None
    try:
        return {
            "band": (float(data["band"][0]), float(data["band"][1])),
            "cv_max": float(data["cv_max"]),
            "fp_min": float(data["fp_per100_min"]),
            "img_min": float(data["img_per100_min"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def detect_literary_upgrade(encoder_p_ai: float | None, text: str) -> bool:
    """W17b Variant B: literary-AI upgrade rule (verdict-changing, gated).

    Fires when ALL hold: formal_zh gate did NOT fire (caller ensures),
    encoder p_ai in the ambiguity band, sentence-CV low (AI-like uniformity),
    AND literary features present (first-person + imagery density — the
    precondition that cut casual false-upgrades 5x in the dossier).
    Measured: 26% AI-literary catch, 0/40 human-literary, 2% casual-AI,
    0/62 human-formal. Fail-safe: errors -> False.
    """
    try:
        cfg = literary_upgrade_config()
        if cfg is None or encoder_p_ai is None:
            return False
        if not (cfg["band"][0] <= float(encoder_p_ai) <= cfg["band"][1]):
            return False
        if _sent_cv(text) > cfg["cv_max"]:
            return False
        n = max(1, len(text))
        fp = (text.count("我") + text.count("我们")) * 100 / n
        img = sum(text.count(w) for w in _LITERARY_IMAGERY) * 100 / n
        return fp >= cfg["fp_min"] and img >= cfg["img_min"]
    except Exception:  # noqa: BLE001 — upgrade must never break detection
        return False


def detect_literary_ambiguity(encoder_p_ai: float | None, text: str) -> bool:
    """W17: literary-ambiguity band check (caveat-only rule).

    Fires when: encoder score sits in the ambiguity band AND sentence-length
    CV indicates AI-like uniformity (callers apply this only when the
    formal_zh gate did NOT fire — formal text has its own rules). Returns
    True => attach caveat + compress confidence. Never changes the verdict.
    Pure function; errors -> False (fail-safe).
    """
    try:
        if encoder_p_ai is None or not (LITERARY_BAND[0] <= encoder_p_ai <= LITERARY_BAND[1]):
            return False
        import re as _re

        sents = [s for s in _re.split(r"[。！？\n]+", text) if s.strip()]
        if not sents:
            return False
        lens = [len(s) for s in sents]
        mean = sum(lens) / len(lens)
        if mean <= 0:
            return False
        var = sum((x - mean) ** 2 for x in lens) / len(lens)
        cv = var ** 0.5 / mean
        return cv <= LITERARY_CV_MAX
    except Exception:  # noqa: BLE001 — caveat must never break detection
        return False


def _calibration_dir() -> Path:
    """Resolve models/calibration for dev-checkout AND installed layouts.

    Order: settings.model_dir (honors MODEL_DIR env; CWD-relative 'models'
    default matches Docker WORKDIR mount) -> repo-root via parents[3]
    (editable/dev checkout). The old parents[3]-only resolution broke under
    a wheel install (site-packages layout shifts the parent chain).
    """
    from pathlib import Path

    try:
        from aigc_detector.config import settings

        d = Path(settings.model_dir) / "calibration"
        if d.is_dir():
            return d
    except Exception:  # noqa: BLE001 — config import must never break loaders
        pass
    return Path(__file__).resolve().parents[3] / "models" / "calibration"


def formal_temperature() -> float | None:
    """Load the fitted formal-register temperature (W11), if present.

    Artifact: models/calibration/global_temperature.json (written by
    scripts/fit_global_temperature.py; fitted on the 382-doc formal probe
    corpus, class-balanced NLL). Returns None when absent/unapplied —
    callers then skip calibration (T=1, behavior unchanged).
    """
    import json

    path = _calibration_dir() / "global_temperature.json"
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

    path = _calibration_dir() / "binoculars_floor.json"
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
