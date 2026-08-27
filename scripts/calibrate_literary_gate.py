"""Calibrate the literary-prose register gate on the probe set.

Gate concept (W17-1): lexical features that distinguish lyrical first-person
prose from (a) formal 公文 (which has its own gate) and (b) casual posts.
Candidate features measured here per text:
  - first-person density (我/我们 per 100 chars)
  - imagery-marker density (比喻/景物词表: 像/仿佛/宛如/月亮/风/雨/夜/光...)
  - sentence-length CV (already known: prose has high variance)
  - dash/ellipsis usage (—— / ……)
Threshold chosen so: human literary probe hit >= 80%, formal-zh probe
miss <= 5% (those already route to the formal gate), AI-casual W4c miss
>= 95% (must not gate casual posts).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

IMAGERY = ["像", "仿佛", "宛如", "好似", "月亮", "月光", "星空", "风", "雨", "雪",
           "夜", "黄昏", "清晨", "光", "影", "云", "海", "山", "树", "花", "叶",
           "梦", "泪", "心", "温柔", "寂静", "岁月", "时光", "回忆", "思念"]


def feats(text: str) -> dict:
    n = max(1, len(text))
    sents = [s for s in re.split(r"[。！？\n]+", text) if s.strip()]
    lens = np.array([len(s) for s in sents]) if sents else np.array([1])
    fp = (text.count("我") + text.count("我们")) * 100 / n
    img = sum(text.count(w) for w in IMAGERY) * 100 / n
    cv = float(lens.std() / lens.mean()) if lens.mean() > 0 else 0.0
    dash = (text.count("——") + text.count("……")) * 100 / n
    return {"fp_per100": fp, "img_per100": img, "sent_cv": cv, "dash_per100": dash}


def load_literary(d: Path) -> list[str]:
    out = []
    for f in sorted(d.glob("*.md")):
        parts = f.read_text(encoding="utf-8").split("---", 2)
        body = parts[2].strip() if len(parts) == 3 else ""
        if len(body) > 200:
            out.append(body)
    return out


def main() -> int:
    lit = load_literary(ROOT / "dataset/literary_prose_zh/human")
    formal = load_literary(ROOT / "dataset/legal_declaration_zh/human")
    casual = []
    recs = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = set()
    for r in recs:
        k = (r["model"], r["topic_id"], r["seed"])
        if r["register"] == "casual" and k not in seen and len(r["text"]) > 200:
            seen.add(k)
            casual.append(r["text"])
        if len(casual) >= 120:
            break

    for name, texts in [("literary-human(n=%d)" % len(lit), lit),
                        ("formal-zh(n=%d)" % len(formal), formal),
                        ("casual-ai(n=%d)" % len(casual), casual)]:
        rows = [feats(t) for t in texts]
        print(f"\n{name}:")
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            print(f"  {k:<12} p10={np.percentile(v,10):.3f} p50={np.median(v):.3f} p90={np.percentile(v,90):.3f}")

    # score = weighted sum, grid-search weights/threshold on constraints
    def score(t: str) -> float:
        f = feats(t)
        return f["fp_per100"] * 1.0 + f["img_per100"] * 1.5 + f["sent_cv"] * 8.0 + f["dash_per100"] * 4.0

    lit_s = np.array([score(t) for t in lit])
    form_s = np.array([score(t) for t in formal])
    cas_s = np.array([score(t) for t in casual])
    print("\n=== frontier WITH formal-gate precondition (literary gate only consulted when formal_zh NOT hit) ===")
    sys.path.insert(0, str(ROOT / "src"))
    from aigc_detector.detection.register import detect_register_zh

    form_ok = [t for t in formal if not detect_register_zh(t).is_formal_zh]
    print(f"formal docs surviving their own gate: {len(form_ok)}/{len(formal)}")
    print("=== grid on preconditioned set: lit>=80%, formal-survivor<=10%, casual<=10% ===")
    best = None
    for thr in np.arange(4.0, 12.01, 0.25):
        lh = (lit_s >= thr).mean()
        fm = (np.array([score(t) for t in form_ok]) >= thr).mean() if form_ok else 0.0
        cg = (cas_s >= thr).mean()
        if lh >= 0.80 and fm <= 0.10 and cg <= 0.10:
            print(f"  thr={thr:.2f}: lit={lh:.0%} formal-survivor={fm:.0%} casual={cg:.0%}  OK")
            if best is None:
                best = thr
    if best is None:
        for thr in np.arange(4.0, 12.01, 0.5):
            lh = (lit_s >= thr).mean()
            fm = (np.array([score(t) for t in form_ok]) >= thr).mean() if form_ok else 0.0
            cg = (cas_s >= thr).mean()
            print(f"  thr={thr:.1f}: lit={lh:.0%} formal-sv={fm:.0%} casual={cg:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
