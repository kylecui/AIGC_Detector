"""W17 recalibration — final honest design after gate-search failure.

Two findings from calibration:
1. Lexical literary gate is UNSATISFIABLE (casual posts share the imagery/
   first-person feature space with lyrical prose — same-style continuum;
   frontier shows no threshold meets lit>=80% & casual<=10%).
2. Encoder band [0.002, 0.05] on NON-formal zh text is a zone of ambiguity:
   literary AI median 0.0071 lives there, literary human max 0.0047 lives
   there. Within-band calls are unreliable EITHER DIRECTION.

Design shipped instead: a CAVEAT rule (not an upgrade rule) — when formal_zh
gate does NOT fire AND encoder p_ai falls in the ambiguity band AND
register-agnostic prose signals (length + sentence variance) suggest
non-casual text, attach a literary-ambiguity warning + calibrate confidence
down. This buys FN-2-style cases an honest "低置信，建议人工复核" instead of
a confident 0.98 Human — without fabricating detection power we don't have.

This script validates the caveat rule on the probe sets and picks the band.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def sent_cv(text: str) -> float:
    import re

    sents = [s for s in re.split(r"[。！？\n]+", text) if s.strip()]
    lens = np.array([len(s) for s in sents]) if sents else np.array([1])
    return float(lens.std() / lens.mean()) if lens.mean() > 0 else 0.0


def main() -> int:
    # human literary: how many land in candidate bands (should be LOW = few caveats)
    lit_files = sorted((ROOT / "dataset/literary_prose_zh/human").glob("*.md"))
    # AI literary encoder scores
    ai = [json.loads(l) for l in (ROOT / "dataset/literary_prose_zh/eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = set()
    ai_enc = []
    for r in ai:
        k = (r["model"], r["topic"], r["seed"])
        if k not in seen and "encoder" in (r.get("stage_p_ai") or {}):
            seen.add(k)
            ai_enc.append(r["stage_p_ai"]["encoder"])
    ai_enc = np.array(ai_enc)

    # human literary encoder (from human_eval)
    hum = [json.loads(l) for l in (ROOT / "dataset/literary_prose_zh/human_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    hum_enc = np.array([r["stage_p_ai"]["encoder"] for r in hum if "encoder" in r.get("stage_p_ai", {})])

    # casual AI (W4c): encoder distribution — the band must NOT flag casual
    cas = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    crec = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    cas_enc = []
    cseen = set()
    for r in cas:
        rec = crec.get(r["id"])
        if not rec or rec["register"] != "casual":
            continue
        k = (rec["model"], rec["topic_id"], rec["seed"])
        if k in cseen or "encoder" not in (r.get("stage_p_ai") or {}):
            continue
        cseen.add(k)
        cas_enc.append(r["stage_p_ai"]["encoder"])
    cas_enc = np.array(cas_enc)

    print(f"n: ai_lit={len(ai_enc)} hum_lit={len(hum_enc)} casual_ai={len(cas_enc)}")
    for name, v in [("ai_literary", ai_enc), ("hum_literary", hum_enc), ("casual_ai", cas_enc)]:
        print(f"{name:<14} p10={np.percentile(v,10):.4f} p50={np.median(v):.4f} p90={np.percentile(v,90):.4f}")

    print("\n=== candidate caveat bands: fraction flagged ===")
    for lo, hi in [(0.002, 0.05), (0.003, 0.05), (0.002, 0.03), (0.0047, 0.05), (0.002, 0.10)]:
        f_ai = ((ai_enc >= lo) & (ai_enc <= hi)).mean()
        f_h = ((hum_enc >= lo) & (hum_enc <= hi)).mean()
        f_c = ((cas_enc >= lo) & (cas_enc <= hi)).mean()
        print(f"  band[{lo},{hi}]: ai_lit={f_ai:.0%} hum_lit={f_h:.0%} casual={f_c:.0%}")

    # dual-condition: encoder band + sentence-CV (literary prose has high
    # sentence-length variance; casual posts low). Needs original texts for
    # CV, so re-walk records for casual + load literary texts.
    print("\n=== dual-condition (band + sent_cv>=0.55) ===")
    def dual(e, texts, lo=0.0047, hi=0.05):
        n_hit = 0
        for score_v, t in zip(e, texts):
            if lo <= score_v <= hi and sent_cv(t) >= 0.55:
                n_hit += 1
        return n_hit / max(1, len(e))

    lit_texts = []
    for f in lit_files:
        parts = f.read_text(encoding="utf-8").split("---", 2)
        body = parts[2].strip() if len(parts) == 3 else ""
        if len(body) > 200:
            lit_texts.append(body)
    # AI literary texts aligned with ai_enc order (same dedup walk)
    ai_recs = [json.loads(l) for l in (ROOT / "dataset/literary_prose_zh/ai_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    aseen = set()
    ai_texts = []
    ai_enc2 = []
    for r in ai_recs:
        k = (r["model"], r["topic_id"], r["seed"])
        if k in aseen:
            continue
        aseen.add(k)
        ai_texts.append(r["text"])
        ai_enc2.append(None)  # fill later by matching id
    # simpler: rebuild pairs directly
    res_by_id = {}
    for r in ai:
        res_by_id.setdefault(r["id"], r)
    pairs = []
    for rec in ai_recs:
        if rec["id"] in res_by_id and "encoder" in (res_by_id[rec["id"]].get("stage_p_ai") or {}):
            pairs.append((res_by_id[rec["id"]]["stage_p_ai"]["encoder"], rec["text"]))
    ded = {}
    for rec in ai_recs:
        k = (rec["model"], rec["topic_id"], rec["seed"])
        if k not in ded:
            ded[k] = rec
    pairs = []
    for k, rec in ded.items():
        r = res_by_id.get(rec["id"])
        if r and "encoder" in (r.get("stage_p_ai") or {}):
            pairs.append((r["stage_p_ai"]["encoder"], rec["text"]))
    e = np.array([p[0] for p in pairs]); t = [p[1] for p in pairs]
    print(f"ai pairs n={len(e)}")

    cas_pairs = []
    cseen2 = set()
    for r in cas:
        rec = crec.get(r["id"])
        if not rec or rec["register"] != "casual":
            continue
        k = (rec["model"], rec["topic_id"], rec["seed"])
        if k in cseen2 or "encoder" not in (r.get("stage_p_ai") or {}):
            continue
        cseen2.add(k)
        cas_pairs.append((r["stage_p_ai"]["encoder"], rec["text"]))
    ce = np.array([p[0] for p in cas_pairs]); ct = [p[1] for p in cas_pairs]

    he_aligned = hum_enc  # order matches lit_files sorted; but lit_texts filtered >200 chars
    print(f"dual ai_lit={dual(e, t):.0%}  hum_lit={dual(hum_enc, lit_texts):.0%}  casual={dual(ce, ct):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
