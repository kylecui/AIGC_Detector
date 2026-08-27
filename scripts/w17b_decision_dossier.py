"""W17b decision package: deployment-side FP stress test for the literary
encoder-recalibration path (threshold ~0.0047 on the encoder stage).

The question: if we lower the LITERARY upgrade threshold to the zero-FP
band edge, what happens to NON-literary real traffic? The probe says 0/40
human literary, but deployment text is mostly casual/general — the 500-doc
AI-casual set + 102 human formal docs + 40 human literary are our best
deployment proxies. This script measures the upgrade rule's behavior on
all available deployment-proxy sets under candidate thresholds, with the
dual condition (band + CV) AND a literary-features precondition variant.

NOT a deployment: this is the decision dossier for the user.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aigc_detector.detection.register import detect_register_zh  # noqa: E402


def sent_cv(text: str) -> float:
    sents = [s for s in re.split(r"[。！？\n]+", text) if s.strip()]
    if not sents:
        return 1.0
    lens = [len(s) for s in sents]
    m = sum(lens) / len(lens)
    return (sum((x - m) ** 2 for x in lens) / len(lens)) ** 0.5 / m if m > 0 else 1.0


def load_eval(dataset: str) -> dict:
    """Return {key: (encoder_p_ai, text, label_side)} for a probe set."""
    out = {}
    if dataset == "ai_literary":
        res_by_id = {}
        for l in (ROOT / "dataset/literary_prose_zh/eval_results.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                res_by_id.setdefault(r["id"], r)
        ded = {}
        for l in (ROOT / "dataset/literary_prose_zh/ai_records.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                rec = json.loads(l)
                ded.setdefault((rec["model"], rec["topic_id"], rec["seed"]), rec)
        for k, rec in ded.items():
            r = res_by_id.get(rec["id"])
            if r and "encoder" in (r.get("stage_p_ai") or {}):
                out[k] = (r["stage_p_ai"]["encoder"], rec["text"], "ai")
    elif dataset == "hum_literary":
        evals = {json.loads(l)["id"]: json.loads(l) for l in
                 (ROOT / "dataset/literary_prose_zh/human_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        for f in sorted((ROOT / "dataset/literary_prose_zh/human").glob("*.md")):
            e = evals.get(f.stem)
            if e and "encoder" in (e.get("stage_p_ai") or {}):
                body = f.read_text(encoding="utf-8").split("---", 2)[2]
                out[f.stem] = (e["stage_p_ai"]["encoder"], body, "human")
    elif dataset == "casual_ai":
        res = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        crec = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        seen = set()
        for r in res:
            rec = crec.get(r["id"])
            if not rec or rec["register"] != "casual":
                continue
            k = (rec["model"], rec["topic_id"], rec["seed"])
            if k in seen or "encoder" not in (r.get("stage_p_ai") or {}):
                continue
            seen.add(k)
            out[k] = (r["stage_p_ai"]["encoder"], rec["text"], "ai")
    elif dataset == "hum_formal":
        # W5-era: ensemble evals only have p_ai; encoder stage scores for the
        # formal probe were never extracted for all — use human_stage_scores
        for l in (ROOT / "reports/human_stage_scores.jsonl").read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            enc = (r.get("stage_p_ai") or {}).get("encoder")
            if enc is None:
                continue
            src = ROOT / "dataset/legal_declaration_zh/human" / r["file"]
            if not src.exists():
                continue
            body = src.read_text(encoding="utf-8").split("---", 2)[2]
            out[r["file"]] = (enc, body, "human")
    return out


def main() -> int:
    sets = {ds: load_eval(ds) for ds in
            ["ai_literary", "hum_literary", "casual_ai", "hum_formal"]}
    for ds, d in sets.items():
        print(f"{ds}: {len(d)} docs with encoder scores")

    print("\n=== W17b candidate: UPGRADE rule (non-formal + band + low-CV -> AI verdict) ===")
    print("Variant A: band [0.0047, 0.05] + cv<=0.45 (same as caveat, but UPGRADES)")
    print("Variant B: band [0.0047, 0.05] + cv<=0.45 + literary-feature precondition (fp>=0.5/100 & img>=1.0/100)")

    IMAGERY = ["像", "仿佛", "宛如", "好似", "月亮", "月光", "星空", "风", "雨", "雪",
               "夜", "黄昏", "清晨", "光", "影", "云", "海", "山", "树", "花", "叶",
               "梦", "泪", "心", "温柔", "寂静", "岁月", "时光", "回忆", "思念"]

    def lit_feat(text: str) -> bool:
        n = max(1, len(text))
        fp = (text.count("我") + text.count("我们")) * 100 / n
        img = sum(text.count(w) for w in IMAGERY) * 100 / n
        return fp >= 0.5 and img >= 1.0

    for variant, extra in [("A: band+cv", lambda t: True), ("B: band+cv+litfeat", lit_feat)]:
        print(f"\n--- Variant {variant} ---")
        for ds, d in sets.items():
            n_hit = 0
            for _, (e, t, side) in d.items():
                if side == "human" and detect_register_zh(t).is_formal_zh:
                    continue  # formal humans route to formal rules first
                if 0.0047 <= e <= 0.05 and sent_cv(t) <= 0.45 and extra(t):
                    n_hit += 1
            n = len(d)
            role = "CAUGHT" if ds.startswith("ai") else "FALSE-UPGRADE"
            print(f"  {ds:<14} {n_hit}/{n} = {n_hit/n:.0%}  ({role})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
