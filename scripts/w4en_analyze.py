"""W4-EN analysis: contract evasion replication on the EN detection path.

Core questions (protocol v1.1 decisive core):
  Q1: Does formal-register contract evasion (A > B) replicate on EN?
  Q2: Does the casual blind spot appear on EN (C/D vs human baseline 0.29 flag)?
  Q3: Cross-path contrast: which stages drive EN results (linguistic .85 /
      statistical .15 path, no encoder/binoculars weight)?

Usage: uv run python scripts/w4en_analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent.parent
res = [
    json.loads(line)
    for line in
    (ROOT / "dataset/paired_generation_v1/w4en_eval_results.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
recs = {
    json.loads(line)["id"]: json.loads(line)
    for line in
    (ROOT / "dataset/paired_generation_v1/w4en_records.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}


def auroc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


hum_en = [
    r["p_ai"]
    for r in json.loads((ROOT / "reports/human_probe_results_en_human.json").read_text(encoding="utf-8"))
]
hv = np.array(hum_en)

cell = {}
for r in res:
    rec = recs.get(r["id"])
    if not rec:
        continue
    cell.setdefault((r["model"], r["arm"]), []).append(r)

print(f"evaluations: {len(res)} | EN human baseline: flag@0.47 = {(hv >= 0.47).mean():.0%} "
      f"({(hv >= 0.47).sum()}/{len(hv)})")
print(f"\n{'cell':<26}{'n':>4}{'mean p_ai':>11}{'miss@0.47':>11}{'vs-human AUROC':>15}")
report = {"human_baseline": {"n": len(hv), "flag_rate": float((hv >= 0.47).mean())}, "cells": {}}
for (m, a) in sorted(cell):
    xs = np.array([r["p_ai"] for r in cell[(m, a)]])
    miss = float((xs < 0.47).mean())
    au = auroc(xs, hv)
    name = f"{m.split('/')[-1]}-{a}"
    report["cells"][name] = {
        "n": len(xs), "mean": round(float(xs.mean()), 3),
        "miss": round(miss, 3), "auroc_vs_human": round(au, 3),
    }
    print(f"{name:<26}{len(xs):>4}{xs.mean():>11.3f}{miss:>11.0%}{au:>15.3f}")

# Q1: paired topic-level A-B contrast per model (topic-stratified)
print("\n=== Q1: formal contract contrast (A-B, topic-level paired Wilcoxon) ===")

report["contrasts"] = {}
for m in sorted({k[0] for k in cell}):
    by_topic = {}
    for r in res:
        rec = recs.get(r["id"])
        if rec and r["model"] == m and r["arm"] in ("A", "B"):
            by_topic.setdefault((rec["topic_id"], r["arm"]), []).append(r["p_ai"])
    topics_a = {t for t, a in by_topic if a == "A"}
    topics_b = {t for t, a in by_topic if a == "B"}
    common = sorted(topics_a & topics_b)
    if len(common) < 6:
        continue
    diffs = [np.mean(by_topic[(t, "A")]) - np.mean(by_topic[(t, "B")]) for t in common]
    w = wilcoxon(diffs).pvalue if any(d != 0 for d in diffs) else 1.0
    name = f"{m.split('/')[-1]}-A-B"
    report["contrasts"][name] = {
        "n_topics": len(common), "mean_diff": round(float(np.mean(diffs)), 3), "wilcoxon_p": float(w),
    }
    print(f"{m.split('/')[-1]:<18} n_topics={len(common)} diff={np.mean(diffs):+.3f} p={w:.2e}")
    # casual too
    by_topic_c = {}
    for r in res:
        rec = recs.get(r["id"])
        if rec and r["model"] == m and r["arm"] in ("C", "D"):
            by_topic_c.setdefault((rec["topic_id"], r["arm"]), []).append(r["p_ai"])
    ca = {t for t, a in by_topic_c if a == "C"}
    cb = {t for t, a in by_topic_c if a == "D"}
    cc = sorted(ca & cb)
    if len(cc) >= 6:
        diffs_c = [np.mean(by_topic_c[(t, "C")]) - np.mean(by_topic_c[(t, "D")]) for t in cc]
        wc = wilcoxon(diffs_c).pvalue if any(d != 0 for d in diffs_c) else 1.0
        name = f"{m.split('/')[-1]}-C-D"
        report["contrasts"][name] = {
            "n_topics": len(cc), "mean_diff": round(float(np.mean(diffs_c)), 3), "wilcoxon_p": float(wc),
        }
        print(f"{'  casual C-D:':<18} n_topics={len(cc)} diff={np.mean(diffs_c):+.3f} p={wc:.2e}")

out_path = ROOT / "dataset/paired_generation_v1/w4en_analysis.json"
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwritten: dataset/paired_generation_v1/w4en_analysis.json")
