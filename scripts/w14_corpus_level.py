"""W14 (2-week time-box, day 1): corpus-level detection via k-sample pooling.

Chakraborty et al. (arXiv:2304.04736): multi-sample pooling from the same
generator restores separability via Chernoff information even when single-doc
distributions overlap. Operational question for our formal register: if a
reviewer receives a BATCH of k documents from one unknown source (all human
or all from one model+arm), does the mean detector score separate sources
better than single-doc?

Design (CPU-only, cached scores):
- Human corpus pool: 62 main-store + 20 matched-era = 82 human formal docs
- AI corpus pools: each (model, arm) cell from W4c (100 docs = 20 topics x 5 seeds)
- For k in {1,2,5,10,20}: bootstrap B=2000 corpus scores per side
  (AI side: topic-stratified sampling — k distinct topics, one random seed
  each — avoids within-topic correlation inflating pooling gains)
- Metrics: AUROC(corpus scores) + TPR@FPR5% (threshold = 95th pct human)
- Verdict rule (pre-set): pooling "recovers" a cell if TPR@FPR5% >= 0.8 at
  some k<=10 while k=1 TPR < 0.5

Usage: uv run python scripts/w14_corpus_level.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).parent.parent
B = 2000
RNG = np.random.default_rng(20260818)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def wilson_p95(scores: np.ndarray) -> float:
    return float(np.percentile(scores, 95))


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> int:
    # --- pools ---
    human = [r["p_ai"] for r in json.loads(
        (ROOT / "reports/human_probe_trial_results.json").read_text(encoding="utf-8"))]
    human += [r["p_ai"] for r in json.loads(
        (ROOT / "reports/human_probe_results_matched_era.json").read_text(encoding="utf-8"))]
    h = np.array(human)

    res = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_eval_results.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}  # (model,arm)->topic->[p_ai]
    for r in res:
        cells.setdefault((r["model"], r["arm"]), {}).setdefault(r["topic_id"], []).append(r["p_ai"])

    focus = [
        ("THUDM/GLM-4-9B-0414", "B"), ("deepseek-ai/DeepSeek-V3.2", "B"),
        ("Qwen/Qwen3-8B", "B"), ("Qwen/Qwen3-14B", "B"),
        ("THUDM/GLM-4-9B-0414", "A"), ("deepseek-ai/DeepSeek-V3.2", "A"),
    ]
    ks = [1, 2, 5, 10, 20]
    report: dict = {"B": B, "human_pool_n": len(h), "cells": {}}

    print(f"human pool: {len(h)} docs | pooling bootstrap B={B} (topic-stratified)")
    print("pooling variants: mean-p_ai and mean-logit (evidence-sum)")
    print(f"\n{'cell':<28}{'k':>4}{'AUROC(p)':>10}{'TPR(p)':>8}{'AUROC(lg)':>11}{'TPR(lg)':>9}")

    for model, arm in focus:
        topics = cells[(model, arm)]
        tlist = sorted(topics)
        key = f"{model.split('/')[-1]}-{arm}"
        report["cells"][key] = {}
        for k in ks:
            ai_p = np.empty(B)
            h_p = np.empty(B)
            ai_lg = np.empty(B)  # mean of PER-DOC logits (evidence sum), not logit of mean
            h_lg = np.empty(B)
            for b in range(B):
                # topic-stratified: k distinct topics, one random seed each
                pick_t = RNG.choice(len(tlist), size=min(k, len(tlist)), replace=False)
                picks = [topics[tlist[i]][RNG.integers(len(topics[tlist[i]]))] for i in pick_t]
                hp = RNG.choice(h, size=k, replace=True)
                ai_p[b] = np.mean(picks)
                h_p[b] = np.mean(hp)
                ai_lg[b] = np.mean(to_logit(np.array(picks)))
                h_lg[b] = np.mean(to_logit(hp))
            au = auroc(ai_p, h_p)
            thr = wilson_p95(h_p)
            tpr = float((ai_p > thr).mean())
            au_lg = auroc(ai_lg, h_lg)
            thr_lg = wilson_p95(h_lg)
            tpr_lg = float((ai_lg > thr_lg).mean())
            report["cells"][key][k] = {"auroc": round(au, 3), "tpr@fpr5": round(tpr, 3),
                                       "auroc_logit": round(au_lg, 3), "tpr@fpr5_logit": round(tpr_lg, 3)}
            print(f"{key:<28}{k:>4}{au:>10.3f}{tpr:>8.3f}{au_lg:>11.3f}{tpr_lg:>9.3f}")
        r1 = report["cells"][key][1]["tpr@fpr5"]
        best = max(report["cells"][key][k]["tpr@fpr5"] for k in ks)
        best_lg = max(report["cells"][key][k]["tpr@fpr5_logit"] for k in ks)
        rec_k = [k for k in ks if max(report["cells"][key][k]["tpr@fpr5"],
                                      report["cells"][key][k]["tpr@fpr5_logit"]) >= 0.8]
        verdict = ("RECOVERED at k=" + str(min(rec_k)) if rec_k and max(r1, best if r1 > 0.5 else 0) < 0.5
                   else "recovered (already good at k=1)" if rec_k
                   else f"NOT recovered by k=20 (best TPR p={best:.2f}/logit={best_lg:.2f})")
        report["cells"][key]["verdict"] = verdict
        print(f"{'':<28}-> {verdict}\n")

    out = ROOT / "reports/w14_corpus_level.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
