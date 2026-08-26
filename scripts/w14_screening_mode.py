"""W14 continuation: batch-screening mode + per-stage pooling.

Day-2 analyses inside the 2-week time-box, motivated by day-1's findings
(mean-pooling cannot recover evasive cells; DeepSeek inversion makes corpus
VERDICTS confidently wrong):

  A. SCREENING MODE — operational question: for a batch of k same-source docs,
     flag the batch for HUMAN REVIEW if ANY single doc exceeds threshold.
     This renounces classification (which inverted) and asks only "does the
     batch contain at least one catchable doc?" Metrics: batch-flag TPR vs
     batch-flag FPR (a human batch of k is flagged if any member > threshold),
     threshold set from human single-doc 95th percentile.
  B. PER-STAGE POOLING — the ensemble may dilute a stage that still carries
     signal: pool binoculars_zh / statistical_zh per-doc scores separately
     (k-mean) and compute AUROC vs pooled human counterpart scores.

Usage: uv run python scripts/w14_screening_mode.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).parent.parent
B = 2000
RNG = np.random.default_rng(20260819)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-pooling", action="store_true",
                    help="Part B: needs human-side STAGE scores (not yet cached) — "
                         "currently invalid; run only after extracting them.")
    args = ap.parse_args()
    hum_p = np.array([r["p_ai"] for r in json.loads(
        (ROOT / "reports/human_probe_trial_results.json").read_text(encoding="utf-8"))] +
        [r["p_ai"] for r in json.loads(
            (ROOT / "reports/human_probe_results_matched_era.json").read_text(encoding="utf-8"))])

    res = [json.loads(line) for line in (ROOT / "dataset/paired_generation_v1/w4c_eval_results.jsonl")
           .read_text(encoding="utf-8").splitlines() if line.strip()]
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    stage_cells: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
    for r in res:
        key = (r["model"], r["arm"])
        cells.setdefault(key, {}).setdefault(r["topic_id"], []).append(r["p_ai"])
        for st, v in (r.get("stage_p_ai") or {}).items():
            stage_cells.setdefault(key, {}).setdefault(st, {}).setdefault(
                r["topic_id"], []).append(v)

    thr = float(np.percentile(hum_p, 95))  # single-doc threshold from human 95th pct

    print(f"=== A. batch-SCREENING mode (threshold={thr:.3f} = human p95) ===")
    print("(batch flagged if ANY member > thr; renounces classification)")
    print(f"\n{'cell':<28}{'k':>4}{'batchTPR':>10}{'batchFPR':>10}")
    ks = [1, 5, 10, 20]
    screen_report: dict = {}
    for model, arm in [("THUDM/GLM-4-9B-0414", "B"), ("deepseek-ai/DeepSeek-V3.2", "B"),
                       ("Qwen/Qwen3-8B", "B"), ("Qwen/Qwen3-14B", "B"),
                       ("deepseek-ai/DeepSeek-V3.2", "A"), ("THUDM/GLM-4-9B-0414", "A")]:
        topics = cells[(model, arm)]
        tlist = sorted(topics)
        key = f"{model.split('/')[-1]}-{arm}"
        screen_report[key] = {}
        for k in ks:
            tpr = fpr = 0.0
            for _ in range(B):
                pick = RNG.choice(len(tlist), size=min(k, len(tlist)), replace=False)
                ai_pick = [topics[tlist[i]][RNG.integers(len(topics[tlist[i]]))] for i in pick]
                h_pick = RNG.choice(hum_p, size=k, replace=True)
                tpr += max(ai_pick) > thr
                fpr += max(h_pick) > thr
            tpr, fpr = tpr / B, fpr / B
            screen_report[key][k] = {"tpr": round(tpr, 3), "fpr": round(fpr, 3)}
            print(f"{key:<28}{k:>4}{tpr:>10.3f}{fpr:>10.3f}")
        print()

    stage_report: dict = {}
    if args.stage_pooling:
        print("=== B. per-stage k=10 pooling (AI stage vs human ENSEMBLE — descriptive only; "
              "valid version needs cached human stage scores) ===")
        print(f"\n{'cell':<28}{'stage':<14}{'AUROC k=1':>10}{'AUROC k=10':>11}")
        for model, arm in [("THUDM/GLM-4-9B-0414", "B"), ("deepseek-ai/DeepSeek-V3.2", "B"),
                           ("Qwen/Qwen3-14B", "B")]:
            key = f"{model.split('/')[-1]}-{arm}"
            stage_report[key] = {}
            for st in ("statistical", "encoder", "binoculars", "linguistic"):
                sc = stage_cells.get((model, arm), {}).get(st)
                if not sc:
                    continue
                tlist = sorted(sc)
                if len(tlist) < 10:
                    continue
                ai1, h1, ai10, h10 = [], [], [], []
                for _ in range(800):
                    pick = RNG.choice(len(tlist), size=1, replace=False)
                    ai1.append(sc[tlist[pick[0]]][RNG.integers(len(sc[tlist[pick[0]]]))])
                    h1.append(RNG.choice(hum_p))
                    pick = RNG.choice(len(tlist), size=10, replace=False)
                    ai10.append(np.mean([sc[tlist[i]][RNG.integers(len(sc[tlist[i]]))] for i in pick]))
                    h10.append(RNG.choice(hum_p, size=10, replace=True).mean())
                a1 = auroc(np.array(ai1), np.array(h1))
                a10 = auroc(np.array(ai10), np.array(h10))
                stage_report[key][st] = {"auroc_k1": round(a1, 3), "auroc_k10": round(a10, 3)}
                print(f"{key:<28}{st:<14}{a1:>10.3f}{a10:>11.3f}")
            print()

    out = ROOT / "reports/w14_screening_mode.json"
    out.write_text(json.dumps({"screening": screen_report, "stage_pooling": stage_report,
                               "threshold": thr, "B": B}, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
