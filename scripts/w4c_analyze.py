"""W4c analysis: 2x2 decomposition + capability dose-response + cluster-robust SE.

Reads dataset/paired_generation_v1/w4c_{records,eval_results}.jsonl.
Three pre-registered hypotheses (write-up BEFORE seeing data is in the plan;
this script computes all three regardless of outcome — negative results are
results):

  H1 (replication)    : arm B < arm A on formal register, per model, with
                        seed-level dispersion now visible (5 seeds/topic)
  H2 (THE new cell)   : arm D vs arm C on casual register
                        - D ~ C (no evasion) -> formality necessary co-factor
                        - D << C             -> contract-specific gate
  H3 (dose-response)  : arm-B mean p_ai monotone non-increasing across the
                        capability ladder [7B, 8B, 9B, 14B, MoE]

Inference: topic-level clustering (20 topics per cell is the effective n for
between-topic claims); report cluster-robust SEs via topic-block bootstrap,
plus per-topic paired Wilcoxon as sensitivity.

Usage: uv run python scripts/w4c_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

DATA = Path("dataset/paired_generation_v1")
RESULTS = DATA / "w4c_eval_results.jsonl"
OUT = DATA / "w4c_analysis.json"

LADDER = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "THUDM/GLM-4-9B-0414",
          "Qwen/Qwen3-14B", "deepseek-ai/DeepSeek-V3.2"]


def topic_bootstrap_ci(cell_p_ai: dict[str, list[float]], n_boot: int = 2000,
                       seed: int = 7) -> tuple[float, float, float]:
    """Mean + topic-block bootstrap 95% CI (resample topics, not seeds)."""
    topics = list(cell_p_ai)
    if not topics:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(topics), size=len(topics), replace=True)
        vals = [v for i in pick for v in cell_p_ai[topics[i]]]
        means.append(float(np.mean(vals)))
    all_vals = [v for vs in cell_p_ai.values() for v in vs]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(all_vals)), float(lo), float(hi)


def paired_topic_diff(cell_x: dict[str, list[float]],
                      cell_y: dict[str, list[float]]) -> dict:
    """Per-topic mean difference x - y, then topic-level Wilcoxon + bootstrap."""
    from scipy.stats import wilcoxon

    common = sorted(set(cell_x) & set(cell_y))
    diffs = [np.mean(cell_x[t]) - np.mean(cell_y[t]) for t in common]
    out = {"n_topics": len(common),
           "mean_diff": float(np.mean(diffs)) if diffs else float("nan")}
    if len(diffs) >= 6 and any(d != 0 for d in diffs):
        out["wilcoxon_p_two_sided"] = float(wilcoxon(diffs).pvalue)
    rng = np.random.default_rng(11)
    # topic-level bootstrap of the mean diff (resample topics, NOT permutation —
    # permutation preserves the multiset and hence the mean, degenerating the CI)
    diffs_arr = np.array(diffs, dtype=float)
    boots = [float(np.mean(rng.choice(diffs_arr, size=len(diffs_arr), replace=True)))
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["boot_ci95"] = [float(lo), float(hi)]
    return out


def main() -> int:
    res = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # cell_p_ai[(register, arm)][model][topic] = [p_ai per seed]
    cell: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
    for r in res:
        key = (r["register"], r["arm"])
        cell.setdefault(key, {}).setdefault(r["model"], {}).setdefault(r["topic_id"], []).append(r["p_ai"])

    report: dict = {"n_total": len(res), "cells": {}}

    print(f"loaded {len(res)} evaluations")
    print(f"\n{'cell':<18}{'model':<22}{'n':>4}{'mean':>8}{'95% CI (topic-boot)':>22}")
    for key in sorted(cell):
        reg, arm = key
        for m in LADDER:
            if m not in cell[key]:
                continue
            mean, lo, hi = topic_bootstrap_ci(cell[key][m])
            n = sum(len(v) for v in cell[key][m].values())
            report["cells"][f"{reg}|{arm}|{m}"] = {"n": n, "mean": mean, "ci95": [lo, hi]}
            print(f"{reg+'-'+arm:<18}{m.split('/')[-1]:<22}{n:>4}{mean:>8.3f}   [{lo:.3f}, {hi:.3f}]")

    # ---- H1/H2: per-model arm contrasts within register ----
    report["contrasts"] = {}
    for reg, (x_arm, y_arm, tag) in [("formal", ("A", "B", "H1-replication")),
                                     ("casual", ("C", "D", "H2-formality-vs-contract"))]:
        print(f"\n=== {tag}: {x_arm} vs {y_arm} ({reg}) ===")
        print(f"{'model':<22}{'Δ(x-y)':>8}{'wilcoxon p':>12}{'perm CI95':>20}")
        for m in LADDER:
            kx, ky = (reg, x_arm), (reg, y_arm)
            if m not in cell.get(kx, {}) or m not in cell.get(ky, {}):
                continue
            d = paired_topic_diff(cell[kx][m], cell[ky][m])
            report["contrasts"][f"{tag}|{m}"] = d
            wp = f"{d.get('wilcoxon_p_two_sided', float('nan')):.2e}"
            ci = d.get("boot_ci95", [float("nan")] * 2)
            print(f"{m.split('/')[-1]:<22}{d['mean_diff']:>+8.3f}{wp:>12}   "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # ---- H3: dose-response on arm B (formal-contract) ----
    b_means = []
    print("\n=== H3 dose-response (formal-contract arm B, ladder order) ===")
    for m in LADDER:
        mean, lo, hi = topic_bootstrap_ci(cell.get(("formal", "B"), {}).get(m, {}))
        b_means.append((m, mean, lo, hi))
        print(f"  {m.split('/')[-1]:<22} {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    pts = [m for _, m, _, _ in b_means if not np.isnan(m)]
    monotone = all(pts[i] >= pts[i + 1] - 0.02 for i in range(len(pts) - 1))
    # Spearman of ladder position vs mean
    from scipy.stats import spearmanr

    xs = [i for i, (_, m, _, _) in enumerate(b_means) if not np.isnan(m)]
    rho, p = spearmanr(xs, pts)
    report["h3_dose_response"] = {
        "means": [{"model": m, "mean": mu, "ci": [lo, hi]} for m, mu, lo, hi in b_means],
        "monotone_nonincreasing_tol02": bool(monotone),
        "spearman_rho": float(rho), "spearman_p": float(p),
    }
    print(f"  monotone(non-increasing, tol .02): {monotone}; spearman rho={rho:+.2f} p={p:.3f}")

    # ---- H2 verdict summary (the decisive readout) ----
    d_contrasts = {k.split("/")[-1]: d for k, d in report["contrasts"].items()
                   if k.startswith("H2")}
    if d_contrasts:
        above = [m for m, d in d_contrasts.items() if d["mean_diff"] > 0.1
                 and d.get("wilcoxon_p_two_sided", 1) < 0.05]
        print(f"\nH2 decisive readout: casual-register contract evasion (C-D>0.1, p<0.05) in models: {above or 'NONE'}")
        report["h2_verdict"] = {
            "models_with_casual_evasion": above,
            "interpretation": ("contract-specific gate (evasion without formality)"
                               if above else
                               "formality necessary co-factor (no casual evasion) — "
                               "consistent with Perplexity Trap C1-C3"),
        }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
