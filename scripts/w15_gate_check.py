"""W15 formal gate check: W12 dual-layer gate on the enablement decision.

L1 disaster guard: FPR(enabled) vs FPR(baseline) + 25pp.
L2 paired test: per-doc verdict-change asymmetry on the 82-doc human probe
   (McNemar-style sign test on discordant pairs; floor is one-directional
   Human->AI, so the paired test reduces to: are the NEW flags numerous
   enough to be systematic? — plus median p_ai shift).

Uses the exact OR-rule semantics of the deployed code
(flag if ensemble >= 0.47 OR binoculars >= 0.46) with the completed
82-doc binoculars coverage.

Usage: uv run python scripts/w15_gate_check.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent

CUTOFF = 0.46
THR = 0.47


def wilson(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    # human side: ensemble + binoculars (merged 82)
    ens = {}
    for src in ["reports/human_probe_trial_results.json", "reports/human_probe_results_matched_era.json"]:
        for r in json.loads((ROOT / src).read_text(encoding="utf-8")):
            ens[r["file"]] = r["p_ai"]
    stage = [
        json.loads(line)
        for line in (ROOT / "reports/human_stage_scores.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bino = {r["file"]: r["stage_p_ai"]["binoculars"] for r in stage if "binoculars" in r["stage_p_ai"]}
    thr_raw = 0.8536432310785527
    forced = [
        json.loads(line)
        for line in (ROOT / "reports/human_binoculars_forced.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for r in forced:
        ratio = r["binoculars_p_ai"] / thr_raw
        bino[r["file"]] = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(5.0 * (ratio - 1.0)))))
    assert len(bino) == 82

    base_flags = {f for f, p in ens.items() if p >= THR}
    new_flags = {f for f, p in ens.items() if (p >= THR) or (bino.get(f, 0) >= CUTOFF)}
    fp_base, fp_new = len(base_flags), len(new_flags)
    added = new_flags - base_flags

    lo, hi = wilson(fp_new, 82)
    l1_pass = (fp_new / 82) <= (fp_base / 82) + 0.25

    # L2: discordant pairs (only Human->AI direction possible)
    n_added = len(added)
    n_removed = 0
    # sign test: under H0 (random direction), P(all n_added discordants one-sided)
    # one-directional rule -> trivially all additions; test whether count is
    # large relative to binomial(82, small) — report exact one-sided binomial
    # p for observing >= n_added flips under p_flip = base flip propensity
    # (conservative: p_flip estimated as 0 — the honest statement is the
    # count itself + the FPR delta CI)
    from scipy.stats import binom

    p_l2 = float(binom.sf(n_added - 1, 82, 0.02)) if n_added else 1.0  # vs 2% accidental-fire propensity

    print("=== W12 dual-layer gate: W15 enablement ===")
    print(f"L1 disaster: FPR {fp_base}/82 ({fp_base/82:.1%}) -> {fp_new}/82 ({fp_new/82:.1%}) "
          f"[Wilson {lo:.1%}, {hi:.1%}]  +25pp threshold -> {'PASS' if l1_pass else 'FAIL'}")
    print(f"L2 paired: verdict flips Human->AI on humans: {sorted(added)} (n={n_added}, removed={n_removed})")
    print(f"   new-flag files: {[f for f in sorted(added)]}")
    print(f"   one-sided binomial p vs 2% propensity: {p_l2:.3f} -> "
          f"{'PASS (not systematic)' if p_l2 > 0.05 else 'FAIL (systematic)'}")
    verdict = "PASS" if l1_pass and p_l2 > 0.05 else "FAIL"
    print(f"\nGATE VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
