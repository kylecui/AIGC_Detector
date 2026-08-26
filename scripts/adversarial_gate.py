"""W7/W12 automated promotion gate for adversarial candidates.

Evaluates a candidate encoder-zh adapter (models/encoder-zh-adversarial-candidate)
against the production baseline using the W5 probe set + FN-1 replay, then
emits PASS/FAIL. Production adapter is NEVER touched by this script; promotion
(on PASS) is a human copy action printed at the end.

Gate design (v3.1 W12 — dual-layer, replacing the +5pp single-layer screen
that was shown statistically incoherent at n=62: ~84% pass at true baseline,
17% pass even at true FPR=30%):

  L1 DISASTER GUARD  : candidate FPR point-estimate > baseline + 25pp -> FAIL
                       (no statistics needed for egregious regressions)
  L2 PAIRED TEST     : Wilcoxon signed-rank + sign test on per-doc p_ai
                       differences (candidate − production) over the SAME 62
                       human docs — the paired design recovers power that
                       point-estimate comparison loses at small n.
                       FAIL if paired test shows significant p_ai INCREASE
                       (one-sided, alpha=0.05) AND median shift > +0.05.
                       (Direction: higher p_ai on human docs = worse FPR.)
  Legacy +5pp point screen is REPORTED but non-gating.

Report-only: G3 formality coupling, FN-1 replay, per-family rates.

Usage:
    uv run python scripts/adversarial_gate.py [--candidate models/encoder-zh-adversarial-candidate]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
BASELINE_FPR = 12 / 62  # reports/probe_baseline_2026-08.md
LEGACY_TOLERANCE = 0.05    # old +5pp screen — reported, non-gating
DISASTER_MARGIN = 0.25     # L1: candidate FPR > baseline + 25pp -> FAIL
PAIRED_ALPHA = 0.05        # L2 one-sided significance
PAIRED_MEDIAN_MARGIN = 0.05  # L2: median p_ai shift must also exceed this


def dual_layer_gate(cand_p_ai: dict[str, float], prod_p_ai: dict[str, float],
                    cand_fpr: float) -> dict:
    """W12 dual-layer gate on joined per-doc p_ai maps (same docs, two systems).

    Pure CPU; used both live (after candidate eval) and offline (archived
    reports) — see tmp/gate_selftest.py for the synthetic self-test.
    """
    from scipy.stats import wilcoxon

    common = sorted(set(cand_p_ai) & set(prod_p_ai))
    diffs = [cand_p_ai[f] - prod_p_ai[f] for f in common]

    # L1 disaster guard
    l1_fail = cand_fpr > BASELINE_FPR + DISASTER_MARGIN
    l1 = {"layer": "L1_disaster_guard",
          "cand_fpr": cand_fpr, "threshold": BASELINE_FPR + DISASTER_MARGIN,
          "verdict": "FAIL" if l1_fail else "PASS"}

    # L2 paired test (one-sided: candidate p_ai INCREASE = harm)
    l2: dict = {"layer": "L2_paired_test", "n_pairs": len(diffs)}
    if len(diffs) >= 10 and any(d != 0 for d in diffs):

        stat = wilcoxon(diffs, alternative="greater")
        l2["wilcoxon_p_one_sided"] = float(stat.pvalue)
        sign_pos = sum(1 for d in diffs if d > 0)
        sign_neg = sum(1 for d in diffs if d < 0)
        l2["n_pos"], l2["n_neg"] = sign_pos, sign_neg
        n_nz = sign_pos + sign_neg
        if n_nz:
            from scipy.stats import binom

            l2["sign_test_p_one_sided"] = float(
                min(1.0, binom.cdf(min(sign_pos, sign_neg), n_nz, 0.5))
                if sign_neg < sign_pos else
                min(1.0, 2 * binom.cdf(sign_neg, n_nz, 0.5))
            )
        diffs_sorted = sorted(diffs)
        m = len(diffs_sorted)
        l2["median_shift"] = (diffs_sorted[m // 2] if m % 2
                              else (diffs_sorted[m // 2 - 1] + diffs_sorted[m // 2]) / 2)
        sig = min(l2.get("wilcoxon_p_one_sided", 1.0),
                  l2.get("sign_test_p_one_sided", 1.0))
        l2_fail = (sig < PAIRED_ALPHA) and (l2["median_shift"] > PAIRED_MEDIAN_MARGIN)
        l2["verdict"] = "FAIL" if l2_fail else "PASS"
    else:
        l2["verdict"] = "PASS"  # insufficient pairs — gate cannot fire; report only

    verdict = "FAIL" if (l1_fail or l2.get("verdict") == "FAIL") else "PASS"
    return {
        "verdict": verdict,
        "l1": l1, "l2": l2,
        "legacy_screen_report_only": {
            "cand_fpr": cand_fpr,
            "threshold": BASELINE_FPR + LEGACY_TOLERANCE,
            "would_have_passed": cand_fpr <= BASELINE_FPR + LEGACY_TOLERANCE,
        },
        "n_joined": len(common),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="models/encoder-zh-adversarial-candidate")
    args = ap.parse_args()
    cand = Path(args.candidate)
    if not (cand / "adapter_config.json").exists():
        print(f"no candidate adapter at {cand}")
        return 1

    from evaluate_paired_experiment import build_pipeline

    from aigc_detector.training.adversarial import formality_score

    pipeline = build_pipeline(adapter_zh=cand)
    print("pipeline ready (candidate encoder)")

    # --- G1+report: probe set ---
    hum_dir = ROOT / "dataset/legal_declaration_zh/human"
    rows = []
    for f in sorted(hum_dir.glob("*.md")):
        parts = f.read_text(encoding="utf-8").split("---", 2)
        body = parts[2].strip() if len(parts) == 3 else ""
        if len(body) < 50:
            continue
        res = pipeline.detect(body)
        rows.append({"file": f.name, "label": res.predicted_label,
                     "p_ai": res.p_ai, "formality": formality_score(body)})
    n = len(rows)
    flagged = sum(1 for r in rows if r["label"] == "AI-generated")
    fpr = flagged / n

    # --- W12 dual-layer gate (replaces single-layer G1) ---
    prod = json.loads((ROOT / "reports/human_probe_trial_results.json").read_text(encoding="utf-8"))
    prod_p = {r["file"]: r["p_ai"] for r in prod}
    gate_result = dual_layer_gate({r["file"]: r["p_ai"] for r in rows}, prod_p, fpr)
    g1 = gate_result["verdict"] == "PASS"
    print(f"\nL1 disaster guard: cand FPR {fpr:.1%} vs threshold "
          f"{gate_result['l1']['threshold']:.1%} -> {gate_result['l1']['verdict']}")
    l2 = gate_result["l2"]
    if "wilcoxon_p_one_sided" in l2:
        print(f"L2 paired test: n={l2['n_pairs']} pos/neg={l2['n_pos']}/{l2['n_neg']} "
              f"median_shift={l2['median_shift']:+.4f} "
              f"wilcoxon_p={l2['wilcoxon_p_one_sided']:.2e} -> {l2['verdict']}")
    print(f"legacy +5pp screen (report-only): would_have_passed="
          f"{gate_result['legacy_screen_report_only']['would_have_passed']}")

    # --- G3: formality coupling (report vs baseline needs both; candidate-only here) ---
    from scipy.stats import spearmanr
    rho, _ = spearmanr([r["p_ai"] for r in rows], [r["formality"] for r in rows])
    print(f"G3 candidate |Spearman(p_ai, formality)| = {abs(rho):.3f} "
          f"(compare vs production run of this gate; lower is better)")

    # --- G2: FN-1 replay ---
    from aigc_detector.api.routes import _detect_segments, _segment_highlights
    fn1 = (ROOT / "tests/fixtures/fn1_declaration.txt").read_text(encoding="utf-8").strip()
    segments, _ = _detect_segments(pipeline, fn1)
    hl = _segment_highlights(segments)
    g2 = hl is not None and hl["max_p_ai"] >= 0.8
    doc = pipeline.detect(fn1)
    print(f"G2 FN-1 replay: doc={doc.predicted_label} conf={doc.confidence:.3f}; "
          f"max seg p_ai={hl['max_p_ai'] if hl else None} -> {'PASS' if g2 else 'FAIL'}")

    verdict = "PASS" if (g1 and g2) else "FAIL"
    out = ROOT / f"reports/adversarial_gate_{date.today():%Y-%m-%d}.json"
    out.write_text(json.dumps({
        "candidate": str(cand), "verdict": verdict,
        "gate": gate_result,
        "g1_probe_fpr": fpr, "baseline_fpr": BASELINE_FPR,
        "g3_formality_spearman": float(rho),
        "g2_fn1_max_seg": hl["max_p_ai"] if hl else None,
        "fn1_doc": {"label": doc.predicted_label, "conf": doc.confidence},
        "per_doc": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}  (report: {out})")
    if verdict == "PASS":
        print("PROMOTION (human action): copy candidate dir over models/encoder-zh after review,")
        print("  e.g.:  Copy-Item -Recurse -Force models/encoder-zh-adversarial-candidate/* models/encoder-zh/")
        print("  then re-run: uv run pytest tests/ -q  &&  scripts/defensibility_report.py")
    else:
        print("Candidate REJECTED — production adapter untouched. Keep report for ADR trail.")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
