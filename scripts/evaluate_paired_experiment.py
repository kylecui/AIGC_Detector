"""Evaluate paired-generation experiment through the in-process detection pipeline.

W4 of .sisyphus/plans/fn1-countermeasures-and-paired-experiment.md

Loads the exact pipeline the API service uses (replicates api/main.py lifespan
construction, no HTTP), evaluates every record in
dataset/paired_generation_v1/pilot_records.jsonl (checkpointed), then computes
paired statistics (Wilcoxon signed-rank, paired t, Cliff's delta) per stage and
for the ensemble, arm A (free-form) vs arm B (contract-constrained).

Usage:
    uv run python scripts/evaluate_paired_experiment.py [--budget-seconds 540]
    (re-run to resume; evaluated ids are skipped; stats computed when complete)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.detection.pipeline import DetectionPipeline  # noqa: E402

DATA_DIR = Path("dataset/paired_generation_v1")
RECORDS = DATA_DIR / "pilot_records.jsonl"
RESULTS = DATA_DIR / "eval_results.jsonl"
SUMMARY = DATA_DIR / "summary.json"


def build_pipeline(adapter_zh: Path | None = None) -> DetectionPipeline:
    """Assemble via the single declarative entry point (v0.2a).

    Delegates to PlanRunner (plans/default.yaml) — the same assembly the API
    lifespan uses. adapter_zh overrides the production encoder-zh adapter
    (candidate gating).
    """
    from aigc_detector.plan import PlanRunner

    wrapped = PlanRunner.default().build(adapter_zh=adapter_zh).pipeline
    return wrapped._inner if hasattr(wrapped, "_inner") else wrapped


def done_ids() -> set[str]:
    ids: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def paired_stats(pairs: list[tuple[float, float]]) -> dict:
    """pairs: (arm_A, arm_B) per topic. Returns paired test results."""
    import numpy as np
    from scipy import stats as st

    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    d = a - b  # positive => A more AI-detectable than B
    n = len(d)
    out: dict = {
        "n": n,
        "mean_A": float(a.mean()),
        "mean_B": float(b.mean()),
        "mean_diff_A_minus_B": float(d.mean()),
        "n_pos": int((d > 0).sum()),
        "n_neg": int((d < 0).sum()),
        "n_zero": int((d == 0).sum()),
    }
    if n >= 5 and np.any(d != 0):
        w = st.wilcoxon(a, b, zero_method="wilcox")
        out["wilcoxon_p"] = float(w.pvalue)
    if n >= 2:
        t = st.ttest_rel(a, b)
        out["paired_t_p"] = float(t.pvalue)
        out["t_stat"] = float(t.statistic)
        if float(d.std(ddof=1)) > 0:
            out["pratts_dz"] = float(d.mean() / d.std(ddof=1))  # paired effect size
        try:  # sign test on positive direction
            k = out["n_pos"]
            n_nz = n - out["n_zero"]
            if n_nz > 0:
                out["sign_test_p"] = float(
                    min(1.0, 2 * st.binom.cdf(min(k, n_nz - k), n_nz, 0.5))
                )
        except Exception:
            pass
    return out


def analyze() -> bool:
    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in RECORDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    results = {
        json.loads(line)["id"]: json.loads(line)
        for line in RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    by_model_topic: dict[str, dict[str, dict[str, dict]]] = {}
    for rid, res in results.items():
        rec = records.get(rid)
        if not rec:
            continue
        by_model_topic.setdefault(rec["model"], {}).setdefault(rec["topic_id"], {})[
            rec["arm"]
        ] = {
            "p_ai": res["p_ai"],
            **{f"{k}_p_ai": v for k, v in (res.get("stage_p_ai") or {}).items()},
        }

    metrics = {"ensemble_p_ai": "p_ai", "statistical": "statistical_p_ai",
               "linguistic": "linguistic_p_ai", "encoder": "encoder_p_ai",
               "binoculars": "binoculars_p_ai"}
    summary: dict = {"models": {}}
    any_complete = False

    for model, topics in sorted(by_model_topic.items()):
        complete = {t: v for t, v in topics.items() if "A" in v and "B" in v}
        if not complete:
            continue
        any_complete = True
        print(f"\n=== {model} (pairs: {len(complete)}) ===")
        print(f"{'metric':<16}{'mean A':>8}{'mean B':>8}{'A-B':>8}{'wilcoxon p':>11}{'sign p':>9}{'d_z':>8}")
        mstats: dict = {"n_pairs": len(complete), "tests": {}}
        for name, key in metrics.items():
            pairs = [(v["A"][key], v["B"][key]) for v in complete.values()
                     if key in v["A"] and key in v["B"]]
            if not pairs:
                continue
            s = paired_stats(pairs)
            mstats["tests"][name] = s
            wp = f"{s['wilcoxon_p']:.4f}" if "wilcoxon_p" in s else "-"
            sp = f"{s['sign_test_p']:.2e}" if "sign_test_p" in s else "-"
            dz = f"{s['pratts_dz']:+.2f}" if "pratts_dz" in s else "-"
            print(f"{name:<16}{s['mean_A']:>8.4f}{s['mean_B']:>8.4f}{s['mean_diff_A_minus_B']:>+8.4f}{wp:>11}{sp:>9}{dz:>8}"
                  f"  [{s['n_pos']}+/{s['n_neg']}-/{s.get('n_zero', 0)}0]")
        summary["models"][model] = mstats

    if not any_complete:
        print("no complete pairs yet")
        return False
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {SUMMARY}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=540)
    args = ap.parse_args()

    records = [json.loads(line) for line in
               RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = done_ids()
    todo = [r for r in records if r["id"] not in done]
    print(f"records: {len(records)}, evaluated: {len(done)}, pending: {len(todo)}")

    if todo:
        pipeline = build_pipeline()
        print("pipeline ready; evaluating")
        t0 = time.time()
        n = 0
        with RESULTS.open("a", encoding="utf-8") as fh:
            for r in todo:
                if time.time() - t0 > args.budget_seconds:
                    print("budget exhausted; re-run to resume")
                    break
                res = pipeline.detect(r["text"])
                stage_p_ai = {}
                for stage, info in (res.breakdown or {}).items():
                    if isinstance(info, dict) and "p_ai" in info:
                        stage_p_ai[stage] = float(info["p_ai"])
                out = {
                    "id": r["id"],
                    "topic_id": r["topic_id"],
                    "arm": r["arm"],
                    "p_ai": float(res.p_ai),
                    "predicted_label": res.predicted_label,
                    "confidence": float(res.confidence),
                    "detected_language": res.detected_language,
                    "stages_used": list(res.stages_used),
                    "stage_p_ai": stage_p_ai,
                    "processing_time_ms": res.processing_time_ms,
                }
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                print(f"  {r['topic_id']}/{r['arm']} p_ai={res.p_ai:.4f} "
                      f"({res.processing_time_ms:.0f}ms, {', '.join(res.stages_used)})")
        print(f"evaluated {n} records")
    else:
        print("all records evaluated")

    analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
