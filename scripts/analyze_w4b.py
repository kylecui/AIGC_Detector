"""W4b replication analysis: paired stats on replication topics only (t41..t60).

Reuses paired_stats from evaluate_paired_experiment. Reads
dataset/paired_generation_v1/{pilot_records.jsonl, eval_results.jsonl},
filters to replication topic ids, computes per-model paired stats for
ensemble and each stage, writes dataset/paired_generation_v1/summary_w4b.json.

Usage:
    uv run python scripts/analyze_w4b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_paired_experiment import paired_stats  # noqa: E402

DATA_DIR = Path("dataset/paired_generation_v1")
RECORDS = DATA_DIR / "pilot_records.jsonl"
RESULTS = DATA_DIR / "eval_results.jsonl"
OUT = DATA_DIR / "summary_w4b.json"

REP_TOPICS = {f"t{i}" for i in range(41, 61)}

METRICS = {
    "ensemble_p_ai": "p_ai",
    "statistical": "statistical_p_ai",
    "linguistic": "linguistic_p_ai",
    "encoder": "encoder_p_ai",
    "binoculars": "binoculars_p_ai",
}


def main() -> int:
    records = {}
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["id"]] = r
    results = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            results[r["id"]] = r

    # model -> topic -> arm -> metric values (replication topics only)
    by_model_topic: dict[str, dict[str, dict[str, dict]]] = {}
    n_short = 0
    for rid, rec in records.items():
        if rec["topic_id"] not in REP_TOPICS:
            continue
        res = results.get(rid)
        if not res:
            continue
        if rec["char_len"] < 200:
            n_short += 1
        by_model_topic.setdefault(rec["model"], {}).setdefault(rec["topic_id"], {})[
            rec["arm"]
        ] = {
            "p_ai": res["p_ai"],
            **{f"{k}_p_ai": v for k, v in (res.get("stage_p_ai") or {}).items()},
        }

    summary: dict = {
        "experiment": "W4b replication (t41-t60, fresh sampling)",
        "spec_version": "v1.2-replication",
        "n_replication_records": sum(
            len(arms) for m in by_model_topic.values() for arms in m.values()
        ),
        "n_short_outputs_lt200": n_short,
        "models": {},
    }

    for model, topics in sorted(by_model_topic.items()):
        complete = {t: v for t, v in topics.items() if "A" in v and "B" in v}
        print(f"\n=== {model} (replication pairs: {len(complete)}) ===")
        print(f"{'metric':<16}{'mean A':>8}{'mean B':>8}{'A-B':>8}"
              f"{'wilcoxon p':>11}{'sign p':>9}{'d_z':>8}")
        mstats: dict = {"n_pairs": len(complete), "tests": {}}
        for name, key in METRICS.items():
            pairs = [(v["A"][key], v["B"][key]) for v in complete.values()
                     if key in v["A"] and key in v["B"]]
            if not pairs:
                continue
            s = paired_stats(pairs)
            mstats["tests"][name] = s
            wp = f"{s['wilcoxon_p']:.4f}" if "wilcoxon_p" in s else "-"
            sp = f"{s['sign_test_p']:.2e}" if "sign_test_p" in s else "-"
            dz = f"{s['pratts_dz']:+.2f}" if "pratts_dz" in s else "-"
            print(f"{name:<16}{s['mean_A']:>8.4f}{s['mean_B']:>8.4f}"
                  f"{s['mean_diff_A_minus_B']:>+8.4f}{wp:>11}{sp:>9}{dz:>8}"
                  f"  [{s['n_pos']}+/{s['n_neg']}-/{s.get('n_zero', 0)}0]"
                  f"  (n={s['n']})")
        summary["models"][model] = mstats

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
