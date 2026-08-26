"""W3b reopening analysis: binoculars-floor gate — full-probe threshold sweep.

Gate design under evaluation: in formal_zh register, raise the effective
score to max(ensemble, binoculars_floor) where binoculars_floor is a fixed
cutoff on the binoculars stage's p_ai. Sweep the cutoff over the complete
probe (82 human with full binoculars coverage + all W4c formal AI cells +
FN-1 anchor) and report per-cutoff: human FPR / AI-formal miss rates per
cell / FN-1 status.

D4 discipline: all rates with Wilson CIs; no FPR@1% claims at this n.

Usage: uv run python scripts/w3b_floor_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def wilson(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    # human side: ensemble p_ai + binoculars p_ai (merged 82)
    hum_ens = {}
    for src in ["reports/human_probe_trial_results.json",
                "reports/human_probe_results_matched_era.json"]:
        for r in json.loads((ROOT / src).read_text(encoding="utf-8")):
            hum_ens[r["file"]] = r["p_ai"]
    stage = [json.loads(l) for l in (ROOT / "reports/human_stage_scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    h_bino = {r["file"]: r["stage_p_ai"]["binoculars"] for r in stage if "binoculars" in r["stage_p_ai"]}
    THR = 0.8536432310785527
    for r in [json.loads(l) for l in (ROOT / "reports/human_binoculars_forced.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]:
        ratio = r["binoculars_p_ai"] / THR
        h_bino[r["file"]] = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(5.0 * (ratio - 1.0)))))
    assert len(h_bino) == 82, f"coverage {len(h_bino)}"

    # AI formal cells with both ensemble + binoculars
    res = [json.loads(l) for l in (ROOT / "dataset/paired_generation_v1/w4c_eval_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cells = {}
    for r in res:
        if r["register"] != "formal":
            continue
        b = (r.get("stage_p_ai") or {}).get("binoculars")
        if b is None:
            continue
        key = f"{r['model'].split('/')[-1]}-{r['arm']}"
        cells.setdefault(key, []).append((r["p_ai"], b))

    FN1_ENS, FN1_BINO = 0.1091, 0.3431

    cutoffs = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
    print(f"{'cutoff':>7}{'humanFPR':>10}{'[CI95]':>16}", end="")
    for k in sorted(cells):
        print(f"{k:>14}", end="")
    print(f"{'FN-1':>7}")

    rows_out = []
    for cut in cutoffs:
        fp = sum(1 for f, b in h_bino.items() if max(hum_ens.get(f, 0.0), cut if b >= cut else 0.0) >= 0.47
                 or (b >= cut))  # floor fires => flagged
        # simpler: gate flags if ensemble>=0.47 OR binoculars>=cut
        fp = sum(1 for f, b in h_bino.items()
                 if hum_ens.get(f, 0.0) >= 0.47 or b >= cut)
        lo, hi = wilson(fp, 82)
        line = f"{cut:>7.2f}{fp/82:>9.0%}  [{lo:.0%},{hi:.0%}]"
        miss_cells = {}
        for k, pairs in sorted(cells.items()):
            miss = sum(1 for e, b in pairs if e < 0.47 and b < cut)
            miss_cells[k] = miss / len(pairs)
            line += f"{miss / len(pairs):>13.0%}"
        fn1 = "CAUGHT" if FN1_BINO >= cut else "miss"
        line += f"{fn1:>8}"
        rows_out.append({"cutoff": cut, "human_fp": fp, "human_fpr": fp / 82,
                         "ci95": [lo, hi], "cell_miss": miss_cells, "fn1": fn1})
        print(line)

    (ROOT / "reports/w3b_floor_analysis.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten: reports/w3b_floor_analysis.json")
    print("\nNote: gate = flag if ensemble>=0.47 OR binoculars-stage>=cutoff (formal_zh only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
