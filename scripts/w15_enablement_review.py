"""W15 enablement review: live end-to-end verification through the real route chain.

Matrix (all through pipeline.detect + _register_caveat + _apply_binoculars_floor
+ _calibrate_confidence, exactly as routes.py orders them):
  1. FN-1 fixture        -> caveat fires; floor MUST NOT fire (bino 0.343); verdict stays Human; conf calibrated
  2. GLM-B contract doc  -> caveat fires; floor MUST fire (bino ~0.6); decision_rule present; latency measured
  3. Human formal doc    -> caveat fires; floor must not fire (bino < 0.46); verdict Human
  4. Casual zh doc       -> no caveat; floor skipped entirely

Usage: uv run python scripts/w15_enablement_review.py
Exit code 0 = all checks PASS.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_paired_experiment import build_pipeline  # noqa: E402

from aigc_detector.api.routes import (  # noqa: E402
    _apply_binoculars_floor,
    _calibrate_confidence,
    _register_caveat,
)


def run_case(pipeline, name: str, text: str, expect: dict) -> bool:
    t0 = time.time()
    result = pipeline.detect(text)
    det_ms = (time.time() - t0) * 1000

    caveat = _register_caveat(text)
    t1 = time.time()
    decision = _apply_binoculars_floor(result, caveat, text, pipeline)
    floor_ms = (time.time() - t1) * 1000
    confidence, calibration = _calibrate_confidence(caveat, result.confidence, result.p_ai)

    bino = (result.breakdown or {}).get("binoculars", {}).get("p_ai")
    print(f"\n--- {name} ---")
    print(f"verdict={result.predicted_label}  p_ai={result.p_ai:.4f}  conf={confidence:.4f}")
    print(f"caveat={'YES' if caveat else 'no'}  bino_stage={bino if bino is None else round(bino, 4)}  "
          f"decision_rule={'FIRED' if decision else 'no'}")
    if calibration:
        print(f"calibration: T={calibration['T']} raw={calibration['confidence_raw']}")
    print(f"latency: detect={det_ms:.0f}ms floor={floor_ms:.0f}ms")

    ok = True
    for key, want in expect.items():
        got = {
            "verdict": result.predicted_label,
            "caveat": bool(caveat),
            "fired": bool(decision),
        }[key]
        if got != want:
            print(f"  CHECK FAIL: {key} = {got}, expected {want}")
            ok = False
    if ok:
        print("  checks: PASS")
    return ok


def main() -> int:
    pipeline = build_pipeline()

    # 1. FN-1 fixture — the honest boundary
    fn1 = (ROOT / "tests/fixtures/fn1_declaration.txt").read_text(encoding="utf-8").strip()
    ok1 = run_case(pipeline, "FN-1 (edited AI text — must stay Human)", fn1,
                   {"verdict": "Human-written", "caveat": True, "fired": False})

    # 2. GLM-B contract doc the ENSEMBLE MISSES (p_ai < 0.47) — the floor's
    # actual job: catch raw contract generation the ensemble dilutes away
    w4c_records_path = ROOT / "dataset/paired_generation_v1/w4c_records.jsonl"
    recs = [
        json.loads(line)
        for line in w4c_records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    w4c_evals_path = ROOT / "dataset/paired_generation_v1/w4c_eval_results.jsonl"
    evals = {
        json.loads(line)["id"]: json.loads(line)
        for line in w4c_evals_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    glm_b_miss = next(
        r for r in recs
        if r["model"] == "THUDM/GLM-4-9B-0414" and r["arm"] == "B" and r["register"] == "formal"
        and evals.get(r["id"], {}).get("predicted_label") != "AI-generated"
    )
    ok2 = run_case(
        pipeline,
        f"GLM-B contract doc MISSED by ensemble ({glm_b_miss['topic_id']}) — floor MUST fire",
        glm_b_miss["text"],
        {"verdict": "AI-generated", "caveat": True, "fired": True},
    )

    # 3. Human formal doc that FIRES the register gate (zibo lawyers' 严正声明)
    zibo_path = ROOT / "dataset/legal_declaration_zh/human/10-association-zibo-lawyers-statement.md"
    zibo = zibo_path.read_text(encoding="utf-8")
    body3 = zibo.split("---", 2)[2].strip()
    ok3 = run_case(pipeline, "Human formal (zibo 严正声明 — gate hits, floor must not fire)", body3,
                   {"verdict": "Human-written", "caveat": True, "fired": False})

    # 4. Casual zh (no caveat -> floor never consulted)
    casual = next(r for r in recs if r["register"] == "casual" and r["model"] == "THUDM/GLM-4-9B-0414")
    ok4 = run_case(pipeline, "Casual zh post (no caveat — floor skipped)", casual["text"],
                   {"caveat": False, "fired": False})

    verdict = all([ok1, ok2, ok3, ok4])
    print(f"\n=== W15 LIVE VERIFICATION: {'PASS' if verdict else 'FAIL'} ===")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
