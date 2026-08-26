"""W4c evaluation: run in-process pipeline over w4c_records.jsonl (checkpointed).

Same weights as the service. Resumes via w4c_eval_results.jsonl.

Usage: uv run python scripts/w4c_evaluate.py [--budget-seconds 470]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_paired_experiment import build_pipeline  # noqa: E402

DATA = Path("dataset/paired_generation_v1")
RECORDS = DATA / "w4c_records.jsonl"
RESULTS = DATA / "w4c_eval_results.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=470)
    ap.add_argument("--records", default="dataset/paired_generation_v1/w4c_records.jsonl")
    ap.add_argument("--results", default="dataset/paired_generation_v1/w4c_eval_results.jsonl")
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done: set[str] = set()
    results_path = Path(args.results)
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [r for r in records if r["id"] not in done]
    print(f"records: {len(records)}, evaluated: {len(done)}, pending: {len(todo)}")

    if not todo:
        print("all evaluated")
        return 0

    pipeline = build_pipeline()
    print("pipeline ready")
    t0 = time.time()
    n = 0
    with results_path.open("a", encoding="utf-8") as fh:
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
                "register": r["register"],
                "arm": r["arm"],
                "model": r["model"],
                "seed": r["seed"],
                "p_ai": float(res.p_ai),
                "predicted_label": res.predicted_label,
                "confidence": float(res.confidence),
                "stage_p_ai": stage_p_ai,
                "stages_used": list(res.stages_used),
                "processing_time_ms": res.processing_time_ms,
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(todo)} evaluated")
    print(f"evaluated {n} this run (total {len(done) + n}/{len(records)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
