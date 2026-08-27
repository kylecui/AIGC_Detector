"""Evaluate the literary-prose AI records through the in-process pipeline.

Checkpointed into dataset/literary_prose_zh/eval_results.jsonl.
Usage: uv run python scripts/literary_prose_eval.py [--budget-seconds 470]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_paired_experiment import build_pipeline  # noqa: E402

REC = Path("dataset/literary_prose_zh/ai_records.jsonl")
RES = Path("dataset/literary_prose_zh/eval_results.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=470)
    args = ap.parse_args()

    records = [json.loads(l) for l in REC.read_text(encoding="utf-8").splitlines() if l.strip()]
    done: set[str] = set()
    if RES.exists():
        for line in RES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [r for r in records if r["id"] not in done]
    print(f"records: {len(records)}, evaluated: {len(done)}, pending: {len(todo)}")
    if not todo:
        print("all evaluated")
        return 0

    pipeline = build_pipeline()
    t0 = time.time()
    n = 0
    with RES.open("a", encoding="utf-8") as fh:
        for r in todo:
            if time.time() - t0 > args.budget_seconds:
                print("budget exhausted; re-run to resume")
                break
            res = pipeline.detect(r["text"])
            stage = {k: float(v["p_ai"]) for k, v in (res.breakdown or {}).items()
                     if isinstance(v, dict) and "p_ai" in v}
            fh.write(json.dumps({
                "id": r["id"], "model": r["model"], "topic": r["topic_id"], "seed": r["seed"],
                "p_ai": float(res.p_ai), "label": res.predicted_label,
                "conf": float(res.confidence), "stage_p_ai": stage,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
    print(f"evaluated {n} this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
