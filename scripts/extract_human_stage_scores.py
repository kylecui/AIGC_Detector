"""Extract per-STAGE scores for the human formal probe (82 docs).

Adjudicates the W14 day-2 binoculars hint: AI evasive cells' binoculars stage
scores pool to AUROC ~1.0 *descriptively* (vs human ENSEMBLE scores — invalid
cross-side). The missing comparison is binoculars-vs-binoculars. This pass
records every stage score the pipeline actually produces for human docs.

Usage: uv run python scripts/extract_human_stage_scores.py [--budget-seconds 460]
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

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "reports/human_stage_scores.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=460)
    args = ap.parse_args()

    docs = []
    for sub, tag in [("dataset/legal_declaration_zh/human", "zh_main"),
                     ("dataset/legal_declaration_zh/human_matched_era", "zh_matched")]:
        for f in sorted((ROOT / sub).glob("*.md")):
            parts = f.read_text(encoding="utf-8").split("---", 2)
            body = parts[2].strip() if len(parts) == 3 else ""
            if len(body) > 50:
                docs.append({"file": f.name, "slice": tag, "body": body})

    done: set[str] = set()
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["file"])
    todo = [d for d in docs if d["file"] not in done]
    print(f"docs: {len(docs)}, done: {len(done)}, pending: {len(todo)}")
    if not todo:
        print("all extracted")
        return 0

    pipeline = build_pipeline()
    print("pipeline ready")
    t0 = time.time()
    n = 0
    with CKPT.open("a", encoding="utf-8") as fh:
        for d in todo:
            if time.time() - t0 > args.budget_seconds:
                print("budget exhausted; re-run to resume")
                break
            res = pipeline.detect(d["body"])
            stage = {k: float(v["p_ai"]) for k, v in (res.breakdown or {}).items()
                     if isinstance(v, dict) and "p_ai" in v}
            rec = {"file": d["file"], "slice": d["slice"], "p_ai": float(res.p_ai),
                   "label": res.predicted_label, "stages_used": list(res.stages_used),
                   "stage_p_ai": stage}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
    print(f"extracted {n} this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
