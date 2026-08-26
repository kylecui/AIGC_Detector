"""W5 trial-batch evaluation: run the detection pipeline on human formal docs.

Evaluates every markdown file in dataset/legal_declaration_zh/human/ through
the in-process pipeline (same weights as the service) and reports per-doc
p_ai/label — the trial question: how many REAL human formal documents does
the detector flag as AI (trial FPR preview, n=10, descriptive only)?

Usage:
    uv run python scripts/eval_human_probe.py [--dir dataset/legal_declaration_zh/human]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_paired_experiment import build_pipeline  # noqa: E402

HEADER_RE = re.compile(r"^---\s*$(.*?)^---\s*$", re.S | re.M)


def extract_body(md_text: str) -> str:
    """Strip the YAML-ish header block; return the document body text."""
    m = HEADER_RE.search(md_text)
    body = md_text[m.end():] if m else md_text
    return body.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dataset/legal_declaration_zh/human")
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint/results filenames (e.g. matched_era)")
    ap.add_argument("--budget-seconds", type=int, default=480,
                    help="Stop when exceeded; re-run resumes (checkpointed JSONL).")
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    files = sorted(Path(args.dir).glob("*.md"))
    if not files:
        print(f"no .md files in {args.dir}")
        return 1
    ckpt = Path(f"reports/human_probe_eval_checkpoint{tag}.jsonl")
    out = Path(f"reports/human_probe_results{tag or '_trial'}.json")
    done: set[str] = set()
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["file"])
    todo = [f for f in files if f.name not in done]
    print(f"docs: {len(files)}, evaluated: {len(done)}, pending: {len(todo)}")

    rows: list[dict] = []
    if todo:
        pipeline = build_pipeline()
        print("pipeline ready")
        t0 = time.time()
        with ckpt.open("a", encoding="utf-8") as fh:
            for f in todo:
                if time.time() - t0 > args.budget_seconds:
                    print("budget exhausted; re-run to resume")
                    break
                body = extract_body(f.read_text(encoding="utf-8"))
                if len(body) < 50:
                    print(f"  SKIP {f.name}: body only {len(body)} chars (header parse?)")
                    continue
                res = pipeline.detect(body)
                row = {
                    "file": f.name,
                    "chars": len(body),
                    "label": res.predicted_label,
                    "p_ai": round(res.p_ai, 4),
                    "confidence": round(res.confidence, 4),
                    "stages": list(res.stages_used),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                print(
                    f"  {f.name[:44]:<46} {res.predicted_label:<14} "
                    f"p_ai={res.p_ai:.4f} conf={res.confidence:.3f}"
                )
    else:
        print("all docs evaluated")

    # aggregate over checkpoint (complete picture)
    all_rows = [
        json.loads(line)
        for line in ckpt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    flagged = sum(1 for r in all_rows if r["label"] == "AI-generated")
    n = len(all_rows)
    if n:
        print(f"\nresult so far: {flagged}/{n} human formal docs flagged as AI "
              f"(flag rate = {flagged / n:.0%}, n={n})")
    out = Path(f"reports/human_probe_results{tag or '_trial'}.json")
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {out} (n={n}/{len(files)}; re-run if pending remain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
