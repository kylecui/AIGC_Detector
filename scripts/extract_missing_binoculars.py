"""Force-run the zh binoculars detector on the 40 human docs lacking scores.

Completes the same-side comparison: are the ensemble-FLAGGED hard humans
(incl. CCB 0.987) also low-binoculars, or do they invade the AI range?
Usage: uv run python scripts/extract_missing_binoculars.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_paired_experiment import build_pipeline  # noqa: E402

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "reports/human_stage_scores.jsonl"
OUT = ROOT / "reports/human_binoculars_forced.jsonl"


def main() -> int:
    rows = [
        json.loads(line)
        for line in CKPT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    have = {r["file"] for r in rows if "binoculars" in r["stage_p_ai"]}
    all_docs = {}
    for sub, tag in [("dataset/legal_declaration_zh/human", "zh_main"),
                     ("dataset/legal_declaration_zh/human_matched_era", "zh_matched")]:
        for f in sorted((ROOT / sub).glob("*.md")):
            parts = f.read_text(encoding="utf-8").split("---", 2)
            body = parts[2].strip() if len(parts) == 3 else ""
            if len(body) > 50:
                all_docs[f.name] = (tag, body)
    missing = [(fn, *all_docs[fn]) for fn in all_docs if fn not in have]
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["file"])
    todo = [m for m in missing if m[0] not in done]
    print(f"missing binoculars: {len(missing)}, already forced: {len(done)}, todo: {len(todo)}")
    if not todo:
        return 0

    pipeline = build_pipeline()
    bino = pipeline.binoculars_detectors.get("zh")
    if bino is None:
        print("ERROR: zh binoculars detector unavailable")
        return 1
    if not bino.is_loaded:  # property, not method
        print("loading binoculars-zh (Qwen2 pair, ~80s)...")
        bino.load()
    with OUT.open("a", encoding="utf-8") as fh:
        for fn, tag, body in todo:
            try:
                p_ai = float(bino.compute_score(body))
            except Exception as e:  # noqa: BLE001 — record and continue
                print(f"  {fn[:46]:<48} FAILED: {e}")
                continue
            rec = {"file": fn, "slice": tag, "binoculars_p_ai": p_ai}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {fn[:46]:<48} bino={p_ai:.4f}")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
