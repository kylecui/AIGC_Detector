"""Casual-zh human probe evaluation + W17b Variant B FP终验 in one pass.

Evaluates all human casual posts through the in-process pipeline, then:
(a) ensemble flag rate (baseline FPR)
(b) W17b Variant B upgrade-rule FP rate on the NEW set (the enablement gate: <=5%)
(c) W17 caveat FP rate (secondary)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_paired_experiment import build_pipeline  # noqa: E402

ROOT = Path(__file__).parent.parent
HUM = ROOT / "dataset/casual_zh/human"
CKPT = ROOT / "dataset/casual_zh/human_eval.jsonl"


def main() -> int:
    files = sorted(HUM.glob("*.md"))
    done: set[str] = set()
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [f for f in files if f.stem not in done]
    print(f"files: {len(files)}, done: {len(done)}, todo: {len(todo)}")
    if todo:
        pipeline = build_pipeline()
        t0 = time.time()
        with CKPT.open("a", encoding="utf-8") as fh:
            for f in todo:
                if time.time() - t0 > 460:
                    print("budget exhausted; re-run")
                    break
                body = f.read_text(encoding="utf-8").split("---", 2)[2].strip()
                if len(body) < 50:
                    continue
                r = pipeline.detect(body)
                stage = {k: float(v["p_ai"]) for k, v in (r.breakdown or {}).items()
                         if isinstance(v, dict) and "p_ai" in v}
                fh.write(json.dumps({
                    "id": f.stem, "p_ai": float(r.p_ai), "label": r.predicted_label,
                    "conf": float(r.confidence), "stage_p_ai": stage,
                }, ensure_ascii=False) + "\n")
                fh.flush()
        print("re-run if todo remains; now analyzing")
    rows = [json.loads(l) for l in CKPT.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    flagged = sum(1 for r in rows if r["label"] == "AI-generated")
    print(f"\ncasual-human baseline: flag {flagged}/{n} = {flagged/n:.0%}")

    # W17b Variant B FP on this set
    from aigc_detector.detection import register as reg

    cfg = {"band": (0.0047, 0.05), "cv_max": 0.45, "fp_min": 0.5, "img_min": 1.0}
    orig = reg.literary_upgrade_config
    reg.literary_upgrade_config = lambda: cfg
    up_fp = 0
    for f in files:
        body = f.read_text(encoding="utf-8").split("---", 2)[2].strip()
        e = next((r["stage_p_ai"]["encoder"] for r in rows
                  if r["id"] == f.stem and "encoder" in r.get("stage_p_ai", {})), None)
        if e is not None and reg.detect_literary_upgrade(e, body):
            up_fp += 1
    reg.literary_upgrade_config = orig
    print(f"W17b-VariantB upgrade FP: {up_fp}/{n} = {up_fp/n:.1%}  (gate: <=5%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
