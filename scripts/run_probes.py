"""ProbeKit v0.2c: execute the probes declared in a DetectionPlan.

Each plan probe is a deployment-behavior contract. This runner executes them
and emits PASS/FAIL/SKIP per probe (machine-readable report to reports/).
GPU probes are skipped (recorded as SKIP) when --cpu-only or no CUDA device
is present — a SKIP never passes a release; the verify gate treats
"SKIP on a GPU probe" as a blocking warning for releases, acceptable for
config-edit smoke runs.

Usage:
    uv run python scripts/run_probes.py [--plan plans/default.yaml] [--cpu-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402


def probe_fn1_known_bad(spec: dict, root: Path) -> tuple[str, str]:
    from aigc_detector.api.routes import _detect_segments, _segment_highlights
    from aigc_detector.plan import PlanRunner

    fixture = (root / spec["fixture"]).read_text(encoding="utf-8").strip()
    bundle = PlanRunner.default().build()
    result = bundle.pipeline.detect(fixture)
    want_label = spec["asserts"]["verdict"]
    if result.predicted_label != want_label:
        return "FAIL", f"verdict={result.predicted_label} expected {want_label}"
    segments, _ = _detect_segments(bundle.pipeline, fixture)
    hl = _segment_highlights(segments)
    if hl is None or hl["max_p_ai"] < spec["asserts"]["max_segment_p_ai_gte"]:
        mx = hl["max_p_ai"] if hl else None
        return "FAIL", f"max segment p_ai={mx} < {spec['asserts']['max_segment_p_ai_gte']}"
    return "PASS", f"verdict={result.predicted_label}, max_seg={hl['max_p_ai']:.4f}"


def probe_gate_coverage(spec: dict, root: Path, gate: str, lo: int, hi: int | None, files_min: int) -> tuple[str, str]:
    from aigc_detector.detection.register import detect_register_en_formal, detect_register_zh

    detect = detect_register_zh if gate == "zh" else detect_register_en_formal
    d = root / spec["dataset"]
    files = sorted(d.glob("*.md"))
    if len(files) < files_min:
        return "FAIL", f"dataset too small: {len(files)} < {files_min}"
    hits = 0
    for f in files:
        body = f.read_text(encoding="utf-8").split("---", 2)[2]
        if len(body) > 50:
            r = detect(body)
            hit = r.is_formal_zh if hasattr(r, "is_formal_zh") else r[0]
            hits += bool(hit)
    if hits < lo:
        return "FAIL", f"coverage regressed: {hits}/{len(files)} < {lo}"
    if hi is not None and hits > hi:
        return "FAIL", f"gate over-firing: {hits}/{len(files)} > {hi}"
    return "PASS", f"{hits}/{len(files)} in [{lo}, {hi if hi is not None else 'inf'}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="plans/default.yaml")
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(__file__).parent.parent
    results: dict[str, dict] = {}

    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_gpu = False

    for spec in plan.get("probes", []):
        pid, req = spec["id"], spec.get("requires", "cpu")
        t0 = time.time()
        if req == "gpu" and (args.cpu_only or not has_gpu):
            results[pid] = {"status": "SKIP", "reason": "gpu unavailable or --cpu-only", "seconds": 0}
            print(f"SKIP  {pid}  (gpu)")
            continue
        try:
            if pid == "fn1-known-bad":
                status, detail = probe_fn1_known_bad(spec, root)
            elif pid == "zh-formal-gate-coverage":
                e = spec["expects"]
                status, detail = probe_gate_coverage(spec, root, "zh", e["formal_hit_subset_gte"], None, e["files_gte"])
            elif pid == "en-formal-gate-coverage":
                e = spec["expects"]
                status, detail = probe_gate_coverage(
                    spec, root, "en", e["lexical_hit_gte"], e.get("lexical_hit_lte"), e["files_gte"])
            else:
                status, detail = "SKIP", f"unknown probe id: {pid}"
        except Exception as e:  # noqa: BLE001 — probe errors are probe failures
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        results[pid] = {"status": status, "detail": detail, "seconds": round(time.time() - t0, 1)}
        print(f"{status:<5} {pid}  {detail}  ({results[pid]['seconds']}s)")

    n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
    verdict = "FAIL" if n_fail else "PASS"
    out = root / f"reports/probes-{date.today():%Y-%m-%d}.json"
    out.write_text(json.dumps({"plan": plan.get("plan"), "verdict": verdict, "probes": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPROBES {verdict} ({n_fail} failed) -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
