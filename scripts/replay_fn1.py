"""FN-1 known-bad replay: D5/D3 acceptance for W2.

Loads the in-process pipeline (same weights as the service), detects the
FN-1 fixture (AI-drafted compliance declaration that scored Human-written
0.89 — see DETECTOR_NOTES_2026-08.md), and asserts the 0.856 segment now
surfaces via segment_highlights:

    acceptance: max_p_ai >= 0.8 AND that segment appears in top_k_segments

Also prints the full user-visible output for the D7 end-to-end review
transcript (docs/research). Fixture anchor values are FROZEN by design;
provenance: single failure case recorded 2026-08-17.

Usage:
    uv run python scripts/replay_fn1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _segment_highlights  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "fn1_declaration.txt"


def main() -> int:
    # Reuse the paired-experiment evaluator's in-process pipeline builder
    # (identical construction to api/main.py lifespan; GPU required).
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_paired_experiment import build_pipeline

    text = FIXTURE.read_text(encoding="utf-8").strip()
    print(f"fixture: {FIXTURE.name} ({len(text)} chars)")

    pipeline = build_pipeline()
    print("pipeline ready; detecting FN-1 fixture ...")
    result = pipeline.detect(text)

    # Segment detection (mirrors route logic without HTTP)
    from aigc_detector.api.routes import _detect_segments

    segments, seg_ms = _detect_segments(pipeline, text)
    hl = _segment_highlights(segments)
    assert hl is not None, "no scored segments returned"

    print("\n===== FN-1 REPLAY — user-visible output =====")
    print(f"判定: {result.predicted_label}  置信度: {result.confidence:.4f}  P(AI): {result.p_ai:.4f}")
    print(f"语言: {result.detected_language}  阶段: {', '.join(result.stages_used)}")
    print(f"\n局部AI痕迹（辅助信号）: 最高分段 P(AI) = {hl['max_p_ai']:.4f}  (共 {hl['n_segments']} 段)")
    for s in hl["top_k_segments"]:
        print(f"  Segment #{s['index']}  P(AI)={s['p_ai']:.4f}  {s['text_snippet'][:50]}…")

    ok_max = hl["max_p_ai"] >= 0.8
    ok_visible = any(s["p_ai"] >= 0.8 for s in hl["top_k_segments"])
    print(f"\nacceptance max_p_ai>=0.8: {'PASS' if ok_max else 'FAIL'}")
    print(f"acceptance top_k contains >=0.8 segment: {'PASS' if ok_visible else 'FAIL'}")

    out = {
        "fixture": str(FIXTURE),
        "doc": {"label": result.predicted_label, "confidence": result.confidence, "p_ai": result.p_ai},
        "segment_highlights": hl,
        "segment_time_ms": round(seg_ms, 1),
        "per_segment": [{"index": s["index"], "p_ai": s["p_ai"]} for s in segments],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/fn1_replay_w2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nwritten: reports/fn1_replay_w2.json")
    return 0 if (ok_max and ok_visible) else 1


if __name__ == "__main__":
    sys.exit(main())
