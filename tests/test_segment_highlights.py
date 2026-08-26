"""W2 unit tests: segment_highlights auxiliary signal.

Covers the pure helper `_segment_highlights` (ordering, top-k truncation,
missing-p_ai tolerance, snippet length) plus schema-level non-breaking
compatibility (field is optional; old payloads without it still validate).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _segment_highlights  # noqa: E402
from aigc_detector.api.schemas import DetectionResponse  # noqa: E402


def _seg(i: int, p_ai: float, text: str = "x" * 120) -> dict:
    return {"index": i, "text": text, "p_ai": p_ai, "predicted_label": "x", "confidence": 0.9}


class TestSegmentHighlights:
    def test_max_and_ordering(self):
        segs = [_seg(0, 0.12), _seg(1, 0.856), _seg(2, 0.31)]
        hl = _segment_highlights(segs)
        assert hl is not None
        assert hl["max_p_ai"] == 0.856
        assert [s["index"] for s in hl["top_k_segments"]] == [1, 2, 0]
        assert hl["n_segments"] == 3

    def test_top_k_truncates_to_three(self):
        segs = [_seg(i, 0.1 * i) for i in range(6)]
        hl = _segment_highlights(segs)
        assert len(hl["top_k_segments"]) == 3
        assert hl["n_segments"] == 6  # n_segments counts ALL segments

    def test_snippet_length_capped(self):
        segs = [_seg(0, 0.9, text="字" * 500)]
        hl = _segment_highlights(segs)
        assert len(hl["top_k_segments"][0]["text_snippet"]) == 80

    def test_empty_and_all_unscored_return_none(self):
        assert _segment_highlights([]) is None
        junk = [{"index": 0, "text": "abc"}]  # no numeric p_ai
        assert _segment_highlights(junk) is None

    def test_single_segment(self):
        hl = _segment_highlights([_seg(0, 0.42)])
        assert hl["max_p_ai"] == 0.42
        assert hl["top_k_segments"][0]["index"] == 0


class TestSchemaBackwardCompat:
    def test_response_without_highlights_still_valid(self):
        """Old payloads (pre-W2) must keep validating: field is optional."""
        resp = DetectionResponse(
            predicted_label="Human-written",
            confidence=0.89,
            p_ai=0.11,
            detected_language="zh",
            stages_used=["encoder"],
            breakdown={},
            segments=[{"index": 0, "p_ai": 0.2}],
        )
        assert resp.segment_highlights is None

    def test_response_with_highlights_valid(self):
        resp = DetectionResponse(
            predicted_label="Human-written",
            confidence=0.89,
            p_ai=0.11,
            detected_language="zh",
            segment_highlights={
                "max_p_ai": 0.856,
                "top_k_segments": [{"index": 7, "p_ai": 0.856, "text_snippet": "…"}],
                "n_segments": 8,
            },
        )
        assert resp.segment_highlights["max_p_ai"] == 0.856
