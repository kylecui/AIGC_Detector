"""W1 regression test: segment construction and per-segment detection contract.

Guards against the class of bug suspected in FN-1 analysis (later shown to be
a test-script key error, but the contract is worth pinning): segments must
carry an integer `index` ascending from 0, plus text/char_start/char_end, and
per-segment detection must propagate pipeline outputs into each segment dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.api.routes import _build_segments, _detect_segments  # noqa: E402

ZH_TEXT = (
    "人工智能技术的发展正在改变信息传播的格局。基于大语言模型的内容生成能力显著提升，"
    "信息真实性验证面临新的挑战。构建可靠的检测系统尤为重要。"
    "本文从统计特征、语言学建模与深度语义理解三个维度展开阐述。"
    "实验表明多模型集成框架在多语言环境上表现稳定。"
    "该工作为网络信息生态治理提供了技术支撑。"
    "未来将进一步扩展到更多文体与语言。"
)


class _StubDetection:
    def __init__(self, p_ai: float):
        self.predicted_label = "ai-generated" if p_ai >= 0.5 else "human-written"
        self.confidence = 0.9
        self.p_ai = p_ai
        self.detected_language = "zh"
        self.stages_used = ["statistical"]
        self.breakdown = {"statistical": {"p_ai": p_ai}}
        self.processing_time_ms = 1.0


class _StubPipeline:
    def detect(self, text: str) -> _StubDetection:
        return _StubDetection(0.3)


class TestBuildSegments:
    def test_segments_have_ascending_integer_index(self):
        segs = _build_segments(ZH_TEXT)
        assert segs, "expected at least one segment"
        indexes = [s["index"] for s in segs]
        assert indexes == list(range(len(segs))), f"index not ascending: {indexes}"
        for s in segs:
            assert isinstance(s["index"], int)
            assert s["text"]
            assert isinstance(s["char_start"], int) and isinstance(s["char_end"], int)

    def test_segment_offsets_cover_text_without_overlap(self):
        segs = _build_segments(ZH_TEXT)
        for prev, cur in zip(segs, segs[1:]):
            assert prev["char_end"] <= cur["char_start"], "segments overlap"

    def test_short_text_yields_single_segment(self):
        segs = _build_segments("这是一段很短的文本。")
        assert len(segs) == 1
        assert segs[0]["index"] == 0

    def test_max_segments_bound(self):
        long_text = "这是一个用于测试的句子，长度足够。" * 100
        segs = _build_segments(long_text)
        assert len(segs) <= 8


class TestDetectSegments:
    def test_detection_propagates_to_segments(self):
        segs, elapsed = _detect_segments(_StubPipeline(), ZH_TEXT)
        assert isinstance(elapsed, float) and elapsed >= 0
        assert segs, "expected segments"
        for s in segs:
            assert s["predicted_label"] == "human-written"
            assert s["p_ai"] == 0.3
            assert "index" in s and isinstance(s["index"], int)
            assert "processing_time_ms" in s
