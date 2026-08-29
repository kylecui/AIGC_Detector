"""AAWM watermark-verification diagnostic stage (third-party integration demo).

Bridges kylecui/acrostic-agent-watermark (AAWM) into the detection
framework's stage contract. When the operator configures AAWM credentials,
every detection response gains watermark evidence alongside the statistical
verdict — the provenance thesis of the paper (W14: "provenance must
migrate from text statistics to process evidence") becomes product code.

Contract role: DIAGNOSTIC (evidence-only, never votes — the watermark
verdict is far MORE authoritative than the statistical one, but requiring
credentials keeps the statistical pipeline unchanged for non-AAWM users).

Config (models/calibration/aawm_stage.json):
  {"enabled": false, "key": "path/to/key.json", "registry": "path/to/reg.json",
   "calibration": "path/to/calibration.json"}

When disabled/missing credentials: neutral result (contract discipline).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

stage_id = "aawm"


def _config() -> dict | None:
    """Load the AAWM stage artifact if enabled."""
    try:
        here = Path(__file__).resolve()
        for base in (here.parents[1] / "models/calibration",):  # wheel: pkg root
            p = base / "aawm_stage.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("enabled"):
                    return data
        # repo layout fallback
        p = here.parents[4] / "models/calibration/aawm_stage.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("enabled"):
                return data
    except (OSError, ValueError):
        pass
    return None


class AAWMStage:
    """Watermark verification via AAWM trace (existence + attribution)."""

    stage_id = stage_id

    def __init__(self) -> None:
        self._loaded = False
        self._wm = None
        self._salt = None

    def load(self) -> None:
        cfg = _config()
        if cfg is None:
            self._loaded = True  # neutral mode: loaded, but no credentials
            return
        try:
            from aawm.plugins import Watermarker

            self._wm = Watermarker.from_config(cfg["key"], cfg.get("registry"))
            self._salt = cfg.get("session_salt")
            self._loaded = True
        except Exception:  # noqa: BLE001 — third-party dep missing -> neutral
            self._wm = None
            self._loaded = True

    def unload(self) -> None:
        self._wm = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str, language: str | None = None) -> dict[str, Any]:
        try:
            if not self.is_loaded:
                self.load()
            if self._wm is None:
                return {"p_ai": 0.5, "label": "Human-written", "confidence": 0.0,
                        "evidence": {"note": "AAWM credentials not configured — "
                                              "watermark verification unavailable (neutral)"}}
            trace = self._wm.trace(text, session_salt=self._salt)
            uid = getattr(trace, "user_id", None) or (trace.get("user_id") if isinstance(trace, dict) else None)
            exists = getattr(trace, "watermark_present", None)
            if exists is None and isinstance(trace, dict):
                exists = trace.get("watermark_present")
            # Watermark present => authoritative AI/agent-generation evidence
            # (far stronger than statistical detection) — reported as evidence
            # with p_ai 0.99; attribution included when decodable.
            if exists:
                return {"p_ai": 0.99, "label": "AI-generated", "confidence": 0.99,
                        "model": "aawm-watermark-trace",
                        "evidence": {"watermark_present": True, "user_id": uid,
                                     "note": "AAWM水印检出：agent级溯源证据（密钥验证），权威性高于统计检测"}}
            return {"p_ai": 0.5, "label": "Human-written", "confidence": 0.0,
                    "model": "aawm-watermark-trace",
                    "evidence": {"watermark_present": False,
                                 "note": "未检出AAWM水印（不排除其他水印方案或无水印AI文本）"}}
        except Exception as e:  # noqa: BLE001 — contract: degrade, never raise
            return {"p_ai": 0.5, "label": "Human-written", "confidence": 0.0,
                    "evidence": {"note": f"stage error: {type(e).__name__}: {e}"}}
