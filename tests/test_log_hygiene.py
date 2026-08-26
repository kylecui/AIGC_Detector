"""P0-4 tests: log hygiene guard.

Contract: request text must never reach log output — neither directly (long
strings truncated) nor via payload dicts (sensitive keys redacted). The
guard is defense-in-depth on top of the primary rule (routes log only
lengths/counts), verified here at the formatter level.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aigc_detector.utils.log_hygiene import SanitizingFormatter  # noqa: E402

SECRET_TEXT = "THIS-IS-USER-REQUEST-TEXT-" + "x" * 500


def _emit(fmt: SanitizingFormatter, msg, args=None) -> str:
    rec = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=args, exc_info=None,
    )
    return fmt.format(rec)


class TestSanitizingFormatter:
    def test_long_string_truncated_no_content(self):
        fmt = SanitizingFormatter("%(message)s")
        out = _emit(fmt, SECRET_TEXT)
        assert "x" * 200 not in out
        assert out.startswith("THIS-IS-USER")  # first 120 chars visible, rest redacted
        assert "redacted" in out

    def test_sensitive_dict_keys_redacted(self):
        fmt = SanitizingFormatter("%(message)s")
        payload = {"filename": "doc.pdf", "text": SECRET_TEXT, "chars": 1234}
        out = _emit(fmt, payload)
        assert SECRET_TEXT[:200] not in out
        assert "<redacted:str>" in out
        assert "1234" in out  # non-sensitive fields survive

    def test_nested_content_key_redacted(self):
        fmt = SanitizingFormatter("%(message)s")
        out = _emit(fmt, {"document_content": "hello", "n": 1})
        assert "hello" not in out

    def test_normal_short_messages_untouched(self):
        fmt = SanitizingFormatter("%(message)s")
        out = _emit(fmt, "File upload: 'doc.pdf' (1024 bytes, .pdf)")
        assert out == "File upload: 'doc.pdf' (1024 bytes, .pdf)"
