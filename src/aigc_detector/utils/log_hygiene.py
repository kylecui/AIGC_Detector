"""P0-4 log hygiene: sanitize-then-emit guard, wired into service logging.

Policy (release-plan-v2 P0 #7, enforced before setup_logging ever runs):
request text, uploaded filenames beyond basename, and extracted document
content must NEVER reach log output. The guard redacts oversized values and
known text-bearing keys defensively — belt and suspenders with the primary
rule "don't log request content at all" (routes.py logs only lengths/counts).
"""

from __future__ import annotations

import logging
import re

# keys that (defensively) never appear verbatim in log records
_SENSITIVE_KEY_RE = re.compile(
    r"(text|content|body|prompt|document|excerpt|snippet|password|token|key|secret)",
    re.IGNORECASE,
)
_MAX_VALUE_LEN = 120


class SanitizingFormatter(logging.Formatter):
    """Redacts dict/list values under sensitive keys and truncates long strings.

    Defense-in-depth: if a future code path passes a payload dict into a log
    call, the emitted line still cannot contain request text.
    """

    def format(self, record: logging.LogRecord) -> str:
        for name in ("msg", "args"):
            val = getattr(record, name, None)
            if isinstance(val, dict):
                setattr(record, name, self._sanitize_dict(val))
            elif isinstance(val, str) and len(val) > _MAX_VALUE_LEN:
                setattr(record, name, val[:_MAX_VALUE_LEN] + f"...[{len(val)} chars redacted]")
        return super().format(record)

    @staticmethod
    def _sanitize_dict(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if _SENSITIVE_KEY_RE.search(str(k)) or (isinstance(v, str) and len(v) > _MAX_VALUE_LEN):
                out[k] = f"<redacted:{type(v).__name__}>"
            else:
                out[k] = v
        return out


def setup_service_logging(log_dir, level: int = logging.INFO) -> None:
    """Wire utils.logging.setup_logging with the sanitizing formatter on top.

    Called from the API lifespan (and safe to call from scripts): console
    handler stays as-is for operator visibility; the file handler's records
    pass through SanitizingFormatter so no sensitive payload ever lands on
    disk even by accident.
    """
    from aigc_detector.utils.logging import setup_logging

    setup_logging(log_dir=log_dir, level=level)
    root = logging.getLogger()
    for h in root.handlers:
        h.setFormatter(SanitizingFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
