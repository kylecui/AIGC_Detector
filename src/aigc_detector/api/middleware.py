"""API middleware: rate limiting, error handling, request IDs, auth, CORS.

References:
    - DESIGN.md §5.3 (rate limiting, OOM handling)
    - DEVPLAN.md Phase 4 task 4.5

Hardening additions (v0.2):
    - ``RequestIDMiddleware``: propagates/generates X-Request-ID and binds it
      into a contextvar so the SanitizingFormatter can append ``[rid=...]``.
    - ``APIKeyAuthMiddleware``: optional constant-time API-key check on the
      detect endpoints (``settings.api_key``); everything else is exempt.
    - CORS via ``settings.cors_origins`` (comma-separated; empty = disabled).
    - 429 responses increment the Prometheus counters.
"""

from __future__ import annotations

import contextvars
import hmac
import logging
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aigc_detector.api.metrics import metrics_registry
from aigc_detector.config import settings

logger = logging.getLogger(__name__)

# Rate limiter: keyed by client IP
limiter = Limiter(key_func=get_remote_address)


# ======================================================================
# Request-ID context (item 5)
# ======================================================================

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("aigc_request_id", default=None)

_RID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


def set_request_id(rid: str) -> contextvars.Token:
    """Bind *rid* into the logging context (used by middleware + tests)."""
    return _request_id_var.set(rid)


def get_request_id() -> str | None:
    """Current request id, or None outside a request context."""
    return _request_id_var.get()


def _sanitize_request_id(raw: str) -> str:
    cleaned = _RID_SANITIZE_RE.sub("", raw.strip())[:64]
    return cleaned or uuid.uuid4().hex[:12]


class RequestIDMiddleware:
    """Echo or generate X-Request-ID; bind it to the logging contextvar."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = _sanitize_request_id(Headers(scope=scope).get("x-request-id", ""))
        token = _request_id_var.set(rid)

        async def send_with_rid(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["x-request-id"] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_with_rid)
        finally:
            _request_id_var.reset(token)


# ======================================================================
# API-key auth (item 3)
# ======================================================================

# Protect-list: ONLY these paths require X-API-Key when settings.api_key is
# set. Health/readiness/metrics/docs/static stay open by design.
_PROTECTED_PATHS = ("/api/v1/detect", "/api/v1/detect/file")


class APIKeyAuthMiddleware:
    """Optional constant-time API-key guard for the detect endpoints."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            expected = settings.api_key  # read at request time (env/test friendly)
            if expected and scope.get("path", "") in _PROTECTED_PATHS:
                provided = Headers(scope=scope).get("x-api-key", "")
                ok = bool(provided) and hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
                if not ok:
                    response = JSONResponse(status_code=401, content={"detail": "invalid api key"})
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


_auth_disabled_notice_logged = False


def log_auth_disabled_once() -> None:
    """One-time startup warning when auth is off (api_key unset)."""
    global _auth_disabled_notice_logged
    if not settings.api_key and not _auth_disabled_notice_logged:
        logger.warning("AUTH DISABLED (api_key not set)")
        _auth_disabled_notice_logged = True


# ======================================================================
# Exception handlers
# ======================================================================


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate-limit violations (counts into metrics)."""
    endpoint = "detect_file" if request.url.path.endswith("/detect/file") else "detect"
    metrics_registry.inc_request(endpoint, 429)
    logger.warning("Rate limit exceeded for %s", get_remote_address(request))
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


async def oom_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Handle CUDA out-of-memory errors gracefully."""
    error_msg = str(exc)
    if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
        logger.error("CUDA OOM error: %s", error_msg)
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU out of memory. Please try again later or use shorter text."},
        )
    # Re-raise non-OOM RuntimeErrors
    raise exc


def setup_middleware(app: FastAPI) -> None:
    """Register all middleware and exception handlers on the FastAPI app."""
    # SlowAPI rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # CUDA OOM handler
    app.add_exception_handler(RuntimeError, oom_error_handler)

    # Middleware stack (Starlette: last added = outermost).
    # Execution order: CORS -> RequestID -> Auth -> routes, so error
    # responses (401) still carry CORS + request-id headers.
    app.add_middleware(APIKeyAuthMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Optional CORS (empty setting = disabled, same-origin only)
    raw_origins = (settings.cors_origins or "").strip()
    if raw_origins:
        origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.info("CORS enabled for %d origin(s)", len(origins))

    logger.info("Middleware configured: rate limiter + OOM handler + request-id + auth + cors")
