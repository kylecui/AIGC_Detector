"""API middleware: rate limiting and error handling.

References:
    - DESIGN.md §5.3 (rate limiting, OOM handling)
    - DEVPLAN.md Phase 4 task 4.5
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Rate limiter: keyed by client IP
limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate-limit violations."""
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

    logger.info("Middleware configured: rate limiter + OOM handler")
